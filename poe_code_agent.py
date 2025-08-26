import json
import re
from collections.abc import Generator
from typing import Any

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text
from smolagents import (
    CODEAGENT_RESPONSE_FORMAT,
    ActionOutput,
    ActionStep,
    AgentExecutionError,
    AgentGenerationError,
    AgentParsingError,
    ChatMessage,
    ChatMessageStreamDelta,
    CodeAgent,
    LogLevel,
    ToolCall,
    ToolOutput,
    agglomerate_stream_deltas,
    fix_final_answer_code,
    parse_code_blobs,
    truncate_content,
)


def extract_code_from_text(text: str, code_block_tags: tuple[str, str]) -> str | None:
    """Extract code from the LLM's output.

    Returns the *last* code block whose opening tag appears at the beginning
    of a line.
    """
    # UPDATED: Revised pattern to match only if the code block starts at the beginning of the line
    # It helps for thinking models prefixing with >
    # BEFORE:
    # pattern = rf"{code_block_tags[0]}(.*?){code_block_tags[1]}"
    # AFTER:
    pattern = (
        rf"(?m)^{re.escape(code_block_tags[0])}(.*?){re.escape(code_block_tags[1])}"
    )
    # -------

    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        # UPDATED: Revised logic to return just the last match
        # It helps for thinking models, which draft the code block
        # BEFORE:
        # return "\n\n".join(match.strip() for match in matches)
        # AFTER:
        return matches[-1].strip()
        # -------
    return None


def extract_internal_structure_text(text: str) -> str:
    """
    Return the substring starting with the first occurrence of
    ``{\\s*"thought":``. .
    """
    # Find the start of the target JSON object
    start_match = re.search(r'\{\s*"thought"\s*:', text)
    if not start_match:
        return text

    # Slice from the opening brace to the end of the string
    return text[start_match.start() :]


class PoeCodeAgent(CodeAgent):
    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()
        ### Generate model output ###
        memory_step.model_input_messages = input_messages
        stop_sequences = ["Observation:", "Calling tools:"]
        if self.code_block_tags[1] not in self.code_block_tags[0]:
            # If the closing tag is contained in the opening tag, adding it as a stop sequence would cut short any code generation
            stop_sequences.append(self.code_block_tags[1])
        try:
            additional_args: dict[str, Any] = {}
            if self._use_structured_outputs_internally:
                additional_args["response_format"] = CODEAGENT_RESPONSE_FORMAT
            if self.stream_outputs:
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=stop_sequences,
                    **additional_args,
                )
                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                with Live(
                    "", console=self.logger.console, vertical_overflow="visible"
                ) as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        live.update(
                            Markdown(
                                agglomerate_stream_deltas(
                                    chat_message_stream_deltas
                                ).render_as_markdown()
                            )
                        )
                        yield event
                chat_message = agglomerate_stream_deltas(chat_message_stream_deltas)
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
            else:
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=stop_sequences,
                    **additional_args,
                )
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
                self.logger.log_markdown(
                    content=output_text,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            if not self._use_structured_outputs_internally:
                # This adds the end code sequence (i.e. the closing code block tag) to the history.
                # This will nudge subsequent LLM calls to finish with this end code sequence, thus efficiently stopping generation.
                if output_text and not output_text.strip().endswith(
                    self.code_block_tags[1]
                ):
                    output_text += self.code_block_tags[1]
                    memory_step.model_output_message.content = output_text

            memory_step.token_usage = chat_message.token_usage
            memory_step.model_output = output_text
        except Exception as e:
            raise AgentGenerationError(
                f"Error in generating model output:\n{e}", self.logger
            ) from e

        ### Parse output ###
        try:
            if self._use_structured_outputs_internally:
                # UPDATED: add a pre-processing step to extract the json data.
                # It helps with thinking models, which add their thoughts before
                # the structured output.
                # BEFORE:
                # code_action = json.loads(output_text)["code"]
                # AFTER:
                pre_processed_output_text = extract_internal_structure_text(output_text)
                code_action = json.loads(pre_processed_output_text)["code"]
                # -------
                code_action = (
                    extract_code_from_text(code_action, self.code_block_tags)
                    or code_action
                )
            else:
                code_action = parse_code_blobs(output_text, self.code_block_tags)
            code_action = fix_final_answer_code(code_action)
            memory_step.code_action = code_action
        except Exception as e:
            error_msg = (
                f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            )
            raise AgentParsingError(error_msg, self.logger)

        tool_call = ToolCall(
            name="python_interpreter",
            arguments=code_action,
            id=f"call_{len(self.memory.steps)}",
        )
        yield tool_call
        memory_step.tool_calls = [tool_call]

        ### Execute action ###
        self.logger.log_code(
            title="Executing parsed code:", content=code_action, level=LogLevel.INFO
        )
        try:
            code_output = self.python_executor(code_action)
            execution_outputs_console = []
            if len(code_output.logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(code_output.logs),
                ]
            observation = "Execution logs:\n" + code_output.logs
        except Exception as e:
            if (
                hasattr(self.python_executor, "state")
                and "_print_outputs" in self.python_executor.state
            ):
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(
                        Group(*execution_outputs_console), level=LogLevel.INFO
                    )
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(code_output.output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        if not code_output.is_final_answer:
            execution_outputs_console += [
                Text(
                    f"Out: {truncated_output}",
                ),
            ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = code_output.output
        yield ActionOutput(
            output=code_output.output, is_final_answer=code_output.is_final_answer
        )
