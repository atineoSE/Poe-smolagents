import os
import re

import psutil
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, models, tool, utils

load_dotenv()


# Function to override framework parsing. It matches the LAST code block only.
# This effectively prevents the framework from tripping on thinking tokens
# where the model is drafting the code block.
def extract_last_code_from_text(
    text: str, code_block_tags: tuple[str, str]
) -> str | None:
    """Extract code from the LLM's output.

    Returns the *last* code block whose opening tag appears at the beginning
    of a line.
    """
    # Revised pattern to match only if the code block starts at the beginning of the line
    pattern = (
        rf"(?m)^{re.escape(code_block_tags[0])}(.*?){re.escape(code_block_tags[1])}"
    )
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        # Revised logic to return just the last match
        return matches[-1].strip()
    return None


utils.extract_code_from_text = extract_last_code_from_text

# Monkey patch the function which describes whether model supports stop parameters
# to always return False.
# This effectively prevents the Poe API call from failing on some models,
# e.g. DeepSeek-v3.1
models.supports_stop_parameter = lambda model_id: False

model = OpenAIServerModel(
    model_id=os.environ.get("MODEL", ""),
    api_base="https://api.poe.com/v1",
    api_key=os.environ.get("API_KEY"),
)


@tool
def get_system_info() -> dict:
    """
    Collects basic system metrics.

    Returns:
        dict: A dictionary with the following keys:
            - ``cpu_percent`` (float): Current CPU usage percentage.
            - ``memory_total_mb`` (int): Total physical memory in megabytes.
            - ``memory_used_mb`` (int): Amount of physical memory currently used (MiB).
            - ``memory_percent`` (float): Percentage of physical memory in use.
            - ``swap_total_mb`` (int): Total swap space in megabytes.
            - ``swap_used_mb`` (int): Amount of swap currently used (MiB).
            - ``swap_percent`` (float): Percentage of swap space in use.

            All numeric values are derived from ``psutil`` and are rounded down
            to the nearest integer where applicable.
    """
    cpu_percent = psutil.cpu_percent(interval=1)
    virtual_mem = psutil.virtual_memory()
    swap_mem = psutil.swap_memory()

    return {
        "cpu_percent": cpu_percent,
        "memory_total_mb": virtual_mem.total // (1024 * 1024),
        "memory_used_mb": virtual_mem.used // (1024 * 1024),
        "memory_percent": virtual_mem.percent,
        "swap_total_mb": swap_mem.total // (1024 * 1024),
        "swap_used_mb": swap_mem.used // (1024 * 1024),
        "swap_percent": swap_mem.percent,
    }


if __name__ == "__main__":
    from smolagents import CodeAgent

    agent = CodeAgent(tools=[get_system_info], model=model, verbosity_level=2)

    # Agent "hello world"
    agent.run("Tell me about you", max_steps=3)

    # Tool use "hello world"
    agent.run("Tell me the system specs", max_steps=3)
