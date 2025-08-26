import os
import traceback

from dotenv import load_dotenv
from smolagents import LiteLLMModel, OpenAIServerModel

load_dotenv()

if __name__ == "__main__":
    from smolagents import CodeAgent

    # OpenAI model -- works
    model = OpenAIServerModel(
        model_id="GPT-5",
        api_base="https://api.poe.com/v1",
        api_key=os.environ.get("API_KEY"),
    )
    agent = CodeAgent(tools=[], model=model, verbosity_level=2)
    agent.run("Tell me about you", max_steps=3)

    # DeepSeek model as OpenAI compatible model
    # Fails with error code: 500: Internal server error
    try:
        model = OpenAIServerModel(
            model_id="DeepSeek-v3.1",
            api_base="https://api.poe.com/v1",
            api_key=os.environ.get("API_KEY"),
        )
        agent = CodeAgent(tools=[], model=model, verbosity_level=2)
        agent.run("Tell me about you", max_steps=3)
    except Exception as e:
        print(
            f"DeepSeek model as OpenAI compatible model failed with error: {str(e)}.\n{traceback.format_exc()}"
        )

    # DeepSeek model as LiteLLModel
    # Fails with litellm.InternalServerError: InternalServerError: OpenAIException - Internal server error
    try:
        model = LiteLLMModel(
            model_id="openai/DeepSeek-v3.1",
            api_base="https://api.poe.com/v1",
            api_key=os.environ.get("API_KEY"),
            num_ctx=8192,
        )
        agent = CodeAgent(tools=[], model=model, verbosity_level=2)
        agent.run("Tell me about you", max_steps=3)
    except Exception as e:
        print(
            f"DeepSeek model as LiteLLMModel failed with error: {str(e)}.\n{traceback.format_exc()}"
        )


