import os
import re

from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, models, utils

load_dotenv()


# Function to override framework parsing. It matches the LAST code block only.
# This effectively prevents the framework from tripping on thinking tokens
# where the model is drafting the code block.
def extract_last_code_from_text(
    text: str, code_block_tags: tuple[str, str]
) -> str | None:
    """Extract code from the LLM's output."""
    pattern = rf"{code_block_tags[0]}(.*?){code_block_tags[1]}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Override: return just the last match
    return None


utils.extract_code_from_text = extract_last_code_from_text

# Monkey patch the function which describes whether model supports stop parameters
# to always return False.
# This effectively prevents the Poe API call from failing on some models,
# e.g. DeepSeek-v3.1
models.supports_stop_parameter = lambda model_id: False

model = OpenAIServerModel(
    model_id=os.environ.get("MODEL", ""),
    api_base=os.environ.get("API_BASE_URL"),
    api_key=os.environ.get("API_KEY"),
)

if __name__ == "__main__":
    from smolagents import CodeAgent

    agent = CodeAgent(tools=[], model=model, verbosity_level=2)
    agent.run("Tell me about you", max_steps=3)
