import os
import re

from dotenv import load_dotenv
from smolagents import OpenAIServerModel, models, utils

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
