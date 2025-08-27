import os

from dotenv import load_dotenv
from smolagents import OpenAIServerModel, models

load_dotenv()

# Monkey patch the function which describes whether model supports stop parameters
# to always return False.
# This effectively prevents the Poe API call from failing on some models,
# e.g. DeepSeek-v3.1
models.supports_stop_parameter = lambda model_id: False

model_id = os.environ.get("MODEL")
if not model_id:
    raise ValueError("Could not find MODEL variable in the environment")
api_key = os.environ.get("API_KEY")
if not api_key:
    raise ValueError("Could not find API_KEY variable in the environment")

model = OpenAIServerModel(
    model_id=model_id, api_base="https://api.poe.com/v1", api_key=api_key
)
