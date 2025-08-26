import os

from dotenv import load_dotenv
from smolagents import OpenAIServerModel, models

load_dotenv()

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
