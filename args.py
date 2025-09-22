import argparse

parser = argparse.ArgumentParser(description="Run the script with a specific model ID.")
parser.add_argument(
    "-m",
    "--model-id",
    type=str,
    required=True,
    help="Identifier of the model to use (e.g., 'Claude-Sonnet-3.7', 'Gemini-2.5-Flash')",
)
parser.add_argument(
    "-a",
    "--agent-type",
    type=str,
    choices=["tool-calling", "code"],
    default="code",
    help="Identifier of kindf of agent to use",
)
args = parser.parse_args()
