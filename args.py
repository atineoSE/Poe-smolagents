import argparse

parser = argparse.ArgumentParser(description="Run the script with a specific model ID.")
parser.add_argument(
    "--model-id",
    type=str,
    required=True,
    help="Identifier of the model to use (e.g., 'Claude-Sonnet-3.7', 'Gemini-2.5-Flash')",
)
args = parser.parse_args()
