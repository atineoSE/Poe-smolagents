from llm import model
from poe_code_agent import PoeCodeAgent

if __name__ == "__main__":
    agent = PoeCodeAgent(tools=[], model=model, verbosity_level=2)
    agent.run("Tell me about you", max_steps=3)
