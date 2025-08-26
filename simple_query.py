from smolagents import CodeAgent

from llm import model

if __name__ == "__main__":
    agent = CodeAgent(tools=[], model=model, verbosity_level=2)
    agent.run("Tell me about you", max_steps=3)
