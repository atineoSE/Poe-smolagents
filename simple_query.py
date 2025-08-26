import inspect

from llm import model
from poe_code_agent import PoeCodeAgent

print(PoeCodeAgent.__module__)  # should point to your file
print(inspect.getsource(PoeCodeAgent._generate_model_output))

if __name__ == "__main__":
    agent = PoeCodeAgent(tools=[], model=model, verbosity_level=2)
    agent.run("Tell me about you", max_steps=3)
