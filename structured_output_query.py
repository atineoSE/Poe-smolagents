import psutil
from pydantic import BaseModel
from smolagents import tool

from llm import model
from poe_code_agent import PoeCodeAgent


class System(BaseModel):
    """
    Model representing basic system metrics.

    Attributes
    ----------
    cpu_percent: float
        The current CPU utilization percentage.
    memory_total_mb: int
        Total physical memory in megabytes.
    memory_used_mb: int
        Amount of memory currently used in megabytes.
    """

    cpu_percent: float
    memory_total_mb: int
    memory_used_mb: int


@tool
def get_system_info() -> System:
    """
    Collects basic system metrics.

    Returns: A System object with the system information.
    """
    virtual_mem = psutil.virtual_memory()
    return System(
        cpu_percent=psutil.cpu_percent(interval=1),
        memory_total_mb=virtual_mem.total // (1024 * 1024),
        memory_used_mb=virtual_mem.used // (1024 * 1024),
    )


if __name__ == "__main__":
    agent = PoeCodeAgent(
        tools=[get_system_info],
        model=model,
        verbosity_level=2,
        use_structured_outputs_internally=True,
    )
    result = agent.run("What is the system information?", max_steps=3)
    print(f"Result: {result}")
    print(f"Result type: {type(result).__name__}")
    agent.run("Are there more than 20 MB of memory free?", max_steps=3, reset=False)
