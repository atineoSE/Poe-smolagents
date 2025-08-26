import psutil
from smolagents import tool

from llm import model
from poe_code_agent import PoeCodeAgent


@tool
def get_system_info() -> dict:
    """
    Collects basic system metrics.

    Returns:
        dict: A dictionary with the following keys:
            - ``cpu_percent`` (float): Current CPU usage percentage.
            - ``memory_total_mb`` (int): Total physical memory in megabytes.
            - ``memory_used_mb`` (int): Amount of physical memory currently used (MiB).
            - ``memory_percent`` (float): Percentage of physical memory in use.
            - ``swap_total_mb`` (int): Total swap space in megabytes.
            - ``swap_used_mb`` (int): Amount of swap currently used (MiB).
            - ``swap_percent`` (float): Percentage of swap space in use.

            All numeric values are derived from ``psutil`` and are rounded down
            to the nearest integer where applicable.
    """
    cpu_percent = psutil.cpu_percent(interval=1)
    virtual_mem = psutil.virtual_memory()
    swap_mem = psutil.swap_memory()

    return {
        "cpu_percent": cpu_percent,
        "memory_total_mb": virtual_mem.total // (1024 * 1024),
        "memory_used_mb": virtual_mem.used // (1024 * 1024),
        "memory_percent": virtual_mem.percent,
        "swap_total_mb": swap_mem.total // (1024 * 1024),
        "swap_used_mb": swap_mem.used // (1024 * 1024),
        "swap_percent": swap_mem.percent,
    }


if __name__ == "__main__":
    agent = PoeCodeAgent(tools=[get_system_info], model=model, verbosity_level=2)
    agent.run("Tell me the system specs", max_steps=3)
