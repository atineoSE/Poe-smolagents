import psutil
from pydantic import BaseModel
from smolagents import Tool, tool


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


class SystemInfoTool(Tool):
    name = "system_info_tool"
    description = "Gets system information"
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        return f"The system information is: {get_system_info().model_dump()}"
