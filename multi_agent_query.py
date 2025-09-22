from args import args
from stats import dump_stats
from tool import SystemInfoTool
from wrapped_agents import WrappedCodeAgent, WrappedToolCallingAgent, get_agent_model

if __name__ == "__main__":
    model_id = args.model_id
    model = get_agent_model(model_id)
    agent_type = args.agent_type
    manager_agent_name = f"manager_{agent_type.replace('-', '_')}_agent"
    provider_agent_name = f"provider_{agent_type.replace('-', '_')}_agent"
    agent_class = WrappedCodeAgent if agent_type == "code" else WrappedToolCallingAgent

    provider_agent = agent_class(
        tools=[SystemInfoTool()],
        model=model,
        max_steps=2,
        verbosity_level=2,
        name=provider_agent_name,
        description="A provider agent, which can fetch system information.",
    )

    manager_agent = agent_class(
        tools=[],
        model=model,
        verbosity_level=2,
        max_steps=2,
        name=manager_agent_name,
        description="A manager agent, which can manage a provider agent",
        managed_agents=[provider_agent],
    )

    manager_agent.run("What is the system information?")

    dump_stats(manager_agent)
