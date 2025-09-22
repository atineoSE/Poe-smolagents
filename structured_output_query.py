from args import args
from stats import dump_stats
from tool import get_system_info
from wrapped_agents import WrappedCodeAgent, WrappedToolCallingAgent, get_agent_model

if __name__ == "__main__":
    model_id = args.model_id
    model = get_agent_model(model_id)
    agent_type = args.agent_type
    agent_name = f"{agent_type.replace('-', '_')}_agent"
    if agent_type == "code":
        agent = WrappedCodeAgent(
            tools=[get_system_info],
            model=model,
            verbosity_level=2,
            name=agent_name,
            use_structured_outputs_internally=True,
        )
    else:
        agent = WrappedToolCallingAgent(
            tools=[get_system_info],
            model=model,
            verbosity_level=2,
            name=agent_name,
        )

    result = agent.run("What is the system information?", max_steps=3)

    print(f"Result: {result}")
    print(f"Result type: {type(result).__name__}")
    agent.run("Are there more than 20 MB of memory free?", max_steps=3, reset=False)

    dump_stats(agent)
