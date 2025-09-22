from args import args
from stats import dump_stats
from wrapped_agents import WrappedCodeAgent, WrappedToolCallingAgent, get_agent_model

if __name__ == "__main__":
    model_id = args.model_id
    model = get_agent_model(model_id)
    agent_type = args.agent_type
    agent_class = WrappedCodeAgent if agent_type == "code" else WrappedToolCallingAgent

    agent_name = f"{agent_type.replace('-', '_')}_agent"
    agent = agent_class(
        tools=[],
        model=model,
        verbosity_level=2,
        name=agent_name,
    )
    agent.run("Tell me about you", max_steps=3)

    dump_stats(agent)
