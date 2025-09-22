from args import args
from stats import dump_stats
from wrapped_agents import WrappedCodeAgent, get_agent_model

if __name__ == "__main__":
    model_id = args.model_id
    model = get_agent_model(model_id)
    agent = WrappedCodeAgent(
        tools=[],
        model=model,
        verbosity_level=2,
        name="base_agent",
    )
    agent.run("Tell me about you", max_steps=3)

    dump_stats(agent)
