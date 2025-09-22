from wrapped_agents import get_all_messages


def dump_stats(agent):
    print("Dumping agent messages:")
    print("***********************")
    all_messages = get_all_messages(agent)
    for message in all_messages:
        print(message)

    print(f"Total input tokens = {agent.total_input_tokens}")
