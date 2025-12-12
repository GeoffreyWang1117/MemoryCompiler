#!/usr/bin/env python3
"""Example: LangChain integration.

This example demonstrates how to use MemoryCompiler with LangChain
for enhanced conversation memory management.

Note: Requires langchain and langchain-openai installed.
"""

# Note: This is a demonstration - LangChain import may fail if not installed


def demo_basic_usage():
    """Demonstrate basic LangChain memory usage."""
    print("=" * 60)
    print("Basic LangChain Memory Usage")
    print("=" * 60)

    from memory_compiler.integrations import MemoryCompilerMemory

    # Create memory with MemoryCompiler backend
    memory = MemoryCompilerMemory(
        token_budget=2048,
        memory_key="history",
        human_prefix="Human",
        ai_prefix="AI",
    )

    # Simulate conversation
    conversations = [
        ("Hi, I'm Alex and I work at Tesla.", "Nice to meet you, Alex!"),
        ("I'm an engineer working on autopilot.", "That's fascinating work!"),
        ("We're making great progress on FSD.", "Full self-driving is exciting!"),
        ("I've been there 3 years now.", "Sounds like a great experience."),
    ]

    for human, ai in conversations:
        memory.save_context({"input": human}, {"output": ai})
        print(f"Human: {human}")
        print(f"AI: {ai}")

    print("\n" + "-" * 40)
    print("Memory Variables:")
    print("-" * 40)

    variables = memory.load_memory_variables({})
    print(variables["history"])

    print("\n" + "-" * 40)
    print("Memory Statistics:")
    print("-" * 40)

    stats = memory.get_stats()
    print(f"Total turns: {stats['total_turns']}")
    print(f"Token estimate: {stats['estimated_tokens']}")
    print(f"Entities: {stats['entities']}")
    print(f"Facts: {stats['facts']}")


def demo_with_chain():
    """Demonstrate using memory with a LangChain chain."""
    print("\n" + "=" * 60)
    print("Memory with LangChain Chain (Conceptual)")
    print("=" * 60)

    # This is a conceptual example - requires OpenAI API key
    example_code = '''
    from langchain.chains import ConversationChain
    from langchain_openai import ChatOpenAI
    from memory_compiler.integrations import MemoryCompilerMemory

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Initialize compressed memory
    memory = MemoryCompilerMemory(
        token_budget=2048,
        memory_key="history",
    )

    # Create conversation chain
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True,
    )

    # Have a conversation
    responses = []
    responses.append(chain.run("Hi, I'm Sarah from New York."))
    responses.append(chain.run("I work as a data scientist."))
    responses.append(chain.run("What do you remember about me?"))

    # Memory automatically compresses long conversations
    print(memory.load_memory_variables({})["history"])
    '''

    print("Example code for using with ConversationChain:")
    print(example_code)


def demo_memory_persistence():
    """Demonstrate persisting memory across sessions."""
    print("\n" + "=" * 60)
    print("Memory Persistence")
    print("=" * 60)

    from memory_compiler.integrations import LangChainMemoryAdapter
    from memory_compiler.storage import SQLiteMemoryStore

    # Create adapter with persistent storage
    store = SQLiteMemoryStore(":memory:")  # Use file path for persistence

    adapter = LangChainMemoryAdapter(
        token_budget=2048,
        store=store,
    )

    # Simulate first session
    session_id = "user_123_session_1"
    adapter.save_session(
        session_id,
        [
            ("I'm John and I love hiking.", "Hiking is great exercise!"),
            ("My favorite trail is in Yosemite.", "Yosemite has beautiful trails."),
        ],
    )
    print(f"Saved session: {session_id}")

    # Load session in "new" context
    loaded_memory = adapter.load_session(session_id)
    if loaded_memory:
        print(f"\nLoaded session {session_id}:")
        variables = loaded_memory.load_memory_variables({})
        print(variables.get("history", "No history"))

    # List all sessions
    sessions = adapter.list_sessions()
    print(f"\nAll sessions: {sessions}")


def demo_adapter_features():
    """Demonstrate advanced adapter features."""
    print("\n" + "=" * 60)
    print("Advanced Adapter Features")
    print("=" * 60)

    from memory_compiler.integrations import MemoryCompilerMemory

    memory = MemoryCompilerMemory(
        token_budget=1024,
        memory_key="chat_history",
        return_messages=False,  # Return string, not message objects
    )

    # Add many turns
    for i in range(10):
        memory.save_context(
            {"input": f"This is message {i} from the user."},
            {"output": f"This is response {i} from the assistant."},
        )

    print(f"Added 10 conversation turns")

    # Get compressed output
    variables = memory.load_memory_variables({})
    print(f"\nCompressed history length: {len(variables['chat_history'])} chars")
    print(f"\nCompressed history preview:")
    print(variables["chat_history"][:500] + "...")

    # Access the underlying IR
    if memory._current_ir:
        print(f"\nUnderlying Memory IR:")
        print(f"  Entities: {len(list(memory._current_ir.iter_entities()))}")
        print(f"  Facts: {len(list(memory._current_ir.iter_facts()))}")

    # Clear memory
    memory.clear()
    print("\nMemory cleared!")


def main():
    """Run all LangChain integration examples."""
    print("LangChain Integration Examples")
    print("Note: Some examples are conceptual and require additional setup.\n")

    try:
        demo_basic_usage()
        demo_with_chain()
        demo_memory_persistence()
        demo_adapter_features()

    except ImportError as e:
        print(f"Import error: {e}")
        print("\nTo run these examples, install LangChain:")
        print("  pip install langchain langchain-openai")

    print("\n" + "=" * 60)
    print("LangChain integration examples completed!")


if __name__ == "__main__":
    main()
