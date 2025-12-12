#!/usr/bin/env python3
"""Example: RAG-style memory retrieval.

This example demonstrates how to use the MemoryCompiler's RAG retrieval
capabilities to index conversations and retrieve relevant context.
"""

from memory_compiler import MemoryCompiler
from memory_compiler.retrieval import (
    HybridRetriever,
    MemoryRetriever,
    RetrievalConfig,
    RetrievalStrategy,
)


def main():
    # Initialize components
    compiler = MemoryCompiler(use_mock=True)
    retriever = HybridRetriever(
        semantic_weight=0.6,
        keyword_weight=0.4,
    )

    # Sample conversations to index
    conversations = {
        "session_python": """
        User: I'm learning Python programming. Can you explain decorators?
        Assistant: Decorators are functions that modify other functions. They use the @syntax.
        User: What about list comprehensions?
        Assistant: List comprehensions provide a concise way to create lists: [x*2 for x in range(10)].
        User: I prefer Python over Java for its simplicity.
        Assistant: Many developers appreciate Python's clean syntax and readability.
        """,
        "session_travel": """
        User: I'm planning a trip to Japan next month.
        Assistant: Japan is wonderful! Are you interested in Tokyo or Kyoto?
        User: Both! I love Japanese food, especially ramen and sushi.
        Assistant: Tokyo has amazing ramen shops. Try Ichiran in Shibuya.
        User: I also want to see Mount Fuji.
        Assistant: You can see Fuji from Hakone, about 1.5 hours from Tokyo.
        """,
        "session_work": """
        User: I have a deadline for the quarterly report next Friday.
        Assistant: Would you like help organizing the report structure?
        User: Yes, I need to include sales figures and projections.
        Assistant: I suggest starting with an executive summary, then detailed sections.
        User: My manager Bob wants charts and visualizations.
        Assistant: Charts definitely help communicate data effectively.
        """,
    }

    # Extract and index each conversation
    print("Indexing conversations...")
    for session_id, dialogue in conversations.items():
        # Extract Memory IR
        ir = compiler.extract(dialogue)

        # Index for retrieval
        chunks = retriever.index_memory(session_id, ir)
        print(f"  - {session_id}: {chunks} chunks indexed")

    print("\n" + "=" * 60)

    # Test queries
    queries = [
        "What programming language does the user prefer?",
        "Where is the user traveling to?",
        "What is the user's work deadline?",
        "Tell me about Python decorators",
        "Japanese food recommendations",
        "Who is the user's manager?",
    ]

    print("\nRetrieving relevant context for queries:\n")

    for query in queries:
        print(f"Query: {query}")
        print("-" * 40)

        results = retriever.retrieve(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result.source_key}] (score: {result.score:.3f})")
            print(f"     {result.text[:80]}...")

        # Format as context
        context = retriever.format_context(results, max_tokens=500)
        print(f"\n  Formatted context:\n{context}\n")
        print()

    # Demonstrate filtered retrieval
    print("=" * 60)
    print("\nFiltered retrieval (only Python session):\n")

    results = retriever.retrieve(
        "What does the user like?",
        top_k=3,
        filter_keys=["session_python"],
    )

    for result in results:
        print(f"  - {result.text}")

    # Show statistics
    print("\n" + "=" * 60)
    print("\nRetriever Statistics:")
    stats = retriever.get_stats()
    print(f"  Semantic chunks: {stats['semantic']['total_chunks']}")
    print(f"  Keyword documents: {stats['keyword']['total_documents']}")
    print(f"  Vocabulary size: {stats['keyword']['vocabulary_size']}")


if __name__ == "__main__":
    main()
