#!/usr/bin/env python3
"""Example: Using the MemoryCompiler API.

This example demonstrates how to interact with the MemoryCompiler
REST API using httpx (or requests).

To run the API server:
    python -m memory_compiler.api.app

Or:
    uvicorn memory_compiler.api.app:app --reload
"""

import httpx

API_BASE = "http://localhost:8000"


def compress_dialogue():
    """Compress a dialogue using the API."""
    print("=" * 60)
    print("Compressing Dialogue")
    print("=" * 60)

    response = httpx.post(
        f"{API_BASE}/compress",
        json={
            "messages": [
                {"role": "user", "content": "Hi, I'm John and I work at Google."},
                {"role": "assistant", "content": "Nice to meet you, John!"},
                {"role": "user", "content": "I'm a software engineer specializing in ML."},
                {"role": "assistant", "content": "Machine learning is a great field!"},
                {"role": "user", "content": "I've been there for 5 years now."},
            ],
            "token_budget": 512,
            "output_format": "narrative",
        },
        timeout=30.0,
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Original tokens: {data['original_tokens']}")
        print(f"Compressed tokens: {data['compressed_tokens']}")
        print(f"Compression ratio: {data['compression_ratio']:.2f}x")
        print(f"\nCompressed text:\n{data['compressed_text']}")
        print(f"\nEntities: {len(data['entities'])}")
        for entity in data["entities"]:
            print(f"  - {entity['name']} ({entity['type']})")
        print(f"\nFacts: {len(data['facts'])}")
        for fact in data["facts"]:
            print(f"  - {fact['subject']} {fact['predicate']} {fact['object']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def create_and_retrieve_memory():
    """Create a memory and retrieve it later."""
    print("\n" + "=" * 60)
    print("Creating and Retrieving Memory")
    print("=" * 60)

    # Create memory
    response = httpx.post(
        f"{API_BASE}/memories",
        json={
            "key": "john_session_001",
            "messages": [
                {"role": "user", "content": "My name is John and I love Python."},
                {"role": "assistant", "content": "Python is a great language!"},
                {"role": "user", "content": "I'm building a chatbot for my startup."},
            ],
            "metadata": {"user_id": "john", "topic": "programming"},
            "compress": True,
            "token_budget": 512,
        },
        timeout=30.0,
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Memory created: {data['key']}")
        print(f"Entities: {data['entity_count']}")
        print(f"Facts: {data['fact_count']}")
    else:
        print(f"Error creating memory: {response.text}")
        return

    # Retrieve memory
    response = httpx.get(
        f"{API_BASE}/memories/john_session_001",
        params={"format": "structured"},
        timeout=30.0,
    )

    if response.status_code == 200:
        data = response.json()
        print(f"\nRetrieved memory: {data['key']}")
        print(f"Compressed text:\n{data['compressed_text']}")


def semantic_retrieval():
    """Demonstrate semantic retrieval across memories."""
    print("\n" + "=" * 60)
    print("Semantic Retrieval")
    print("=" * 60)

    # Create some memories first
    memories = [
        {
            "key": "tech_session",
            "messages": [
                {"role": "user", "content": "I'm interested in machine learning."},
                {"role": "assistant", "content": "ML is transforming many industries."},
            ],
        },
        {
            "key": "travel_session",
            "messages": [
                {"role": "user", "content": "I want to visit Tokyo next year."},
                {"role": "assistant", "content": "Japan is beautiful in spring!"},
            ],
        },
    ]

    for mem in memories:
        httpx.post(
            f"{API_BASE}/memories",
            json=mem,
            timeout=30.0,
        )

    # Retrieve relevant context
    response = httpx.post(
        f"{API_BASE}/retrieve",
        json={
            "query": "artificial intelligence and deep learning",
            "top_k": 5,
            "include_context": True,
            "max_context_tokens": 1024,
        },
        timeout=30.0,
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Query: {data['query']}")
        print(f"\nResults: {len(data['results'])}")
        for result in data["results"]:
            print(f"  - [{result['source_key']}] {result['text'][:60]}...")
            print(f"    Score: {result['score']:.3f}")

        if data["context"]:
            print(f"\nFormatted context:\n{data['context']}")


def search_examples():
    """Demonstrate search functionality."""
    print("\n" + "=" * 60)
    print("Search Examples")
    print("=" * 60)

    # Search by entity
    response = httpx.get(
        f"{API_BASE}/search/entity",
        params={"name": "John"},
        timeout=30.0,
    )

    if response.status_code == 200:
        keys = response.json()
        print(f"Memories with 'John': {keys}")

    # Search by fact
    response = httpx.get(
        f"{API_BASE}/search/fact",
        params={"subject": "John", "predicate": "love"},
        timeout=30.0,
    )

    if response.status_code == 200:
        keys = response.json()
        print(f"Memories with facts about John loving: {keys}")


def get_stats():
    """Get storage statistics."""
    print("\n" + "=" * 60)
    print("Storage Statistics")
    print("=" * 60)

    response = httpx.get(f"{API_BASE}/memories/stats", timeout=30.0)

    if response.status_code == 200:
        stats = response.json()
        print(f"Total memories: {stats['num_memories']}")
        print(f"Total entities: {stats['total_entities']}")
        print(f"Total facts: {stats['total_facts']}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Storage type: {stats['storage_type']}")


def main():
    """Run all examples."""
    print("MemoryCompiler API Client Examples")
    print("Make sure the API server is running at", API_BASE)
    print()

    try:
        # Check health
        response = httpx.get(f"{API_BASE}/health", timeout=5.0)
        if response.status_code != 200:
            print("API server is not available!")
            return
        print(f"API Status: {response.json()['status']}")
    except httpx.ConnectError:
        print(f"Cannot connect to API server at {API_BASE}")
        print("Start the server with: uvicorn memory_compiler.api.app:app --reload")
        return

    compress_dialogue()
    create_and_retrieve_memory()
    semantic_retrieval()
    search_examples()
    get_stats()


if __name__ == "__main__":
    main()
