"""LlamaIndex integration for MemoryCompiler.

This module provides adapters to use MemoryCompiler as a chat store
and memory backend for LlamaIndex applications.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.pipeline import CompressionResult, MemoryCompiler


class MemoryCompilerChatStore:
    """LlamaIndex-compatible chat store using MemoryCompiler.

    This class implements a chat store interface that automatically
    compresses conversation history to fit within context limits.

    Example with LlamaIndex:
        >>> from llama_index.core.memory import ChatMemoryBuffer
        >>> from memory_compiler.integrations import MemoryCompilerChatStore
        >>>
        >>> chat_store = MemoryCompilerChatStore(token_budget=2000)
        >>> memory = ChatMemoryBuffer.from_defaults(chat_store=chat_store)

    Attributes:
        compiler: The MemoryCompiler instance.
        token_budget: Maximum tokens for compressed output.
    """

    def __init__(
        self,
        token_budget: int = 2048,
        output_format: str = "narrative",
        use_mock: bool = True,
        **compiler_kwargs: Any,
    ) -> None:
        self.token_budget = token_budget

        self.compiler = MemoryCompiler(
            token_budget=token_budget,
            output_format=output_format,
            use_mock_extraction=use_mock,
            **compiler_kwargs,
        )

        self._stores: Dict[str, List[Dict[str, str]]] = {}
        self._compressed_cache: Dict[str, CompressionResult] = {}

    def set_messages(self, key: str, messages: List[Any]) -> None:
        """Set messages for a conversation key.

        Args:
            key: Conversation identifier.
            messages: List of ChatMessage objects.
        """
        dialogue = self._messages_to_dialogue(messages)
        self._stores[key] = dialogue
        # Invalidate cache
        if key in self._compressed_cache:
            del self._compressed_cache[key]

    def get_messages(self, key: str) -> List[Any]:
        """Get messages for a conversation key.

        Returns compressed messages if the conversation is long.
        """
        if key not in self._stores:
            return []

        dialogue = self._stores[key]

        # If short enough, return as-is
        estimated_tokens = sum(len(t["content"].split()) for t in dialogue) * 4 // 3
        if estimated_tokens <= self.token_budget:
            return self._dialogue_to_messages(dialogue)

        # Compress and return
        if key not in self._compressed_cache:
            self._compressed_cache[key] = self.compiler.compress(dialogue)

        result = self._compressed_cache[key]

        # Return compressed context plus recent messages
        compressed_messages = self._create_compressed_messages(result, dialogue)
        return compressed_messages

    def add_message(self, key: str, message: Any) -> None:
        """Add a message to a conversation.

        Args:
            key: Conversation identifier.
            message: ChatMessage to add.
        """
        if key not in self._stores:
            self._stores[key] = []

        turn = self._message_to_turn(message)
        self._stores[key].append(turn)

        # Invalidate cache
        if key in self._compressed_cache:
            del self._compressed_cache[key]

    def delete_messages(self, key: str) -> Optional[List[Any]]:
        """Delete all messages for a conversation.

        Returns the deleted messages.
        """
        if key in self._stores:
            dialogue = self._stores.pop(key)
            if key in self._compressed_cache:
                del self._compressed_cache[key]
            return self._dialogue_to_messages(dialogue)
        return None

    def delete_message(self, key: str, idx: int) -> Optional[Any]:
        """Delete a specific message by index."""
        if key in self._stores and 0 <= idx < len(self._stores[key]):
            turn = self._stores[key].pop(idx)
            if key in self._compressed_cache:
                del self._compressed_cache[key]
            return self._turn_to_message(turn)
        return None

    def delete_last_message(self, key: str) -> Optional[Any]:
        """Delete the last message in a conversation."""
        if key in self._stores and self._stores[key]:
            return self.delete_message(key, len(self._stores[key]) - 1)
        return None

    def get_keys(self) -> List[str]:
        """Get all conversation keys."""
        return list(self._stores.keys())

    def _messages_to_dialogue(self, messages: List[Any]) -> List[Dict[str, str]]:
        """Convert LlamaIndex messages to dialogue format."""
        dialogue = []
        for msg in messages:
            turn = self._message_to_turn(msg)
            dialogue.append(turn)
        return dialogue

    def _message_to_turn(self, message: Any) -> Dict[str, str]:
        """Convert a single message to dialogue turn."""
        # Handle different message types
        if hasattr(message, "role") and hasattr(message, "content"):
            role = str(message.role).lower()
            if role in ["user", "human"]:
                role = "user"
            elif role in ["assistant", "ai", "chatbot"]:
                role = "assistant"
            return {"role": role, "content": str(message.content)}

        elif isinstance(message, dict):
            return {
                "role": message.get("role", "user"),
                "content": message.get("content", ""),
            }

        else:
            return {"role": "user", "content": str(message)}

    def _dialogue_to_messages(self, dialogue: List[Dict[str, str]]) -> List[Any]:
        """Convert dialogue to LlamaIndex messages."""
        try:
            from llama_index.core.llms import ChatMessage, MessageRole

            messages = []
            for turn in dialogue:
                role = turn.get("role", "user")
                content = turn.get("content", "")

                if role == "user":
                    msg_role = MessageRole.USER
                elif role == "assistant":
                    msg_role = MessageRole.ASSISTANT
                elif role == "system":
                    msg_role = MessageRole.SYSTEM
                else:
                    msg_role = MessageRole.USER

                messages.append(ChatMessage(role=msg_role, content=content))

            return messages

        except ImportError:
            # Return as dicts if LlamaIndex not installed
            return dialogue

    def _turn_to_message(self, turn: Dict[str, str]) -> Any:
        """Convert a dialogue turn to LlamaIndex message."""
        return self._dialogue_to_messages([turn])[0]

    def _create_compressed_messages(
        self,
        result: CompressionResult,
        original_dialogue: List[Dict[str, str]],
    ) -> List[Any]:
        """Create message list with compressed context."""
        try:
            from llama_index.core.llms import ChatMessage, MessageRole

            messages = []

            # Add compressed summary as system message
            messages.append(
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=f"Previous conversation summary:\n{result.text}",
                )
            )

            # Add recent turns (last 4 messages)
            recent = original_dialogue[-4:]
            for turn in recent:
                role = MessageRole.USER if turn["role"] == "user" else MessageRole.ASSISTANT
                messages.append(ChatMessage(role=role, content=turn["content"]))

            return messages

        except ImportError:
            return [{"role": "system", "content": result.text}]

    def get_compressed_context(self, key: str) -> str:
        """Get compressed context for a conversation."""
        if key not in self._stores:
            return ""

        if key not in self._compressed_cache:
            self._compressed_cache[key] = self.compiler.compress(self._stores[key])

        return self._compressed_cache[key].text

    def get_stats(self, key: str) -> Dict[str, Any]:
        """Get compression statistics for a conversation."""
        if key not in self._stores:
            return {}

        if key not in self._compressed_cache:
            self._compressed_cache[key] = self.compiler.compress(self._stores[key])

        result = self._compressed_cache[key]
        return {
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "compression_ratio": result.compression_ratio,
            "num_messages": len(self._stores[key]),
        }


class LlamaIndexMemoryAdapter:
    """Adapter for integrating MemoryCompiler with LlamaIndex memory systems.

    Provides utilities for converting between formats and compressing
    existing LlamaIndex chat histories.
    """

    def __init__(self, compiler: Optional[MemoryCompiler] = None) -> None:
        self.compiler = compiler or MemoryCompiler(use_mock_extraction=True)

    def compress_chat_history(
        self,
        messages: List[Any],
        token_budget: Optional[int] = None,
    ) -> CompressionResult:
        """Compress a list of LlamaIndex ChatMessages.

        Args:
            messages: List of ChatMessage objects.
            token_budget: Optional token budget.

        Returns:
            CompressionResult with compressed chat history.
        """
        dialogue = []
        for msg in messages:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                role = str(msg.role).lower()
                if "user" in role or "human" in role:
                    role = "user"
                elif "assistant" in role or "ai" in role:
                    role = "assistant"
                else:
                    role = "system"
                dialogue.append({"role": role, "content": str(msg.content)})

        if token_budget:
            self.compiler.token_budget = token_budget

        return self.compiler.compress(dialogue)

    def create_memory_buffer(
        self,
        dialogue: List[Dict[str, str]],
        token_limit: int = 2048,
    ) -> Any:
        """Create a LlamaIndex ChatMemoryBuffer with compressed history.

        Args:
            dialogue: Dialogue history.
            token_limit: Token limit for the buffer.

        Returns:
            ChatMemoryBuffer with compressed initial history.
        """
        try:
            from llama_index.core.memory import ChatMemoryBuffer

            # Create chat store
            chat_store = MemoryCompilerChatStore(token_budget=token_limit)
            chat_store.set_messages("default", dialogue)

            # Create memory buffer
            memory = ChatMemoryBuffer.from_defaults(
                chat_store=chat_store,
                chat_store_key="default",
                token_limit=token_limit,
            )

            return memory

        except ImportError:
            logger.error("llama_index not installed")
            raise

    def enhance_chat_engine(
        self,
        chat_engine: Any,
        token_budget: int = 2048,
    ) -> Any:
        """Enhance a LlamaIndex chat engine with compressed memory.

        This wraps the chat engine to use MemoryCompiler for
        memory management.

        Args:
            chat_engine: LlamaIndex chat engine.
            token_budget: Token budget for compression.

        Returns:
            Enhanced chat engine.
        """
        # Create wrapper that compresses before each call
        original_chat = chat_engine.chat

        chat_store = MemoryCompilerChatStore(token_budget=token_budget)

        def compressed_chat(message: str, **kwargs: Any) -> Any:
            # Get current history
            if hasattr(chat_engine, "chat_history"):
                chat_store.set_messages("current", chat_engine.chat_history)

            # Get compressed context
            compressed = chat_store.get_compressed_context("current")

            # Inject compressed context
            # This is framework-specific and may need adjustment
            result = original_chat(message, **kwargs)

            return result

        chat_engine.chat = compressed_chat
        chat_engine._memory_compiler_store = chat_store

        return chat_engine
