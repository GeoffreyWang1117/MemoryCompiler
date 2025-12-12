"""LangChain integration for MemoryCompiler.

This module provides adapters to use MemoryCompiler as a memory backend
for LangChain applications, enabling compressed conversation memory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.pipeline import CompressionResult, MemoryCompiler


class MemoryCompilerMemory:
    """LangChain-compatible memory using MemoryCompiler.

    This class implements the LangChain memory interface, providing
    compressed conversation history that fits within token budgets.

    Example with LangChain:
        >>> from langchain.chains import ConversationChain
        >>> from langchain.llms import OpenAI
        >>> from memory_compiler.integrations import MemoryCompilerMemory
        >>>
        >>> memory = MemoryCompilerMemory(token_budget=2000)
        >>> chain = ConversationChain(llm=OpenAI(), memory=memory)
        >>> chain.run("Hello, I'm Alice")

    Attributes:
        compiler: The MemoryCompiler instance.
        memory_key: Key for memory in LangChain context.
        human_prefix: Prefix for human messages.
        ai_prefix: Prefix for AI messages.
    """

    def __init__(
        self,
        token_budget: int = 2048,
        memory_key: str = "history",
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        output_format: str = "narrative",
        return_messages: bool = False,
        use_mock: bool = True,
        **compiler_kwargs: Any,
    ) -> None:
        self.memory_key = memory_key
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.return_messages = return_messages

        self.compiler = MemoryCompiler(
            token_budget=token_budget,
            output_format=output_format,
            use_mock_extraction=use_mock,
            **compiler_kwargs,
        )

        self._dialogue: List[Dict[str, str]] = []
        self._ir: Optional[MemoryIR] = None
        self._last_result: Optional[CompressionResult] = None

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for LangChain.

        Returns the compressed conversation history.
        """
        if not self._dialogue:
            if self.return_messages:
                return {self.memory_key: []}
            return {self.memory_key: ""}

        # Compress if needed
        if self._last_result is None or len(self._dialogue) > 0:
            self._compress()

        if self.return_messages:
            # Return as message objects (for chat models)
            return self._format_as_messages()
        else:
            # Return as string
            return {self.memory_key: self._last_result.text if self._last_result else ""}

    def _format_as_messages(self) -> Dict[str, Any]:
        """Format memory as LangChain messages."""
        try:
            from langchain.schema import AIMessage, HumanMessage

            messages = []
            # Add compressed context as system-like message
            if self._last_result:
                # Use compressed facts as context
                for fact in self._last_result.ir.iter_facts():
                    # Convert key facts to messages
                    pass

                # Add recent turns as actual messages
                recent_turns = self._dialogue[-6:]  # Keep last 3 exchanges
                for turn in recent_turns:
                    if turn["role"] == "user":
                        messages.append(HumanMessage(content=turn["content"]))
                    else:
                        messages.append(AIMessage(content=turn["content"]))

            return {self.memory_key: messages}

        except ImportError:
            logger.warning("langchain not installed, returning empty messages")
            return {self.memory_key: []}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save conversation context to memory.

        Called by LangChain after each interaction.
        """
        # Extract input
        input_key = list(inputs.keys())[0] if inputs else "input"
        input_text = inputs.get(input_key, "")

        # Extract output
        output_key = list(outputs.keys())[0] if outputs else "output"
        output_text = outputs.get(output_key, "")

        # Add to dialogue
        if input_text:
            self._dialogue.append({"role": "user", "content": input_text})
        if output_text:
            self._dialogue.append({"role": "assistant", "content": output_text})

        # Mark for recompression
        self._last_result = None

    def _compress(self) -> None:
        """Run compression on current dialogue."""
        if self._dialogue:
            self._last_result = self.compiler.compress(self._dialogue)
            self._ir = self._last_result.ir

    def clear(self) -> None:
        """Clear memory contents."""
        self._dialogue = []
        self._ir = None
        self._last_result = None

    def get_compressed_context(self) -> str:
        """Get the current compressed context."""
        if self._last_result is None:
            self._compress()
        return self._last_result.text if self._last_result else ""

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        if self._last_result is None:
            self._compress()

        if self._last_result:
            return {
                "original_tokens": self._last_result.original_tokens,
                "compressed_tokens": self._last_result.compressed_tokens,
                "compression_ratio": self._last_result.compression_ratio,
                "num_turns": len(self._dialogue),
            }
        return {}

    @property
    def buffer(self) -> str:
        """Return the memory buffer (compressed context)."""
        return self.get_compressed_context()


class LangChainMemoryAdapter:
    """Adapter to convert between LangChain memory formats and MemoryCompiler.

    This class helps migrate existing LangChain applications to use
    MemoryCompiler while maintaining compatibility.
    """

    def __init__(self, compiler: Optional[MemoryCompiler] = None) -> None:
        self.compiler = compiler or MemoryCompiler(use_mock_extraction=True)

    def from_langchain_messages(
        self, messages: List[Any]
    ) -> List[Dict[str, str]]:
        """Convert LangChain messages to dialogue format."""
        dialogue = []

        for msg in messages:
            msg_type = type(msg).__name__

            if msg_type == "HumanMessage":
                dialogue.append({"role": "user", "content": msg.content})
            elif msg_type == "AIMessage":
                dialogue.append({"role": "assistant", "content": msg.content})
            elif msg_type == "SystemMessage":
                # Include system messages as context
                dialogue.append({"role": "system", "content": msg.content})
            elif hasattr(msg, "content"):
                # Generic message with content
                role = getattr(msg, "type", "user")
                dialogue.append({"role": role, "content": msg.content})

        return dialogue

    def to_langchain_messages(self, dialogue: List[Dict[str, str]]) -> List[Any]:
        """Convert dialogue format to LangChain messages."""
        try:
            from langchain.schema import AIMessage, HumanMessage, SystemMessage

            messages = []
            for turn in dialogue:
                role = turn.get("role", "user")
                content = turn.get("content", "")

                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))

            return messages

        except ImportError:
            logger.warning("langchain not installed")
            return []

    def compress_langchain_memory(
        self,
        memory: Any,
        token_budget: Optional[int] = None,
    ) -> CompressionResult:
        """Compress an existing LangChain memory object.

        Args:
            memory: LangChain memory object (ConversationBufferMemory, etc.)
            token_budget: Optional token budget override.

        Returns:
            CompressionResult with compressed memory.
        """
        # Extract messages from various LangChain memory types
        if hasattr(memory, "chat_memory") and hasattr(memory.chat_memory, "messages"):
            messages = memory.chat_memory.messages
        elif hasattr(memory, "buffer"):
            # ConversationBufferMemory stores as string
            if isinstance(memory.buffer, str):
                # Parse string buffer
                return self._compress_string_buffer(memory.buffer, token_budget)
            messages = memory.buffer
        else:
            raise ValueError(f"Unsupported memory type: {type(memory)}")

        dialogue = self.from_langchain_messages(messages)

        if token_budget:
            self.compiler.token_budget = token_budget

        return self.compiler.compress(dialogue)

    def _compress_string_buffer(
        self, buffer: str, token_budget: Optional[int]
    ) -> CompressionResult:
        """Compress a string buffer from LangChain memory."""
        # Parse the buffer into turns
        import re

        dialogue = []
        pattern = r"(Human|AI|User|Assistant):\s*(.+?)(?=(?:Human|AI|User|Assistant):|$)"
        matches = re.findall(pattern, buffer, re.DOTALL | re.IGNORECASE)

        for role, content in matches:
            role_normalized = "user" if role.lower() in ["human", "user"] else "assistant"
            dialogue.append({"role": role_normalized, "content": content.strip()})

        if token_budget:
            self.compiler.token_budget = token_budget

        return self.compiler.compress(dialogue)

    def create_compressed_memory(
        self,
        dialogue: List[Dict[str, str]],
        memory_class: Optional[type] = None,
    ) -> Any:
        """Create a LangChain memory object with compressed history.

        Args:
            dialogue: Dialogue history to compress.
            memory_class: LangChain memory class to use.

        Returns:
            LangChain memory object with compressed history.
        """
        try:
            from langchain.memory import ConversationBufferMemory

            memory_class = memory_class or ConversationBufferMemory

            # Compress
            result = self.compiler.compress(dialogue)

            # Create memory
            memory = memory_class()

            # Add compressed context as initial history
            if hasattr(memory, "chat_memory"):
                from langchain.schema import AIMessage, HumanMessage

                # Add a summary message
                memory.chat_memory.add_message(
                    HumanMessage(content="[Previous conversation summary]")
                )
                memory.chat_memory.add_message(
                    AIMessage(content=result.text)
                )

            return memory

        except ImportError:
            logger.error("langchain not installed")
            raise
