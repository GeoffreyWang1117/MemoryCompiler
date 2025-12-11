"""LLM-based conversation summarizer for MemoryCompiler.

This module provides LLM-based summarization capabilities that can be
used as an alternative or complement to the IR-based compression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR


@dataclass
class SummaryResult:
    """Result from conversation summarization."""

    summary: str
    key_points: list[str] = field(default_factory=list)
    entities_mentioned: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    original_tokens: int = 0
    summary_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        """Compute compression ratio."""
        if self.original_tokens == 0:
            return 1.0
        return self.summary_tokens / self.original_tokens


class ConversationSummarizer:
    """LLM-based conversation summarizer.

    This class uses language models to generate natural language summaries
    of conversations. It can work standalone or with Memory IR to produce
    enhanced summaries.

    Supports multiple summarization styles:
    - executive: High-level overview
    - detailed: Comprehensive summary with key points
    - bullet: Bullet-point format
    - narrative: Flowing narrative style
    - qa: Question-answer format covering key info

    Example:
        >>> summarizer = ConversationSummarizer(model_name="Qwen/Qwen2.5-7B-Instruct")
        >>> result = summarizer.summarize(dialogue)
        >>> print(result.summary)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        max_summary_tokens: int = 500,
        style: str = "detailed",
        use_mock: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_summary_tokens = max_summary_tokens
        self.style = style
        self.use_mock = use_mock

        self._model = None
        self._tokenizer = None

        if not use_mock:
            self._load_model()

    def _load_model(self) -> None:
        """Load the language model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading summarization model: {self.model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                trust_remote_code=True,
            )

            logger.info("Summarization model loaded")

        except ImportError:
            logger.warning("transformers not installed, using mock summarization")
            self.use_mock = True
        except Exception as e:
            logger.warning(f"Failed to load model: {e}, using mock summarization")
            self.use_mock = True

    def summarize(
        self,
        dialogue: list[dict[str, str]],
        ir: MemoryIR | None = None,
        style: str | None = None,
        max_tokens: int | None = None,
    ) -> SummaryResult:
        """Summarize a conversation.

        Args:
            dialogue: List of dialogue turns.
            ir: Optional Memory IR for enhanced summarization.
            style: Summarization style override.
            max_tokens: Maximum tokens for summary.

        Returns:
            SummaryResult with summary and metadata.
        """
        style = style or self.style
        max_tokens = max_tokens or self.max_summary_tokens

        # Estimate original tokens
        original_text = "\n".join(f"{t['role']}: {t['content']}" for t in dialogue)
        original_tokens = len(original_text.split()) * 4 // 3  # Rough estimate

        if self.use_mock:
            return self._mock_summarize(dialogue, ir, style, original_tokens)

        # Build prompt based on style
        prompt = self._build_prompt(dialogue, ir, style, max_tokens)

        # Generate summary
        summary_text = self._generate(prompt, max_tokens)

        # Parse result
        return self._parse_summary(summary_text, style, original_tokens)

    def _build_prompt(
        self,
        dialogue: list[dict[str, str]],
        ir: MemoryIR | None,
        style: str,
        max_tokens: int,
    ) -> str:
        """Build the summarization prompt."""
        # Format dialogue
        dialogue_text = "\n".join(
            f"{turn['role'].upper()}: {turn['content']}"
            for turn in dialogue
        )

        # Style-specific instructions
        style_instructions = {
            "executive": """Provide a brief executive summary (2-3 sentences) capturing the main purpose and outcome of this conversation.""",

            "detailed": """Provide a detailed summary that includes:
1. Main topic and purpose
2. Key information exchanged
3. Important decisions or conclusions
4. Any action items or next steps""",

            "bullet": """Summarize this conversation as a bullet-point list covering:
- Main participants and their roles
- Key topics discussed
- Important facts stated
- Conclusions or outcomes""",

            "narrative": """Write a flowing narrative summary of this conversation, as if you were telling someone what was discussed. Make it natural and readable.""",

            "qa": """Summarize this conversation by listing the most important questions and their answers that emerged from the discussion.""",
        }

        instruction = style_instructions.get(style, style_instructions["detailed"])

        # Add IR context if available
        ir_context = ""
        if ir:
            entities = [e.name for e in list(ir.iter_entities())[:10]]
            facts = [f.natural_language for f in list(ir.iter_facts())[:10]]

            if entities:
                ir_context += f"\n\nKey entities: {', '.join(entities)}"
            if facts:
                ir_context += f"\n\nKey facts:\n" + "\n".join(f"- {f}" for f in facts)

        prompt = f"""Summarize the following conversation.{ir_context}

## Conversation:
{dialogue_text}

## Instructions:
{instruction}

Keep the summary under {max_tokens} tokens.

## Summary:
"""
        return prompt

    def _generate(self, prompt: str, max_tokens: int) -> str:
        """Generate text using the LLM."""
        messages = [{"role": "user", "content": prompt}]

        if hasattr(self._tokenizer, "apply_chat_template"):
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt

        inputs = self._tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return response.strip()

    def _parse_summary(
        self, text: str, style: str, original_tokens: int
    ) -> SummaryResult:
        """Parse the generated summary."""
        import re

        # Extract key points if present
        key_points = []
        bullet_pattern = r"[-•*]\s*(.+?)(?=\n[-•*]|\n\n|$)"
        matches = re.findall(bullet_pattern, text, re.DOTALL)
        if matches:
            key_points = [m.strip() for m in matches if m.strip()]

        # Extract entities (capitalized words)
        entity_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        entities = list(set(re.findall(entity_pattern, text)))

        # Estimate summary tokens
        summary_tokens = len(text.split()) * 4 // 3

        return SummaryResult(
            summary=text,
            key_points=key_points[:10],
            entities_mentioned=entities[:15],
            topics=[],
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            metadata={"style": style},
        )

    def _mock_summarize(
        self,
        dialogue: list[dict[str, str]],
        ir: MemoryIR | None,
        style: str,
        original_tokens: int,
    ) -> SummaryResult:
        """Generate mock summary for testing."""
        # Extract basic info from dialogue
        num_turns = len(dialogue)

        # Find key information
        entities = []
        topics = []
        key_points = []

        import re

        for turn in dialogue:
            content = turn.get("content", "")

            # Extract names
            name_match = re.search(r"(?:I'm|my name is|I am) (\w+)", content, re.I)
            if name_match:
                entities.append(name_match.group(1))

            # Extract topics (capitalized phrases)
            cap_words = re.findall(r"\b[A-Z][a-z]+(?:\s+[a-z]+)*\b", content)
            topics.extend(cap_words[:2])

        # Use IR if available
        if ir:
            for entity in list(ir.iter_entities())[:5]:
                if entity.name not in entities:
                    entities.append(entity.name)

            for fact in list(ir.iter_facts())[:5]:
                key_points.append(fact.natural_language)

        # Generate summary based on style
        entities_str = ", ".join(entities[:5]) if entities else "participants"
        topics_str = ", ".join(set(topics[:3])) if topics else "various topics"

        if style == "executive":
            summary = f"A {num_turns}-turn conversation involving {entities_str} discussing {topics_str}."
        elif style == "bullet":
            lines = [f"- Participants: {entities_str}"]
            lines.append(f"- Topics: {topics_str}")
            lines.append(f"- Duration: {num_turns} exchanges")
            for point in key_points[:3]:
                lines.append(f"- {point}")
            summary = "\n".join(lines)
        elif style == "narrative":
            summary = f"In this conversation, {entities_str} engaged in a discussion about {topics_str}. Over the course of {num_turns} exchanges, they covered various aspects of the topic."
            if key_points:
                summary += f" Key information shared includes: {key_points[0]}."
        else:
            summary = f"Summary of conversation ({num_turns} turns):\n"
            summary += f"Participants: {entities_str}\n"
            summary += f"Main topics: {topics_str}\n"
            if key_points:
                summary += "Key points:\n" + "\n".join(f"- {p}" for p in key_points[:5])

        summary_tokens = len(summary.split()) * 4 // 3

        return SummaryResult(
            summary=summary,
            key_points=key_points,
            entities_mentioned=entities,
            topics=list(set(topics)),
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            metadata={"style": style, "mock": True},
        )

    def summarize_with_ir(
        self,
        ir: MemoryIR,
        style: str | None = None,
    ) -> SummaryResult:
        """Summarize directly from Memory IR.

        This method uses the structured IR to generate a summary,
        which can be more accurate than raw dialogue summarization.
        """
        # Convert IR to dialogue if needed
        dialogue = [
            {"role": turn.role, "content": turn.content}
            for turn in ir.dialogue_turns
        ]

        return self.summarize(dialogue, ir=ir, style=style)

    def incremental_summarize(
        self,
        previous_summary: SummaryResult,
        new_turns: list[dict[str, str]],
    ) -> SummaryResult:
        """Update summary with new dialogue turns.

        Useful for streaming scenarios where new turns are added.
        """
        # Combine previous summary with new turns for context
        context = f"Previous summary: {previous_summary.summary}\n\n"
        context += "New exchanges:\n"
        context += "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in new_turns
        )

        prompt = f"""{context}

Update the summary to incorporate the new exchanges. Keep it concise.

Updated summary:
"""

        if self.use_mock:
            # Simple mock update
            new_summary = previous_summary.summary
            new_summary += f" The conversation continued with {len(new_turns)} more exchanges."

            return SummaryResult(
                summary=new_summary,
                key_points=previous_summary.key_points,
                entities_mentioned=previous_summary.entities_mentioned,
                original_tokens=previous_summary.original_tokens + len(new_turns) * 20,
                summary_tokens=len(new_summary.split()),
                metadata={"incremental": True},
            )

        updated_text = self._generate(prompt, self.max_summary_tokens)

        return SummaryResult(
            summary=updated_text,
            key_points=previous_summary.key_points,
            entities_mentioned=previous_summary.entities_mentioned,
            original_tokens=previous_summary.original_tokens + len(new_turns) * 20,
            summary_tokens=len(updated_text.split()),
            metadata={"incremental": True},
        )
