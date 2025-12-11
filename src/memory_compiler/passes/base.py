"""Base class for optimization passes.

This module defines the interface that all optimization passes must implement,
following the compiler pass pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from memory_compiler.ir.memory_ir import MemoryIR


@dataclass
class PassResult:
    """Result of running an optimization pass.

    Attributes:
        modified: Whether the IR was modified by this pass.
        entities_removed: Number of entities removed.
        facts_removed: Number of facts removed.
        facts_merged: Number of facts merged.
        events_removed: Number of events removed.
        events_compressed: Number of event sequences compressed.
        stats: Additional pass-specific statistics.
    """

    modified: bool = False
    entities_removed: int = 0
    facts_removed: int = 0
    facts_merged: int = 0
    events_removed: int = 0
    events_compressed: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    def __add__(self, other: PassResult) -> PassResult:
        """Combine two pass results."""
        return PassResult(
            modified=self.modified or other.modified,
            entities_removed=self.entities_removed + other.entities_removed,
            facts_removed=self.facts_removed + other.facts_removed,
            facts_merged=self.facts_merged + other.facts_merged,
            events_removed=self.events_removed + other.events_removed,
            events_compressed=self.events_compressed + other.events_compressed,
            stats={**self.stats, **other.stats},
        )


class OptimizationPass(ABC):
    """Abstract base class for optimization passes.

    Each pass transforms a Memory IR in place, potentially removing,
    merging, or modifying memory units to achieve compression while
    preserving important information.

    Subclasses must implement the `run` method.
    """

    def __init__(self, name: str | None = None, **config: Any) -> None:
        """Initialize the pass.

        Args:
            name: Optional name for the pass (defaults to class name).
            **config: Pass-specific configuration options.
        """
        self.name = name or self.__class__.__name__
        self.config = config

    @abstractmethod
    def run(self, ir: MemoryIR) -> PassResult:
        """Run the optimization pass on the IR.

        This method modifies the IR in place and returns statistics
        about what was changed.

        Args:
            ir: The Memory IR to optimize.

        Returns:
            PassResult with statistics about modifications.
        """
        pass

    def should_run(self, ir: MemoryIR) -> bool:
        """Check if this pass should run on the given IR.

        Can be overridden to skip passes under certain conditions.

        Args:
            ir: The Memory IR to check.

        Returns:
            True if the pass should run.
        """
        return True

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"


class PassManager:
    """Manages a sequence of optimization passes.

    The PassManager runs passes in order, tracking results and
    optionally running passes to a fixed point.
    """

    def __init__(self, passes: list[OptimizationPass] | None = None) -> None:
        """Initialize the pass manager.

        Args:
            passes: Initial list of passes to run.
        """
        self.passes: list[OptimizationPass] = passes or []

    def add_pass(self, pass_: OptimizationPass) -> None:
        """Add a pass to the pipeline."""
        self.passes.append(pass_)

    def run(self, ir: MemoryIR, max_iterations: int = 1) -> PassResult:
        """Run all passes on the IR.

        Args:
            ir: The Memory IR to optimize.
            max_iterations: Maximum number of iterations (for fixed-point).

        Returns:
            Combined PassResult from all passes.
        """
        total_result = PassResult()

        for iteration in range(max_iterations):
            iteration_result = PassResult()

            for pass_ in self.passes:
                if pass_.should_run(ir):
                    result = pass_.run(ir)
                    iteration_result = iteration_result + result

            total_result = total_result + iteration_result

            # Stop if no modifications were made
            if not iteration_result.modified:
                break

        return total_result

    def run_to_fixed_point(self, ir: MemoryIR, max_iterations: int = 10) -> PassResult:
        """Run passes repeatedly until no more changes are made.

        Args:
            ir: The Memory IR to optimize.
            max_iterations: Safety limit on iterations.

        Returns:
            Combined PassResult from all iterations.
        """
        return self.run(ir, max_iterations=max_iterations)

    def __len__(self) -> int:
        return len(self.passes)

    def __repr__(self) -> str:
        pass_names = [p.name for p in self.passes]
        return f"PassManager(passes={pass_names})"
