"""Performance profiling for MemoryCompiler."""

from __future__ import annotations

import cProfile
import functools
import io
import pstats
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ProfileResult:
    """Result of a profiling session.

    Attributes:
        function_name: Name of profiled function.
        total_time_ms: Total execution time.
        calls: Number of function calls.
        time_per_call_ms: Average time per call.
        memory_delta_mb: Memory change during execution.
        top_functions: Top time-consuming functions.
        call_stack: Call stack information.
    """

    function_name: str
    total_time_ms: float
    calls: int = 1
    time_per_call_ms: float = 0.0
    memory_delta_mb: float = 0.0
    top_functions: List[Dict[str, Any]] = field(default_factory=list)
    call_stack: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "total_time_ms": self.total_time_ms,
            "calls": self.calls,
            "time_per_call_ms": self.time_per_call_ms,
            "memory_delta_mb": self.memory_delta_mb,
            "top_functions": self.top_functions,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Profile: {self.function_name}",
            f"  Total time: {self.total_time_ms:.2f}ms",
            f"  Calls: {self.calls}",
            f"  Time/call: {self.time_per_call_ms:.2f}ms",
            f"  Memory delta: {self.memory_delta_mb:.2f}MB",
        ]

        if self.top_functions:
            lines.append("  Top functions:")
            for func in self.top_functions[:5]:
                lines.append(
                    f"    {func['name']}: {func['cumtime_ms']:.2f}ms "
                    f"({func['calls']} calls)"
                )

        return "\n".join(lines)


class Profiler:
    """Performance profiler for MemoryCompiler operations.

    Provides detailed profiling of function execution including
    time, memory usage, and call stacks.

    Example:
        >>> profiler = Profiler()
        >>> result = profiler.profile(my_function, arg1, arg2)
        >>> print(result.summary())
    """

    def __init__(
        self,
        enabled: bool = True,
        track_memory: bool = True,
        detailed: bool = False,
    ) -> None:
        """Initialize profiler.

        Args:
            enabled: Whether profiling is enabled.
            track_memory: Whether to track memory usage.
            detailed: Whether to capture detailed call stacks.
        """
        self.enabled = enabled
        self.track_memory = track_memory
        self.detailed = detailed
        self._results: List[ProfileResult] = []

    def profile(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> tuple[Any, ProfileResult]:
        """Profile a function call.

        Args:
            func: Function to profile.
            *args: Function arguments.
            **kwargs: Function keyword arguments.

        Returns:
            Tuple of (function result, ProfileResult).
        """
        if not self.enabled:
            result = func(*args, **kwargs)
            return result, ProfileResult(
                function_name=func.__name__,
                total_time_ms=0,
            )

        # Memory tracking
        memory_before = 0
        if self.track_memory:
            memory_before = self._get_memory_usage()

        # CPU profiling
        if self.detailed:
            profiler = cProfile.Profile()
            profiler.enable()

        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if self.detailed:
                profiler.disable()

        # Memory after
        memory_after = 0
        if self.track_memory:
            memory_after = self._get_memory_usage()

        # Build result
        profile_result = ProfileResult(
            function_name=func.__name__,
            total_time_ms=elapsed_ms,
            time_per_call_ms=elapsed_ms,
            memory_delta_mb=(memory_after - memory_before) / (1024 * 1024),
        )

        if self.detailed:
            profile_result.top_functions = self._extract_top_functions(profiler)
            profile_result.call_stack = self._get_call_stack(profiler)

        self._results.append(profile_result)
        return result, profile_result

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0

    def _extract_top_functions(
        self,
        profiler: cProfile.Profile,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Extract top time-consuming functions."""
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")

        # Get top functions
        top_functions = []
        for func_info, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:top_n]:
            filename, lineno, func_name = func_info
            top_functions.append({
                "name": func_name,
                "file": filename,
                "line": lineno,
                "calls": nc,
                "tottime_ms": tt * 1000,
                "cumtime_ms": ct * 1000,
            })

        return top_functions

    def _get_call_stack(self, profiler: cProfile.Profile) -> str:
        """Get formatted call stack."""
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)
        return stream.getvalue()

    def get_results(self) -> List[ProfileResult]:
        """Get all profiling results."""
        return self._results.copy()

    def clear_results(self) -> None:
        """Clear profiling results."""
        self._results.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all profiling results."""
        if not self._results:
            return {"count": 0}

        total_time = sum(r.total_time_ms for r in self._results)
        total_memory = sum(r.memory_delta_mb for r in self._results)

        return {
            "count": len(self._results),
            "total_time_ms": total_time,
            "avg_time_ms": total_time / len(self._results),
            "total_memory_mb": total_memory,
            "slowest": max(self._results, key=lambda r: r.total_time_ms).to_dict(),
        }


def profile(
    enabled: bool = True,
    track_memory: bool = False,
    log_result: bool = True,
) -> Callable[[F], F]:
    """Decorator to profile a function.

    Example:
        >>> @profile(track_memory=True)
        ... def slow_function():
        ...     time.sleep(0.1)
        ...     return "done"

    Args:
        enabled: Whether profiling is enabled.
        track_memory: Whether to track memory.
        log_result: Whether to log the result.

    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        profiler = Profiler(enabled=enabled, track_memory=track_memory)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result, profile_result = profiler.profile(func, *args, **kwargs)

            if log_result and profile_result.total_time_ms > 10:
                logger.debug(profile_result.summary())

            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if log_result and elapsed_ms > 10:
                logger.debug(f"Profile: {func.__name__} - {elapsed_ms:.2f}ms")

            return result

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore

    return decorator


class PerformanceTimer:
    """Context manager for timing code blocks.

    Example:
        >>> with PerformanceTimer("data_loading") as timer:
        ...     data = load_data()
        >>> print(f"Took {timer.elapsed_ms}ms")
    """

    def __init__(self, name: str, log: bool = True) -> None:
        """Initialize timer.

        Args:
            name: Name for the timed block.
            log: Whether to log on exit.
        """
        self.name = name
        self.log = log
        self.start_time: float = 0
        self.end_time: float = 0

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.perf_counter() - self.start_time) * 1000

    def __enter__(self) -> "PerformanceTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()

        if self.log:
            logger.debug(f"Timer [{self.name}]: {self.elapsed_ms:.2f}ms")

        return False


class MemoryTracker:
    """Track memory usage over time.

    Example:
        >>> tracker = MemoryTracker()
        >>> tracker.snapshot("before")
        >>> # do work
        >>> tracker.snapshot("after")
        >>> print(tracker.diff("before", "after"))
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._snapshots: Dict[str, int] = {}

    def snapshot(self, name: str) -> int:
        """Take a memory snapshot.

        Args:
            name: Snapshot name.

        Returns:
            Current memory usage in bytes.
        """
        try:
            import psutil
            memory = psutil.Process().memory_info().rss
        except ImportError:
            memory = 0

        self._snapshots[name] = memory
        return memory

    def diff(self, name1: str, name2: str) -> Dict[str, Any]:
        """Get difference between two snapshots.

        Args:
            name1: First snapshot name.
            name2: Second snapshot name.

        Returns:
            Difference information.
        """
        mem1 = self._snapshots.get(name1, 0)
        mem2 = self._snapshots.get(name2, 0)
        diff = mem2 - mem1

        return {
            "from": name1,
            "to": name2,
            "diff_bytes": diff,
            "diff_mb": diff / (1024 * 1024),
            "snapshot1_mb": mem1 / (1024 * 1024),
            "snapshot2_mb": mem2 / (1024 * 1024),
        }

    def clear(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()
