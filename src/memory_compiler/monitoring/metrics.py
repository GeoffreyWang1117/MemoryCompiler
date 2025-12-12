"""Metrics collection for MemoryCompiler."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value.

    Attributes:
        name: Metric name.
        value: Current value.
        metric_type: Type of metric.
        labels: Metric labels.
        timestamp: When recorded.
        description: Metric description.
    """

    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class CompressionMetrics:
    """Metrics specific to compression operations.

    Attributes:
        original_tokens: Original token count.
        compressed_tokens: Compressed token count.
        compression_ratio: Achieved compression ratio.
        entities_extracted: Number of entities extracted.
        facts_extracted: Number of facts extracted.
        passes_applied: Optimization passes applied.
        latency_ms: Processing latency in milliseconds.
    """

    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    entities_extracted: int = 0
    facts_extracted: int = 0
    passes_applied: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "entities_extracted": self.entities_extracted,
            "facts_extracted": self.facts_extracted,
            "passes_applied": self.passes_applied,
            "latency_ms": self.latency_ms,
        }


@dataclass
class PerformanceMetrics:
    """System performance metrics.

    Attributes:
        requests_total: Total number of requests.
        requests_success: Successful requests.
        requests_failed: Failed requests.
        avg_latency_ms: Average latency.
        p50_latency_ms: 50th percentile latency.
        p95_latency_ms: 95th percentile latency.
        p99_latency_ms: 99th percentile latency.
        active_sessions: Current active sessions.
        memory_usage_mb: Memory usage in MB.
    """

    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    active_sessions: int = 0
    memory_usage_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requests_total": self.requests_total,
            "requests_success": self.requests_success,
            "requests_failed": self.requests_failed,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "active_sessions": self.active_sessions,
            "memory_usage_mb": self.memory_usage_mb,
        }


class MetricsCollector:
    """Collect and aggregate metrics.

    Provides a central place for recording and querying metrics
    across the MemoryCompiler system.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.increment("compressions_total")
        >>> collector.observe("compression_latency", 150.5)
        >>> metrics = collector.get_all()
    """

    def __init__(self, namespace: str = "memory_compiler") -> None:
        """Initialize collector.

        Args:
            namespace: Metric namespace prefix.
        """
        self.namespace = namespace
        self._metrics: Dict[str, MetricValue] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._lock = Lock()
        self._start_time = time.time()

        # Initialize default metrics
        self._init_default_metrics()

    def _init_default_metrics(self) -> None:
        """Initialize default metrics."""
        defaults = [
            ("compressions_total", MetricType.COUNTER, "Total compression operations"),
            ("compressions_success", MetricType.COUNTER, "Successful compressions"),
            ("compressions_failed", MetricType.COUNTER, "Failed compressions"),
            ("extractions_total", MetricType.COUNTER, "Total extraction operations"),
            ("tokens_processed", MetricType.COUNTER, "Total tokens processed"),
            ("tokens_saved", MetricType.COUNTER, "Tokens saved by compression"),
            ("entities_extracted", MetricType.COUNTER, "Total entities extracted"),
            ("facts_extracted", MetricType.COUNTER, "Total facts extracted"),
            ("active_sessions", MetricType.GAUGE, "Current active sessions"),
            ("memory_usage_bytes", MetricType.GAUGE, "Memory usage in bytes"),
        ]

        for name, mtype, desc in defaults:
            self._metrics[name] = MetricValue(
                name=f"{self.namespace}_{name}",
                value=0.0,
                metric_type=mtype,
                description=desc,
            )

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name.
            value: Value to add.
            labels: Optional labels.
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = MetricValue(
                    name=f"{self.namespace}_{name}",
                    value=0.0,
                    metric_type=MetricType.COUNTER,
                    labels=labels or {},
                )

            self._metrics[name].value += value
            self._metrics[name].timestamp = time.time()

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric.

        Args:
            name: Metric name.
            value: Gauge value.
            labels: Optional labels.
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = MetricValue(
                    name=f"{self.namespace}_{name}",
                    value=value,
                    metric_type=MetricType.GAUGE,
                    labels=labels or {},
                )
            else:
                self._metrics[name].value = value
                self._metrics[name].timestamp = time.time()

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a value for histogram/summary.

        Args:
            name: Metric name.
            value: Observed value.
            labels: Optional labels.
        """
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = []

            self._histograms[name].append(value)

            # Keep last 10000 observations
            if len(self._histograms[name]) > 10000:
                self._histograms[name] = self._histograms[name][-10000:]

    def get(self, name: str) -> Optional[MetricValue]:
        """Get a metric by name.

        Args:
            name: Metric name.

        Returns:
            MetricValue or None.
        """
        return self._metrics.get(name)

    def get_all(self) -> Dict[str, MetricValue]:
        """Get all metrics."""
        with self._lock:
            return dict(self._metrics)

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics.

        Args:
            name: Histogram name.

        Returns:
            Statistics dict with count, sum, avg, percentiles.
        """
        with self._lock:
            values = self._histograms.get(name, [])

            if not values:
                return {
                    "count": 0,
                    "sum": 0.0,
                    "avg": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                }

            sorted_values = sorted(values)
            count = len(sorted_values)

            return {
                "count": count,
                "sum": sum(sorted_values),
                "avg": sum(sorted_values) / count,
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "p50": sorted_values[int(count * 0.50)],
                "p95": sorted_values[int(count * 0.95)],
                "p99": sorted_values[int(count * 0.99)],
            }

    def record_compression(self, metrics: CompressionMetrics) -> None:
        """Record compression operation metrics.

        Args:
            metrics: Compression metrics to record.
        """
        self.increment("compressions_total")
        self.increment("tokens_processed", metrics.original_tokens)
        self.increment("tokens_saved", metrics.original_tokens - metrics.compressed_tokens)
        self.increment("entities_extracted", metrics.entities_extracted)
        self.increment("facts_extracted", metrics.facts_extracted)
        self.observe("compression_latency_ms", metrics.latency_ms)
        self.observe("compression_ratio", metrics.compression_ratio)

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get aggregated performance metrics."""
        latency_stats = self.get_histogram_stats("compression_latency_ms")

        return PerformanceMetrics(
            requests_total=int(self._metrics.get("compressions_total", MetricValue("", 0, MetricType.COUNTER)).value),
            requests_success=int(self._metrics.get("compressions_success", MetricValue("", 0, MetricType.COUNTER)).value),
            requests_failed=int(self._metrics.get("compressions_failed", MetricValue("", 0, MetricType.COUNTER)).value),
            avg_latency_ms=latency_stats["avg"],
            p50_latency_ms=latency_stats["p50"],
            p95_latency_ms=latency_stats["p95"],
            p99_latency_ms=latency_stats["p99"],
            active_sessions=int(self._metrics.get("active_sessions", MetricValue("", 0, MetricType.GAUGE)).value),
        )

    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string.
        """
        lines = []

        for name, metric in self._metrics.items():
            full_name = f"{self.namespace}_{name}"

            # Add HELP line
            if metric.description:
                lines.append(f"# HELP {full_name} {metric.description}")

            # Add TYPE line
            lines.append(f"# TYPE {full_name} {metric.metric_type.value}")

            # Add metric value
            labels_str = ""
            if metric.labels:
                label_pairs = [f'{k}="{v}"' for k, v in metric.labels.items()]
                labels_str = "{" + ",".join(label_pairs) + "}"

            lines.append(f"{full_name}{labels_str} {metric.value}")

        # Add histogram metrics
        for name, values in self._histograms.items():
            if not values:
                continue

            full_name = f"{self.namespace}_{name}"
            stats = self.get_histogram_stats(name)

            lines.append(f"# TYPE {full_name} histogram")
            lines.append(f"{full_name}_count {stats['count']}")
            lines.append(f"{full_name}_sum {stats['sum']}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._histograms.clear()
            self._init_default_metrics()

    @property
    def uptime_seconds(self) -> float:
        """Get collector uptime in seconds."""
        return time.time() - self._start_time


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def record_compression_metrics(metrics: CompressionMetrics) -> None:
    """Record compression metrics to global collector."""
    get_metrics_collector().record_compression(metrics)


class MetricsMiddleware:
    """Middleware for automatic metrics collection."""

    def __init__(self, collector: Optional[MetricsCollector] = None) -> None:
        """Initialize middleware.

        Args:
            collector: Metrics collector to use.
        """
        self.collector = collector or get_metrics_collector()

    def wrap_function(
        self,
        func: Callable,
        operation_name: str,
    ) -> Callable:
        """Wrap a function to collect metrics.

        Args:
            func: Function to wrap.
            operation_name: Name for metrics.

        Returns:
            Wrapped function.
        """
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            self.collector.increment(f"{operation_name}_total")

            try:
                result = func(*args, **kwargs)
                self.collector.increment(f"{operation_name}_success")
                return result

            except Exception as e:
                self.collector.increment(f"{operation_name}_failed")
                raise

            finally:
                latency = (time.perf_counter() - start_time) * 1000
                self.collector.observe(f"{operation_name}_latency_ms", latency)

        return wrapper
