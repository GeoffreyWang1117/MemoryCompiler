"""Metrics and monitoring for MemoryCompiler.

Provides observability through:
- Prometheus metrics
- OpenTelemetry tracing
- Structured logging
- Performance profiling
"""

from memory_compiler.monitoring.metrics import (
    MetricsCollector,
    CompressionMetrics,
    PerformanceMetrics,
)
from memory_compiler.monitoring.tracing import (
    Tracer,
    SpanContext,
    trace,
)
from memory_compiler.monitoring.profiler import (
    Profiler,
    ProfileResult,
    profile,
)
from memory_compiler.monitoring.health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
)

__all__ = [
    "MetricsCollector",
    "CompressionMetrics",
    "PerformanceMetrics",
    "Tracer",
    "SpanContext",
    "trace",
    "Profiler",
    "ProfileResult",
    "profile",
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
]
