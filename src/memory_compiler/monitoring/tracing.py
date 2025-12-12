"""Distributed tracing support for MemoryCompiler."""

from __future__ import annotations

import contextvars
import functools
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from loguru import logger

# Context variable for current span
_current_span: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar(
    "current_span", default=None
)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class SpanContext:
    """Context for distributed tracing.

    Attributes:
        trace_id: Unique trace identifier.
        span_id: Unique span identifier.
        parent_span_id: Parent span ID if any.
        baggage: Baggage items for propagation.
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for propagation."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanContext":
        """Create from dictionary."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            baggage=data.get("baggage", {}),
        )

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation."""
        return {
            "X-Trace-ID": self.trace_id,
            "X-Span-ID": self.span_id,
            "X-Parent-Span-ID": self.parent_span_id or "",
        }

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional["SpanContext"]:
        """Create from HTTP headers."""
        trace_id = headers.get("X-Trace-ID")
        span_id = headers.get("X-Span-ID")

        if not trace_id:
            return None

        return cls(
            trace_id=trace_id,
            span_id=span_id or cls._generate_id(),
            parent_span_id=headers.get("X-Parent-Span-ID") or None,
        )

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique ID."""
        return uuid.uuid4().hex[:16]


@dataclass
class Span:
    """A single trace span.

    Attributes:
        name: Span operation name.
        context: Span context.
        start_time: When span started.
        end_time: When span ended.
        status: Span status (ok, error).
        attributes: Span attributes.
        events: Span events.
        children: Child spans.
    """

    name: str
    context: SpanContext
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "ok"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    children: List["Span"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_status(self, status: str, message: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        if message:
            self.attributes["status_message"] = message

    def end(self) -> None:
        """End the span."""
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
            "children": [c.to_dict() for c in self.children],
        }


class Tracer:
    """Tracer for distributed tracing.

    Provides span creation and management for tracing
    operations across the MemoryCompiler system.

    Example:
        >>> tracer = Tracer("memory_compiler")
        >>> with tracer.start_span("compression") as span:
        ...     span.set_attribute("tokens", 1000)
        ...     # do work
    """

    def __init__(
        self,
        service_name: str = "memory_compiler",
        exporter: Optional[Callable[[Span], None]] = None,
    ) -> None:
        """Initialize tracer.

        Args:
            service_name: Name of the service.
            exporter: Optional function to export completed spans.
        """
        self.service_name = service_name
        self.exporter = exporter or self._default_exporter
        self._active_spans: Dict[str, Span] = {}

    def _default_exporter(self, span: Span) -> None:
        """Default span exporter (logs)."""
        if span.duration_ms > 100:  # Only log slow spans
            logger.debug(
                f"Span completed: {span.name} "
                f"[{span.duration_ms:.2f}ms] "
                f"status={span.status}"
            )

    def start_span(
        self,
        name: str,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "SpanContextManager":
        """Start a new span.

        Args:
            name: Span operation name.
            parent: Optional parent span.
            attributes: Initial attributes.

        Returns:
            SpanContextManager for use with 'with' statement.
        """
        # Get parent from context if not provided
        if parent is None:
            parent = _current_span.get()

        # Create context
        if parent:
            context = SpanContext(
                trace_id=parent.context.trace_id,
                span_id=SpanContext._generate_id(),
                parent_span_id=parent.context.span_id,
            )
        else:
            context = SpanContext(
                trace_id=SpanContext._generate_id(),
                span_id=SpanContext._generate_id(),
            )

        span = Span(
            name=name,
            context=context,
            attributes={"service": self.service_name, **(attributes or {})},
        )

        if parent:
            parent.children.append(span)

        self._active_spans[context.span_id] = span
        return SpanContextManager(self, span)

    def _end_span(self, span: Span) -> None:
        """End and export a span."""
        span.end()

        if span.context.span_id in self._active_spans:
            del self._active_spans[span.context.span_id]

        # Export root spans (spans without parents in our active set)
        if span.context.parent_span_id is None:
            self.exporter(span)

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return _current_span.get()

    def inject_context(self, carrier: Dict[str, str]) -> None:
        """Inject trace context into a carrier (e.g., HTTP headers).

        Args:
            carrier: Dictionary to inject headers into.
        """
        span = _current_span.get()
        if span:
            carrier.update(span.context.to_headers())

    def extract_context(
        self,
        carrier: Dict[str, str],
    ) -> Optional[SpanContext]:
        """Extract trace context from a carrier.

        Args:
            carrier: Dictionary containing trace headers.

        Returns:
            SpanContext if found.
        """
        return SpanContext.from_headers(carrier)


class SpanContextManager:
    """Context manager for spans."""

    def __init__(self, tracer: Tracer, span: Span) -> None:
        self.tracer = tracer
        self.span = span
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> Span:
        self._token = _current_span.set(self.span)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.span.set_status("error", str(exc_val))
            self.span.add_event("exception", {
                "type": exc_type.__name__,
                "message": str(exc_val),
            })

        self.tracer._end_span(self.span)

        if self._token:
            _current_span.reset(self._token)

        return False  # Don't suppress exceptions


# Global tracer instance
_global_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get the global tracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def trace(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator to trace a function.

    Example:
        >>> @trace("process_dialogue")
        ... def process(dialogue: str) -> str:
        ...     # processing
        ...     return result

    Args:
        name: Span name (defaults to function name).
        attributes: Initial span attributes.

    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.start_span(span_name, attributes=attributes) as span:
                # Add function info
                span.set_attribute("function", func.__name__)
                span.set_attribute("module", func.__module__)

                result = func(*args, **kwargs)

                return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.start_span(span_name, attributes=attributes) as span:
                span.set_attribute("function", func.__name__)
                span.set_attribute("module", func.__module__)
                span.set_attribute("async", True)

                result = await func(*args, **kwargs)

                return result

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore

    return decorator


class OpenTelemetryExporter:
    """Export spans to OpenTelemetry collector."""

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        service_name: str = "memory_compiler",
    ) -> None:
        """Initialize exporter.

        Args:
            endpoint: OTLP endpoint URL.
            service_name: Service name for traces.
        """
        self.endpoint = endpoint
        self.service_name = service_name
        self._batch: List[Span] = []
        self._batch_size = 100

    def export(self, span: Span) -> None:
        """Export a span.

        Args:
            span: Span to export.
        """
        self._batch.append(span)

        if len(self._batch) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        """Flush pending spans."""
        if not self._batch:
            return

        # Convert to OTLP format
        spans_data = [self._to_otlp(s) for s in self._batch]

        try:
            # In production, send to collector
            # For now, just log
            logger.debug(f"Would export {len(spans_data)} spans to {self.endpoint}")
        except Exception as e:
            logger.error(f"Failed to export spans: {e}")
        finally:
            self._batch.clear()

    def _to_otlp(self, span: Span) -> Dict[str, Any]:
        """Convert span to OTLP format."""
        return {
            "traceId": span.context.trace_id,
            "spanId": span.context.span_id,
            "parentSpanId": span.context.parent_span_id,
            "name": span.name,
            "startTimeUnixNano": int(span.start_time * 1e9),
            "endTimeUnixNano": int((span.end_time or time.time()) * 1e9),
            "status": {"code": 1 if span.status == "ok" else 2},
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in span.attributes.items()
            ],
        }
