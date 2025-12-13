"""Tests for monitoring module."""

import pytest
import time
from memory_compiler.monitoring import (
    MetricsCollector,
    CompressionMetrics,
    PerformanceMetrics,
    Tracer,
    Span,
    trace,
    Profiler,
    profile,
    HealthChecker,
    HealthStatus,
)


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_record_compression_metrics(self):
        """Test recording compression metrics."""
        collector = MetricsCollector()

        metrics = CompressionMetrics(
            input_tokens=1000,
            output_tokens=300,
            compression_ratio=0.3,
            entity_count=5,
            fact_count=10,
            processing_time_ms=150.0,
        )

        collector.record_compression(metrics)

        stats = collector.get_stats()
        assert stats["total_compressions"] >= 1

    def test_record_performance_metrics(self):
        """Test recording performance metrics."""
        collector = MetricsCollector()

        metrics = PerformanceMetrics(
            operation="extract",
            duration_ms=50.0,
            memory_used_mb=100.0,
        )

        collector.record_performance(metrics)

        stats = collector.get_stats()
        assert "extract" in str(stats)

    def test_increment_counter(self):
        """Test incrementing counters."""
        collector = MetricsCollector()

        collector.increment("requests")
        collector.increment("requests")
        collector.increment("errors")

        stats = collector.get_stats()
        assert stats.get("requests", 0) >= 2

    def test_record_gauge(self):
        """Test recording gauge values."""
        collector = MetricsCollector()

        collector.set_gauge("active_sessions", 5)
        collector.set_gauge("active_sessions", 10)

        stats = collector.get_stats()
        assert stats.get("active_sessions", 0) == 10

    def test_record_histogram(self):
        """Test recording histogram values."""
        collector = MetricsCollector()

        for i in range(10):
            collector.observe("latency", i * 10)

        stats = collector.get_stats()
        assert "latency" in str(stats)

    def test_export_prometheus(self):
        """Test Prometheus format export."""
        collector = MetricsCollector()
        collector.increment("test_counter")

        prometheus_output = collector.export_prometheus()

        assert "test_counter" in prometheus_output

    def test_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()
        collector.increment("counter")
        collector.reset()

        stats = collector.get_stats()
        assert stats.get("counter", 0) == 0


class TestTracer:
    """Tests for Tracer."""

    def test_create_span(self):
        """Test creating a span."""
        tracer = Tracer(service_name="test-service")

        with tracer.span("test_operation") as span:
            span.set_attribute("key", "value")

        assert span.name == "test_operation"
        assert span.end_time is not None

    def test_nested_spans(self):
        """Test nested spans."""
        tracer = Tracer(service_name="test-service")

        with tracer.span("parent") as parent:
            with tracer.span("child") as child:
                pass

        # Child should reference parent
        assert child.parent_id == parent.span_id

    def test_span_attributes(self):
        """Test span attributes."""
        tracer = Tracer(service_name="test-service")

        with tracer.span("operation") as span:
            span.set_attribute("user_id", "123")
            span.set_attribute("count", 42)

        assert span.attributes["user_id"] == "123"
        assert span.attributes["count"] == 42

    def test_span_events(self):
        """Test span events."""
        tracer = Tracer(service_name="test-service")

        with tracer.span("operation") as span:
            span.add_event("processing_started")
            span.add_event("processing_completed", {"items": 10})

        assert len(span.events) == 2

    def test_span_error(self):
        """Test span error recording."""
        tracer = Tracer(service_name="test-service")

        try:
            with tracer.span("failing_operation") as span:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert span.status == "ERROR"

    def test_trace_decorator(self):
        """Test trace decorator."""
        tracer = Tracer(service_name="test-service")

        @trace(tracer)
        def traced_function(x, y):
            return x + y

        result = traced_function(1, 2)
        assert result == 3

    def test_get_active_spans(self):
        """Test getting active spans."""
        tracer = Tracer(service_name="test-service")

        with tracer.span("active_span"):
            active = tracer.get_active_spans()
            assert len(active) >= 1

    def test_export_spans(self):
        """Test exporting spans."""
        tracer = Tracer(service_name="test-service")

        with tracer.span("exportable"):
            pass

        spans = tracer.export_spans()
        assert len(spans) >= 1


class TestProfiler:
    """Tests for Profiler."""

    def test_profile_function(self):
        """Test profiling a function."""
        profiler = Profiler()

        with profiler.profile("test_operation"):
            time.sleep(0.01)  # Small delay

        result = profiler.get_last_result()
        assert result is not None
        assert result.duration_ms >= 10

    def test_profile_decorator(self):
        """Test profile decorator."""
        profiler = Profiler()

        @profile(profiler)
        def slow_function():
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"

        stats = profiler.get_stats()
        assert "slow_function" in str(stats)

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        profiler = Profiler(track_memory=True)

        with profiler.profile("memory_test"):
            data = [i for i in range(10000)]

        result = profiler.get_last_result()
        assert result.memory_delta_mb is not None

    def test_get_summary(self):
        """Test getting profile summary."""
        profiler = Profiler()

        for i in range(5):
            with profiler.profile("repeated_op"):
                pass

        summary = profiler.get_summary("repeated_op")

        assert summary["count"] == 5
        assert "avg_duration_ms" in summary

    def test_clear_results(self):
        """Test clearing profile results."""
        profiler = Profiler()

        with profiler.profile("op"):
            pass

        profiler.clear()

        assert len(profiler.get_all_results()) == 0


class TestHealthChecker:
    """Tests for HealthChecker."""

    def test_register_check(self):
        """Test registering a health check."""
        checker = HealthChecker()

        def my_check():
            return True, "OK"

        checker.register("my_service", my_check)

        result = checker.check("my_service")
        assert result.status == HealthStatus.HEALTHY

    def test_failing_check(self):
        """Test failing health check."""
        checker = HealthChecker()

        def failing_check():
            return False, "Service unavailable"

        checker.register("failing_service", failing_check)

        result = checker.check("failing_service")
        assert result.status == HealthStatus.UNHEALTHY
        assert "unavailable" in result.message

    def test_check_all(self):
        """Test checking all registered services."""
        checker = HealthChecker()

        checker.register("service1", lambda: (True, "OK"))
        checker.register("service2", lambda: (True, "OK"))

        results = checker.check_all()

        assert len(results) == 2
        assert all(r.status == HealthStatus.HEALTHY for r in results.values())

    def test_overall_health(self):
        """Test overall health status."""
        checker = HealthChecker()

        checker.register("healthy", lambda: (True, "OK"))
        checker.register("unhealthy", lambda: (False, "Down"))

        overall = checker.get_overall_health()

        # Should be degraded since one service is down
        assert overall.status != HealthStatus.HEALTHY

    def test_exception_handling(self):
        """Test exception handling in health check."""
        checker = HealthChecker()

        def throwing_check():
            raise RuntimeError("Check failed")

        checker.register("throwing", throwing_check)

        result = checker.check("throwing")
        assert result.status == HealthStatus.UNHEALTHY

    def test_timeout(self):
        """Test health check timeout."""
        checker = HealthChecker(default_timeout=0.1)

        def slow_check():
            time.sleep(1)
            return True, "OK"

        checker.register("slow", slow_check)

        result = checker.check("slow")
        # Should timeout and be unhealthy
        assert result.status == HealthStatus.UNHEALTHY

    def test_degraded_status(self):
        """Test degraded health status."""
        checker = HealthChecker()

        def degraded_check():
            return True, "OK", {"warning": "High latency"}

        checker.register("degraded", degraded_check)

        result = checker.check("degraded")
        # Custom checks can return extra info
        assert result.status == HealthStatus.HEALTHY

    def test_export_health_json(self):
        """Test exporting health as JSON."""
        checker = HealthChecker()
        checker.register("service", lambda: (True, "OK"))

        json_output = checker.export_json()

        assert "service" in json_output
        assert "status" in json_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
