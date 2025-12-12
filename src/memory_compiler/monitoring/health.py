"""Health checking for MemoryCompiler components."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a component.

    Attributes:
        name: Component name.
        status: Health status.
        message: Status message.
        latency_ms: Check latency.
        last_check: Last check timestamp.
        details: Additional details.
    """

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    last_check: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check,
            "details": self.details,
        }

    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY


@dataclass
class HealthReport:
    """Overall health report.

    Attributes:
        status: Overall status.
        components: Individual component health.
        timestamp: Report timestamp.
        version: Service version.
        uptime_seconds: Service uptime.
    """

    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: float = field(default_factory=time.time)
    version: str = ""
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "components": [c.to_dict() for c in self.components],
        }

    @property
    def is_healthy(self) -> bool:
        """Check if overall status is healthy."""
        return self.status == HealthStatus.HEALTHY


# Type for health check functions
HealthCheckFn = Callable[[], ComponentHealth]


class HealthChecker:
    """Health checker for MemoryCompiler components.

    Manages health checks for all system components and
    provides aggregated health reports.

    Example:
        >>> checker = HealthChecker()
        >>> checker.register("storage", check_storage_health)
        >>> checker.register("model", check_model_health)
        >>> report = checker.check_all()
        >>> print(report.status)
    """

    def __init__(
        self,
        version: str = "1.0.0",
        check_timeout: float = 5.0,
    ) -> None:
        """Initialize health checker.

        Args:
            version: Service version.
            check_timeout: Timeout for individual checks.
        """
        self.version = version
        self.check_timeout = check_timeout
        self._checks: Dict[str, HealthCheckFn] = {}
        self._last_report: Optional[HealthReport] = None
        self._start_time = time.time()

    def register(
        self,
        name: str,
        check_fn: HealthCheckFn,
    ) -> None:
        """Register a health check.

        Args:
            name: Component name.
            check_fn: Function that returns ComponentHealth.
        """
        self._checks[name] = check_fn
        logger.debug(f"Registered health check: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a health check.

        Args:
            name: Component name.

        Returns:
            True if unregistered.
        """
        if name in self._checks:
            del self._checks[name]
            return True
        return False

    def check(self, name: str) -> ComponentHealth:
        """Run a single health check.

        Args:
            name: Component name to check.

        Returns:
            Component health status.
        """
        if name not in self._checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Health check not registered",
            )

        check_fn = self._checks[name]
        start_time = time.perf_counter()

        try:
            result = check_fn()
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            result.last_check = time.time()
            return result

        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

    def check_all(self) -> HealthReport:
        """Run all health checks.

        Returns:
            Aggregated health report.
        """
        components = []

        for name in self._checks:
            health = self.check(name)
            components.append(health)

        # Determine overall status
        overall_status = self._aggregate_status(components)

        report = HealthReport(
            status=overall_status,
            components=components,
            version=self.version,
            uptime_seconds=time.time() - self._start_time,
        )

        self._last_report = report
        return report

    def _aggregate_status(
        self,
        components: List[ComponentHealth],
    ) -> HealthStatus:
        """Aggregate component statuses into overall status."""
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in components]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN

    def get_last_report(self) -> Optional[HealthReport]:
        """Get the last health report."""
        return self._last_report

    @property
    def uptime_seconds(self) -> float:
        """Get checker uptime in seconds."""
        return time.time() - self._start_time


# Built-in health checks
def create_memory_check(
    threshold_mb: float = 1000,
) -> HealthCheckFn:
    """Create a memory usage health check.

    Args:
        threshold_mb: Memory threshold in MB.

    Returns:
        Health check function.
    """
    def check() -> ComponentHealth:
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            if memory_mb > threshold_mb:
                return ComponentHealth(
                    name="memory",
                    status=HealthStatus.DEGRADED,
                    message=f"High memory usage: {memory_mb:.1f}MB",
                    details={"usage_mb": memory_mb, "threshold_mb": threshold_mb},
                )

            return ComponentHealth(
                name="memory",
                status=HealthStatus.HEALTHY,
                message=f"Memory usage: {memory_mb:.1f}MB",
                details={"usage_mb": memory_mb},
            )

        except ImportError:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="psutil not installed",
            )

    return check


def create_disk_check(
    path: str = "/",
    threshold_percent: float = 90,
) -> HealthCheckFn:
    """Create a disk usage health check.

    Args:
        path: Path to check.
        threshold_percent: Usage threshold percentage.

    Returns:
        Health check function.
    """
    def check() -> ComponentHealth:
        try:
            import psutil
            usage = psutil.disk_usage(path)
            percent = usage.percent

            if percent > threshold_percent:
                return ComponentHealth(
                    name="disk",
                    status=HealthStatus.DEGRADED,
                    message=f"High disk usage: {percent}%",
                    details={
                        "path": path,
                        "percent": percent,
                        "free_gb": usage.free / (1024**3),
                    },
                )

            return ComponentHealth(
                name="disk",
                status=HealthStatus.HEALTHY,
                message=f"Disk usage: {percent}%",
                details={
                    "path": path,
                    "percent": percent,
                    "free_gb": usage.free / (1024**3),
                },
            )

        except ImportError:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.UNKNOWN,
                message="psutil not installed",
            )

    return check


def create_storage_check(
    store: Any,
) -> HealthCheckFn:
    """Create a storage backend health check.

    Args:
        store: Storage backend to check.

    Returns:
        Health check function.
    """
    def check() -> ComponentHealth:
        try:
            # Try to list keys
            keys = store.list_keys()

            return ComponentHealth(
                name="storage",
                status=HealthStatus.HEALTHY,
                message=f"Storage accessible, {len(keys)} items",
                details={"item_count": len(keys)},
            )

        except Exception as e:
            return ComponentHealth(
                name="storage",
                status=HealthStatus.UNHEALTHY,
                message=f"Storage error: {str(e)}",
            )

    return check


def create_model_check(
    model_loader: Callable,
) -> HealthCheckFn:
    """Create a model availability health check.

    Args:
        model_loader: Function to load/check model.

    Returns:
        Health check function.
    """
    def check() -> ComponentHealth:
        try:
            model_loader()

            return ComponentHealth(
                name="model",
                status=HealthStatus.HEALTHY,
                message="Model available",
            )

        except Exception as e:
            return ComponentHealth(
                name="model",
                status=HealthStatus.UNHEALTHY,
                message=f"Model error: {str(e)}",
            )

    return check


# Global health checker
_global_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker."""
    global _global_checker
    if _global_checker is None:
        _global_checker = HealthChecker()

        # Register default checks
        _global_checker.register("memory", create_memory_check())

    return _global_checker


def quick_health_check() -> Dict[str, Any]:
    """Perform a quick health check.

    Returns:
        Health status dictionary.
    """
    checker = get_health_checker()
    report = checker.check_all()
    return report.to_dict()
