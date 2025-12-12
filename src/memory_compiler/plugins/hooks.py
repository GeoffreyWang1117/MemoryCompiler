"""Hook system for pipeline customization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class HookType(Enum):
    """Types of hooks in the pipeline."""

    PRE_EXTRACTION = "pre_extraction"
    POST_EXTRACTION = "post_extraction"
    PRE_OPTIMIZATION = "pre_optimization"
    POST_OPTIMIZATION = "post_optimization"
    PRE_GENERATION = "pre_generation"
    POST_GENERATION = "post_generation"
    ON_ERROR = "on_error"
    ON_COMPLETE = "on_complete"


@dataclass
class HookContext:
    """Context passed to hook functions.

    Attributes:
        stage: Current pipeline stage.
        data: Data being processed.
        metadata: Additional metadata.
        errors: Any errors that occurred.
    """

    stage: HookType
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[Exception] = field(default_factory=list)


# Type alias for hook functions
HookFunction = Callable[[HookContext], Optional[HookContext]]


@dataclass
class RegisteredHook:
    """A registered hook with metadata."""

    name: str
    hook_type: HookType
    function: HookFunction
    priority: int = 0  # Higher priority runs first
    enabled: bool = True


class HookManager:
    """Manage and execute pipeline hooks.

    Allows registering custom functions to be called at various
    points in the compression pipeline.

    Example:
        >>> manager = HookManager()
        >>>
        >>> @manager.hook(HookType.POST_EXTRACTION)
        ... def log_entities(ctx):
        ...     print(f"Extracted {len(ctx.data.entities)} entities")
        ...     return ctx
        >>>
        >>> # Later, execute hooks
        >>> manager.execute(HookType.POST_EXTRACTION, context)
    """

    def __init__(self) -> None:
        """Initialize hook manager."""
        self._hooks: Dict[HookType, List[RegisteredHook]] = {
            ht: [] for ht in HookType
        }

    def register(
        self,
        hook_type: HookType,
        function: HookFunction,
        name: Optional[str] = None,
        priority: int = 0,
    ) -> str:
        """Register a hook function.

        Args:
            hook_type: Type of hook.
            function: Hook function to call.
            name: Optional hook name.
            priority: Execution priority (higher = earlier).

        Returns:
            Hook name for later reference.
        """
        name = name or f"{hook_type.value}_{len(self._hooks[hook_type])}"

        hook = RegisteredHook(
            name=name,
            hook_type=hook_type,
            function=function,
            priority=priority,
        )

        self._hooks[hook_type].append(hook)
        self._hooks[hook_type].sort(key=lambda h: h.priority, reverse=True)

        logger.debug(f"Registered hook: {name} for {hook_type.value}")
        return name

    def unregister(self, name: str) -> bool:
        """Unregister a hook by name.

        Args:
            name: Hook name.

        Returns:
            True if unregistered.
        """
        for hook_type, hooks in self._hooks.items():
            for i, hook in enumerate(hooks):
                if hook.name == name:
                    hooks.pop(i)
                    logger.debug(f"Unregistered hook: {name}")
                    return True

        return False

    def enable(self, name: str) -> bool:
        """Enable a hook by name."""
        for hooks in self._hooks.values():
            for hook in hooks:
                if hook.name == name:
                    hook.enabled = True
                    return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a hook by name."""
        for hooks in self._hooks.values():
            for hook in hooks:
                if hook.name == name:
                    hook.enabled = False
                    return True
        return False

    def execute(
        self,
        hook_type: HookType,
        context: HookContext,
        stop_on_error: bool = False,
    ) -> HookContext:
        """Execute all hooks of a given type.

        Args:
            hook_type: Type of hooks to execute.
            context: Context to pass to hooks.
            stop_on_error: Whether to stop on first error.

        Returns:
            Updated context after all hooks.
        """
        hooks = self._hooks[hook_type]

        for hook in hooks:
            if not hook.enabled:
                continue

            try:
                result = hook.function(context)
                if result is not None:
                    context = result

            except Exception as e:
                logger.error(f"Error in hook {hook.name}: {e}")
                context.errors.append(e)

                if stop_on_error:
                    break

        return context

    def hook(
        self,
        hook_type: HookType,
        name: Optional[str] = None,
        priority: int = 0,
    ) -> Callable[[HookFunction], HookFunction]:
        """Decorator to register a hook function.

        Example:
            >>> @manager.hook(HookType.POST_EXTRACTION)
            ... def my_hook(ctx):
            ...     # Process context
            ...     return ctx
        """

        def decorator(fn: HookFunction) -> HookFunction:
            self.register(hook_type, fn, name=name or fn.__name__, priority=priority)
            return fn

        return decorator

    def list_hooks(
        self,
        hook_type: Optional[HookType] = None,
    ) -> List[RegisteredHook]:
        """List registered hooks.

        Args:
            hook_type: Optional filter by type.

        Returns:
            List of registered hooks.
        """
        if hook_type:
            return list(self._hooks[hook_type])

        all_hooks = []
        for hooks in self._hooks.values():
            all_hooks.extend(hooks)
        return all_hooks

    def clear(self, hook_type: Optional[HookType] = None) -> None:
        """Clear all hooks or hooks of a specific type."""
        if hook_type:
            self._hooks[hook_type].clear()
        else:
            for hooks in self._hooks.values():
                hooks.clear()


# Global hook manager instance
_global_hook_manager: Optional[HookManager] = None


def get_hook_manager() -> HookManager:
    """Get the global hook manager."""
    global _global_hook_manager
    if _global_hook_manager is None:
        _global_hook_manager = HookManager()
    return _global_hook_manager


def hook(
    hook_type: HookType,
    name: Optional[str] = None,
    priority: int = 0,
) -> Callable[[HookFunction], HookFunction]:
    """Decorator to register a hook with the global manager.

    Example:
        >>> @hook(HookType.POST_EXTRACTION)
        ... def log_extraction(ctx):
        ...     print(f"Extraction complete")
        ...     return ctx
    """
    return get_hook_manager().hook(hook_type, name, priority)


# Pre-built hooks
def create_logging_hook(
    log_level: str = "debug",
    message_template: str = "Pipeline stage: {stage}",
) -> HookFunction:
    """Create a logging hook.

    Args:
        log_level: Logging level.
        message_template: Message template with {stage} placeholder.

    Returns:
        Hook function.
    """

    def logging_hook(ctx: HookContext) -> HookContext:
        message = message_template.format(stage=ctx.stage.value)

        if log_level == "debug":
            logger.debug(message)
        elif log_level == "info":
            logger.info(message)
        elif log_level == "warning":
            logger.warning(message)

        return ctx

    return logging_hook


def create_metrics_hook(
    metrics_collector: Dict[str, Any],
) -> HookFunction:
    """Create a metrics collection hook.

    Args:
        metrics_collector: Dictionary to collect metrics into.

    Returns:
        Hook function.
    """
    import time

    def metrics_hook(ctx: HookContext) -> HookContext:
        stage = ctx.stage.value
        timestamp = time.time()

        if stage not in metrics_collector:
            metrics_collector[stage] = {
                "count": 0,
                "last_timestamp": 0,
            }

        metrics_collector[stage]["count"] += 1
        metrics_collector[stage]["last_timestamp"] = timestamp

        return ctx

    return metrics_hook


def create_validation_hook(
    validators: List[Callable[[Any], bool]],
    error_message: str = "Validation failed",
) -> HookFunction:
    """Create a validation hook.

    Args:
        validators: List of validation functions.
        error_message: Error message on failure.

    Returns:
        Hook function.
    """

    def validation_hook(ctx: HookContext) -> HookContext:
        for validator in validators:
            if not validator(ctx.data):
                raise ValueError(error_message)

        return ctx

    return validation_hook
