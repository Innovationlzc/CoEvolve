"""Runtime adapters that connect SkillGraphs to concrete scenes."""

__all__ = [
    "DEFAULT_ISAAC_PROXY_TASK",
    "IsaacProxyExecutionConfig",
    "IsaacProxyRuntimeResult",
    "ManualUsdSceneRuntime",
    "ManualUsdRuntimeResult",
    "execute_isaac_proxy_skill_graph",
]


def __getattr__(name: str):
    if name in {
        "DEFAULT_ISAAC_PROXY_TASK",
        "IsaacProxyExecutionConfig",
        "IsaacProxyRuntimeResult",
        "execute_isaac_proxy_skill_graph",
    }:
        from . import isaac_proxy_runtime

        return getattr(isaac_proxy_runtime, name)
    if name in {"ManualUsdSceneRuntime", "ManualUsdRuntimeResult"}:
        from . import manual_usd_runtime

        return getattr(manual_usd_runtime, name)
    raise AttributeError(name)
