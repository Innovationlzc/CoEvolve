"""OSP alignment lite task registration."""

import gymnasium as gym


gym.register(
    id="CoEvolve-OSP-Alignment-Lite-Direct-v0",
    entry_point=f"{__name__}.osp_alignment_env:OSPAlignmentLiteEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osp_alignment_env_cfg:OSPAlignmentLiteEnvCfg",
    },
)

