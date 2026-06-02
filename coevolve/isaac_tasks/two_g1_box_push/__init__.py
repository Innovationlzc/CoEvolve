"""Two-G1 box-push lite task registration."""

import gymnasium as gym


gym.register(
    id="CoEvolve-TwoG1BoxPush-Direct-v0",
    entry_point=f"{__name__}.two_g1_box_push_env:TwoG1BoxPushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.two_g1_box_push_env_cfg:TwoG1BoxPushEnvCfg",
    },
)

