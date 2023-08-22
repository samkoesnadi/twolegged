from gymnasium.envs.registration import register
register(
    id='TwoLegged-v0',
    entry_point='twolegged.envs.twolegged_env:TwoLeggedEnv',
    max_episode_steps=150,
)
