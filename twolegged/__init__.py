from gym.envs.registration import register
register(
    id='TwoLegged-v0',
    entry_point='twolegged.envs:TwoLeggedEnv'
)
