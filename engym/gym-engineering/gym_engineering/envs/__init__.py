from gym.envs.registration import register

register(
    id='stackbars-v0',
    entry_point='gym_engineering.envs.engineering:StackedBarsEnv',
    max_episode_steps=100,
)

register(
    id='selfbalancingbot-v0',
    entry_point='gym_engineering.envs.engineering:BalancebotEnv',
    max_episode_steps=600,
)
