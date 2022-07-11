from gym.envs.registration import register

register(
    id='StackedBars-v0',
    entry_point='gym_engineering.envs.engineering:StackedBarsEnv',
    max_episode_steps=100,
)
