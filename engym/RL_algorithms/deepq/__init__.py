from . import models  # noqa F401
from .deepq_learner import DEEPQ  # noqa F401
from .deepq import learn  # noqa F401
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa F401

def wrap_atari_dqn(env):
    from engym.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
