"""Policy V3 macro interface and a runnable rigid-fixture smoke environment."""

import numpy as np

from ..policy_v3.environment import MacroBagGraspEnv
from ..policy_v3.geometry import CandidateRules, InfoMode
from ..policy_v3.yam_backend import YamBackend


def make_smoke_env(seed=7):
    """Use the real Yam_B controller with the local rigid software fixture."""
    backend = YamBackend(seed, scripted=True)
    return MacroBagGraspEnv(backend, np.random.default_rng(seed), info_mode=InfoMode.V0_ONLY,
                           rules=CandidateRules(12))
