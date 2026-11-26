"""Policy V3 scorer variants with train-only normalization and FiLM."""

from collections import defaultdict

import numpy as np
import torch
from torch import nn

from .data import ReplayBuffer
from .models import ProprioHistoryEncoder, mlp
from .training import _concordance


class NormalizedHistoryEncoder(ProprioHistoryEncoder):
    def __init__(self):
        super().__init__()
        from .history import HISTORY_DIM

        self.register_buffer("channel_mean", torch.zeros(HISTORY_DIM))
        self.register_buffer("channel_scale", torch.ones(HISTORY_DIM))

    @torch.no_grad()
    def fit_channels(self, train_rows, profile):
        frames = [profile.frames(row.history()) for row in ReplayBuffer.bce_records(train_rows) if row.valid_len]
        if not frames:
            raise ValueError("An informed scorer requires nonempty valid training histories")
        data = np.concatenate(frames).astype(np.float64)
        self.channel_mean.copy_(torch.as_tensor(data.mean(axis=0)))
        std = data.std(axis=0)
        self.channel_scale.copy_(torch.as_tensor(np.where(std > 1e-6, std, 1.)))

    def forward(self, history, padding_mask, a_try):
        return super().forward((history - self.channel_mean) / self.channel_scale, padding_mask, a_try)


class FiLMGraspScorer(nn.Module):
    def __init__(self, *, informed=True):
        super().__init__()
        self.base = mlp(37 + 11, 64, 64)
        self.film = nn.Linear(64, 128) if informed else None
        self.head = nn.Sequential(nn.SiLU(), nn.Linear(64, 1))
        self.register_buffer("grasp_scale", torch.tensor([12.] * 3 + [1.] * 6 + [12., 1.]))
        if self.film is not None:
            nn.init.normal_(self.film.weight, std=.01)
            nn.init.zeros_(self.film.bias)

    def logits(self, public, history, grasp):
        x = self.base(torch.cat((public, grasp * self.grasp_scale), dim=-1))
        if self.film is not None:
            scale, shift = self.film(history).chunk(2, dim=-1)
            x = (1 + scale) * x + shift
        return self.head(x).squeeze(-1)

    def forward(self, public, history, grasp):
        return self.logits(public, history, grasp).sigmoid()


def validation_auc_delta(rows, informed, blind, *, seed=0, repeats=1000):
    """One-sided paired bootstrap over initial states, using identical validation rows."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.initial_state_id].append(index)
    labels = np.asarray([row.task_success for row in rows], dtype=bool)
    if len(groups) < 2 or not labels.any() or labels.all():
        raise ValueError("AUC comparison requires held-out initial states with both labels")

    def auc(scores, ids):
        wins, pairs = _concordance(np.asarray(scores)[ids], labels[ids])
        return wins / pairs if pairs else None

    def delta(ids):
        a, b = auc(informed, ids), auc(blind, ids)
        return None if a is None else a - b

    rng = np.random.default_rng(seed)
    members = list(groups.values())
    estimates = []
    for _ in range(repeats):
        ids = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        difference = delta(ids)
        if difference is not None:
            estimates.append(difference)
    if len(estimates) < .9 * repeats:
        raise ValueError("Too many grouped bootstrap resamples lack both classes")
    return dict(delta=float(delta(np.arange(len(rows)))), lower_95=float(np.quantile(estimates, .05)),
                initial_states=len(groups), usable_resamples=len(estimates))


def matched_rgb_score_range(rgb, state_ids, scores):
    """Check scores across distinct hidden states with identical RGB and candidate columns."""
    rgb, scores = np.asarray(rgb), np.asarray(scores)
    if (rgb.ndim != 4 or rgb.shape[-1] != 3 or len(rgb) < 3 or len(state_ids) != len(rgb)
            or len(set(state_ids)) != len(state_ids) or not np.all(rgb == rgb[0])):
        raise ValueError("Distinct held-out states with identical RGB frames are required")
    if scores.ndim != 2 or scores.shape[0] != len(rgb) or not scores.shape[1] or not np.isfinite(scores).all():
        raise ValueError("Provide a rectangular score array for the same candidate set across states")
    return float(np.ptp(scores, axis=0).max())
