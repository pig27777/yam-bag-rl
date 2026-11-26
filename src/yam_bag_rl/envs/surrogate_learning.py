"""CPU TCN-FiLM and masked contextual PPO for device-collected macro episodes."""

from dataclasses import dataclass
from math import ceil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..policy_v3.models import GraspScorer, PrivilegedCritic, ProbeActor
from .surrogate_env import ACTION_MASK, TryBatch

EFFORT = np.r_[np.arange(24, 36), 38]
COMMANDS = np.r_[np.arange(55, 62), np.arange(12, 18)]
HISTORY_WIDTH = 62


@dataclass(frozen=True)
class BatchData:
    initial_id: np.ndarray
    regime: np.ndarray
    v0: np.ndarray
    actions: np.ndarray
    tried: TryBatch
    choices: np.ndarray
    success: np.ndarray
    valid: np.ndarray
    private: np.ndarray

    def __len__(self):
        return len(self.v0)

    def take(self, index):
        t = self.tried
        return BatchData(self.initial_id[index], self.regime[index], self.v0[index], self.actions[index],
                         TryBatch(t.history[index], t.valid[index], t.windows[index], t.timestamps[index],
                                  t.candidates[index]), self.choices[index], self.success[index], self.valid[index],
                         self.private[index])


def concatenate(data):
    names = ("initial_id", "regime", "v0", "actions", "choices", "success", "valid", "private")
    arrays = {name: np.concatenate([getattr(row, name) for row in data]) for name in names}
    t = {name: np.concatenate([getattr(row.tried, name) for row in data])
         for name in ("history", "valid", "windows", "timestamps", "candidates")}
    return BatchData(tried=TryBatch(**t), **arrays)


def split_initial_states(data, seed):
    groups = np.unique(data.initial_id)
    group_order = np.random.default_rng(seed).permutation(groups)
    first, second = round(.70 * len(groups)), round(.85 * len(groups))
    train, valid, test = np.split(group_order, (first, second))
    return tuple(data.take(np.flatnonzero(np.isin(data.initial_id, group))) for group in (train, valid, test))


def command_design(history):
    cmd = history[..., COMMANDS]
    return torch.cat((torch.ones_like(cmd[..., :1]), cmd, cmd.square()), dim=-1)


class CausalResidual(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.dilation = 1
        self.conv = nn.Conv1d(input_dim, hidden, 3)
        self.skip = nn.Conv1d(input_dim, hidden, 1) if input_dim != hidden else nn.Identity()

    def forward(self, x):
        h = self.conv(F.pad(x, (2 * self.dilation, 0)))
        return self.skip(x) + F.silu(h)


class ResidualEncoder(nn.Module):
    def __init__(self, active, *, include_effort):
        super().__init__()
        self.include_effort = include_effort
        self.register_buffer("active", torch.as_tensor(active, dtype=torch.long))
        self.register_buffer("baseline", torch.zeros(1 + 2 * len(COMMANDS), HISTORY_WIDTH))
        self.register_buffer("mu", torch.zeros(HISTORY_WIDTH))
        self.register_buffer("sigma", torch.ones(HISTORY_WIDTH))
        self.blocks = nn.ModuleList()
        for i, dilation in enumerate((1, 2, 4)):
            block = CausalResidual(len(active) if i == 0 else 32, 32)
            block.dilation = dilation
            block.conv.dilation = (dilation,)
            self.blocks.append(block)
        self.action = nn.Sequential(nn.Linear(13, 32), nn.SiLU())
        self.dropout = nn.Dropout(.20)
        self.output = nn.Sequential(nn.Linear(128 + 32, 32), nn.LayerNorm(32))

    @classmethod
    def fit_statistics(cls, data, include_effort):
        choose = data.valid & data.tried.valid.any(axis=1)
        raw = torch.from_numpy(data.tried.history[choose])
        mask = torch.from_numpy(data.tried.valid[choose])
        approach = (torch.from_numpy(data.tried.windows[choose]) == 1) & mask
        design = command_design(raw)
        x = design[approach].double()
        y = raw[approach].double()
        if not len(x):
            raise ValueError("Collect valid pre-contact approach observations before fitting normalizers")
        penalty = .015 * torch.eye(x.shape[1], dtype=x.dtype)
        baseline = torch.linalg.solve(x.T @ x + penalty, x.T @ y).float()
        residual = raw - design @ baseline
        valid_history = raw[mask]
        residual_valid = residual[mask]
        active = np.flatnonzero((valid_history.abs().amax(dim=0) > 0.).numpy())
        if not include_effort:
            active = active[~np.isin(active, EFFORT)]
        encoder = cls(active, include_effort=include_effort)
        encoder.baseline.copy_(baseline)
        encoder.mu.copy_(residual_valid.mean(dim=0))
        encoder.sigma.copy_(residual_valid.std(dim=0, unbiased=False).clamp_min(.001))
        return encoder

    def _embed(self, sequence, valid, windows):
        x = sequence.transpose(1, 2).masked_fill(~valid[:, None], 0.)
        for layer in self.blocks:
            x = layer(x).masked_fill(~valid[:, None], 0.)
        pooled = []
        for phase in (2, 3):
            selected = (valid & (windows == phase))[:, None, :]
            mean = (x * selected).sum(-1) / selected.sum(-1).clamp_min(1)
            high = x.masked_fill(~selected, -torch.inf).amax(-1)
            high = torch.where(selected.any(-1), high, 0.)
            pooled.extend((mean, high))
        return torch.cat(pooled, dim=-1)

    def forward(self, history, valid, windows, actions):
        predicted = command_design(history) @ self.baseline
        residual = ((history - predicted - self.mu) / self.sigma).clamp(-12., 12.)
        zero = (-self.mu / self.sigma).expand_as(residual).clamp(-12., 12.)
        actual, control = self._embed(residual[..., self.active], valid, windows), self._embed(
            zero[..., self.active], valid, windows)
        latent = self.output(torch.cat((self.dropout(actual - control), self.action(actions)), dim=-1))
        return latent * valid.any(dim=-1, keepdim=True)


class SurrogateScorer(nn.Module):
    def __init__(self, encoder=None, *, condition_actions=True):
        super().__init__()
        self.encoder = encoder
        self.condition_actions = condition_actions
        self.head = GraspScorer(fusion_mode="film", public_dim=25)

    def forward(self, v0, actions, history, valid, windows, candidates):
        probe = actions if self.condition_actions else torch.zeros_like(actions)
        if self.encoder is None:
            latent = v0.new_zeros((len(v0), 32))
        else:
            latent = self.encoder(history, valid, windows, probe)
        return self.head.logits(torch.cat((v0, probe), dim=-1), latent, candidates)


def scorer_input(v0, action, tried):
    return tuple(torch.from_numpy(x) for x in (v0, action, tried.history, tried.valid, tried.windows,
                                                tried.candidates))


@torch.inference_mode()
def score_batches(scorer, v0, actions, tried, batch_size=384):
    scorer.eval()
    inputs = scorer_input(v0, actions, tried)
    logits = [scorer(*(value[start:start + batch_size] for value in inputs))
              for start in range(0, len(v0), batch_size)]
    return torch.cat(logits).numpy()


def _auc(label, score):
    positive, negative = score[label.astype(bool)], score[~label.astype(bool)]
    if not len(positive) or not len(negative):
        return None
    order = np.sort(negative)
    left = np.searchsorted(order, positive, side="left")
    right = np.searchsorted(order, positive, side="right")
    return float(np.mean((left + right) * .5 / len(negative)))


@torch.inference_mode()
def metrics(scorer, data):
    data = data.take(np.flatnonzero(data.valid))
    if not len(data):
        return dict(n=0, bce=None, auroc=None, brier=None, mean_candidate_percentile=None)
    logits = score_batches(scorer, data.v0, data.actions, data.tried)
    scores = logits[np.arange(len(data)), data.choices]
    y = data.success.astype(np.float32)
    p = torch.from_numpy(scores).sigmoid().numpy()
    rank = ((logits < scores[:, None]).sum(1) + .5 * (logits == scores[:, None]).sum(1)) / logits.shape[1]
    return dict(n=len(data), bce=float(np.mean(np.logaddexp(0., scores) - y * scores)), auroc=_auc(y, p),
                brier=float(np.mean((p - y)**2)), mean_candidate_percentile=float(rank.mean()),
                successful_candidate_percentile=float(rank[y == 1].mean()) if y.any() else None)


def fit_scorer(scorer, train, validation, *, epochs, batch_size=384, encoder_lr=.001, head_lr=.0003,
               patience=9, seed=0):
    mask = np.flatnonzero(train.valid)
    if not len(mask) or not validation.valid.any():
        raise ValueError("Both train and validation need valid episodes")
    params = [dict(params=scorer.head.parameters(), lr=head_lr)]
    if scorer.encoder is not None:
        params.append(dict(params=scorer.encoder.parameters(), lr=encoder_lr))
    opt = torch.optim.AdamW(params, weight_decay=.01)
    rng = np.random.default_rng(seed)
    all_inputs = scorer_input(train.v0, train.actions, train.tried)
    choice = torch.from_numpy(train.choices.astype(np.int64))
    labels = torch.from_numpy(train.success.astype(np.float32))
    records = []
    best_state, best_loss, since = None, float("inf"), 0
    for epoch in range(1, epochs + 1):
        scorer.train()
        shuffled = rng.permutation(mask)
        total_loss = 0.
        for idx in np.array_split(shuffled, ceil(len(shuffled) / batch_size)):
            logits = scorer(*(v[idx] for v in all_inputs)).gather(1, choice[idx, None]).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, labels[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * len(idx)
        val = metrics(scorer, validation)
        records.append(dict(epoch=epoch, train_bce=total_loss / len(mask), validation=val))
        if val["bce"] < best_loss - 1e-6:
            best_loss = val["bce"]
            best_state = {k: v.detach().clone() for k, v in scorer.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= patience:
                break
    scorer.load_state_dict(best_state)
    scorer.eval()
    return records


@torch.inference_mode()
def linear_probe(scorer, train, test):
    if scorer.encoder is None:
        return None

    def embed(data):
        idx = np.flatnonzero(data.valid)
        v = data.take(idx)
        _, action, h, valid, window, _ = scorer_input(v.v0, v.actions, v.tried)
        latent = scorer.encoder(h, valid, window, action).numpy()
        return np.c_[np.ones(len(v)), latent], v.private[:, :4]

    x, y = embed(train)
    a, b = embed(test)
    weight = np.linalg.solve(x.T @ x + .03 * np.eye(x.shape[1]), x.T @ y)
    r2 = 1. - ((a @ weight - b)**2).sum(0) / np.maximum(1e-8, ((b - b.mean(0))**2).sum(0))
    return dict(zip(("mass", "lateral_bias", "fill", "longitudinal_bias"), map(float, r2), strict=True))


class MaskedPPO:
    def __init__(self, seed, action_mask=ACTION_MASK):
        torch.manual_seed(seed)
        self.actor = ProbeActor(action_mask)
        self.critic = PrivilegedCritic(14)
        self.opt = torch.optim.Adam((dict(params=self.actor.parameters(), lr=.0003),
                                     dict(params=self.critic.parameters(), lr=.001)))
        self.rng = torch.Generator().manual_seed(seed + 28)

    @torch.inference_mode()
    def sample(self, public, *, deterministic=False):
        u, a, logp = self.actor.sample(torch.from_numpy(public), self.rng, deterministic=deterministic)
        return u.numpy(), a.numpy(), logp.numpy()

    def update(self, data, u, old_logp, reward, batch_size):
        idx = np.flatnonzero(data.valid)
        if not len(idx):
            raise ValueError("PPO requires valid on-policy macro episodes")
        v0, s0 = torch.from_numpy(data.v0[idx]), torch.from_numpy(data.private[idx])
        u, old_logp, reward = (torch.from_numpy(x[idx].astype(np.float32)) for x in (u, old_logp, reward))
        with torch.no_grad():
            advantage = reward - self.critic(s0, v0)
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
            old = self.actor.distribution(v0)
            old_mean, old_std = old.loc.clone(), old.scale.clone()
        active = self.actor.action_mask
        logs = []
        early_stop = False
        for _ in range(4):
            for mb in torch.randperm(len(idx), generator=self.rng).split(batch_size):
                dist = self.actor.distribution(v0[mb])
                logp = self.actor.log_prob(dist, u[mb])
                dlog = logp - old_logp[mb]
                ratio = dlog.exp()
                a = advantage[mb]
                policy = -torch.minimum(ratio * a, ratio.clamp(.8, 1.2) * a).mean()
                sample = dist.rsample()
                entropy = -self.actor.log_prob(dist, sample).mean()
                value = F.mse_loss(self.critic(s0[mb], v0[mb]), reward[mb])
                loss = policy + .5 * value - .01 * entropy
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_((*self.actor.parameters(), *self.critic.parameters()), .5)
                self.opt.step()
                with torch.no_grad():
                    new = self.actor.distribution(v0)
                    old_dist = torch.distributions.Normal(old_mean[:, active], old_std[:, active])
                    new_dist = torch.distributions.Normal(new.loc[:, active], new.scale[:, active])
                    kl = torch.distributions.kl_divergence(old_dist, new_dist).sum(-1).mean()
                    logs.append((float(policy), float(value), float(entropy), float(kl)))
                if kl > .02:
                    early_stop = True
                    break
            if early_stop:
                break
        stat = dict(zip(("actor_loss", "value_loss", "entropy", "kl"), np.mean(logs, axis=0).tolist(), strict=True))
        return stat | dict(kl_early_stop=early_stop, optimizer_steps=len(logs))
