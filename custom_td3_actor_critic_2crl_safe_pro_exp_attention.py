"""
Custom TD3 implementation for Stable Baselines 3 with Constrained Reinforcement Learning (CRL):
1. Huber Loss for the critic for more robust training.
2. Adaptive Lagrange multiplier λ for smoothness constraints (instead of fixed penalty).
3. Separate learning rate schedules for the actor and critic.
4. Constraint-based penalty that adapts based on actual policy smoothness.
"""

import collections
import csv
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.nn import functional as F
import config
from attention_policy import (
    normalize_raw_actions_to_portfolio_weights,
    collect_basket_amortization_diagnostics,
    PortfolioWorldModel,
    _maybe_compile_module,
    _maybe_compile_method,
)

from stable_baselines3 import TD3
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.utils import polyak_update
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union

from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.td3.policies import TD3Policy


CURRICULUM_ROLE_TO_CODE = {
    "anchor": 1,
    "frontier": 2,
    "uniform_probe": 3,
    "rehearsal": 4,
}
CURRICULUM_CODE_TO_ROLE = {
    code: role for role, code in CURRICULUM_ROLE_TO_CODE.items()
}

QWM_TRAINING_STAT_KEYS = (
    "loss",
    "reconstruction_loss",
    "posterior_reconstruction_loss",
    "prior_reconstruction_loss",
    "state_loss",
    "reward_loss",
    "prior_state_loss",
    "prior_reward_loss",
    "kl_loss",
    "observation_loss",
    "local_history_loss",
    "episode_history_loss",
    "context_loss",
    "prior_std_mean",
    "posterior_std_mean",
    "latent_mean_gap",
    "gradient_norm",
)


def _qwm_assert_finite(value: torch.Tensor, message: str) -> None:
    """Stop before AdamW without poisoning CUDA's execution context.

    Device-side assertions are not recoverable optimizer validation: a later
    innocent CUDA operation reports the failure and the process cannot safely
    continue. QWM instead performs one explicit health synchronization after
    backward, replacing the former many per-metric synchronizations.
    """
    finite = torch.isfinite(value).all()
    if not bool(finite):
        raise FloatingPointError(message)


def qwm_search_depth(
    world_updates: int,
    *,
    initial_depth: int,
    maximum_depth: int,
    depth_three_updates: int,
    depth_four_updates: int,
) -> int:
    """Return the staged QWM depth supported by accumulated real-data fitting."""
    depth = max(1, int(initial_depth))
    maximum_depth = max(depth, int(maximum_depth))
    if maximum_depth >= 3 and int(world_updates) >= int(depth_three_updates):
        depth = max(depth, 3)
    if maximum_depth >= 4 and int(world_updates) >= int(depth_four_updates):
        depth = max(depth, 4)
    return min(depth, maximum_depth)


def _qwm_repeat_observations(
    observations: Dict[str, torch.Tensor], count: int
) -> Dict[str, torch.Tensor]:
    return {
        key: value.repeat_interleave(int(count), dim=0)
        if torch.is_tensor(value)
        else value
        for key, value in observations.items()
    }


def _qwm_select_observations(
    observations: Dict[str, torch.Tensor], indices: torch.Tensor
) -> Dict[str, torch.Tensor]:
    return {
        key: value[indices] if torch.is_tensor(value) else value
        for key, value in observations.items()
    }


def _qwm_mark_compiled_step(
    module: Any,
    method_name: str,
    reference: torch.Tensor,
) -> None:
    """Declare a new CUDA-Graph replay after prior outputs are owned.

    PyTorch's compiled callables reuse static output storage.  The QWM tree
    invokes actor, world model, and critic several times inside one environment
    decision, so automatic iteration discovery cannot infer these boundaries.
    """
    if not (
        torch.is_tensor(reference)
        and reference.device.type == "cuda"
        and getattr(module, f"_compiled_{method_name}", False)
        and hasattr(torch, "compiler")
        and hasattr(torch.compiler, "cudagraph_mark_step_begin")
    ):
        return
    torch.compiler.cudagraph_mark_step_begin()


def _qwm_materialize_compiled_transition(
    next_observations: Dict[str, torch.Tensor],
    predicted_rewards: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Move CUDA-Graph outputs into tree-owned, float32 storage.

    ``Tensor.float()`` aliases its input when it is already float32.  An
    explicit copy is required because the next replay of the compiled world
    model is allowed to overwrite its static output buffers.
    """
    owned_observations = {
        key: (
            value.to(dtype=torch.float32, copy=True)
            if torch.is_tensor(value) and value.is_floating_point()
            else value.clone()
            if torch.is_tensor(value)
            else value
        )
        for key, value in next_observations.items()
    }
    return owned_observations, predicted_rewards.to(
        dtype=torch.float32, copy=True
    )


def qwm_tree_search(
    observations: Dict[str, torch.Tensor],
    *,
    propose_actions,
    candidate_values,
    transition,
    depth: int,
    candidates: int,
    leaf_candidates: int,
    model_samples: int,
    beam_width: int,
    search_discount: float,
    leaf_aggregation: str = "max",
    successor_aggregation: str = "mean",
    transition_candidates=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Score root actions with the paper's Q/value tree and per-root beams.

    ``transition`` returns K successor rows for every input state-action row and
    a reward tensor shaped ``[input_rows, K]``.  The production world model
    draws those rows from its learned conditional latent prior, so K is a real
    successor-scenario axis rather than K copies of one deterministic output.
    ``successor_aggregation`` implements the paper's aggregation over that K
    axis independently of the N action-candidate aggregation.
    """
    depth = int(depth)
    candidates = int(candidates)
    leaf_candidates = int(leaf_candidates)
    model_samples = int(model_samples)
    beam_width = int(beam_width)
    if depth < 1 or candidates < 1 or leaf_candidates < 1:
        raise ValueError("QWM depth and candidate counts must be positive")
    if model_samples < 1 or beam_width < 1:
        raise ValueError("QWM model_samples and beam_width must be positive")
    if not 0.0 <= float(search_discount) <= 1.0:
        raise ValueError("QWM search_discount must be in [0, 1]")
    if leaf_aggregation not in {"max", "mean"}:
        raise ValueError("QWM leaf_aggregation must be 'max' or 'mean'")
    if successor_aggregation not in {"max", "mean"}:
        raise ValueError("QWM successor_aggregation must be 'max' or 'mean'")

    def aggregate_successors(values: torch.Tensor) -> torch.Tensor:
        if successor_aggregation == "max":
            return values.max(dim=-1).values
        return values.mean(dim=-1)

    root_actions = propose_actions(observations, candidates)
    if root_actions.ndim != 3:
        raise ValueError("QWM proposer must return [batch, candidates, action_dim]")
    batch_size, root_count, action_dim = root_actions.shape
    roots = batch_size * root_count
    root_values = candidate_values(observations, root_actions).reshape(roots)
    root_action_rows = root_actions.reshape(roots, action_dim)
    if transition_candidates is None:
        root_state = _qwm_repeat_observations(observations, root_count)
        frontier, root_rewards = transition(
            root_state, root_action_rows, model_samples
        )
    else:
        frontier, root_rewards = transition_candidates(
            observations, root_actions, model_samples
        )
    root_rewards = root_rewards.reshape(roots, model_samples)

    parent_count = model_samples
    cumulative_priority = root_values.unsqueeze(1).expand(roots, parent_count)
    levels = []
    for current_depth in range(1, depth):
        frontier_actions = propose_actions(frontier, candidates)
        width = int(frontier_actions.shape[1])
        q_values = candidate_values(frontier, frontier_actions).reshape(
            roots, parent_count, width
        )
        direct_values = q_values.max(dim=-1).values
        priority = cumulative_priority.unsqueeze(-1) + (
            float(search_discount) ** current_depth
        ) * q_values
        survivors = min(beam_width, parent_count * width)
        selected = priority.reshape(roots, -1).topk(
            survivors, dim=1
        ).indices
        selected_parents = torch.div(
            selected, width, rounding_mode="floor"
        )

        root_offsets = (
            torch.arange(roots, device=selected.device).unsqueeze(1)
            * parent_count
            * width
        )
        global_selected = (root_offsets + selected).reshape(-1)
        parent_offsets = (
            torch.arange(roots, device=selected.device).unsqueeze(1)
            * parent_count
        )
        global_selected_parents = (
            parent_offsets + selected_parents
        ).reshape(-1)
        selected_states = _qwm_select_observations(
            frontier, global_selected_parents
        )
        selected_actions = frontier_actions.reshape(-1, action_dim)[
            global_selected
        ]
        next_frontier, selected_rewards = transition(
            selected_states, selected_actions, model_samples
        )
        selected_rewards = selected_rewards.reshape(
            roots, survivors, model_samples
        )
        levels.append(
            (direct_values, selected_parents, selected_rewards)
        )
        frontier = next_frontier
        cumulative_priority = priority.reshape(roots, -1).gather(
            1, selected
        ).unsqueeze(-1).expand(-1, -1, model_samples).reshape(
            roots, survivors * model_samples
        )
        parent_count = survivors * model_samples

    leaf_actions = propose_actions(frontier, leaf_candidates)
    leaf_q = candidate_values(frontier, leaf_actions)
    if leaf_aggregation == "mean":
        value = leaf_q.mean(dim=-1).reshape(roots, parent_count)
    else:
        value = leaf_q.max(dim=-1).values.reshape(roots, parent_count)

    for direct_values, selected_parents, selected_rewards in reversed(levels):
        survivors = int(selected_parents.shape[1])
        successor_values = value.reshape(roots, survivors, model_samples)
        rollout_values = aggregate_successors(
            selected_rewards
            + float(search_discount) * successor_values
        )
        best_rollout = torch.full_like(direct_values, -torch.inf)
        best_rollout.scatter_reduce_(
            1,
            selected_parents,
            rollout_values,
            reduce="amax",
            include_self=True,
        )
        value = torch.where(
            torch.isfinite(best_rollout),
            0.5 * (direct_values + best_rollout),
            direct_values,
        )

    root_successor_values = value.reshape(roots, model_samples)
    model_mediated_root = aggregate_successors(
        root_rewards
        + float(search_discount) * root_successor_values
    )
    root_scores = 0.5 * (root_values + model_mediated_root)
    return root_actions, root_scores.reshape(batch_size, root_count)


def _membership_key(row: np.ndarray) -> bytes:
    """Compact, exact key for a binary basket-membership row."""
    binary = np.asarray(row, dtype=np.float32).reshape(-1) > 0.5
    return np.packbits(binary, bitorder="little").tobytes()


def _private_membership_row_from_observation(
    observation: Dict[str, np.ndarray],
    *,
    env_index: int = 0,
) -> Optional[np.ndarray]:
    """Resolve replay provenance without making membership a neural feature.

    Current environments emit an exact private multi-hot row.  Checkpoints
    written during the Path-2 migration may instead contain an all-zero row
    because neural membership disablement was incorrectly applied to replay
    metadata as well.  Their aligned ``asset_ids`` still provide enough
    information to reconstruct the canonical basket key.
    """
    if not isinstance(observation, dict):
        return None
    membership_value = observation.get("basket_membership")
    membership_dim = int(getattr(config, "BASKET_MEMBERSHIP_DIM", 0))
    membership = None
    if membership_value is not None:
        membership_array = np.asarray(membership_value)
        if membership_array.ndim >= 2:
            if env_index >= membership_array.shape[0]:
                return None
            membership_array = membership_array[env_index]
        membership = np.asarray(membership_array, dtype=np.float32).reshape(-1)
        membership_dim = int(membership.size)
        if membership_dim > 0 and np.any(membership > 0.5):
            return membership
    if membership_dim <= 0:
        return membership

    asset_ids_value = observation.get("asset_ids")
    if asset_ids_value is None:
        return membership
    asset_ids_array = np.asarray(asset_ids_value)
    if asset_ids_array.ndim >= 2:
        if env_index >= asset_ids_array.shape[0]:
            return membership
        asset_ids_array = asset_ids_array[env_index]
    reconstructed = np.zeros(membership_dim, dtype=np.float32)
    for asset_id in np.asarray(asset_ids_array).reshape(-1):
        asset_index = int(asset_id)
        if 0 <= asset_index < membership_dim:
            reconstructed[asset_index] = 1.0
    return reconstructed if np.any(reconstructed > 0.5) else membership


def build_foundation_candidates(
    observations: Dict[str, torch.Tensor],
    actor_actions: torch.Tensor,
) -> torch.Tensor:
    """Build the fixed sixteen-policy counterfactual library as weights."""
    mask = observations["mask"].float()
    previous = observations["prev_action"].float()
    batch, assets = mask.shape
    action_dim = assets + 1
    candidates = []
    candidates.append(previous)
    cash = previous.new_zeros((batch, action_dim))
    cash[:, 0] = 1.0
    candidates.append(cash)
    equal = previous.new_zeros((batch, action_dim))
    equal[:, 1:] = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    candidates.append(equal)
    for asset_index in range(assets):
        one_hot = previous.new_zeros((batch, action_dim))
        one_hot[:, asset_index + 1] = mask[:, asset_index]
        one_hot[:, 0] = 1.0 - mask[:, asset_index]
        candidates.append(one_hot)

    episode_history = observations["episode_feature_history"].float()
    momentum_name = "episode_cross_sectional_momentum_rank"
    momentum_offset = int(
        getattr(config, "MDAEN_EPISODE_FEATURE_COUNT", episode_history.shape[-1])
        - len(getattr(config, "EPISODE_RELATIVE_FEATURES", ()))
    )
    momentum_index = momentum_offset + list(
        getattr(config, "EPISODE_RELATIVE_FEATURES", ())
    ).index(momentum_name)
    momentum = episode_history[:, :, -1, momentum_index]
    local_history = observations["local_feature_history"].float()
    volatility = local_history[:, :, -1, -2].abs()

    def _tilt(scores):
        weights = torch.softmax(scores.masked_fill(~mask.bool(), -1e4), dim=1)
        result = previous.new_zeros((batch, action_dim))
        result[:, 1:] = weights * mask
        return result

    candidates.extend((_tilt(momentum), _tilt(-momentum), _tilt(-volatility)))
    candidates.append(
        normalize_raw_actions_to_portfolio_weights(actor_actions, mask)
    )
    quasi = torch.sin(
        torch.arange(1, assets + 1, device=mask.device, dtype=mask.dtype)
        .unsqueeze(0)
        * 12.9898
        + torch.arange(batch, device=mask.device, dtype=mask.dtype).unsqueeze(1)
        * 78.233
    ).abs() + 0.05
    dirichlet_like = previous.new_zeros((batch, action_dim))
    quasi = quasi * mask
    dirichlet_like[:, 1:] = quasi / quasi.sum(dim=1, keepdim=True).clamp(min=1e-6)
    candidates.append(dirichlet_like)
    result = torch.stack(candidates, dim=1)
    expected = int(getattr(config, "FOUNDATION_CANDIDATE_COUNT", 16))
    if result.shape[1] != expected:
        raise RuntimeError(
            f"foundation candidate contract expected {expected}, got {result.shape[1]}"
        )
    return result


def basket_relative_forecast_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    return_reference: float,
    huber_beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row error in the ordering of the actually available assets.

    Absolute forecasts remain separately supervised so a common change in
    all eight returns can still affect the choice between equities and cash.
    This short-horizon lesson is never used as a full Bellman-Q target.
    """
    if predictions.shape != targets.shape or targets.shape != valid_mask.shape:
        raise ValueError("Relative forecast tensors must share [batch, asset, channel]")
    if return_reference <= 0.0 or huber_beta <= 0.0:
        raise ValueError("Relative forecast scales must be positive")
    valid = valid_mask.float()
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    predicted_mean = (predictions * valid).sum(dim=1, keepdim=True) / counts
    realized_mean = (targets * valid).sum(dim=1, keepdim=True) / counts
    residual = (
        (predictions - predicted_mean) - (targets - realized_mean)
    ) / return_reference
    per_asset = F.smooth_l1_loss(
        residual, torch.zeros_like(residual), reduction="none", beta=huber_beta
    )
    per_sample = (per_asset * valid).sum(dim=(1, 2)) / valid.sum(
        dim=(1, 2)
    ).clamp_min(1.0)
    return per_sample, valid.sum(dim=(1, 2)) > 1.0


def build_foundation_action_targets(
    observations: Dict[str, torch.Tensor],
    candidate_actions: torch.Tensor,
    component_outcomes: torch.Tensor,
    component_masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose dense portfolio outcomes from stored unit asset returns."""
    scale = float(getattr(config, "FINANCIAL_REWARD_SCALE", 10.0))
    returns = component_outcomes.float()
    if returns.shape[-1] == 1:
        returns = returns[..., 0]
    returns = returns / max(scale, 1e-8)
    valid = component_masks.bool()
    if valid.shape[-1] == 1:
        valid = valid[..., 0]
    batch, steps, assets = returns.shape
    mask = observations["mask"].float()
    observation_batch = int(mask.shape[0])
    candidate_batch = int(candidate_actions.shape[0])
    if observation_batch != batch or candidate_batch != batch:
        raise ValueError(
            "foundation action-target rows must be aligned: "
            f"observations={observation_batch}, candidates={candidate_batch}, "
            f"component_outcomes={batch}"
        )
    candidate_count = candidate_actions.shape[1]
    # Candidate rows are already executable portfolio weights.  Applying the
    # actor-logit transform again would silently change the counterfactual.
    weights = candidate_actions.float().clamp(min=0.0)
    weights = torch.cat(
        (weights[..., :1], weights[..., 1:] * mask.unsqueeze(1)), dim=-1
    )
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    asset_weights = weights[..., 1:]
    safe_returns = torch.where(valid, returns, torch.zeros_like(returns))
    cumulative_assets = torch.cumprod(
        1.0 + safe_returns.clamp(min=-0.95, max=5.0), dim=1
    ) - 1.0
    previous = observations["prev_action"].float()
    turnover = 0.5 * (weights - previous.unsqueeze(1)).abs().sum(dim=-1)
    threshold = float(getattr(config, "MIN_TRADE_THRESHOLD", 0.02))
    transaction_cost = torch.where(
        turnover > threshold,
        turnover * float(getattr(config, "TRANSACTION_COST", 0.0)),
        torch.zeros_like(turnover),
    )
    horizons = tuple(int(v) for v in getattr(
        config, "FOUNDATION_HORIZONS", (1, 5, 15, 30)
    ))
    net_horizon_returns = []
    for horizon in horizons:
        gross = torch.einsum(
            "bka,ba->bk", asset_weights, cumulative_assets[:, horizon - 1]
        )
        net_horizon_returns.append(
            torch.log1p((gross - transaction_cost).clamp(min=-0.95)) * scale
        )
    step_portfolio_returns = torch.einsum(
        "bka,bsa->bks", asset_weights, safe_returns
    )
    downside = torch.sqrt(
        torch.square(
            torch.minimum(
                step_portfolio_returns, torch.zeros_like(step_portfolio_returns)
            )
        ).mean(dim=-1).clamp(min=1e-12)
    ) * scale
    targets = torch.stack(
        (*net_horizon_returns, turnover, downside), dim=-1
    )
    horizon_valid = valid[:, : max(horizons)].all(dim=1) & mask.bool()
    row_valid = horizon_valid.any(dim=1)
    utility = targets[..., 3] - float(
        getattr(config, "RISK_ADJUSTMENT_BETA", 2.0)
    ) * torch.clamp(
        targets[..., 5]
        - float(getattr(config, "DOWNSIDE_DEVIATION_TARGET", 0.0)) * scale,
        min=0.0,
    )
    return targets, utility, returns, valid & mask.bool().unsqueeze(1)



class BasketReplayEvidenceUnavailable(ValueError):
    """No basket currently satisfies the financial replay maturity contract."""


class _DenseReplayPositionIndex:
    """Mutable O(1) replay-row index with a zero-copy NumPy sampling view.

    The August-24 role sampler stored row positions in Python sets.  Although
    set insertion was cheap, every subsequent sample converted and sorted the
    growing set.  Because the active basket is also the basket receiving the
    newest row, its cache was invalidated on every environment step.  The
    resulting add->sample loop was therefore O(number of stored rows).

    This structure keeps a dense integer array plus a position-to-offset map.
    Deletion swaps the final element into the removed slot, so add, discard,
    and array access are all O(1).  Sampling does not require chronological or
    sorted order, only the exact set of currently live circular-buffer rows.
    """

    def __init__(self, initial_capacity: int = 64):
        capacity = max(1, int(initial_capacity))
        self._values = np.empty(capacity, dtype=np.int64)
        self._offset_by_position: Dict[int, int] = {}
        self._size = 0

    def __len__(self) -> int:
        return int(self._size)

    def __bool__(self) -> bool:
        return self._size > 0

    def __iter__(self):
        return iter(self._values[: self._size])

    def __contains__(self, position: int) -> bool:
        return int(position) in self._offset_by_position

    def __eq__(self, other) -> bool:
        if isinstance(other, _DenseReplayPositionIndex):
            return self._offset_by_position.keys() == other._offset_by_position.keys()
        try:
            return set(map(int, self.array_view())) == set(map(int, other))
        except TypeError:
            return False

    def add(self, position: int) -> None:
        position = int(position)
        if position in self._offset_by_position:
            return
        if self._size == self._values.size:
            enlarged = np.empty(max(1, self._values.size * 2), dtype=np.int64)
            enlarged[: self._size] = self._values[: self._size]
            self._values = enlarged
        self._offset_by_position[position] = self._size
        self._values[self._size] = position
        self._size += 1

    def discard(self, position: int) -> None:
        position = int(position)
        offset = self._offset_by_position.pop(position, None)
        if offset is None:
            return
        final_offset = self._size - 1
        if offset != final_offset:
            final_position = int(self._values[final_offset])
            self._values[offset] = final_position
            self._offset_by_position[final_position] = offset
        self._size = final_offset

    def array_view(self) -> np.ndarray:
        """Return live rows without allocating, sorting, or scanning replay."""
        return self._values[: self._size]


class ValueExperienceReplayScorer(nn.Module):
    """Query-conditioned, permutation-equivariant replay-utility model.

    NERS supplies the local-row plus global-candidate-set contract.  The added
    query is a disjoint, identity-neutral sample of the current replay tape:
    it represents the Bellman problem the critic is presently trying to solve.
    A row is therefore not globally ``useful`` in the abstract; it is useful
    when its FAM/AAM state-action situation is expected to improve that query.
    No basket key, asset id, or membership coordinate enters this calculation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("replay scorer dimensions must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.local = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        context_dim = 4 * hidden_dim
        self.score = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.transfer = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Uniform replay is the safe initial policy.  Financial and Bellman
        # feedback must earn every later deviation from it.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        nn.init.zeros_(self.transfer[-1].weight)
        nn.init.zeros_(self.transfer[-1].bias)

    def forward(
        self,
        candidate_features: torch.Tensor,
        query_features: Optional[torch.Tensor] = None,
        *,
        return_transfer: bool = False,
    ):
        if candidate_features.dim() != 2:
            raise ValueError("replay candidates must be [candidate, feature]")
        if candidate_features.shape[-1] != self.input_dim:
            raise ValueError(
                "replay feature width mismatch: expected "
                f"{self.input_dim}, received {candidate_features.shape[-1]}"
            )
        if query_features is None:
            query_features = candidate_features
        if query_features.dim() != 2 or query_features.shape[-1] != self.input_dim:
            raise ValueError(
                "replay query must be [query, feature] with the candidate width"
            )
        if query_features.shape[0] == 0:
            raise ValueError("replay query must contain at least one row")
        local = self.local(candidate_features)
        query_local = self.local(query_features)
        candidate_context = local.mean(dim=0, keepdim=True).expand_as(local)
        # Candidate-specific cross-query attention preserves a multimodal query.
        # A candidate near one unresolved regime is no longer compared with the
        # synthetic arithmetic mean of every hard and coverage row.  No learned
        # parameters are added here, so existing replay-scorer checkpoints keep
        # the same state-dict shapes.
        normalized_local = F.normalize(local, dim=-1, eps=1e-6)
        normalized_query = F.normalize(query_local, dim=-1, eps=1e-6)
        query_attention = torch.softmax(
            normalized_local @ normalized_query.transpose(0, 1) / 0.25,
            dim=1,
        )
        query_context = query_attention @ query_local
        # The product is a parameter-efficient local/query compatibility term.
        # Candidate and query permutations preserve the corresponding scores.
        context = torch.cat(
            (local, candidate_context, query_context, local * query_context),
            dim=-1,
        )
        utility = self.score(context).squeeze(-1)
        if return_transfer:
            return utility, self.transfer(context).squeeze(-1)
        return utility


def build_query_conditioned_transfer_groups(
    selected_features: torch.Tensor,
    query_features: torch.Tensor,
    group_count: int,
) -> List[np.ndarray]:
    """Partition selected rows into identity-neutral query neighborhoods.

    Query prototypes are chosen by deterministic farthest-point traversal, then
    every selected row is assigned to its nearest prototype in cosine space.
    The result depends on behavioral features, never sampler append order,
    curriculum role, basket membership, or asset identity.
    """
    if selected_features.dim() != 2 or query_features.dim() != 2:
        raise ValueError("transfer grouping expects rank-two feature tensors")
    if selected_features.shape[1] != query_features.shape[1]:
        raise ValueError("selected/query feature widths must match")
    row_count = int(selected_features.shape[0])
    query_count = int(query_features.shape[0])
    groups = min(max(1, int(group_count)), row_count, query_count)
    if row_count <= 0 or query_count <= 0:
        return []
    selected = F.normalize(selected_features.detach().float(), dim=-1, eps=1e-6)
    query = F.normalize(query_features.detach().float(), dim=-1, eps=1e-6)
    prototype_indices = [0]
    minimum_distance = 1.0 - query @ query[0]
    while len(prototype_indices) < groups:
        next_index = int(torch.argmax(minimum_distance).item())
        if next_index in prototype_indices:
            remaining = [
                index for index in range(query_count)
                if index not in prototype_indices
            ]
            if not remaining:
                break
            next_index = remaining[0]
        prototype_indices.append(next_index)
        distance = 1.0 - query @ query[next_index]
        minimum_distance = torch.minimum(minimum_distance, distance)
    prototypes = query[
        torch.as_tensor(prototype_indices, device=query.device, dtype=torch.long)
    ]
    assignments = torch.argmax(selected @ prototypes.transpose(0, 1), dim=1)
    result = [
        torch.nonzero(assignments == group, as_tuple=False)
        .squeeze(-1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
        for group in range(len(prototype_indices))
    ]
    # Degenerate representations can leave a prototype empty.  Preserve the
    # exact-cover contract with a deterministic balanced fallback.
    if any(rows.size == 0 for rows in result):
        order = torch.argsort(assignments).detach().cpu().numpy()
        result = [
            rows.astype(np.int64, copy=False)
            for rows in np.array_split(order, len(prototype_indices))
            if rows.size
        ]
    return result


def transfer_compatible_gradient(
    group_gradient: torch.Tensor,
    query_gradient: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Project only the query-interfering component out of a group gradient."""
    dot = torch.sum(group_gradient * query_gradient)
    if float(dot.detach().item()) >= 0.0:
        return group_gradient
    query_square_norm = torch.sum(query_gradient.square()).clamp_min(epsilon)
    return group_gradient - (dot / query_square_norm) * query_gradient


def transfer_compatible_gradient_sequence(
    source_gradients: Tuple[Optional[torch.Tensor], ...],
    reference_gradients: Tuple[Optional[torch.Tensor], ...],
    epsilon: float = 1e-12,
) -> Tuple[Optional[torch.Tensor], ...]:
    """Project interference in the global parameter vector, not tensor by tensor.

    Gradient compatibility is defined over the concatenated shared-parameter
    space.  Performing a separate projection for every parameter tensor changes
    that geometry and can over-amplify coordinates with small optimizer moments.
    """
    dot = None
    reference_square_norm = None
    for source, reference in zip(source_gradients, reference_gradients):
        if source is None or reference is None:
            continue
        term = torch.sum(source * reference)
        square = torch.sum(reference.square())
        dot = term if dot is None else dot + term
        reference_square_norm = (
            square
            if reference_square_norm is None
            else reference_square_norm + square
        )
    if (
        dot is None
        or reference_square_norm is None
        or float(dot.detach().item()) >= 0.0
    ):
        return source_gradients
    coefficient = dot / reference_square_norm.clamp_min(epsilon)
    return tuple(
        source
        if source is None or reference is None
        else source - coefficient * reference
        for source, reference in zip(source_gradients, reference_gradients)
    )


def optimizer_aware_meta_gradient_scale(
    optimizer: torch.optim.Optimizer,
    parameters: Tuple[torch.nn.Parameter, ...],
    main_gradients: Tuple[Optional[torch.Tensor], ...],
    meta_gradients: Tuple[Optional[torch.Tensor], ...],
    requested_scale: float,
    *,
    max_adam_step_ratio: float,
    max_relative_parameter_step: float,
) -> Tuple[float, float, float]:
    """Bound MER by a conservative linearized Adam displacement.

    A small raw gradient is not necessarily a small Adam update: a coordinate
    with a tiny second moment can be strongly amplified.  The current Adam
    second moment plus the ordinary TD gradient define a fixed preconditioner.
    Holding that denominator fixed is conservative for a new MER coordinate:
    it does not let MER's own square inflate the denominator and make the
    proposed displacement look safer.  The calculation scans the shared
    parameters once and never mutates parameters or optimizer state.
    """
    requested_scale = max(float(requested_scale), 0.0)
    max_adam_step_ratio = max(float(max_adam_step_ratio), 0.0)
    max_relative_parameter_step = max(
        float(max_relative_parameter_step), 0.0
    )
    if requested_scale <= 0.0:
        return 0.0, 0.0, 0.0

    group_by_parameter = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    base_step_square = None
    unit_meta_step_square = None
    parameter_square = None
    for parameter, main_gradient, meta_gradient in zip(
        parameters, main_gradients, meta_gradients
    ):
        if main_gradient is None or meta_gradient is None:
            continue
        group = group_by_parameter.get(id(parameter))
        if group is None:
            continue
        sign = -1.0 if bool(group.get("maximize", False)) else 1.0
        main = sign * main_gradient.detach()
        meta = sign * meta_gradient.detach()
        weight_decay = float(group.get("weight_decay", 0.0))
        if weight_decay != 0.0:
            main = main + weight_decay * parameter.detach()
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        learning_rate = float(group.get("lr", 1e-3))
        epsilon = float(group.get("eps", 1e-8))
        state = optimizer.state.get(parameter, {})
        exp_avg = state.get("exp_avg", torch.zeros_like(parameter))
        exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(parameter))
        raw_step = state.get("step", 0)
        if torch.is_tensor(raw_step):
            raw_step = int(raw_step.detach().item())
        step = int(raw_step) + 1
        correction1 = 1.0 - float(beta1) ** step
        correction2_sqrt = math.sqrt(1.0 - float(beta2) ** step)
        next_square = float(beta2) * exp_avg_sq + (
            1.0 - float(beta2)
        ) * main.square()
        if bool(group.get("amsgrad", False)):
            max_square = state.get(
                "max_exp_avg_sq", torch.zeros_like(parameter)
            )
            next_square = torch.maximum(max_square, next_square)
        denominator = next_square.sqrt() / correction2_sqrt + epsilon
        step_coefficient = learning_rate / correction1
        base_step = step_coefficient * (
            float(beta1) * exp_avg + (1.0 - float(beta1)) * main
        ) / denominator
        unit_meta_step = step_coefficient * (
            1.0 - float(beta1)
        ) * meta / denominator
        base_term = base_step.square().sum()
        meta_term = unit_meta_step.square().sum()
        parameter_term = parameter.detach().square().sum()
        base_step_square = (
            base_term if base_step_square is None else base_step_square + base_term
        )
        unit_meta_step_square = (
            meta_term
            if unit_meta_step_square is None
            else unit_meta_step_square + meta_term
        )
        parameter_square = (
            parameter_term
            if parameter_square is None
            else parameter_square + parameter_term
        )
    if (
        base_step_square is None
        or unit_meta_step_square is None
        or parameter_square is None
    ):
        return 0.0, 0.0, 0.0
    base_step_norm = float(base_step_square.sqrt().item())
    unit_meta_step_norm = float(unit_meta_step_square.sqrt().item())
    parameter_norm = float(parameter_square.sqrt().item())
    if not np.isfinite(unit_meta_step_norm) or unit_meta_step_norm <= 0.0:
        return 0.0, 0.0, 0.0
    scale = min(
        requested_scale,
        max_adam_step_ratio * base_step_norm / max(unit_meta_step_norm, 1e-12),
        max_relative_parameter_step
        * parameter_norm
        / max(unit_meta_step_norm, 1e-12),
    )
    scale = max(float(scale), 0.0)
    increment_norm = scale * unit_meta_step_norm
    return (
        scale,
        increment_norm / max(base_step_norm, 1e-12),
        increment_norm / max(parameter_norm, 1e-12),
    )


def value_experience_sampling_probabilities(
    logits: torch.Tensor,
    *,
    temperature: float,
    uniform_fraction: float,
) -> torch.Tensor:
    """Convert relative utility into a full-support mixed replay law."""
    if logits.dim() != 1 or logits.numel() == 0:
        raise ValueError("replay logits must be a non-empty vector")
    temperature = max(float(temperature), 1e-4)
    uniform_fraction = float(np.clip(uniform_fraction, 0.0, 1.0))
    finite_logits = torch.nan_to_num(
        logits, nan=0.0, posinf=20.0, neginf=-20.0
    )
    learned = torch.softmax(finite_logits / temperature, dim=0)
    uniform = torch.full_like(learned, 1.0 / float(logits.numel()))
    probabilities = (
        (1.0 - uniform_fraction) * learned
        + uniform_fraction * uniform
    )
    return probabilities / probabilities.sum().clamp(min=1e-12)


def bounded_reptile_parameter_deltas(
    parameters,
    gradients,
    *,
    inner_learning_rate: float,
    meta_step_size: float,
    max_relative_parameter_delta: float,
):
    """Return one bounded, first-order Reptile displacement.

    For one SGD inner step, Reptile's interpolation is exactly a scaled
    gradient descent direction: ``theta + beta * ((theta - alpha*g) - theta)``.
    The returned deltas implement that expression without mutating parameters
    or optimizer moments.  The global relative-norm cap makes this a small
    representation adaptation rather than an untracked second critic update.
    """
    active = [
        (parameter, gradient.detach())
        for parameter, gradient in zip(parameters, gradients)
        if gradient is not None and parameter.requires_grad
    ]
    if not active:
        return tuple(None for _ in parameters), 0.0
    parameter_norm_sq = sum(
        parameter.detach().square().sum() for parameter, _ in active
    )
    gradient_norm_sq = sum(
        gradient.square().sum() for _, gradient in active
    )
    parameter_norm = float(parameter_norm_sq.sqrt().item())
    gradient_norm = float(gradient_norm_sq.sqrt().item())
    if not np.isfinite(gradient_norm) or gradient_norm <= 0.0:
        return tuple(None for _ in parameters), 0.0
    scale = max(0.0, float(inner_learning_rate)) * max(
        0.0, float(meta_step_size)
    )
    raw_norm = scale * gradient_norm
    max_norm = max(0.0, float(max_relative_parameter_delta)) * max(
        parameter_norm, 1e-12
    )
    if raw_norm > max_norm and raw_norm > 0.0:
        scale *= max_norm / raw_norm
    deltas = []
    active_by_identity = {id(parameter): gradient for parameter, gradient in active}
    for parameter in parameters:
        gradient = active_by_identity.get(id(parameter))
        deltas.append(None if gradient is None else -scale * gradient)
    applied_norm = scale * gradient_norm
    return tuple(deltas), float(applied_norm)


def role_preserving_teacher_probabilities(
    task_probabilities: Dict[bytes, float],
    mature_keys,
    anchor_key: bytes,
) -> Dict[bytes, float]:
    """Preserve anchor/transfer mass after exact-task evidence filtering.

    The curriculum chooses the semantic role first: the demonstrated anchor or
    a transfer task.  Evidence filtering is then allowed to redistribute mass
    only *inside* that role.  An immature transfer key can therefore no longer
    donate its probability to the anchor merely because its first completed
    trajectory has not reached replay yet.

    If one role has no mature task at all, the other role temporarily receives
    the complete distribution.  This is the only feasible behavior before the
    first complete transfer episode exists, and the intended role mass is
    restored automatically as soon as evidence becomes available.
    """
    mature_keys = set(mature_keys)
    raw = {
        key: max(float(value), 0.0)
        for key, value in dict(task_probabilities or {}).items()
    }
    total = float(sum(raw.values()))
    if total <= 0.0:
        return {}
    raw = {key: value / total for key, value in raw.items()}

    role_members = {
        "anchor": [anchor_key] if anchor_key in raw else [],
        "transfer": [key for key in raw if key != anchor_key],
    }
    desired_role_mass = {
        role: float(sum(raw[key] for key in keys))
        for role, keys in role_members.items()
    }
    available_roles = {
        role: [key for key in keys if key in mature_keys]
        for role, keys in role_members.items()
    }
    available_role_mass = float(
        sum(
            desired_role_mass[role]
            for role, keys in available_roles.items()
            if keys and desired_role_mass[role] > 0.0
        )
    )
    if available_role_mass <= 0.0:
        return {}

    adjusted: Dict[bytes, float] = {}
    for role, keys in available_roles.items():
        if not keys or desired_role_mass[role] <= 0.0:
            continue
        role_mass = desired_role_mass[role] / available_role_mass
        within_mass = float(sum(raw[key] for key in keys))
        if within_mass <= 0.0:
            within = {key: 1.0 / len(keys) for key in keys}
        else:
            within = {key: raw[key] / within_mass for key in keys}
        for key, conditional_probability in within.items():
            adjusted[key] = role_mass * conditional_probability

    adjusted_total = float(sum(adjusted.values()))
    if adjusted_total <= 0.0:
        return {}
    return {
        key: value / adjusted_total for key, value in adjusted.items()
    }


def sample_curriculum_role_mixed_batch_inds(
    positions_by_key: Dict[bytes, np.ndarray],
    batch_size: int,
    role_task_probabilities: Dict[str, Dict[bytes, float]],
    rng: np.random.Generator,
    *,
    min_unique_transitions_per_role: int = 1,
    target_distinct_tasks_per_role: int = 1,
    max_basket_fraction: float = 1.0,
    positions_by_role_key: Optional[
        Dict[str, Dict[bytes, np.ndarray]]
    ] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sample one universal-MDP batch while preserving curriculum role mass.

    A role is selected first (anchor, frontier, rehearsal, or probe), then an
    exact basket key is used only to locate rows belonging to that role.  Rows
    from all selected roles are finally shuffled into one Bellman update.  The
    critic therefore learns one observable set-conditioned value rule rather
    than taking a sequence of mutually offset exact-basket optimization steps.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    requested_batch_size = batch_size
    available = {
        key: np.asarray(rows, dtype=np.int64).reshape(-1)
        for key, rows in positions_by_key.items()
        if np.asarray(rows).size
    }
    if not available:
        raise BasketReplayEvidenceUnavailable("replay contains no indexed rows")

    role_provenance_enabled = positions_by_role_key is not None
    role_available: Dict[str, Dict[bytes, np.ndarray]] = {}
    if positions_by_role_key:
        for role, role_positions in positions_by_role_key.items():
            cleaned = {
                key: np.asarray(rows, dtype=np.int64).reshape(-1)
                for key, rows in dict(role_positions or {}).items()
                if key in available and np.asarray(rows).size
            }
            if cleaned:
                role_available[str(role)] = cleaned

    minimum_role_rows = max(1, int(min_unique_transitions_per_role))
    role_maps: Dict[str, Dict[bytes, float]] = {}
    role_masses: Dict[str, float] = {}
    unavailable_role_rows: Dict[str, int] = {}
    for role, task_map in dict(role_task_probabilities or {}).items():
        role = str(role)
        raw_map = {
            key: max(float(probability), 0.0)
            for key, probability in dict(task_map or {}).items()
            if float(probability) > 0.0
        }
        desired_mass = float(sum(raw_map.values()))
        available_for_role = (
            role_available.get(role, {})
            if role_provenance_enabled
            else available
        )
        filtered = {
            key: probability
            for key, probability in raw_map.items()
            if key in available_for_role
        }
        # With role provenance, a row belongs to exactly one role and one
        # basket.  Summing lengths is therefore the exact unique-row count and
        # avoids rebuilding a Python set containing the whole replay buffer on
        # every gradient update.  The set fallback is retained for old callers
        # that provide no role index.
        if role_provenance_enabled:
            unique_role_rows = sum(
                len(available_for_role[key]) for key in filtered
            )
        else:
            unique_role_rows = len(
                {
                    int(row)
                    for key in filtered
                    for row in available_for_role[key]
                }
            )
        if unique_role_rows < minimum_role_rows:
            unavailable_role_rows[role] = unique_role_rows
            continue
        if desired_mass > 0.0 and filtered:
            role_maps[role] = filtered
            # Filtering immature exact tasks may redistribute probability
            # within a role, but it must not donate that role's authority to a
            # different role (the defect that previously turned 90/10 into
            # approximately 98/2).
            role_masses[str(role)] = desired_mass
    if not role_maps:
        raise BasketReplayEvidenceUnavailable(
            "no curriculum role has the required unique replay transitions"
        )

    roles = tuple(role_maps)
    masses = np.asarray([role_masses[role] for role in roles], dtype=np.float64)
    masses /= masses.sum()
    max_per_key = max(1, int(math.ceil(requested_batch_size * float(
        np.clip(max_basket_fraction, 1.0 / requested_batch_size, 1.0)
    ))))
    role_capacities = []
    for role in roles:
        available_for_role = (
            role_available.get(role, {})
            if role_provenance_enabled
            else available
        )
        capacity = sum(
            min(len(available_for_role.get(key, ())), max_per_key)
            for key in role_maps[role]
        )
        role_capacities.append(int(capacity))
    effective_batch_size = min(
        requested_batch_size,
        *[
            int(math.floor(capacity / max(float(mass), 1e-12)))
            for capacity, mass in zip(role_capacities, masses)
            if mass > 0.0
        ],
    )
    effective_batch_size = max(1, int(effective_batch_size))

    def allocate_role_counts(total_rows: int) -> np.ndarray:
        exact = masses * float(total_rows)
        counts = np.floor(exact).astype(np.int64)
        remainder_rows = total_rows - int(counts.sum())
        if remainder_rows > 0:
            fractional = exact - counts
            order = np.argsort(
                -(fractional + rng.uniform(0.0, 1e-12, size=len(fractional)))
            )
            counts[order[:remainder_rows]] += 1
        return counts

    role_counts = allocate_role_counts(effective_batch_size)
    while any(
        int(count) > int(capacity)
        for count, capacity in zip(role_counts, role_capacities)
    ):
        effective_batch_size -= 1
        if effective_batch_size <= 0:
            raise BasketReplayEvidenceUnavailable(
                "curriculum roles cannot form a proportion-preserving batch"
            )
        role_counts = allocate_role_counts(effective_batch_size)
    selected: List[int] = []
    selected_set = set()
    selected_per_key: Dict[bytes, int] = collections.defaultdict(int)
    realized_role_counts: Dict[str, int] = collections.defaultdict(int)

    def draw_one(key: bytes, role: Optional[str] = None) -> bool:
        if selected_per_key[key] >= max_per_key:
            return False
        row_source = (
            role_available[role][key]
            if role is not None
            and role in role_available
            and key in role_available[role]
            else available[key]
        )
        row_source = np.asarray(row_source, dtype=np.int64).reshape(-1)
        if row_source.size == 0:
            return False

        # Rows within a role/key are already unique.  Random retries are
        # constant-time while the source still has free rows; only an
        # exhausted or nearly exhausted source falls back to one bounded scan.
        # The old implementation materialized a Python candidate list on every
        # draw, turning one batch into repeated O(rows-in-key) scans.
        retry_count = min(16, max(4, int(row_source.size)))
        for _ in range(retry_count):
            row = int(row_source[int(rng.integers(row_source.size))])
            if row not in selected_set:
                selected.append(row)
                selected_set.add(row)
                selected_per_key[key] += 1
                return True

        start = int(rng.integers(row_source.size))
        for offset in range(row_source.size):
            row = int(row_source[(start + offset) % row_source.size])
            if row not in selected_set:
                selected.append(row)
                selected_set.add(row)
                selected_per_key[key] += 1
                return True
        return False

    for role, requested in zip(roles, role_counts):
        requested = int(requested)
        if requested <= 0:
            continue
        task_map = role_maps[role]
        keys = tuple(task_map)
        probabilities = np.asarray(
            [task_map[key] for key in keys], dtype=np.float64
        )
        probabilities /= probabilities.sum()
        diversity_count = min(
            requested,
            len(keys),
            max(1, int(target_distinct_tasks_per_role)),
        )
        diverse_keys = rng.choice(
            len(keys),
            size=diversity_count,
            replace=False,
            p=probabilities,
        )
        for key_index in np.asarray(diverse_keys).reshape(-1):
            if draw_one(keys[int(key_index)], role=role):
                realized_role_counts[role] += 1
        attempts = 0
        while realized_role_counts[role] < requested and attempts < requested * 20:
            attempts += 1
            key = keys[int(rng.choice(len(keys), p=probabilities))]
            if draw_one(key, role=role):
                realized_role_counts[role] += 1

    # A sparse role contributes only its unique rows.  We deliberately return
    # a smaller batch instead of duplicating evidence or filling the deficit
    # from a role that the current teacher did not authorize.
    if not selected:
        raise BasketReplayEvidenceUnavailable("no unique replay rows were selectable")
    batch_inds = np.asarray(selected, dtype=np.int64)
    rng.shuffle(batch_inds)
    realized_task_counts = {
        key: int(count) for key, count in selected_per_key.items() if count
    }
    diagnostics = {
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": int(effective_batch_size),
        "sampled_rows": int(batch_inds.size),
        "unique_rows": int(np.unique(batch_inds).size),
        "role_counts": dict(realized_role_counts),
        "task_counts": realized_task_counts,
        "distinct_tasks": int(len(realized_task_counts)),
        "maximum_task_fraction": float(
            max(realized_task_counts.values(), default=0) / max(batch_inds.size, 1)
        ),
        "roles_below_unique_transition_floor": unavailable_role_rows,
    }
    return batch_inds, diagnostics


def apply_bounded_auxiliary_gradients(
    main_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    parameters,
    *,
    max_shared_ratio: float,
    project_conflicts: bool = True,
    scale_by_positive_cosine: bool = False,
    alignment_cosine_ema: Optional[float] = None,
    alignment_ema_decay: float = 0.0,
    epsilon: float = 1e-12,
    max_grad_norm: Optional[float] = None,
    main_gradients: Optional[Tuple[Optional[torch.Tensor], ...]] = None,
) -> Dict[str, float]:
    """Backpropagate an auxiliary objective without overruling control.

    Parameters touched by both objectives receive the complete primary
    financial gradient plus a one-sided, norm-bounded predictive gradient.  A
    predictive component that points against the financial gradient is first
    projected onto the financial gradient's normal plane.  Its remaining norm
    is then capped to ``max_shared_ratio`` times the financial norm.

    Parameters used only by the predictor keep their full auxiliary gradient:
    they cannot alter action or Q values directly, so restricting them would
    weaken representation learning without protecting control.
    """
    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not params:
        return {
            "main_norm": 0.0,
            "auxiliary_shared_raw_norm": 0.0,
            "auxiliary_shared_projected_norm": 0.0,
            "auxiliary_private_norm": 0.0,
            "auxiliary_scale": 0.0,
            "effective_shared_ratio": 0.0,
            "cosine": 0.0,
            "smoothed_cosine": 0.0,
            "alignment_authority": 0.0,
            "conflict": 0.0,
            "control_clip_scale": 1.0,
            "private_clip_scale": 1.0,
        }
    max_shared_ratio = float(max_shared_ratio)
    if max_shared_ratio < 0.0:
        raise ValueError("max_shared_ratio must be nonnegative")
    epsilon = max(float(epsilon), torch.finfo(torch.float32).eps)
    alignment_ema_decay = float(alignment_ema_decay)
    if not 0.0 <= alignment_ema_decay < 1.0:
        raise ValueError("alignment_ema_decay must be in [0, 1)")
    for parameter in params:
        parameter.grad = None

    if main_gradients is None:
        main_grads = torch.autograd.grad(
            main_loss,
            params,
            allow_unused=True,
        )
    else:
        if len(main_gradients) != len(params):
            raise ValueError(
                "precomputed main-gradient count does not match parameters"
            )
        main_grads = tuple(main_gradients)
    auxiliary_grads = torch.autograd.grad(
        auxiliary_loss,
        params,
        allow_unused=True,
    )
    shared_indices = [
        index
        for index, (main_grad, auxiliary_grad) in enumerate(
            zip(main_grads, auxiliary_grads)
        )
        if main_grad is not None and auxiliary_grad is not None
    ]

    if shared_indices:
        reference = main_grads[shared_indices[0]]
        main_sq = reference.new_zeros(())
        auxiliary_sq = reference.new_zeros(())
        dot = reference.new_zeros(())
        for index in shared_indices:
            main_grad = main_grads[index].detach()
            auxiliary_grad = auxiliary_grads[index].detach()
            main_sq = main_sq + main_grad.square().sum()
            auxiliary_sq = auxiliary_sq + auxiliary_grad.square().sum()
            dot = dot + (main_grad * auxiliary_grad).sum()
        main_norm = torch.sqrt(main_sq)
        auxiliary_raw_norm = torch.sqrt(auxiliary_sq)
        denominator = (main_norm * auxiliary_raw_norm).clamp_min(epsilon)
        cosine = dot / denominator
        conflict = dot < 0.0
        projection_coefficient = (
            torch.where(
                conflict,
                dot / main_sq.clamp_min(epsilon),
                dot.new_zeros(()),
            )
            if project_conflicts
            else dot.new_zeros(())
        )
        projected = {}
        projected_sq = reference.new_zeros(())
        for index in shared_indices:
            auxiliary_grad = auxiliary_grads[index].detach()
            if project_conflicts:
                auxiliary_grad = (
                    auxiliary_grad
                    - projection_coefficient * main_grads[index].detach()
                )
            projected[index] = auxiliary_grad
            projected_sq = projected_sq + auxiliary_grad.square().sum()
        projected_norm = torch.sqrt(projected_sq)
        smoothed_cosine = (
            cosine
            if alignment_cosine_ema is None
            else (
                alignment_ema_decay
                * cosine.new_tensor(float(alignment_cosine_ema))
                + (1.0 - alignment_ema_decay) * cosine
            )
        )
        alignment_authority = (
            torch.clamp(smoothed_cosine, min=0.0, max=1.0)
            if scale_by_positive_cosine
            else cosine.new_ones(())
        )
        allowed_norm = max_shared_ratio * alignment_authority * main_norm
        auxiliary_scale = torch.where(
            projected_norm <= epsilon,
            projected_norm.new_zeros(()),
            torch.clamp(
                allowed_norm / projected_norm.clamp_min(epsilon),
                max=1.0,
            ),
        )
    else:
        main_norm = main_loss.detach().new_zeros(())
        auxiliary_raw_norm = main_norm.clone()
        projected_norm = main_norm.clone()
        auxiliary_scale = main_norm.clone()
        cosine = main_norm.clone()
        smoothed_cosine = main_norm.clone()
        conflict = main_norm.new_zeros((), dtype=torch.bool)
        projected = {}
        alignment_authority = main_norm.clone()

    private_sq = auxiliary_loss.detach().new_zeros(())
    final_gradients = {}
    control_indices = []
    private_indices = []
    for index, parameter in enumerate(params):
        main_grad = main_grads[index]
        auxiliary_grad = auxiliary_grads[index]
        if main_grad is not None and auxiliary_grad is not None:
            final_grad = (
                main_grad.detach()
                + auxiliary_scale * projected[index]
            )
        elif main_grad is not None:
            final_grad = main_grad.detach()
        elif auxiliary_grad is not None:
            final_grad = auxiliary_grad.detach()
            private_sq = private_sq + final_grad.square().sum()
            private_indices.append(index)
        else:
            continue
        if main_grad is not None:
            control_indices.append(index)
        final_gradients[index] = final_grad

    control_sq = auxiliary_loss.detach().new_zeros(())
    for index in control_indices:
        control_sq = control_sq + final_gradients[index].square().sum()
    control_final_norm = torch.sqrt(control_sq)
    private_norm = torch.sqrt(private_sq)
    control_clip_scale = control_final_norm.new_ones(())
    private_clip_scale = private_norm.new_ones(())
    if max_grad_norm is not None:
        maximum = max(float(max_grad_norm), 0.0)
        control_clip_scale = torch.where(
            control_final_norm > maximum,
            torch.clamp(
                control_final_norm.new_tensor(maximum)
                / control_final_norm.clamp_min(epsilon),
                max=1.0,
            ),
            control_clip_scale,
        )
        private_clip_scale = torch.where(
            private_norm > maximum,
            torch.clamp(
                private_norm.new_tensor(maximum)
                / private_norm.clamp_min(epsilon),
                max=1.0,
            ),
            private_clip_scale,
        )
    control_index_set = set(control_indices)
    for index, parameter in enumerate(params):
        if index not in final_gradients:
            continue
        scale = (
            control_clip_scale if index in control_index_set else private_clip_scale
        )
        parameter.grad = (scale * final_gradients[index]).clone()

    effective_ratio = torch.where(
        main_norm > epsilon,
        auxiliary_scale * projected_norm / main_norm.clamp_min(epsilon),
        main_norm.new_zeros(()),
    )
    statistic_names = (
        "main_norm",
        "auxiliary_shared_raw_norm",
        "auxiliary_shared_projected_norm",
        "auxiliary_private_norm",
        "auxiliary_scale",
        "effective_shared_ratio",
        "cosine",
        "smoothed_cosine",
        "alignment_authority",
        "conflict",
        "control_clip_scale",
        "private_clip_scale",
    )
    statistic_values = torch.stack(
        (
            main_norm,
            auxiliary_raw_norm,
            projected_norm,
            private_norm,
            auxiliary_scale,
            effective_ratio,
            cosine,
            smoothed_cosine,
            alignment_authority,
            conflict.to(main_norm.dtype),
            control_clip_scale,
            private_clip_scale,
        )
    ).tolist()
    return dict(zip(statistic_names, statistic_values))


def _sample_balanced_basket_groups(
    groups: List[np.ndarray],
    batch_size: int,
    baskets_per_batch: int,
    rng: np.random.Generator,
    sample_without_replacement: bool,
    group_selection: str = "uniform",
    group_probabilities: Optional[np.ndarray] = None,
    uniform_mix: float = 0.0,
    allow_partial_unique_batch: bool = False,
) -> np.ndarray:
    """Draw internally coherent blocks from selected basket-index groups.

    ``uniform`` gives every stored task key equal probability. ``occupancy``
    gives a task probability proportional to its number of replay rows.
    ``teacher`` follows the current curriculum probabilities, renormalized over
    keys that are actually present in replay. ``teacher_uniform`` blends that
    law with uniform-over-seen-task coverage.  When partial unique batches are
    enabled, a sparse task contributes each stored transition at most once
    instead of manufacturing a nominal full batch through replacement.
    """
    nonempty_indices = [index for index, group in enumerate(groups) if len(group)]
    nonempty = [
        np.asarray(groups[index], dtype=np.int64)
        for index in nonempty_indices
    ]
    if not nonempty:
        raise ValueError("basket groups must contain at least one replay row")
    batch_size = int(batch_size)
    baskets_per_batch = int(baskets_per_batch)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if baskets_per_batch <= 0:
        raise ValueError("baskets_per_batch must be positive")
    group_selection = str(group_selection).strip().lower()
    if group_selection not in {
        "uniform",
        "occupancy",
        "teacher",
        "teacher_uniform",
    }:
        raise ValueError(
            "group_selection must be 'uniform', 'occupancy', 'teacher', or "
            "'teacher_uniform'"
        )

    n_strata = min(len(nonempty), baskets_per_batch, batch_size)
    selection_probabilities = None
    if group_selection == "occupancy":
        group_sizes = np.asarray(
            [len(group) for group in nonempty], dtype=np.float64
        )
        selection_probabilities = group_sizes / group_sizes.sum()
    elif group_selection in {"teacher", "teacher_uniform"}:
        if group_probabilities is None:
            raise ValueError("teacher group selection requires probabilities")
        selection_probabilities = np.asarray(
            group_probabilities, dtype=np.float64
        ).reshape(-1)
        if len(selection_probabilities) == len(groups):
            selection_probabilities = selection_probabilities[nonempty_indices]
        elif len(selection_probabilities) != len(nonempty):
            raise ValueError("teacher probabilities must match basket groups")
        selection_probabilities = np.clip(
            selection_probabilities, 0.0, None
        )
        probability_sum = float(selection_probabilities.sum())
        if probability_sum <= 0.0:
            group_sizes = np.asarray(
                [len(group) for group in nonempty], dtype=np.float64
            )
            selection_probabilities = group_sizes / group_sizes.sum()
        else:
            selection_probabilities /= probability_sum
        if group_selection == "teacher_uniform":
            uniform_mix = float(np.clip(uniform_mix, 0.0, 1.0))
            uniform_probabilities = np.full(
                len(nonempty), 1.0 / len(nonempty), dtype=np.float64
            )
            selection_probabilities = (
                (1.0 - uniform_mix) * selection_probabilities
                + uniform_mix * uniform_probabilities
            )
    chosen = rng.choice(
        len(nonempty),
        size=n_strata,
        replace=False,
        p=selection_probabilities,
    )
    if (
        n_strata > 1
        and group_selection in {"teacher", "teacher_uniform"}
        and selection_probabilities is not None
    ):
        # Selecting task keys according to the teacher is not sufficient if
        # every selected key is subsequently assigned an equal row block.  In
        # the predictive eight-task path that old rule capped the foundation
        # basket at one eighth of the representation gradient even when the
        # teacher still assigned it most of the task probability.  Preserve
        # broad compositional coverage by granting every selected task one row,
        # then distribute the remaining row authority according to the same
        # teacher-tempered law used to choose the tasks.
        selected_probabilities = selection_probabilities[
            np.asarray(chosen, dtype=np.int64)
        ]
        selected_probability_sum = float(selected_probabilities.sum())
        if selected_probability_sum <= 0.0:
            selected_probabilities = np.full(
                n_strata, 1.0 / n_strata, dtype=np.float64
            )
        else:
            selected_probabilities = (
                selected_probabilities / selected_probability_sum
            )
        draws_per_stratum = np.ones(n_strata, dtype=np.int64)
        remaining_draws = batch_size - n_strata
        if remaining_draws:
            draws_per_stratum += rng.multinomial(
                remaining_draws, selected_probabilities
            )
    else:
        draws_per_stratum = np.full(
            n_strata, batch_size // n_strata, dtype=np.int64
        )
        remainder = batch_size % n_strata
        if remainder:
            extra = rng.choice(n_strata, size=remainder, replace=False)
            draws_per_stratum[extra] += 1

    sampled = []
    for selected_group, draw_count in zip(chosen, draws_per_stratum):
        candidates = nonempty[int(selected_group)]
        effective_draw_count = int(draw_count)
        if (
            sample_without_replacement
            and allow_partial_unique_batch
            and int(candidates.size) < effective_draw_count
        ):
            effective_draw_count = int(candidates.size)
        replace = (
            not sample_without_replacement
            or int(candidates.size) < effective_draw_count
        )
        sampled.append(
            rng.choice(
                candidates,
                size=effective_draw_count,
                replace=replace,
            )
        )
    batch_inds = np.concatenate(sampled).astype(np.int64, copy=False)
    rng.shuffle(batch_inds)
    return batch_inds


def sample_basket_hierarchical_batch_inds(
    memberships: np.ndarray,
    batch_size: int,
    baskets_per_batch: int,
    rng: Optional[np.random.Generator] = None,
    sample_without_replacement: bool = True,
    group_selection: str = "uniform",
    task_probabilities: Optional[Dict[bytes, float]] = None,
    uniform_mix: float = 0.0,
    allow_partial_unique_batch: bool = False,
) -> np.ndarray:
    """Sample a balanced collection of coherent basket blocks.

    memberships: (n_transitions, membership_dim) multi-hot rows.
    Up to ``baskets_per_batch`` distinct membership keys are selected, then the
    batch is divided as evenly as possible among them.  Rows are sampled without
    replacement whenever a stratum contains enough transitions.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    memberships = np.asarray(memberships)
    if memberships.ndim != 2:
        raise ValueError("memberships must be 2-D (n, dim)")
    n = int(memberships.shape[0])
    if n <= 0:
        raise ValueError("memberships must be non-empty")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    groups_by_key: Dict[bytes, List[int]] = {}
    for row_index, row in enumerate(memberships):
        groups_by_key.setdefault(_membership_key(row), []).append(row_index)
    group_items = list(groups_by_key.items())
    groups = [
        np.asarray(indices, dtype=np.int64) for _, indices in group_items
    ]
    group_probabilities = None
    if str(group_selection).strip().lower() in {
        "teacher",
        "teacher_uniform",
    }:
        task_probabilities = task_probabilities or {}
        group_probabilities = np.asarray(
            [float(task_probabilities.get(key, 0.0)) for key, _ in group_items],
            dtype=np.float64,
        )
    return _sample_balanced_basket_groups(
        groups,
        batch_size=batch_size,
        baskets_per_batch=baskets_per_batch,
        rng=rng,
        sample_without_replacement=bool(sample_without_replacement),
        group_selection=group_selection,
        group_probabilities=group_probabilities,
        uniform_mix=uniform_mix,
        allow_partial_unique_batch=allow_partial_unique_batch,
    )


def basket_balanced_mean(
    values: torch.Tensor,
    memberships: Optional[torch.Tensor],
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Legacy exact-basket reduction for non-Path-2 ablations.

    Production Path 2 returns the ordinary row mean before reaching this
    helper. Keeping the old reducer isolated here preserves load/ablation
    compatibility without giving exact baskets equal authority in mixed TD.
    """
    if values.ndim == 0:
        return values
    batch_size = int(values.shape[0])
    scalar_values = values.reshape(batch_size, -1)
    if scalar_values.shape[1] != 1:
        raise ValueError(
            "basket_balanced_mean expects one scalar per replay sample; "
            f"received shape {tuple(values.shape)}"
        )
    scalar_values = scalar_values[:, 0]
    if valid_mask is None:
        valid = torch.ones_like(scalar_values, dtype=torch.bool)
    else:
        valid = valid_mask.reshape(batch_size, -1)
        if valid.shape[1] != 1:
            raise ValueError("valid_mask must provide one value per replay sample")
        valid = valid[:, 0].bool()

    if memberships is None or memberships.ndim != 2 or memberships.shape[0] != batch_size:
        if not valid.any():
            return scalar_values.sum() * 0.0
        return scalar_values[valid].mean()

    _, inverse = torch.unique(memberships.detach(), dim=0, return_inverse=True)
    group_means = []
    for group_index in range(int(inverse.max().item()) + 1):
        group_mask = (inverse == group_index) & valid
        if group_mask.any():
            group_means.append(scalar_values[group_mask].mean())
    if not group_means:
        return scalar_values.sum() * 0.0
    return torch.stack(group_means).mean()


class ComponentOutcomeReplayBufferSamples(NamedTuple):
    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    next_observations: Dict[str, torch.Tensor]
    dones: torch.Tensor
    rewards: torch.Tensor
    component_outcomes: torch.Tensor
    component_outcome_mask: torch.Tensor


class PredictiveSequenceReplayBufferSamples(NamedTuple):
    """Contiguous future observations anchored on a financial replay batch."""

    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    future_observations: Dict[str, torch.Tensor]
    component_outcomes: torch.Tensor
    component_outcome_masks: torch.Tensor
    outcome_valid_steps: torch.Tensor
    valid_steps: torch.Tensor


class ComponentOutcomeReplayBuffer(DictReplayBuffer):
    """Dict replay buffer with typed per-asset financial target metadata."""

    def __init__(
        self,
        *args,
        component_outcome_assets: int,
        component_outcome_dim: int = 1,
        component_outcome_target_kind: str = "scaled_executed_contribution",
        relabel_market_width: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.market_input_contract = config.market_input_contract()
        self.component_outcome_assets = int(component_outcome_assets)
        self.component_outcome_dim = int(component_outcome_dim)
        self._setup_cross_basket_relabel_storage(int(relabel_market_width or 0))
        self._last_sample_env_indices: Optional[np.ndarray] = None
        self.component_outcome_target_kind = str(
            component_outcome_target_kind
        )
        if self.component_outcome_target_kind not in {
            "scaled_executed_contribution",
            "scaled_unit_asset_return",
        }:
            raise ValueError(
                "Unsupported component outcome target kind: "
                f"{self.component_outcome_target_kind}"
            )
        if self.component_outcome_assets <= 0 or self.component_outcome_dim <= 0:
            raise ValueError("Component outcome replay dimensions must be positive")
        storage_shape = (
            self.buffer_size,
            self.n_envs,
            self.component_outcome_assets,
            self.component_outcome_dim,
        )
        self.component_outcomes = np.zeros(storage_shape, dtype=np.float32)
        self.component_outcome_masks = np.zeros(storage_shape, dtype=bool)
        # Private replay provenance, never a neural task feature. It records
        # why the teacher collected a transition so a uniform probe cannot
        # later masquerade as anchor/frontier evidence merely because its
        # canonical basket happens to be identical.
        self.curriculum_role_codes = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.uint8
        )
        # Incremental membership index for hierarchical replay.  Keeping this
        # alongside the circular buffer avoids an O(buffer_size) membership
        # scan on every gradient update.  Production uses n_envs=1; multi-env
        # sampling continues to fall back to the base DictReplayBuffer path.
        self._basket_positions_by_key: Dict[
            bytes, _DenseReplayPositionIndex
        ] = {}
        # Role provenance is maintained alongside the basket index.  The
        # curriculum-role sampler is called from the gradient hot path; it
        # must not reconstruct this mapping by scanning every stored row on
        # every update.
        self._basket_positions_by_role_key: Dict[
            str, Dict[bytes, _DenseReplayPositionIndex]
        ] = {}
        # Compatibility caches are retained for replay checkpoints that still
        # deserialize legacy Python sets. New dense indexes expose their NumPy
        # storage directly and do not use these caches.
        self._basket_position_array_cache: Dict[bytes, np.ndarray] = {}
        self._basket_role_position_array_cache: Dict[
            str, Dict[bytes, np.ndarray]
        ] = {}
        self._basket_key_by_position: List[Optional[bytes]] = [
            None
        ] * self.buffer_size
        self._basket_completed_episodes_by_key: Dict[bytes, int] = {}

    def _setup_cross_basket_relabel_storage(self, market_width: int) -> None:
        """Allocate the exogenous per-transition state used for relabeling."""
        self.relabel_market_width = max(0, int(market_width))
        self.relabel_enabled = self.relabel_market_width > 0
        if not self.relabel_enabled:
            return
        storage_dtype = np.dtype(
            getattr(config, "CROSS_BASKET_RELABEL_STORAGE_DTYPE", np.float16)
        )
        asset_shape = (
            self.buffer_size,
            self.n_envs,
            self.component_outcome_assets,
        )
        market_shape = asset_shape + (self.relabel_market_width,)
        self.relabel_market = np.zeros(market_shape, dtype=storage_dtype)
        self.relabel_next_market = np.zeros(market_shape, dtype=storage_dtype)
        self.relabel_asset_returns = np.zeros(asset_shape, dtype=np.float32)
        self.relabel_price_valid = np.zeros(asset_shape, dtype=bool)
        self.relabel_next_partial_count = np.full(
            asset_shape[:2], -1, dtype=np.int16
        )
        self.relabel_valid = np.zeros(asset_shape[:2], dtype=bool)

    def _validated_relabel_metadata(self, info: Dict[str, Any]):
        """Return one transition's relabel arrays, or None when absent."""
        if not getattr(self, "relabel_enabled", False):
            return None
        required = (
            "relabel_market",
            "relabel_next_market",
            "relabel_asset_returns",
            "relabel_price_valid",
        )
        if any(key not in info for key in required):
            return None
        assets = self.component_outcome_assets
        market = np.asarray(info["relabel_market"])
        next_market = np.asarray(info["relabel_next_market"])
        asset_returns = np.asarray(info["relabel_asset_returns"], dtype=np.float32)
        price_valid = np.asarray(info["relabel_price_valid"], dtype=bool)
        market_shape = (assets, self.relabel_market_width)
        if (
            market.shape != market_shape
            or next_market.shape != market_shape
            or asset_returns.shape != (assets,)
            or price_valid.shape != (assets,)
        ):
            raise ValueError(
                "Cross-basket relabel metadata must have market shape "
                f"{market_shape} and asset shape {(assets,)}; received "
                f"{market.shape}, {next_market.shape}, {asset_returns.shape}, "
                f"{price_valid.shape}."
            )
        price_valid = price_valid & np.isfinite(asset_returns)
        return (
            market,
            next_market,
            np.where(price_valid, asset_returns, 0.0).astype(np.float32),
            price_valid,
            int(info.get("relabel_next_partial_count", -1)),
        )

    def cross_basket_relabel_metadata(
        self,
        batch_inds: np.ndarray,
        env_indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Gather relabel arrays for explicit (position, env) replay rows."""
        batch_inds = np.asarray(batch_inds, dtype=np.int64)
        env_indices = np.asarray(env_indices, dtype=np.int64)
        return {
            "market": self.relabel_market[batch_inds, env_indices],
            "next_market": self.relabel_next_market[batch_inds, env_indices],
            "asset_returns": self.relabel_asset_returns[batch_inds, env_indices],
            "price_valid": self.relabel_price_valid[batch_inds, env_indices],
            "next_partial_count": self.relabel_next_partial_count[
                batch_inds, env_indices
            ].astype(np.int64),
            "valid": self.relabel_valid[batch_inds, env_indices],
        }

    def _invalidate_basket_position_arrays(
        self,
        basket_key: Optional[bytes] = None,
        role: Optional[str] = None,
    ) -> None:
        """Invalidate only cache entries affected by a replay overwrite."""
        if basket_key is None:
            self._basket_position_array_cache.clear()
            self._basket_role_position_array_cache.clear()
            return
        self._basket_position_array_cache.pop(basket_key, None)
        if role is None:
            for role_cache in self._basket_role_position_array_cache.values():
                role_cache.pop(basket_key, None)
        else:
            self._basket_role_position_array_cache.get(role, {}).pop(
                basket_key, None
            )

    def _cached_basket_positions(
        self,
        basket_key: bytes,
        *,
        role: Optional[str] = None,
    ) -> np.ndarray:
        """Return live replay positions without rescanning replay storage."""
        if role is None:
            cache = self._basket_position_array_cache
            source = self._basket_positions_by_key.get(basket_key, ())
        else:
            cache = self._basket_role_position_array_cache.setdefault(role, {})
            source = self._basket_positions_by_role_key.get(role, {}).get(
                basket_key, ()
            )
        if isinstance(source, _DenseReplayPositionIndex):
            return source.array_view()
        cached = cache.get(basket_key)
        if cached is None:
            cached = np.asarray(sorted(source), dtype=np.int64)
            cache[basket_key] = cached
        return cached

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        if any(version in self.market_input_contract for version in (
            ":global_pool_v1:", ":whole_stock_pool_v2:", ":whole_stock_pool_v3:",
            ":whole_stock_pool_task_critic_v4:",
            ":whole_stock_pool_task_advantage_v5:",
            ":whole_stock_pool_task_centered_advantage_v6:",
            ":whole_stock_pool_task_centered_advantage_popart_v7:",
            ":whole_stock_pool_task_direct_q_reference_advantage_v8:",
            ":whole_stock_pool_task_direct_q_reference_advantage_v9:",
        )):
            width = self.observation_space.spaces["mask"].shape[0]
            if width == len(config.COIN_POOL):
                canonical = np.arange(width)
                for state in (obs, next_obs):
                    if not np.all(np.asarray(state["asset_ids"]) == canonical):
                        raise ValueError("Global-pool replay requires canonical asset positions")
                if not np.array_equal(obs["mask"], next_obs["mask"]):
                    raise ValueError("Availability must stay fixed inside an episode transition")
        write_position = self.pos
        new_basket_key = None
        if self.n_envs == 1 and isinstance(obs, dict):
            membership = _private_membership_row_from_observation(obs)
            if membership is not None and np.any(membership > 0.5):
                new_basket_key = _membership_key(membership)
        old_basket_key = None
        old_curriculum_role = None
        old_row_completed_episode = False
        if self.n_envs == 1:
            old_basket_key = self._basket_key_by_position[write_position]
            old_role_code = int(self.curriculum_role_codes[write_position, 0])
            old_curriculum_role = CURRICULUM_CODE_TO_ROLE.get(old_role_code)
            if old_basket_key is not None:
                old_row_completed_episode = bool(
                    np.asarray(self.dones[write_position, 0]).reshape(-1)[0]
                )
        new_row_completed_episode = bool(
            self.n_envs == 1
            and np.asarray(done).reshape(-1).size
            and np.asarray(done).reshape(-1)[0]
        )
        validated_metadata = []
        for env_index in range(self.n_envs):
            info = infos[env_index] if env_index < len(infos) else {}
            received_contract = info.get("market_input_contract", self.market_input_contract)
            if received_contract != self.market_input_contract:
                raise ValueError(
                    f"Replay market input contract mismatch: environment={received_contract}, "
                    f"buffer={self.market_input_contract}"
                )
            curriculum_role = str(info.get("curriculum_role", "")).strip()
            curriculum_role_code = int(
                CURRICULUM_ROLE_TO_CODE.get(curriculum_role, 0)
            )
            received_target_kind = info.get("component_outcome_target_kind")
            if (
                received_target_kind is not None
                and str(received_target_kind)
                != self.component_outcome_target_kind
            ):
                raise ValueError(
                    "Component outcome target kind mismatch: buffer expects "
                    f"{self.component_outcome_target_kind}, environment emitted "
                    f"{received_target_kind}."
                )
            targets = np.asarray(
                info.get(
                    "component_outcomes",
                    np.zeros(
                        (
                            self.component_outcome_assets,
                            self.component_outcome_dim,
                        ),
                        dtype=np.float32,
                    ),
                ),
                dtype=np.float32,
            )
            masks = np.asarray(
                info.get(
                    "component_outcome_mask",
                    np.zeros_like(targets, dtype=bool),
                ),
                dtype=bool,
            )
            expected_shape = (
                self.component_outcome_assets,
                self.component_outcome_dim,
            )
            if targets.shape != expected_shape or masks.shape != expected_shape:
                raise ValueError(
                    "Component outcome metadata must have shape "
                    f"{expected_shape}; received {targets.shape} and {masks.shape}."
                )
            finite_mask = masks & np.isfinite(targets)
            validated_metadata.append(
                (
                    np.where(finite_mask, targets, 0.0),
                    finite_mask,
                    curriculum_role_code,
                    self._validated_relabel_metadata(info),
                )
            )

        # Validate every environment before the base buffer advances its write
        # pointer, so malformed metadata cannot leave a partially committed row.
        super().add(obs, next_obs, action, reward, done, infos)
        for env_index, (targets, finite_mask, role_code, relabel) in enumerate(
            validated_metadata
        ):
            self.component_outcomes[write_position, env_index] = np.where(
                finite_mask, targets, 0.0
            )
            self.component_outcome_masks[write_position, env_index] = finite_mask
            self.curriculum_role_codes[write_position, env_index] = np.uint8(
                role_code
            )
            if getattr(self, "relabel_enabled", False):
                self.relabel_valid[write_position, env_index] = relabel is not None
                if relabel is not None:
                    (
                        market,
                        next_market,
                        asset_returns,
                        price_valid,
                        partial_count,
                    ) = relabel
                    self.relabel_market[write_position, env_index] = market
                    self.relabel_next_market[write_position, env_index] = next_market
                    self.relabel_asset_returns[write_position, env_index] = asset_returns
                    self.relabel_price_valid[write_position, env_index] = price_valid
                    self.relabel_next_partial_count[write_position, env_index] = (
                        partial_count
                    )
        if self.n_envs == 1 and new_basket_key is not None:
            if old_basket_key is not None:
                old_positions = self._basket_positions_by_key.get(old_basket_key)
                if old_positions is not None:
                    old_positions.discard(write_position)
                    if not old_positions:
                        self._basket_positions_by_key.pop(old_basket_key, None)
                if old_row_completed_episode:
                    remaining = max(
                        0,
                        int(
                            self._basket_completed_episodes_by_key.get(
                                old_basket_key, 0
                            )
                        )
                        - 1,
                    )
                    if remaining:
                        self._basket_completed_episodes_by_key[
                            old_basket_key
                        ] = remaining
                    else:
                        self._basket_completed_episodes_by_key.pop(
                            old_basket_key, None
                        )
                if old_curriculum_role is not None:
                    old_role_positions = self._basket_positions_by_role_key.get(
                        old_curriculum_role, {}
                    )
                    old_role_rows = old_role_positions.get(old_basket_key)
                    if old_role_rows is not None:
                        old_role_rows.discard(write_position)
                        if not old_role_rows:
                            old_role_positions.pop(old_basket_key, None)
                    if not old_role_positions:
                        self._basket_positions_by_role_key.pop(
                            old_curriculum_role, None
                        )
                self._invalidate_basket_position_arrays(old_basket_key)
            self._basket_positions_by_key.setdefault(
                new_basket_key, _DenseReplayPositionIndex()
            ).add(write_position)
            self._basket_key_by_position[write_position] = new_basket_key
            new_role_code = int(validated_metadata[0][2]) if validated_metadata else 0
            new_curriculum_role = CURRICULUM_CODE_TO_ROLE.get(new_role_code)
            if new_curriculum_role is not None:
                self._basket_positions_by_role_key.setdefault(
                    new_curriculum_role, {}
                ).setdefault(
                    new_basket_key, _DenseReplayPositionIndex()
                ).add(write_position)
            if new_row_completed_episode:
                self._basket_completed_episodes_by_key[new_basket_key] = (
                    int(
                        self._basket_completed_episodes_by_key.get(
                            new_basket_key, 0
                        )
                    )
                    + 1
                )

    def _rebuild_basket_position_index(self) -> None:
        """Rebuild membership groups when loading an older replay buffer."""
        self._basket_positions_by_key = {}
        self._basket_positions_by_role_key = {}
        self._basket_position_array_cache = {}
        self._basket_role_position_array_cache = {}
        self._basket_key_by_position = [None] * self.buffer_size
        self._basket_completed_episodes_by_key = {}
        if self.n_envs != 1 or "basket_membership" not in self.observations:
            return
        upper = self.buffer_size if self.full else self.pos
        memberships = np.asarray(self.observations["basket_membership"][:upper])
        if memberships.ndim != 3 or memberships.shape[1] != 1:
            return
        for position, row in enumerate(memberships[:, 0, :]):
            if not np.any(np.asarray(row) > 0.5):
                stored_observation = {
                    "basket_membership": row,
                }
                if "asset_ids" in self.observations:
                    stored_observation["asset_ids"] = self.observations[
                        "asset_ids"
                    ][position, 0]
                repaired = _private_membership_row_from_observation(
                    stored_observation
                )
                if repaired is not None and np.any(repaired > 0.5):
                    row = repaired
                    self.observations["basket_membership"][position, 0] = row
            key = _membership_key(row)
            self._basket_positions_by_key.setdefault(
                key, _DenseReplayPositionIndex()
            ).add(position)
            self._basket_key_by_position[position] = key
            role_codes = getattr(self, "curriculum_role_codes", None)
            if role_codes is not None:
                role_code = int(role_codes[position, 0])
                role = CURRICULUM_CODE_TO_ROLE.get(role_code)
                if role is not None:
                    self._basket_positions_by_role_key.setdefault(
                        role, {}
                    ).setdefault(
                        key, _DenseReplayPositionIndex()
                    ).add(position)
            if bool(np.asarray(self.dones[position, 0]).reshape(-1)[0]):
                self._basket_completed_episodes_by_key[key] = (
                    int(self._basket_completed_episodes_by_key.get(key, 0)) + 1
                )

    def sample_hierarchical_batch_inds(
        self,
        batch_size: int,
        baskets_per_batch: int,
        rng: np.random.Generator,
        sample_without_replacement: bool = True,
        group_selection: str = "uniform",
        task_probabilities: Optional[Dict[bytes, float]] = None,
        min_completed_episodes: int = 0,
        uniform_mix: float = 0.0,
        allow_partial_unique_batch: bool = False,
    ) -> np.ndarray:
        if not hasattr(self, "_basket_positions_by_key"):
            self._rebuild_basket_position_index()
        if not hasattr(self, "_basket_completed_episodes_by_key"):
            self._rebuild_basket_position_index()
        min_completed_episodes = max(0, int(min_completed_episodes))
        group_keys = [
            key
            for key in sorted(self._basket_positions_by_key)
            if self._basket_positions_by_key[key]
            and int(self._basket_completed_episodes_by_key.get(key, 0))
            >= min_completed_episodes
        ]
        groups = [
            self._cached_basket_positions(key)
            for key in group_keys
        ]
        if not groups:
            raise BasketReplayEvidenceUnavailable(
                "replay buffer has no basket with the required completed-"
                "episode evidence"
            )
        group_probabilities = None
        if str(group_selection).strip().lower() in {
            "teacher",
            "teacher_uniform",
        }:
            task_probabilities = task_probabilities or {}
            group_probabilities = np.asarray(
                [float(task_probabilities.get(key, 0.0)) for key in group_keys],
                dtype=np.float64,
            )
        return _sample_balanced_basket_groups(
            groups,
            batch_size=batch_size,
            baskets_per_batch=baskets_per_batch,
            rng=rng,
            sample_without_replacement=sample_without_replacement,
            group_selection=group_selection,
            group_probabilities=group_probabilities,
            uniform_mix=uniform_mix,
            allow_partial_unique_batch=allow_partial_unique_batch,
        )

    def sample_curriculum_mixed_batch_inds(
        self,
        batch_size: int,
        role_task_probabilities: Dict[str, Dict[bytes, float]],
        rng: np.random.Generator,
        *,
        min_unique_transitions_per_role: int,
        target_distinct_tasks_per_role: int,
        max_basket_fraction: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return a role-stratified mixture of unique replay transitions."""
        if not hasattr(self, "_basket_positions_by_key"):
            self._rebuild_basket_position_index()
        requested_keys = {
            key
            for task_map in dict(role_task_probabilities or {}).values()
            for key in dict(task_map or {})
        }
        # The teacher law is the only source of eligible task keys.  Avoid
        # materializing cached arrays for unrelated replay baskets, which is
        # important once the active support is larger than the current 1,287
        # task experiment.
        positions_by_key = {
            key: self._cached_basket_positions(key)
            for key in requested_keys
            if self._basket_positions_by_key.get(key)
        }
        # This cache is updated in ``add`` and rebuilt once for legacy
        # checkpoints.  The previous implementation walked every position in
        # every basket here, making replay cost grow linearly with buffer size
        # on each financial update.
        positions_by_role_key = getattr(
            self, "_basket_positions_by_role_key", None
        )
        if positions_by_role_key is None:
            self._rebuild_basket_position_index()
            positions_by_role_key = self._basket_positions_by_role_key
        else:
            positions_by_role_key = {
                role: {
                    key: self._cached_basket_positions(key, role=role)
                    for key, role_positions in role_map.items()
                    if role_positions
                }
                for role, role_map in positions_by_role_key.items()
                if role_map
            }
            # Legacy checkpoints and provenance-neutral unit fixtures may have
            # a valid basket index but no role metadata.  An empty mapping must
            # mean "fall back to basket availability", not "role provenance is
            # enabled but every role has zero rows".
            if not positions_by_role_key:
                positions_by_role_key = None
        return sample_curriculum_role_mixed_batch_inds(
            positions_by_key,
            batch_size,
            role_task_probabilities,
            rng,
            min_unique_transitions_per_role=min_unique_transitions_per_role,
            target_distinct_tasks_per_role=target_distinct_tasks_per_role,
            max_basket_fraction=max_basket_fraction,
            positions_by_role_key=positions_by_role_key,
        )

    def basket_sample_diagnostics(
        self,
        batch_inds: np.ndarray,
        requested_batch_size: int,
    ) -> Dict[str, Any]:
        """Describe the exact-task composition of a replay sample."""
        batch_inds = np.asarray(batch_inds, dtype=np.int64).reshape(-1)
        if batch_inds.size == 0:
            return {}
        if not hasattr(self, "_basket_positions_by_key"):
            self._rebuild_basket_position_index()
        keys = [self._basket_key_by_position[int(position)] for position in batch_inds]
        keys = [key for key in keys if key is not None]
        if not keys:
            return {}
        key_counts = collections.Counter(keys)
        dominant_key, dominant_count = key_counts.most_common(1)[0]
        first_position = next(
            int(position)
            for position in batch_inds
            if self._basket_key_by_position[int(position)] == dominant_key
        )
        if "basket_membership" in self.observations:
            memberships = np.asarray(
                self.observations["basket_membership"][first_position, 0]
            ).reshape(-1)
        else:
            # Identity-neutral production observations may omit the public
            # membership field while the replay buffer maintains a private
            # provenance key reconstructed from asset ids for diagnostics.
            memberships = np.unpackbits(
                np.frombuffer(dominant_key, dtype=np.uint8),
                bitorder="little",
            )[: int(getattr(config, "BASKET_MEMBERSHIP_DIM", 0))]
        membership_ids = tuple(int(index) for index in np.flatnonzero(memberships > 0.5))
        stored_rows = len(self._basket_positions_by_key.get(dominant_key, ()))
        sampled_rows = int(batch_inds.size)
        unique_rows = int(np.unique(batch_inds).size)
        requested_batch_size = int(requested_batch_size)
        minimum_task_rows = min(len(self._basket_positions_by_key.get(key, ()))
                                for key in key_counts)
        minimum_task_episodes = min(int(self._basket_completed_episodes_by_key.get(key, 0))
                                    for key in key_counts)
        return {
            "membership_ids": membership_ids,
            "task_fingerprint": int.from_bytes(dominant_key[:4], "little"),
            "stored_rows": int(stored_rows),
            "sampled_rows": sampled_rows,
            "unique_rows": unique_rows,
            "completed_episodes": int(
                self._basket_completed_episodes_by_key.get(dominant_key, 0)
            ),
            "distinct_tasks": int(len(key_counts)),
            "minimum_task_sampled_rows": int(min(key_counts.values())),
            "minimum_task_stored_rows": int(minimum_task_rows),
            "minimum_task_completed_episodes": int(minimum_task_episodes),
            "dominant_task_fraction": float(dominant_count / max(sampled_rows, 1)),
            "full_batch_ready": float(stored_rows >= requested_batch_size),
            "partial_unique_batch": float(sampled_rows < requested_batch_size),
            "unique_fraction": float(
                unique_rows / max(sampled_rows, 1)
            ),
            "requested_fraction": float(
                sampled_rows / max(requested_batch_size, 1)
            ),
        }

    def _get_samples(self, batch_inds, env=None, env_indices=None):
        if env_indices is None:
            env_indices = np.random.randint(
                0, high=self.n_envs, size=(len(batch_inds),)
            )
        env_indices = np.asarray(env_indices, dtype=np.int64)
        self._last_sample_env_indices = env_indices
        obs = self._normalize_obs(
            {
                key: values[batch_inds, env_indices, :]
                for key, values in self.observations.items()
            },
            env,
        )
        next_obs = self._normalize_obs(
            {
                key: values[batch_inds, env_indices, :]
                for key, values in self.next_observations.items()
            },
            env,
        )
        return ComponentOutcomeReplayBufferSamples(
            observations={key: self.to_torch(value) for key, value in obs.items()},
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations={
                key: self.to_torch(value) for key, value in next_obs.items()
            },
            dones=self.to_torch(
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(
                self._normalize_reward(
                    self.rewards[batch_inds, env_indices].reshape(-1, 1), env
                )
            ),
            component_outcomes=self.to_torch(
                self.component_outcomes[batch_inds, env_indices]
            ),
            component_outcome_mask=self.to_torch(
                self.component_outcome_masks[batch_inds, env_indices]
            ),
        )

    def sample_with_batch_inds(self, batch_size: int, env=None):
        """Ordinary SB3 row sampling while retaining the sampled positions.

        Multi-step SPR must begin from the exact rows used by financial TD.
        SB3's public ``sample`` API intentionally hides those positions, so
        this typed-buffer helper reproduces its uniform index law and returns
        the indices alongside the unchanged sample object.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound <= 0:
            raise ValueError("cannot sample an empty replay buffer")
        if self.optimize_memory_usage:
            if not self.full:
                batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            else:
                batch_inds = (
                    np.random.randint(1, self.buffer_size, size=batch_size)
                    + self.pos
                ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        batch_inds = np.asarray(batch_inds, dtype=np.int64)
        return self._get_samples(batch_inds, env=env), batch_inds

    def _chronological_successor(self, position: int) -> Optional[int]:
        """Return the next stored transition, never crossing the live edge."""
        position = int(position)
        if self.full:
            successor = (position + 1) % self.buffer_size
            # ``pos`` is the oldest row and the next write location. Moving
            # from the newest row into ``pos`` would jump backward in time.
            return None if successor == self.pos else successor
        successor = position + 1
        return successor if successor < self.pos else None

    def get_predictive_sequences(
        self,
        start_batch_inds: np.ndarray,
        horizon: int,
        env=None,
    ) -> PredictiveSequenceReplayBufferSamples:
        """Gather K-step causal replay sequences without crossing resets.

        The target at horizon ``k`` is the stored next observation of the
        transition at ``t+k-1``. A raw replay ``done`` (including a time-limit
        reset) invalidates that horizon and every later horizon, matching SPR's
        instruction to truncate prediction at episode boundaries.
        """
        if self.n_envs != 1:
            raise ValueError("multi-step predictive replay requires n_envs=1")
        if self.optimize_memory_usage:
            raise ValueError(
                "multi-step predictive replay requires explicit next-observation "
                "storage (optimize_memory_usage=False)"
            )
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("predictive horizon must be positive")
        starts = np.asarray(start_batch_inds, dtype=np.int64).reshape(-1)
        batch_size = int(starts.size)
        offsets = np.arange(horizon, dtype=np.int64)[None, :]
        positions = (starts[:, None] + offsets) % self.buffer_size
        if self.full:
            chronological_age = (starts - int(self.pos)) % self.buffer_size
            storage_valid = offsets <= (
                self.buffer_size - 1 - chronological_age[:, None]
            )
        else:
            storage_valid = (starts[:, None] + offsets) < int(self.pos)

        gathered_dones = np.asarray(self.dones[positions, 0]).reshape(
            batch_size, horizon, -1
        )[..., 0].astype(bool)
        previous_terminal = np.concatenate(
            (
                np.zeros((batch_size, 1), dtype=bool),
                np.cumsum(gathered_dones[:, :-1], axis=1) > 0,
            ),
            axis=1,
        )
        same_basket = np.ones((batch_size, horizon), dtype=bool)
        basket_values = self.observations.get("basket_membership")
        if basket_values is not None:
            start_baskets = basket_values[starts, 0]
            gathered_baskets = basket_values[positions, 0]
            same_basket = np.all(
                gathered_baskets == start_baskets[:, None], axis=-1
            )
            same_basket = np.logical_and.accumulate(same_basket, axis=1)
        outcome_valid_steps = storage_valid & ~previous_terminal & same_basket
        valid_steps = outcome_valid_steps & ~gathered_dones

        action_sequences = np.asarray(self.actions[positions, 0]).copy()
        reward_sequences = np.asarray(self.rewards[positions, 0]).reshape(
            batch_size, horizon, -1
        ).copy()
        done_sequences = gathered_dones[..., None].astype(np.float32)
        action_sequences[~outcome_valid_steps] = 0
        reward_sequences[~outcome_valid_steps] = 0
        done_sequences[~outcome_valid_steps] = 0

        # Invalid latent horizons carry the real starting observation so every
        # attention call remains finite.  The explicit validity mask is the
        # sole authority over whether that carrier contributes to a loss.
        future_observations = {}
        for key, values in self.next_observations.items():
            carrier = np.repeat(
                self.observations[key][starts, 0][:, None], horizon, axis=1
            )
            gathered = values[positions, 0]
            broadcast_valid = valid_steps.reshape(
                batch_size, horizon, *([1] * (gathered.ndim - 2))
            )
            future_observations[key] = np.where(
                broadcast_valid, gathered, carrier
            ).astype(values.dtype, copy=False)

        component_outcomes = np.asarray(
            self.component_outcomes[positions, 0]
        ).copy()
        component_outcome_masks = np.asarray(
            self.component_outcome_masks[positions, 0]
        ).copy()
        component_outcome_masks &= outcome_valid_steps[..., None, None]
        component_outcomes[~component_outcome_masks] = 0.0

        if env is not None:
            flattened = {
                key: values.reshape(
                    batch_size * horizon, *values.shape[2:]
                )
                for key, values in future_observations.items()
            }
            normalized = self._normalize_obs(flattened, env)
            future_observations = {
                key: values.reshape(
                    batch_size, horizon, *values.shape[1:]
                )
                for key, values in normalized.items()
            }
        return PredictiveSequenceReplayBufferSamples(
            actions=self.to_torch(action_sequences),
            rewards=self.to_torch(
                self._normalize_reward(reward_sequences, env)
            ),
            dones=self.to_torch(done_sequences),
            future_observations={
                key: self.to_torch(value)
                for key, value in future_observations.items()
            },
            component_outcomes=self.to_torch(component_outcomes),
            component_outcome_masks=self.to_torch(component_outcome_masks),
            outcome_valid_steps=self.to_torch(outcome_valid_steps),
            valid_steps=self.to_torch(valid_steps),
        )


def _finite_float_or_none(value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def normalize_cross_basket_row_mode(value, default: str = "mixed") -> str:
    """Map config aliases onto the two XBR consumer batches.

    ``original`` / ``real`` / ``source`` / ``full`` select the replay batch
    saved before same-minute relabeling.  ``mixed`` / ``relabeled`` /
    ``relabel`` select the batch after the tail replacement.
    """
    text = str(default if value is None else value).strip().lower()
    aliases = {
        "original": "original",
        "real": "original",
        "source": "original",
        "full": "original",
        "mixed": "mixed",
        "relabeled": "mixed",
        "relabel": "mixed",
    }
    if text not in aliases:
        raise ValueError(
            f"Unknown cross-basket row mode {value!r}; expected original/real "
            "or mixed"
        )
    return aliases[text]


def select_cross_basket_consumer_replay(
    mixed_replay_data,
    original_replay_data,
    mode,
    default: str = "mixed",
):
    """Return the replay object one XBR consumer is allowed to train on."""
    if normalize_cross_basket_row_mode(mode, default=default) == "original":
        return original_replay_data
    return mixed_replay_data


def replay_observations_as_float(observations):
    """Cast a dict or tensor observation batch to float without copying keys."""
    if isinstance(observations, dict):
        return {key: value.float() for key, value in observations.items()}
    return observations.float()


def sample_cross_basket_relabel_masks(
    source_masks: np.ndarray,
    eligible: np.ndarray,
    one_swap_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a different same-size basket per row from price-valid assets.

    Returns the new masks, a flag for rows that actually changed, and a flag
    for rows produced by a single-asset swap.
    """
    source_masks = np.asarray(source_masks, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    row_count, width = source_masks.shape
    new_masks = source_masks.copy()
    changed = np.zeros(row_count, dtype=bool)
    swapped = np.zeros(row_count, dtype=bool)
    for row in range(row_count):
        source = source_masks[row]
        count = int(source.sum())
        pool = np.flatnonzero(eligible[row])
        if count == 0 or pool.size <= count:
            continue
        inside = np.flatnonzero(source & eligible[row])
        outside = np.flatnonzero(~source & eligible[row])
        candidate = None
        if (
            rng.random() < float(one_swap_fraction)
            and inside.size == count
            and outside.size > 0
        ):
            candidate = source.copy()
            candidate[rng.choice(inside)] = False
            candidate[rng.choice(outside)] = True
            swapped[row] = True
        else:
            for _ in range(8):
                drawn = np.zeros(width, dtype=bool)
                drawn[rng.choice(pool, size=count, replace=False)] = True
                if not np.array_equal(drawn, source):
                    candidate = drawn
                    break
        if candidate is None:
            continue
        new_masks[row] = candidate
        changed[row] = True
    return new_masks, changed, swapped


def _relabel_market_block(
    market: torch.Tensor,
    features_per_coin: int,
    primary_count: int,
    comparative_offset: int,
    comparative_count: int,
) -> torch.Tensor:
    """Place stored market fields into the per-asset snapshot token layout."""
    rows, width, _ = market.shape
    block = market.new_zeros((rows, width, int(features_per_coin)))
    block[..., :primary_count] = market[..., :primary_count]
    block[..., comparative_offset:comparative_offset + comparative_count] = market[
        ..., primary_count:primary_count + comparative_count
    ]
    return block


def relabel_cross_basket_observations(
    observations: Dict[str, torch.Tensor],
    new_mask: torch.Tensor,
    market: torch.Tensor,
    *,
    features_per_coin: int,
    primary_count: int,
    comparative_offset: int,
    comparative_count: int,
) -> Dict[str, torch.Tensor]:
    """Show a stored decision minute through another basket.

    Assets kept from the original basket retain their stored snapshot,
    including cost basis, because their previous weights are unchanged.
    Added assets receive the exogenous market block with no position, and
    removed assets are zeroed exactly as PortfolioEnv leaves unavailable
    slots.  The weight of removed assets moves to cash in ``prev_action``.
    """
    rows, width = new_mask.shape
    new_active = new_mask > 0.5
    old_active = observations["mask"] > 0.5
    common = new_active & old_active
    added = new_active & ~old_active
    result = {key: value.clone() for key, value in observations.items()}

    snapshot = observations["observations"].reshape(rows, width, features_per_coin)
    added_block = _relabel_market_block(
        market.to(snapshot.dtype),
        features_per_coin,
        primary_count,
        comparative_offset,
        comparative_count,
    )
    relabeled = torch.where(common.unsqueeze(-1), snapshot, torch.zeros_like(snapshot))
    relabeled = torch.where(added.unsqueeze(-1), added_block, relabeled)
    result["observations"] = relabeled.reshape(rows, width * features_per_coin)

    result["mask"] = new_active.to(observations["mask"].dtype)
    if "basket_membership" in result:
        result["basket_membership"] = new_active.to(
            result["basket_membership"].dtype
        )
    if "prev_mask" in result:
        had_previous = (observations["prev_mask"] > 0.5).any(dim=1, keepdim=True)
        result["prev_mask"] = (new_active & had_previous).to(result["prev_mask"].dtype)

    previous = observations["prev_action"].float()
    kept_assets = previous[:, 1:] * new_active.float()
    kept_cash = (1.0 - kept_assets.sum(dim=1, keepdim=True)).clamp(min=0.0)
    result["prev_action"] = torch.cat((kept_cash, kept_assets), dim=1).to(
        observations["prev_action"].dtype
    )

    if "pool_history_valid" in result and "feature_history_mask" in result:
        time_valid = (observations["feature_history_mask"] > 0.5).unsqueeze(1)
        time_valid = time_valid.expand(-1, width, -1)
        stored_valid = result["pool_history_valid"]
        result["pool_history_valid"] = torch.where(
            added.unsqueeze(-1), time_valid.to(stored_valid.dtype), stored_valid
        )
    return result


def relabel_cross_basket_outcomes(
    relabeled_observations: Dict[str, torch.Tensor],
    next_observations: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    next_market: torch.Tensor,
    asset_returns: torch.Tensor,
    price_valid: torch.Tensor,
    next_partial_counts: torch.Tensor,
    name_rank: torch.Tensor,
    *,
    reward_scale: float,
    features_per_coin: int,
    primary_count: int,
    comparative_offset: int,
    comparative_count: int,
    cost_basis_index: int,
    cost_basis_scale: float,
    cost_basis_bounds: Tuple[float, float],
    min_position_weight: float,
    unit_return_targets: bool,
    component_target_clip: float,
) -> Tuple[
    Dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Exact STRUCTURED_CREDIT consequence of a relabeled allocation.

    The reward is ``scale * log(cash + sum_i w_i * (1 + r_i))``, the portfolio
    log growth PortfolioEnv replays (transaction cost is bookkeeping only in
    this reward mode).  Returns the next observation, the reward, component
    outcome targets and mask, and the executed target weights.
    """
    new_active = relabeled_observations["mask"] > 0.5
    rows, width = new_active.shape
    device = new_active.device

    partial = next_partial_counts >= 0
    order = torch.argsort(name_rank.to(device))

    def live_at_next_bar(active: torch.Tensor) -> torch.Tensor:
        active_by_name = active[:, order]
        position_by_name = torch.cumsum(active_by_name.long(), dim=1)
        live_by_name = active_by_name & (
            position_by_name <= next_partial_counts.clamp(min=0).unsqueeze(1)
        )
        live_partial = torch.zeros_like(active)
        live_partial[:, order] = live_by_name
        return torch.where(partial.unsqueeze(1), live_partial, active)

    next_live = live_at_next_bar(new_active)
    priced = next_live & price_valid

    weights = normalize_raw_actions_to_portfolio_weights(actions, new_active.float())
    weights64 = weights.double()
    realized = torch.where(priced, asset_returns.double(), torch.zeros_like(asset_returns.double()))
    gross = weights64.sum(dim=1) + (weights64[:, 1:] * realized).sum(dim=1)
    finite = torch.isfinite(gross) & (gross > 0.0)
    rewards = torch.where(
        finite,
        float(reward_scale) * torch.log(torch.where(finite, gross, torch.ones_like(gross))),
        torch.zeros_like(gross),
    ).float().unsqueeze(1)

    result = {key: value.clone() for key, value in next_observations.items()}
    old_next_active = next_observations["mask"] > 0.5
    added = new_active & ~old_next_active
    stored_snapshot = new_active & old_next_active & live_at_next_bar(old_next_active)
    market_snapshot = new_active & next_live & ~stored_snapshot
    next_snapshot = next_observations["observations"].reshape(rows, width, features_per_coin)
    market_block = _relabel_market_block(
        next_market.to(next_snapshot.dtype),
        features_per_coin,
        primary_count,
        comparative_offset,
        comparative_count,
    )
    relabeled = torch.where(
        stored_snapshot.unsqueeze(-1), next_snapshot, torch.zeros_like(next_snapshot)
    )
    relabeled = torch.where(market_snapshot.unsqueeze(-1), market_block, relabeled)

    # Weighted-average cost basis, following calculate_actual_shares_from_money_split.
    current_snapshot = relabeled_observations["observations"].reshape(
        rows, width, features_per_coin
    )
    cost_basis_now = current_snapshot[..., cost_basis_index].float()
    previous_weights = relabeled_observations["prev_action"][:, 1:].float()
    target_weights = weights[:, 1:]
    entry_over_price = 1.0 / (1.0 + cost_basis_now / float(cost_basis_scale)).clamp(min=1e-6)
    held = previous_weights > float(min_position_weight)
    buying = target_weights > previous_weights
    blended = (
        entry_over_price * previous_weights + (target_weights - previous_weights)
    ) / target_weights.clamp(min=1e-12)
    entry_after = torch.where(
        held,
        torch.where(buying, blended, entry_over_price),
        torch.ones_like(entry_over_price),
    )
    growth = 1.0 + torch.where(priced, asset_returns.float(), torch.zeros_like(asset_returns.float()))
    next_cost_basis = (
        (growth / entry_after.clamp(min=1e-6) - 1.0) * float(cost_basis_scale)
    ).clamp(float(cost_basis_bounds[0]), float(cost_basis_bounds[1]))
    holds_next = new_active & next_live & (target_weights > float(min_position_weight))
    relabeled[..., cost_basis_index] = torch.where(
        holds_next, next_cost_basis, torch.zeros_like(next_cost_basis)
    ).to(relabeled.dtype)
    result["observations"] = relabeled.reshape(rows, width * features_per_coin)

    if "local_feature_history" in result and bool(partial.any()):
        # At the truncated final bar PortfolioEnv writes a zero history frame
        # for basket assets without a row.  That next state is terminal, so it
        # never enters the Bellman target; only the zeroing is reproduced.
        unpriced = (new_active & ~next_live).unsqueeze(-1)
        history = result["local_feature_history"]
        history[:, :, -1, :] = torch.where(
            unpriced, torch.zeros_like(history[:, :, -1, :]), history[:, :, -1, :]
        )

    result["mask"] = new_active.to(next_observations["mask"].dtype)
    if "basket_membership" in result:
        result["basket_membership"] = new_active.to(result["basket_membership"].dtype)
    if "prev_mask" in result:
        result["prev_mask"] = new_active.to(result["prev_mask"].dtype)
    result["prev_action"] = weights.to(next_observations["prev_action"].dtype)
    if "pool_history_valid" in result and "feature_history_mask" in result:
        time_valid = (next_observations["feature_history_mask"] > 0.5).unsqueeze(1)
        time_valid = time_valid.expand(-1, width, -1)
        stored_valid = result["pool_history_valid"]
        result["pool_history_valid"] = torch.where(
            added.unsqueeze(-1), time_valid.to(stored_valid.dtype), stored_valid
        )

    unit_returns = (float(reward_scale) * asset_returns.float()).clamp(
        -float(component_target_clip), float(component_target_clip)
    )
    if unit_return_targets:
        component_values = unit_returns
    else:
        component_values = (
            float(reward_scale) * asset_returns.float() * target_weights
        ).clamp(-float(component_target_clip), float(component_target_clip))
    component_outcomes = torch.where(
        priced, component_values, torch.zeros_like(component_values)
    ).unsqueeze(-1)
    return result, rewards, component_outcomes, priced.unsqueeze(-1), weights


def _build_action_mask_from_observations(
    observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build the cash+asset action mask that matches the environment's active asset set.

    Cash is always active. Asset slots are active when the observation mask says they are.
    """
    if isinstance(observations, dict) and "mask" in observations:
        mask_current = observations["mask"].to(device=device, dtype=dtype)
        batch_size = mask_current.shape[0]
        action_mask = torch.zeros((batch_size, action_dim), device=device, dtype=dtype)
        action_mask[:, 0] = 1.0
        max_fill = min(action_dim - 1, mask_current.shape[1])
        if max_fill > 0:
            action_mask[:, 1:1 + max_fill] = mask_current[:, :max_fill]
        return action_mask

    if isinstance(observations, dict):
        first_tensor = next((value for value in observations.values() if torch.is_tensor(value)), None)
        if first_tensor is None:
            raise ValueError("Could not infer batch size from dict observations.")
        batch_size = first_tensor.shape[0]
    else:
        batch_size = observations.shape[0]

    return torch.ones((batch_size, action_dim), device=device, dtype=dtype)


def _safe_relative_asset_weights(weights: torch.Tensor, action_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert executed portfolio weights into relative risky-sleeve weights.

    When equity allocation is near zero, fall back to a uniform active-asset allocation
    so budget-only perturbations still have a well-defined risky sleeve template.
    """
    asset_mask = action_mask[:, 1:]
    asset_weights = weights[:, 1:] * asset_mask
    equity_allocation = (1.0 - weights[:, 0:1]).clamp(min=0.0)
    relative = torch.zeros_like(asset_weights)

    has_equity = equity_allocation.squeeze(-1) > eps
    if has_equity.any():
        relative[has_equity] = asset_weights[has_equity] / equity_allocation[has_equity].clamp(min=eps)

    no_equity = ~has_equity
    if no_equity.any():
        active_counts = asset_mask[no_equity].sum(dim=1, keepdim=True).clamp(min=1.0)
        relative[no_equity] = asset_mask[no_equity] / active_counts

    relative = relative * asset_mask
    relative_sum = relative.sum(dim=1, keepdim=True).clamp(min=eps)
    return (relative / relative_sum) * asset_mask


def _portfolio_weights_to_raw_actions(
    weights: torch.Tensor,
    action_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Canonical inverse of the environment's hierarchical normalization.

    Cash weight maps directly to the cash logit. Asset weights are converted into
    risky-sleeve relative shares and then mapped back into [-1, 1].
    """
    weights = weights.float()
    action_mask = action_mask.float()
    cash_weight = weights[:, 0:1].clamp(0.0, 1.0)
    asset_mask = action_mask[:, 1:]
    asset_weights = (weights[:, 1:] * asset_mask).clamp(min=0.0)

    raw_actions = weights.new_full(weights.shape, fill_value=-1.0)
    raw_actions[:, 0:1] = (2.0 * cash_weight) - 1.0

    equity_allocation = (1.0 - cash_weight).clamp(min=0.0)
    relative = torch.zeros_like(asset_weights)
    has_equity = equity_allocation.squeeze(-1) > eps
    if has_equity.any():
        relative[has_equity] = asset_weights[has_equity] / equity_allocation[has_equity].clamp(min=eps)

    relative = relative * asset_mask
    relative_sum = relative.sum(dim=1, keepdim=True)
    needs_renorm = relative_sum.squeeze(-1) > eps
    if needs_renorm.any():
        relative[needs_renorm] = relative[needs_renorm] / relative_sum[needs_renorm].clamp(min=eps)

    asset_logits = (2.0 * relative.clamp(0.0, 1.0)) - 1.0
    raw_actions[:, 1:] = torch.where(asset_mask > 0, asset_logits, raw_actions[:, 1:])
    return raw_actions.clamp(-1.0, 1.0)


class TaskConditionedPopArt(nn.Module):
    """Full PopArt target normalization keyed by the global-pool task mask.

    The production critics continue to emit raw financial Q values.  PopArt
    maps both their predictions and Bellman targets into a task-standardized
    coordinate system for optimization.  This is the raw-output
    parameterization of PopArt: changing running statistics cannot change a
    forward Q value, so the paper's output-preservation invariant is exact by
    construction rather than requiring a mutation of a task-specific final
    layer after every statistics update.

    Sparse basket statistics borrow first and second moments from the global
    target stream.  As a basket accumulates evidence, its reliability rises
    continuously and its own moments take over.  This retains the even playing
    field PopArt is meant to provide without treating one replay occurrence as
    a trustworthy estimate for a one-of-1,287 basket.
    """

    def __init__(
        self,
        task_width: int,
        *,
        beta: float = 3e-4,
        min_std: float = 1e-4,
        max_std: float = 1e6,
        prior_count: float = 64.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.task_width = int(task_width)
        if self.task_width <= 0 or self.task_width > 20:
            raise ValueError("PopArt task width must be in [1, 20]")
        self.num_tasks = 1 << self.task_width
        self.beta = float(beta)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.prior_count = float(prior_count)
        self.epsilon = float(epsilon)
        if not 0.0 < self.beta <= 1.0:
            raise ValueError("PopArt beta must be in (0, 1]")
        if not 0.0 < self.min_std <= self.max_std:
            raise ValueError("PopArt standard-deviation bounds are invalid")
        if self.prior_count < 0.0:
            raise ValueError("PopArt prior_count must be non-negative")

        # Float64 accumulators prevent loss of precision over multi-million-step
        # training while returned statistics are cast to the model dtype.
        self.register_buffer(
            "task_first_ema", torch.zeros(self.num_tasks, dtype=torch.float64)
        )
        self.register_buffer(
            "task_second_ema", torch.zeros(self.num_tasks, dtype=torch.float64)
        )
        self.register_buffer(
            "task_debias", torch.zeros(self.num_tasks, dtype=torch.float64)
        )
        self.register_buffer(
            "task_count", torch.zeros(self.num_tasks, dtype=torch.float64)
        )
        self.register_buffer("global_first_ema", torch.zeros((), dtype=torch.float64))
        self.register_buffer("global_second_ema", torch.zeros((), dtype=torch.float64))
        self.register_buffer("global_debias", torch.zeros((), dtype=torch.float64))
        self.register_buffer("global_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("statistics_updates", torch.zeros((), dtype=torch.long))

    def task_ids_from_observations(
        self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        if not isinstance(observations, dict):
            raise TypeError(
                "Task-conditioned PopArt requires dictionary observations with "
                "a global-pool availability mask."
            )
        membership = observations.get("basket_membership")
        if not torch.is_tensor(membership):
            membership = observations.get("mask")
        if not torch.is_tensor(membership):
            raise KeyError(
                "Task-conditioned PopArt could not find basket_membership or mask"
            )
        membership = membership.reshape(membership.shape[0], -1)
        if membership.shape[1] != self.task_width:
            raise ValueError(
                "PopArt task mask width mismatch: "
                f"received={membership.shape[1]}, expected={self.task_width}"
            )
        powers = torch.bitwise_left_shift(
            torch.ones(self.task_width, device=membership.device, dtype=torch.long),
            torch.arange(self.task_width, device=membership.device, dtype=torch.long),
        )
        return ((membership > 0.5).long() * powers).sum(dim=1)

    @torch.no_grad()
    def update(self, targets: torch.Tensor, task_ids: torch.Tensor) -> None:
        values = targets.detach().reshape(-1).to(
            device=self.global_first_ema.device, dtype=torch.float64
        )
        ids = task_ids.detach().reshape(-1).to(
            device=self.task_first_ema.device, dtype=torch.long
        )
        if values.shape[0] != ids.shape[0]:
            raise ValueError("PopArt targets and task IDs must have equal row counts")
        finite = torch.isfinite(values)
        if not bool(finite.any()):
            return
        values = values[finite]
        ids = ids[finite]
        if bool(((ids < 0) | (ids >= self.num_tasks)).any()):
            raise ValueError("PopArt task ID is outside the configured mask space")

        unique_ids, inverse, counts = torch.unique(
            ids, sorted=False, return_inverse=True, return_counts=True
        )
        grouped_first = torch.zeros(
            unique_ids.shape[0], device=values.device, dtype=torch.float64
        )
        grouped_second = torch.zeros_like(grouped_first)
        grouped_first.index_add_(0, inverse, values)
        grouped_second.index_add_(0, inverse, values.square())
        counts64 = counts.to(dtype=torch.float64)
        grouped_first = grouped_first / counts64
        grouped_second = grouped_second / counts64

        keep = 1.0 - self.beta
        self.task_first_ema[unique_ids] = (
            keep * self.task_first_ema[unique_ids]
            + self.beta * grouped_first
        )
        self.task_second_ema[unique_ids] = (
            keep * self.task_second_ema[unique_ids]
            + self.beta * grouped_second
        )
        self.task_debias[unique_ids] = (
            keep * self.task_debias[unique_ids] + self.beta
        )
        self.task_count[unique_ids] += counts64

        self.global_first_ema.mul_(keep).add_(self.beta * values.mean())
        self.global_second_ema.mul_(keep).add_(self.beta * values.square().mean())
        self.global_debias.mul_(keep).add_(self.beta)
        self.global_count.add_(float(values.numel()))
        self.statistics_updates.add_(1)

    def _global_moments(self) -> Tuple[torch.Tensor, torch.Tensor]:
        observed = self.global_debias > self.epsilon
        safe_debias = self.global_debias.clamp(min=self.epsilon)
        first = self.global_first_ema / safe_debias
        second = self.global_second_ema / safe_debias
        first = torch.where(observed, first, torch.zeros_like(first))
        second = torch.where(observed, second, torch.ones_like(second))
        return first, second

    def statistics(
        self,
        task_ids: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = task_ids.detach().reshape(-1).to(
            device=self.task_first_ema.device, dtype=torch.long
        )
        if bool(((ids < 0) | (ids >= self.num_tasks)).any()):
            raise ValueError("PopArt task ID is outside the configured mask space")
        global_first, global_second = self._global_moments()
        task_debias = self.task_debias[ids]
        safe_debias = task_debias.clamp(min=self.epsilon)
        task_first = self.task_first_ema[ids] / safe_debias
        task_second = self.task_second_ema[ids] / safe_debias
        observed = task_debias > self.epsilon
        task_first = torch.where(observed, task_first, global_first)
        task_second = torch.where(observed, task_second, global_second)

        counts = self.task_count[ids]
        if self.prior_count > 0.0:
            reliability = counts / (counts + self.prior_count)
        else:
            reliability = torch.ones_like(counts)
        reliability = torch.where(observed, reliability, torch.zeros_like(reliability))
        first = reliability * task_first + (1.0 - reliability) * global_first
        second = reliability * task_second + (1.0 - reliability) * global_second
        variance = (second - first.square()).clamp(min=self.min_std ** 2)
        sigma = variance.sqrt().clamp(min=self.min_std, max=self.max_std)

        output_dtype = dtype or torch.float32
        output_device = device or task_ids.device
        return (
            first.to(device=output_device, dtype=output_dtype),
            sigma.to(device=output_device, dtype=output_dtype),
            reliability.to(device=output_device, dtype=output_dtype),
        )

    def global_statistics(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        first, second = self._global_moments()
        sigma = (second - first.square()).clamp(
            min=self.min_std ** 2
        ).sqrt().clamp(min=self.min_std, max=self.max_std)
        output_device = device or self.global_first_ema.device
        return (
            first.to(device=output_device, dtype=dtype),
            sigma.to(device=output_device, dtype=dtype),
        )

    def normalize(
        self, values: torch.Tensor, task_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, sigma, reliability = self.statistics(
            task_ids, dtype=values.dtype, device=values.device
        )
        shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        mean_view = mean.view(shape)
        sigma_view = sigma.view(shape)
        return (values - mean_view) / sigma_view, mean, sigma, reliability

    def balanced_advantage(
        self, raw_advantage: torch.Tensor, task_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, task_sigma, reliability = self.statistics(
            task_ids,
            dtype=raw_advantage.dtype,
            device=raw_advantage.device,
        )
        _, global_sigma = self.global_statistics(
            dtype=raw_advantage.dtype, device=raw_advantage.device
        )
        scale = global_sigma / task_sigma.clamp(min=self.min_std)
        shape = (raw_advantage.shape[0],) + (1,) * (raw_advantage.ndim - 1)
        return raw_advantage * scale.view(shape), scale, reliability


def measure_task_gradient_directions(
    per_sample_losses: torch.Tensor,
    task_ids: torch.Tensor,
    parameters: Tuple[torch.nn.Parameter, ...],
    *,
    max_tasks: int,
    min_samples: int,
) -> Optional[Dict[str, float]]:
    """Measure cross-task gradient geometry without changing any gradient."""
    losses = per_sample_losses.reshape(-1)
    ids = task_ids.detach().reshape(-1).to(losses.device)
    if losses.shape[0] != ids.shape[0] or not parameters:
        return None
    unique_ids, counts = torch.unique(ids, return_counts=True)
    eligible = counts >= int(min_samples)
    unique_ids = unique_ids[eligible]
    counts = counts[eligible]
    if unique_ids.numel() < 2:
        return None
    selected_count = min(int(max_tasks), int(unique_ids.numel()))
    selected = unique_ids[torch.topk(counts, selected_count).indices]

    gradients = []
    norms = []
    for task_id in selected:
        task_loss = losses[ids == task_id].mean()
        task_gradients = torch.autograd.grad(
            task_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        detached = tuple(
            gradient.detach() if gradient is not None else None
            for gradient in task_gradients
        )
        squared_norm = sum(
            float(gradient.square().sum().item())
            for gradient in detached
            if gradient is not None
        )
        gradients.append(detached)
        norms.append(math.sqrt(max(squared_norm, 0.0)))

    cosines = []
    for left_index in range(len(gradients)):
        for right_index in range(left_index + 1, len(gradients)):
            dot = 0.0
            left_square = 0.0
            right_square = 0.0
            for left, right in zip(
                gradients[left_index], gradients[right_index]
            ):
                if left is None or right is None:
                    continue
                dot += float((left * right).sum().item())
                left_square += float(left.square().sum().item())
                right_square += float(right.square().sum().item())
            denominator = math.sqrt(left_square * right_square) + 1e-12
            cosines.append(dot / denominator)
    if not cosines:
        return None
    norm_array = np.asarray(norms, dtype=np.float64)
    cosine_array = np.asarray(cosines, dtype=np.float64)
    return {
        "task_count": float(len(gradients)),
        "pair_count": float(len(cosines)),
        "cosine_mean": float(cosine_array.mean()),
        "cosine_min": float(cosine_array.min()),
        "cosine_positive_fraction": float((cosine_array > 0.0).mean()),
        "gradient_norm_mean": float(norm_array.mean()),
        "gradient_norm_cv": float(
            norm_array.std() / max(norm_array.mean(), 1e-12)
        ),
    }


class CustomTD3_AC(TD3):
    """
    Custom TD3 Agent with Constrained Reinforcement Learning (CRL) using adaptive Lagrange multipliers.
    
    Instead of a fixed penalty coefficient, this implementation uses a learnable Lagrange multiplier λ
    that adapts based on constraint violations. This provides principled, adaptive smoothness control.
    
    Key CRL Features:
    - Adaptive λ (lambda) parameter that increases when policy is too erratic
    - Constraint threshold defining maximum allowed action change
    - Automatic penalty adjustment based on measured constraint violations
    """

    @classmethod
    def load(cls, path, env=None, **kwargs):
        """Reject legacy observation/checkpoint schemas with one clear error."""
        import json
        import zipfile
        from contextlib import nullcontext
        from stable_baselines3.common.save_util import open_path
        # Check before constructing networks or loading tensors. Equal shapes
        # do not imply equal semantics after removing basket normalization.
        stream = hasattr(path, "read")
        position = path.tell() if stream else None
        try:
            with (nullcontext(path) if stream else open_path(path, "r", suffix="zip")) as handle:
                with zipfile.ZipFile(handle) as archive:
                    saved = json.loads(archive.read("data")).get(
                        "market_input_contract", "selected_basket_v1"
                    )
        finally:
            if stream:
                path.seek(position)
        if saved != config.market_input_contract():
            raise ValueError(
                f"Market input contract mismatch: checkpoint={saved}, "
                f"configured={config.market_input_contract()}. Start a fresh run; "
                "the contract includes critic Bellman composition and portfolio-history "
                "conditioning; changing either requires a fresh run. "
                "Use EARNMORE_GLOBAL_POOL=0 for compact-slot checkpoints; "
                "also use LOCAL_CAUSAL_INPUTS=0 for historical selected-basket inputs."
            )
        try:
            model = super().load(path, env=env, **kwargs)
        except (KeyError, ValueError, RuntimeError) as exc:
            message = str(exc)
            if "feature_history" in message or "observation space" in message.lower():
                raise RuntimeError(
                    "Foundation schema v2 is a fresh-run cutover.  This "
                    "checkpoint or environment uses the retired single "
                    "feature_history contract; checkpoint and replay conversion "
                    "is intentionally unsupported."
                ) from exc
            if (
                "value_experience_replay_scorer" in message
                and "size mismatch" in message.lower()
            ):
                raise RuntimeError(
                    "Value-experience replay descriptor mismatch. The "
                    "checkpoint uses the retired raw slot-ordered action "
                    "suffix. Start a fresh run so NERS/ERO/MER use the "
                    "permutation-invariant joint asset-allocation descriptor."
                ) from exc
            raise
        expected = int(getattr(config, "FOUNDATION_SCHEMA_VERSION", 0))
        if int(getattr(model, "foundation_schema_version", -1)) != expected:
            raise RuntimeError(
                "Foundation checkpoint schema mismatch: expected "
                f"version {expected}, received "
                f"{getattr(model, 'foundation_schema_version', 'legacy')}. "
                "Start a fresh run; partial checkpoint/replay conversion is "
                "intentionally unsupported."
            )
        if bool(getattr(model, "qwm_enabled", False)):
            expected_qwm = int(getattr(config, "QWM_SCHEMA_VERSION", 1))
            if int(getattr(model, "qwm_schema_version", -1)) != expected_qwm:
                raise RuntimeError(
                    "QWM checkpoint schema mismatch: expected "
                    f"version {expected_qwm}, received "
                    f"{getattr(model, 'qwm_schema_version', 'legacy')}. "
                    "Start a fresh QWM run; partial world-model conversion is "
                    "intentionally unsupported."
                )
        expected_descriptor = int(
            getattr(config, "VALUE_EXPERIENCE_REPLAY_DESCRIPTOR_VERSION", 1)
        )
        if int(
            getattr(model, "value_experience_replay_descriptor_version", -1)
        ) != expected_descriptor:
            raise RuntimeError(
                "Value-experience replay descriptor mismatch: expected "
                f"version {expected_descriptor}, received "
                f"{getattr(model, 'value_experience_replay_descriptor_version', 'legacy')}. "
                "Start a fresh run; slot-ordered replay descriptors are not "
                "compatible with the identity-neutral NERS/ERO/MER geometry."
            )
        return model

    def _excluded_save_params(self) -> List[str]:
        """Exclude transient basket-teacher caches from SB3 model metadata.

        The curriculum manager owns the durable task distribution and
        republishes it before every learning block.  Inside TD3 that law is
        packed into byte membership keys solely for fast replay lookup.  SB3's
        JSON metadata helper cannot represent those keys and, more
        importantly, a best-model checkpoint must not preserve a stale copy of
        the teacher law.  Detached per-basket TD diagnostics are drained by the
        curriculum callback and likewise contain tuple keys that are runtime
        instrumentation rather than model state.
        """
        return super()._excluded_save_params() + [
            "_basket_replay_teacher_probabilities",
            "_basket_replay_task_probabilities",
            "_basket_replay_role_probabilities",
            "_basket_replay_anchor_key",
            "_basket_td_error_stats",
            "_last_basket_replay_sample_stats",
            "_basket_replay_metric_ema",
            "_value_experience_pending_summaries",
            "_value_experience_last_selected_features",
            "_value_experience_last_pre_td_error",
            "_value_experience_last_query_features",
            "_value_experience_last_baseline_features",
            "_value_experience_last_query_pre_td_error",
            "_value_experience_query_data",
            "_value_experience_last_selected_group_ids",
            "_value_experience_last_transfer_targets",
            "_value_experience_importance_weights",
            "_value_experience_pending_mer",
            "_last_value_experience_stats",
            "_value_experience_candidate_reservoir",
            "_foundation_market_sampler",
            # CUDA-resident telemetry is deliberately transient. Durable
            # checkpoints retain the human-readable scalar snapshots below,
            # while avoiding cloud-pickling a live device synchronization
            # accumulator.
            "_qwm_loss_ema_tensor",
            "_qwm_root_telemetry",
            "_qwm_successor_telemetry",
            "_qwm_last_root_family_ids",
        ]

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        actor_learning_rate: Union[float, Schedule] = 3e-4,
        critic_learning_rate: Union[float, Schedule] = 3e-4,
        # CRL-specific parameters for Velocity
        velocity_constraint_threshold: float = 0.05,
        velocity_lambda_lr: float = 5e-2,
        initial_velocity_lambda: float = 0.001,
        # CRL-specific parameters for Acceleration
        acceleration_constraint_threshold: float = 0.05,  # ✅ FIX: Increased from 0.01 to match shared-subspace cost scale
        acceleration_lambda_lr: float = 3e-3,
        initial_acceleration_lambda: float = 0.0001,
        # Existing parameters
        action_reg_coef: float = 0.1,  # Fallback for non-CRL mode
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        max_grad_norm: float = 1.0,
        overall_total_timesteps: Optional[int] = None,
        use_crl: bool = True,  # Enable/disable CRL mode
        auto_adjust_for_attention: bool = True,  # NEW: Auto-detect and adjust for attention models
        lambda_warmup_steps: int = 500,  # ✅ NEW: Number of steps for lambda warm-up
        lambda_decay_factor: float = 0.9999, # ✅ DEPRECATED: No longer used - replaced by bidirectional learning
        # ─── Sigma uncertainty gating ───────────────────────────────────────────
        use_sigma_gating: bool = True,       # Gate signal features by their sigma before critic/actor
        sigma_gate_scale: float = 0.2,       # Steepness of tanh gate: gate = 1 - tanh(|sigma| * scale)
        sigma_input_mode: Optional[str] = None,  # "dampen_input" (legacy Option A) | "passthrough" (Option B default)
        sigma_loss_focus_enabled: bool = False,  # Option B: upweight critic loss for high-sigma regimes
        sigma_loss_focus_alpha: float = 0.75,    # Focus strength α
        sigma_loss_focus_power: float = 1.0,     # Focus curvature p
        sigma_loss_focus_max_weight: float = 3.0,# Max per-sample weight w_max
        sigma_loss_focus_eps: float = 1e-6,      # Numerical epsilon
        # ─── N-step return blending ─────────────────────────────────────────────
        use_n_step_returns: bool = False,     # Accumulate n-step compound rewards before replay insertion
        n_step: int = 3,                     # How many steps to compound
        n_step_blend: float = 0.5,           # alpha: blended_r = alpha*n_step_r + (1-alpha)*1_step_r
        # ─── Critic-only combinatorial semi-bandit credit ─────────────────────
        component_outcome_aux_enabled: Optional[bool] = None,
        component_outcome_target_gradient_ratio: Optional[float] = None,
        component_outcome_gradient_calibration_interval: Optional[int] = None,
        actor_action_advantage_diagnostic_enabled: Optional[bool] = None,
        # ─── Proportional-exp local Boltzmann action distribution ──────────────
        proportional_exp_mode: str = "off",  # off | local_boltzmann | boltzmann_disagreement
        proportional_exp_candidates: int = 5,
        proportional_exp_alpha: float = 0.05,
        proportional_exp_alpha_max: float = 0.25,
        proportional_exp_disagreement_gain: float = 1.0,
        proportional_exp_ema_decay: float = 0.99,
        proportional_exp_cloud_space: str = "portfolio_weights",  # raw_actions | portfolio_weights
        proportional_exp_budget_candidate_ratio: float = 0.34,
        proportional_exp_target_budget_std: Optional[float] = None,
        proportional_exp_target_composition_std: Optional[float] = None,
        proportional_exp_actor_budget_std: Optional[float] = None,
        proportional_exp_actor_composition_std: Optional[float] = None,
        proportional_exp_budget_residual_weight: float = 0.10,
        proportional_exp_composition_residual_weight: float = 0.10,
        proportional_exp_q_temperature_scale: float = 1.0,
        proportional_exp_alpha_floor_ratio: float = 0.01,
        # ─── Q-Learning with a World Model decision-time search ───────────────
        qwm_enabled: bool = False,
        qwm_hidden_dim: Optional[int] = None,
        qwm_max_normalized_residual: Optional[float] = None,
        qwm_latent_dim: Optional[int] = None,
        qwm_kl_weight: Optional[float] = None,
        qwm_kl_free_nats: Optional[float] = None,
        qwm_reward_target_scale: Optional[float] = None,
        qwm_antithetic_samples: Optional[bool] = None,
        qwm_prior_reconstruction_weight: Optional[float] = None,
        qwm_learning_rate: Optional[float] = None,
        qwm_weight_decay: Optional[float] = None,
        qwm_batch_size: Optional[int] = None,
        qwm_pretrain_replay_rows: Optional[int] = None,
        qwm_pretrain_min_unique_baskets: Optional[int] = None,
        qwm_replay_baskets_per_batch: Optional[int] = None,
        qwm_pretrain_updates: Optional[int] = None,
        qwm_online_updates_per_td_step: Optional[int] = None,
        qwm_state_loss_weight: Optional[float] = None,
        qwm_reward_loss_weight: Optional[float] = None,
        qwm_max_grad_norm: Optional[float] = None,
        qwm_initial_depth: Optional[int] = None,
        qwm_max_depth: Optional[int] = None,
        qwm_depth_three_updates: Optional[int] = None,
        qwm_depth_four_updates: Optional[int] = None,
        qwm_candidates: Optional[int] = None,
        qwm_leaf_candidates: Optional[int] = None,
        qwm_model_samples: Optional[int] = None,
        qwm_beam_width: Optional[int] = None,
        qwm_search_discount: Optional[float] = None,
        qwm_leaf_aggregation: Optional[str] = None,
        qwm_successor_aggregation: Optional[str] = None,
        qwm_cuda_autocast: Optional[bool] = None,
        qwm_compile_regions: Optional[bool] = None,
        qwm_online_search_enabled: Optional[bool] = None,
        qwm_evaluation_search_enabled: Optional[bool] = None,
        qwm_crl_aware_scoring: Optional[bool] = None,
        qwm_search_seed: Optional[int] = None,
    ):
        requested_component_aux = bool(
            (
                getattr(config, "COMPONENT_OUTCOME_AUX_ENABLED", False)
                if _init_setup_model
                else False
            )
            if component_outcome_aux_enabled is None
            else component_outcome_aux_enabled
        )
        observation_space = getattr(env, "observation_space", None)
        policy_name = (
            policy.lower()
            if isinstance(policy, str)
            else getattr(policy, "__name__", str(policy)).lower()
        )
        component_aux_supported = (
            isinstance(observation_space, spaces.Dict)
            and "attention" in policy_name
        )
        resolved_component_aux = (
            requested_component_aux and component_aux_supported
        )
        requested_predictive_ssl = bool(
            getattr(config, "PREDICTIVE_SSL_ENABLED", False)
            if _init_setup_model
            else bool((policy_kwargs or {}).get("predictive_moe_enabled", False))
        )
        resolved_predictive_ssl = (
            requested_predictive_ssl and component_aux_supported
        )
        requested_foundation_sequence_replay = bool(
            getattr(config, "FOUNDATION_SEQUENCE_REPLAY_ENABLED", True)
            and getattr(config, "FOUNDATION_LEARNING_ENABLED", False)
            and getattr(config, "FOUNDATION_ONLINE_REHEARSAL_ENABLED", True)
            and requested_predictive_ssl
        )
        resolved_foundation_sequence_replay = bool(
            requested_foundation_sequence_replay and component_aux_supported
        )

        policy_kwargs = dict(policy_kwargs or {})
        replay_buffer_kwargs = dict(replay_buffer_kwargs or {})
        component_outcome_dim = len(
            tuple(getattr(config, "COMPONENT_OUTCOME_HORIZONS", (1,)))
        )
        if resolved_component_aux:
            policy_kwargs["component_outcome_aux_enabled"] = True
            policy_kwargs["component_outcome_dim"] = component_outcome_dim
            policy_kwargs.setdefault(
                "component_outcome_unit_return_targets",
                bool(
                    getattr(
                        config,
                        "COMPONENT_OUTCOME_UNIT_RETURN_TARGETS",
                        False,
                    )
                ),
            )
            policy_kwargs.setdefault(
                "component_forecast_bridge_enabled",
                bool(
                    getattr(
                        config,
                        "COMPONENT_FORECAST_BRIDGE_ENABLED",
                        False,
                    )
                ),
            )
            policy_kwargs.setdefault(
                "component_forecast_bridge_return_reference",
                float(
                    getattr(
                        config,
                        "COMPONENT_FORECAST_BRIDGE_RETURN_REFERENCE",
                        0.01,
                    )
                ),
            )
            policy_kwargs.setdefault(
                "component_forecast_bridge_max_abs",
                float(
                    getattr(
                        config,
                        "COMPONENT_FORECAST_BRIDGE_MAX_ABS",
                        0.10,
                    )
                ),
            )
            policy_kwargs.setdefault(
                "component_forecast_bridge_local_only",
                bool(
                    getattr(
                        config,
                        "COMPONENT_FORECAST_BRIDGE_LOCAL_ONLY",
                        False,
                    )
                ),
            )
        else:
            policy_kwargs["component_outcome_aux_enabled"] = False
            policy_kwargs["component_outcome_unit_return_targets"] = False
            policy_kwargs["component_forecast_bridge_enabled"] = False
            policy_kwargs["component_forecast_bridge_local_only"] = False

        # Chronological outcome storage is a data contract, not permission to
        # optimize the legacy component auxiliary.  Live foundation rehearsal
        # compounds these one-step labels into 1/5/15/30-step targets while the
        # old component head can remain completely disabled.
        if resolved_component_aux or resolved_foundation_sequence_replay:
            if replay_buffer_class is None:
                replay_buffer_class = ComponentOutcomeReplayBuffer
            elif not issubclass(replay_buffer_class, ComponentOutcomeReplayBuffer):
                purpose = (
                    "live foundation sequence rehearsal"
                    if resolved_foundation_sequence_replay
                    else "critic component-outcome supervision"
                )
                raise TypeError(
                    f"{purpose} requires ComponentOutcomeReplayBuffer or a subclass."
                )
            replay_buffer_kwargs.setdefault(
                "component_outcome_assets",
                int(observation_space["mask"].shape[0]),
            )
            replay_buffer_kwargs.setdefault(
                "component_outcome_dim", component_outcome_dim
            )
            replay_buffer_kwargs.setdefault(
                "component_outcome_target_kind",
                (
                    "scaled_unit_asset_return"
                    if bool(
                        getattr(
                            config,
                            "COMPONENT_OUTCOME_UNIT_RETURN_TARGETS",
                            False,
                        )
                    )
                    else "scaled_executed_contribution"
                ),
            )

        if component_aux_supported:
            policy_kwargs["predictive_moe_enabled"] = bool(
                resolved_predictive_ssl
                and getattr(config, "PREDICTIVE_MOE_ENABLED", False)
            )
            policy_kwargs["predictive_moe_num_experts"] = int(
                getattr(config, "PREDICTIVE_MOE_NUM_EXPERTS", 4)
            )
            policy_kwargs["predictive_moe_expert_hidden_dim"] = int(
                getattr(config, "PREDICTIVE_MOE_EXPERT_HIDDEN_DIM", 64)
            )
            policy_kwargs["predictive_moe_router_temperature"] = float(
                getattr(config, "PREDICTIVE_MOE_ROUTER_TEMPERATURE", 1.0)
            )
            policy_kwargs["predictive_moe_max_residual_abs"] = float(
                getattr(config, "PREDICTIVE_MOE_MAX_RESIDUAL_ABS", 0.10)
            )
        else:
            for predictive_key in (
                "predictive_moe_enabled",
                "predictive_moe_num_experts",
                "predictive_moe_expert_hidden_dim",
                "predictive_moe_router_temperature",
                "predictive_moe_max_residual_abs",
            ):
                policy_kwargs.pop(predictive_key, None)

        # We need to call the parent init, but we'll manage the learning rate ourselves.
        # Pass a dummy value for learning_rate to the parent.
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=0.0,  # Dummy value, will be overridden
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,  # We'll set up the model ourselves
        )

        self.max_grad_norm = max_grad_norm
        self.action_reg_coef = action_reg_coef  # Fallback for non-CRL mode
        self.overall_total_timesteps = overall_total_timesteps
        self.component_outcome_aux_enabled = resolved_component_aux
        self.foundation_sequence_replay_enabled = bool(
            resolved_foundation_sequence_replay
        )
        self.predictive_ssl_enabled = bool(resolved_predictive_ssl)
        self.predictive_ssl_online_enabled = bool(
            getattr(config, "PREDICTIVE_SSL_ONLINE_ENABLED", True)
        )
        self.predictive_ssl_critic_loss_weight = float(
            getattr(config, "PREDICTIVE_SSL_CRITIC_LOSS_WEIGHT", 0.01)
        )
        self.predictive_ssl_actor_loss_weight = float(
            getattr(config, "PREDICTIVE_SSL_ACTOR_LOSS_WEIGHT", 0.005)
        )
        self.predictive_ssl_gradient_guard_enabled = bool(
            getattr(config, "PREDICTIVE_SSL_GRADIENT_GUARD_ENABLED", True)
        )
        self.predictive_ssl_project_conflicts = bool(
            getattr(config, "PREDICTIVE_SSL_PROJECT_CONFLICTING_GRADIENTS", True)
        )
        self.predictive_ssl_gradient_epsilon = float(
            getattr(config, "PREDICTIVE_SSL_GRADIENT_EPS", 1e-12)
        )
        self.predictive_ssl_separate_replay_enabled = bool(
            getattr(config, "PREDICTIVE_SSL_SEPARATE_REPLAY_ENABLED", False)
        )
        self.predictive_ssl_horizon = max(
            1, int(getattr(config, "PREDICTIVE_SSL_HORIZON", 1))
        )
        self.predictive_ssl_sequence_batch_size = max(
            1,
            int(
                getattr(
                    config,
                    "PREDICTIVE_SSL_SEQUENCE_BATCH_SIZE",
                    batch_size,
                )
            ),
        )
        self.predictive_ssl_update_interval = max(
            1,
            int(getattr(config, "PREDICTIVE_SSL_UPDATE_INTERVAL", 1)),
        )
        self.predictive_ssl_horizon_discount = float(
            getattr(config, "PREDICTIVE_SSL_HORIZON_DISCOUNT", 1.0)
        )
        if not 0.0 < self.predictive_ssl_horizon_discount <= 1.0:
            raise ValueError("predictive horizon discount must be in (0, 1]")
        self.predictive_ssl_critic_shared_gradient_max_ratio = float(
            getattr(
                config,
                "PREDICTIVE_SSL_CRITIC_SHARED_GRADIENT_MAX_RATIO",
                0.25,
            )
        )
        self.predictive_ssl_actor_shared_gradient_max_ratio = float(
            getattr(
                config,
                "PREDICTIVE_SSL_ACTOR_SHARED_GRADIENT_MAX_RATIO",
                0.10,
            )
        )
        self.predictive_ssl_scale_shared_by_positive_cosine = bool(
            getattr(
                config,
                "PREDICTIVE_SSL_SCALE_SHARED_BY_POSITIVE_COSINE",
                True,
            )
        )
        self.predictive_ssl_gradient_cosine_ema_decay = float(
            getattr(
                config,
                "PREDICTIVE_SSL_GRADIENT_COSINE_EMA_DECAY",
                0.95,
            )
        )
        if not 0.0 <= self.predictive_ssl_gradient_cosine_ema_decay < 1.0:
            raise ValueError("predictive cosine EMA decay must be in [0, 1)")
        # These scalars are ordinary model state and therefore follow SB3
        # save/load.  Starting from zero requires prediction to establish
        # repeated same-batch agreement before it receives substantial shared
        # control authority.
        self._predictive_ssl_critic_cosine_ema = 0.0
        self._predictive_ssl_actor_cosine_ema = 0.0
        self.predictive_ssl_replay_baskets_per_batch = max(
            1,
            int(getattr(config, "PREDICTIVE_SSL_REPLAY_BASKETS_PER_BATCH", 8)),
        )
        self.predictive_ssl_replay_group_selection = str(
            getattr(config, "PREDICTIVE_SSL_REPLAY_GROUP_SELECTION", "uniform")
        ).strip().lower()
        self.predictive_ssl_replay_uniform_mix = float(
            np.clip(
                getattr(config, "PREDICTIVE_SSL_FOUNDATION_UNIFORM_MIX", 0.10),
                0.0,
                1.0,
            )
        )
        self.foundation_learning_enabled = bool(
            resolved_predictive_ssl
            and getattr(config, "FOUNDATION_LEARNING_ENABLED", False)
        )
        self.foundation_online_rehearsal_enabled = bool(
            getattr(config, "FOUNDATION_ONLINE_REHEARSAL_ENABLED", True)
        )
        self.foundation_schema_version = int(
            getattr(config, "FOUNDATION_SCHEMA_VERSION", 0)
        )
        self.market_input_contract = config.market_input_contract()
        self.training_objective_version = config.ATTENTION_TRAINING_OBJECTIVE_VERSION
        self.online_update_version = config.ATTENTION_ONLINE_UPDATE_VERSION
        self.foundation_horizons = tuple(
            int(value)
            for value in getattr(config, "FOUNDATION_HORIZONS", (1, 5, 15, 30))
        )
        self.foundation_max_horizon = max(self.foundation_horizons)
        self.foundation_online_update_interval = max(
            1,
            int(getattr(config, "FOUNDATION_ONLINE_UPDATE_INTERVAL", 2)),
        )
        self.foundation_online_batch_size = max(
            1,
            int(
                getattr(
                    config,
                    "FOUNDATION_ONLINE_BATCH_SIZE",
                    self.predictive_ssl_sequence_batch_size,
                )
            ),
        )
        self.foundation_warmup_min_updates = int(
            getattr(config, "FOUNDATION_WARMUP_MIN_UPDATES", 10_000)
        )
        self.foundation_warmup_max_updates = int(
            getattr(config, "FOUNDATION_WARMUP_MAX_UPDATES", 50_000)
        )
        self.foundation_probe_interval = max(
            1, int(getattr(config, "FOUNDATION_PROBE_INTERVAL", 500))
        )
        self.foundation_log_interval = max(
            1, int(getattr(config, "FOUNDATION_LOG_INTERVAL", 25))
        )
        self.foundation_prefetch_enabled = bool(
            getattr(config, "FOUNDATION_PREFETCH_ENABLED", True)
        )
        self.foundation_batch_reuse_updates = max(
            1,
            int(getattr(config, "FOUNDATION_BATCH_REUSE_UPDATES", 4)),
        )
        self.foundation_probe_patience = max(
            1, int(getattr(config, "FOUNDATION_PROBE_PATIENCE", 10))
        )
        self.foundation_min_relative_improvement = float(
            getattr(config, "FOUNDATION_MIN_RELATIVE_IMPROVEMENT", 0.005)
        )
        self.foundation_warmup_complete = False
        self.foundation_warmup_requested = False
        self.foundation_warmup_updates = 0
        self.foundation_best_probe = math.inf
        self.foundation_stale_probes = 0
        self._foundation_market_sampler = None
        self.foundation_sampler_rng_state = None
        self.foundation_log_dir = None
        self.earnmore_masked_reconstruction_enabled = bool(
            component_aux_supported
            and getattr(
                config, "EARNMORE_MASKED_RECONSTRUCTION_ENABLED", False
            )
        )
        self.earnmore_schema_version = int(
            getattr(config, "EARNMORE_SCHEMA_VERSION", 1)
        )
        self.earnmore_update_interval = max(
            1, int(getattr(config, "EARNMORE_UPDATE_INTERVAL", 1))
        )
        self.earnmore_max_grad_norm = float(
            getattr(config, "EARNMORE_MAX_GRAD_NORM", 1.0)
        )
        self.earnmore_updates = int(getattr(self, "earnmore_updates", 0))
        self.task_popart_enabled = bool(
            component_aux_supported
            and getattr(config, "TASK_POPART_ENABLED", False)
        )
        self.task_popart_beta = float(
            getattr(config, "TASK_POPART_BETA", 3e-4)
        )
        self.task_popart_min_std = float(
            getattr(config, "TASK_POPART_MIN_STD", 1e-4)
        )
        self.task_popart_max_std = float(
            getattr(config, "TASK_POPART_MAX_STD", 1e6)
        )
        self.task_popart_prior_count = float(
            getattr(config, "TASK_POPART_PRIOR_COUNT", 64.0)
        )
        self.task_popart_gradient_interval = max(
            1,
            int(
                getattr(
                    config,
                    "TASK_POPART_GRADIENT_DIAGNOSTICS_INTERVAL",
                    200,
                )
            ),
        )
        self.task_popart_gradient_max_tasks = max(
            2,
            int(
                getattr(
                    config,
                    "TASK_POPART_GRADIENT_DIAGNOSTICS_MAX_TASKS",
                    4,
                )
            ),
        )
        self.task_popart_gradient_min_samples = max(
            1,
            int(
                getattr(
                    config,
                    "TASK_POPART_GRADIENT_DIAGNOSTICS_MIN_SAMPLES",
                    2,
                )
            ),
        )
        self.task_popart: Optional[TaskConditionedPopArt] = None
        if self.verbose > 0 and self.earnmore_masked_reconstruction_enabled:
            print(
                "[EarnMore-TD3] critic-only whole-stock reconstruction enabled "
                f"| interval={self.earnmore_update_interval} "
                f"| mask_ratio=[{getattr(config, 'EARNMORE_MASK_RATIO_MIN', 0.60):.2f}, "
                f"{getattr(config, 'EARNMORE_MASK_RATIO_MAX', 0.80):.2f}] "
                "| source=online 8-of-13 replay"
            )
        if self.verbose > 0 and self.predictive_ssl_enabled:
            print(
                "[PredictiveMoE] Live actor/twin-critic representation "
                f"learning enabled | experts="
                f"{getattr(config, 'PREDICTIVE_MOE_NUM_EXPERTS', 4)} "
                f"| critic_weight={self.predictive_ssl_critic_loss_weight:.4f} "
                f"| actor_weight={self.predictive_ssl_actor_loss_weight:.4f} "
                f"| gradient_guard={self.predictive_ssl_gradient_guard_enabled} "
                f"| critic_shared_cap="
                f"{self.predictive_ssl_critic_shared_gradient_max_ratio:.3f} "
                f"| actor_shared_cap="
                f"{self.predictive_ssl_actor_shared_gradient_max_ratio:.3f} "
                f"| horizon={self.predictive_ssl_horizon} "
                f"| sequence_batch={self.predictive_ssl_sequence_batch_size} "
                f"| predictive_online={self.predictive_ssl_online_enabled} "
                f"| foundation_warmup={self.foundation_learning_enabled} "
                f"| foundation_online="
                f"{self.foundation_online_rehearsal_enabled} "
                f"| foundation_sequence_replay="
                f"{self.foundation_sequence_replay_enabled}"
            )
        self.actor_action_advantage_diagnostic_enabled = bool(
            getattr(
                config,
                "ACTOR_ACTION_ADVANTAGE_DIAGNOSTIC_ENABLED",
                False,
            )
            if actor_action_advantage_diagnostic_enabled is None
            else actor_action_advantage_diagnostic_enabled
        )
        self.crl_policy_step_diagnostic_interval = max(
            1, int(getattr(config, 'CRL_POLICY_STEP_DIAGNOSTIC_INTERVAL', 100))
        )
        self.component_outcome_target_gradient_ratio = float(
            getattr(config, "COMPONENT_OUTCOME_TARGET_GRADIENT_RATIO", 0.15)
            if component_outcome_target_gradient_ratio is None
            else component_outcome_target_gradient_ratio
        )
        self.component_outcome_gradient_calibration_interval = max(
            1,
            int(
                getattr(
                    config,
                    "COMPONENT_OUTCOME_GRADIENT_CALIBRATION_INTERVAL",
                    100,
                )
                if component_outcome_gradient_calibration_interval is None
                else component_outcome_gradient_calibration_interval
            ),
        )
        self.component_outcome_huber_beta = float(
            getattr(config, "COMPONENT_OUTCOME_HUBER_BETA", 1.0)
        )
        self.component_outcome_loss_return_reference = float(
            getattr(config, "COMPONENT_OUTCOME_LOSS_RETURN_REFERENCE", 0.01)
        )
        self.basket_relative_aux_weight = float(
            getattr(config, "BASKET_RELATIVE_AUX_WEIGHT", 0.0)
        )
        if self.component_outcome_loss_return_reference <= 0.0:
            raise ValueError("Component loss return reference must be positive")
        if self.basket_relative_aux_weight < 0.0:
            raise ValueError("Basket-relative auxiliary weight must be non-negative")
        self.component_outcome_gradient_scale_ema = float(
            getattr(config, "COMPONENT_OUTCOME_GRADIENT_SCALE_EMA", 0.90)
        )
        self.component_outcome_gradient_scale_min = float(
            getattr(config, "COMPONENT_OUTCOME_GRADIENT_SCALE_MIN", 1e-4)
        )
        self.component_outcome_gradient_scale_max = float(
            getattr(config, "COMPONENT_OUTCOME_GRADIENT_SCALE_MAX", 1.0)
        )
        self._component_outcome_gradient_scale: Optional[float] = None
        self._component_outcome_td_gradient_norm: Optional[float] = None
        self._component_outcome_aux_gradient_norm: Optional[float] = None
        # Detached instrumentation for the environment-level basket teacher.
        # It never participates in loss reduction, replay sampling, ProExp, or
        # gradient construction.
        self._basket_td_error_stats: Dict[Tuple[int, ...], List[float]] = {}
        if self.verbose > 0 and self.component_outcome_aux_enabled:
            print(
                "[ComponentOutcomeAux] Private causal financial forecast "
                f"enabled | channels={tuple(config.COMPONENT_OUTCOME_HORIZONS)} "
                f"| target_kind="
                f"{self.replay_buffer_kwargs.get('component_outcome_target_kind')} "
                f"| supervise_local_utilities="
                f"{bool(getattr(config, 'COMPONENT_OUTCOME_SUPERVISE_LOCAL_UTILITIES', True))} "
                f"| target_encoder_grad_ratio="
                f"{self.component_outcome_target_gradient_ratio:.3f} "
                f"| max_adaptive_scale="
                f"{self.component_outcome_gradient_scale_max:.3f} "
                f"| calibration_interval="
                f"{self.component_outcome_gradient_calibration_interval}"
            )
        if self.verbose > 0 and bool(
            getattr(config, "CRITIC_CENTERED_ADVANTAGE_ENABLED", False)
        ):
            print(
                "[CenteredCritic] current-portfolio allocation comparison enabled "
                f"| Bellman composition={config.CRITIC_BELLMAN_COMPOSITION} "
                "| direct_q derives its reference baseline from the same Q "
                "| supervision=Bellman TD "
                f"| foundation_learning={self.foundation_learning_enabled}"
            )
        if self.verbose > 0 and self.task_popart_enabled:
            print(
                "[TaskPopArt] full mask-conditioned Bellman and actor "
                "normalization enabled "
                f"| beta={self.task_popart_beta:g} "
                f"| sigma=[{self.task_popart_min_std:g}, "
                f"{self.task_popart_max_std:g}] "
                f"| global_prior_rows={self.task_popart_prior_count:g} "
                f"| gradient_diagnostic_interval="
                f"{self.task_popart_gradient_interval}"
            )

        # CRL-specific parameters
        self.use_crl = use_crl
        self.lambda_warmup_steps = lambda_warmup_steps
        self.lambda_decay_factor = lambda_decay_factor  # ✅ DEPRECATED: Kept for backward compatibility but no longer used
        self.auto_adjust_for_attention = auto_adjust_for_attention

        # Store thresholds
        self.velocity_constraint_threshold = velocity_constraint_threshold
        self.acceleration_constraint_threshold = acceleration_constraint_threshold

        # Store lambda learning rates
        self.velocity_lambda_lr = velocity_lambda_lr
        self.acceleration_lambda_lr = acceleration_lambda_lr

        # ✅ LAMBDA MOMENTUM: Add momentum parameter for stable lambda updates
        self.lambda_momentum = 0.9  # Momentum for lambda updates to prevent oscillations
        # ✅ NEW: Limit how large the acceleration lambda can grow relative to velocity lambda.
        # Couple it to a *reference* velocity lambda (historical peak) so that late-stage
        # decay of λ_velocity does not unintentionally "turn off" the acceleration penalty.
        # Also impose a global safety cap so λ_acceleration can never explode.
        # Provide more late-phase headroom before hitting the cap so acceleration
        # can keep enforcing refinement while still remaining bounded.
        self.max_accel_to_velocity_lambda_ratio = 10.0
        self.max_velocity_lambda_reference = float(initial_velocity_lambda)
        # ✅ NO GLOBAL CAP: The ratio-based cap is sufficient. Empirically, λ_acceleration
        # needs to reach 0.25-0.30 in late refinement, which the ratio naturally allows
        # (10× velocity peak of ~0.03 = 0.30). A hard global ceiling would cause saturation.
        
        # 🧠 ATTENTION-AWARE CRL: Detect attention but use same CRL parameters as SimpleMlpTD3
        self.is_attention_policy = self._detect_attention_policy(policy)
        # No multipliers: attention and MLP use identical thresholds/lambda LRs so curriculum engages the same way.
        self.attention_exploration_bonus = 0.0
        if self.verbose > 0 and self.is_attention_policy:
            print(f"🧠 ATTENTION MODEL DETECTED: Using same CRL parameters as MLP (no threshold/lambda scaling)")
            print(f"   Velocity constraint_threshold: {self.velocity_constraint_threshold:.4f}")
            print(f"   Acceleration constraint_threshold: {self.acceleration_constraint_threshold:.4f}")
        
        # Store the separate learning rate schedules
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        
        # Initialize Lagrange multipliers λ as learnable parameters
        if self.use_crl:
            self.lambda_param_velocity = torch.tensor(initial_velocity_lambda, device=self.device, requires_grad=False)
            self.lambda_param_acceleration = torch.tensor(initial_acceleration_lambda, device=self.device, requires_grad=False)

            # EMA trackers for violation signals to prevent overreaction to noise
            self._viol_ema_v = None
            self._viol_ema_a = None

            # Track constraint violations for monitoring
            self.velocity_constraint_history = []
            self.velocity_lambda_history = []
            self.acceleration_constraint_history = []
            self.acceleration_lambda_history = []
            
            if self.verbose > 0:
                print(f"🎯 Dual CRL Mode Enabled:")
                print(f"   Velocity: threshold={self.velocity_constraint_threshold:.4f}, initial_λ={initial_velocity_lambda:.4f}, λ_lr={self.velocity_lambda_lr:.4f}")
                print(f"   Acceleration: threshold={self.acceleration_constraint_threshold:.4f}, initial_λ={initial_acceleration_lambda:.4f}, λ_lr={self.acceleration_lambda_lr:.4f}")
        else:
            if self.verbose > 0:
                print(f"🚫 CRL Mode Disabled - Using fixed penalty: {action_reg_coef:.4f}")

        # ─── Sigma gating ───────────────────────────────────────────────────────
        self.use_sigma_gating = use_sigma_gating
        self.sigma_gate_scale = sigma_gate_scale
        legacy_mode = "dampen_input" if bool(use_sigma_gating) else "passthrough"
        requested_mode = legacy_mode if sigma_input_mode is None else str(sigma_input_mode).strip().lower()
        if requested_mode not in {"dampen_input", "passthrough"}:
            if self.verbose > 0:
                print(f"[SigmaMode] Unknown sigma_input_mode='{sigma_input_mode}', falling back to '{legacy_mode}'.")
            requested_mode = legacy_mode
        self.sigma_input_mode = requested_mode
        if self.sigma_input_mode == "dampen_input":
            # Keep compatibility for legacy behavior regardless of incoming use_sigma_gating.
            self.use_sigma_gating = True
        self.sigma_loss_focus_enabled = bool(sigma_loss_focus_enabled)
        self.sigma_loss_focus_alpha = float(sigma_loss_focus_alpha)
        self.sigma_loss_focus_power = float(sigma_loss_focus_power)
        self.sigma_loss_focus_max_weight = float(sigma_loss_focus_max_weight)
        self.sigma_loss_focus_eps = float(sigma_loss_focus_eps)
        self._sigma_signal_idx: Optional[torch.Tensor] = None
        self._sigma_gate_idx: Optional[torch.Tensor] = None
        self._sigma_scale_factors: Optional[torch.Tensor] = None
        self._sigma_unique_idx: Optional[torch.Tensor] = None
        if self.verbose > 0:
            print(f"[SigmaMode] Input mode: {self.sigma_input_mode} | Loss focus: {self.sigma_loss_focus_enabled}")

        # ─── Rank-2 basket amortization diagnostics ─────────────────────────────
        self.basket_context_diagnostics_enabled = bool(
            getattr(config, "BASKET_CONTEXT_DIAGNOSTICS_ENABLED", False)
            and getattr(config, "BASKET_CONTEXT_ENABLED", False)
        )
        self.basket_context_diagnostics_interval = max(
            1,
            int(getattr(config, "BASKET_CONTEXT_DIAGNOSTICS_INTERVAL", 8)),
        )
        self.basket_context_diagnostics_sensitivity_interval = max(
            1,
            int(getattr(config, "BASKET_CONTEXT_DIAGNOSTICS_SENSITIVITY_INTERVAL", 50)),
        )
        self.basket_context_diagnostics_sensitivity_delta = float(
            getattr(config, "BASKET_CONTEXT_DIAGNOSTICS_SENSITIVITY_DELTA", 0.75)
        )
        if self.verbose > 0 and self.basket_context_diagnostics_enabled:
            print(
                "[BasketDiag] Enabled: batch g(B) geometry + FiLM/proj norms "
                f"every {self.basket_context_diagnostics_interval} updates; "
                f"g-swap sensitivity every "
                f"{self.basket_context_diagnostics_sensitivity_interval} updates "
                f"(delta={self.basket_context_diagnostics_sensitivity_delta})"
            )

        # Membership remains private replay metadata even though Path 2 removes
        # it from actor/critic inputs.  Replay stratification must consequently
        # not depend on BASKET_MEMBERSHIP_ENABLED, which now describes only the
        # neural conditioner.
        self.basket_replay_stratify_enabled = bool(
            getattr(config, "BASKET_REPLAY_STRATIFY_ENABLED", False)
        )
        self.curriculum_mixed_replay_enabled = bool(
            getattr(config, "CURRICULUM_MIXED_REPLAY_ENABLED", False)
        )
        self.curriculum_mixed_replay_min_unique_transitions_per_role = max(
            1,
            int(
                getattr(
                    config,
                    "CURRICULUM_MIXED_REPLAY_MIN_UNIQUE_TRANSITIONS_PER_ROLE",
                    1,
                )
            ),
        )
        self.curriculum_mixed_replay_target_distinct_tasks_per_role = max(
            1,
            int(
                getattr(
                    config,
                    "CURRICULUM_MIXED_REPLAY_TARGET_DISTINCT_TASKS_PER_ROLE",
                    1,
                )
            ),
        )
        self.curriculum_mixed_replay_max_basket_fraction = float(
            np.clip(
                getattr(config, "CURRICULUM_MIXED_REPLAY_MAX_BASKET_FRACTION", 1.0),
                0.0,
                1.0,
            )
        )
        self.basket_replay_group_selection = str(
            getattr(config, "BASKET_REPLAY_GROUP_SELECTION", "uniform")
        ).strip().lower()
        if self.basket_replay_group_selection not in {
            "uniform",
            "occupancy",
            "teacher",
            "teacher_uniform",
            "curriculum_role_mixed",
        }:
            raise ValueError(
                "BASKET_REPLAY_GROUP_SELECTION must be 'uniform', "
                "'occupancy', 'teacher', 'teacher_uniform', or "
                "'curriculum_role_mixed'"
            )
        if self.predictive_ssl_replay_group_selection not in {
            "uniform",
            "occupancy",
            "teacher_uniform",
        }:
            raise ValueError(
                "PREDICTIVE_SSL_REPLAY_GROUP_SELECTION must be 'uniform', "
                "'occupancy', or 'teacher_uniform'"
            )
        self.basket_replay_sample_without_replacement = bool(
            getattr(
                config,
                "BASKET_REPLAY_SAMPLE_WITHOUT_REPLACEMENT",
                True,
            )
        )
        self.basket_replay_min_completed_episodes = max(
            0,
            int(getattr(config, "BASKET_REPLAY_MIN_COMPLETED_EPISODES", 1)),
        )
        self.basket_replay_allow_partial_unique_batch = bool(
            getattr(config, "BASKET_REPLAY_ALLOW_PARTIAL_UNIQUE_BATCH", True)
        )
        self.basket_replay_diagnostic_ema_alpha = float(
            np.clip(
                getattr(config, "BASKET_REPLAY_DIAGNOSTIC_EMA_ALPHA", 0.05),
                0.0,
                1.0,
            )
        )
        replay_seed = 0 if getattr(self, "seed", None) is None else int(self.seed)
        self._basket_replay_rng = np.random.default_rng(replay_seed)
        self._predictive_replay_rng = np.random.default_rng(replay_seed + 65_537)
        self._basket_replay_teacher_probabilities: Dict[bytes, float] = {}
        self._basket_replay_task_probabilities: Dict[bytes, float] = {}
        self._basket_replay_role_probabilities: Dict[
            str, Dict[bytes, float]
        ] = {}
        self._basket_replay_teacher_anchor_share = 0.0
        self._basket_replay_financial_anchor_share = 0.0
        self._last_basket_replay_sample_stats: Dict[str, Any] = {}
        self._basket_replay_metric_ema: Dict[str, float] = {}
        anchor_symbols = tuple(
            getattr(config, "BASKET_CURRICULUM_ANCHOR", ())
        )
        ticker_to_id = getattr(config, "TICKER_TO_ID", {})
        self._basket_replay_anchor_membership_ids = tuple(
            sorted(
                int(ticker_to_id[symbol])
                for symbol in anchor_symbols
                if symbol in ticker_to_id
            )
        )
        anchor_membership_dim = int(
            getattr(config, "BASKET_MEMBERSHIP_DIM", 0)
        )
        anchor_membership = np.zeros(
            anchor_membership_dim, dtype=np.float32
        )
        if self._basket_replay_anchor_membership_ids:
            anchor_membership[
                np.asarray(
                    self._basket_replay_anchor_membership_ids,
                    dtype=np.int64,
                )
            ] = 1.0
        self._basket_replay_anchor_key = _membership_key(anchor_membership)
        # Value-equivalent experience replay is independent of exact basket
        # provenance.  The scorer is constructed after the TD3 policy so its
        # input width can be derived from the live FAM/AAM critic.
        self.value_experience_replay_enabled = bool(
            getattr(config, "VALUE_EXPERIENCE_REPLAY_ENABLED", False)
        ) and bool(component_aux_supported)
        self.value_experience_replay_descriptor_version = int(
            getattr(config, "VALUE_EXPERIENCE_REPLAY_DESCRIPTOR_VERSION", 1)
        )
        self.value_experience_candidate_multiplier = max(
            1.0,
            float(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_CANDIDATE_MULTIPLIER",
                    2.0,
                )
            ),
        )
        self.value_experience_query_size = max(
            1,
            int(getattr(config, "VALUE_EXPERIENCE_REPLAY_QUERY_SIZE", 64)),
        )
        self.value_experience_candidate_refresh_interval = max(
            1,
            int(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_CANDIDATE_REFRESH_INTERVAL",
                    1,
                )
            ),
        )
        self.value_experience_feedback_interval = max(
            1,
            int(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_FEEDBACK_INTERVAL",
                    1,
                )
            ),
        )
        self.value_experience_query_hard_fraction = float(np.clip(
            getattr(
                config, "VALUE_EXPERIENCE_REPLAY_QUERY_HARD_FRACTION", 0.50
            ),
            0.0,
            1.0,
        ))
        self.value_experience_uniform_fraction = float(
            np.clip(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_UNIFORM_FRACTION",
                    0.25,
                ),
                0.0,
                1.0,
            )
        )
        self.value_experience_temperature = float(
            getattr(config, "VALUE_EXPERIENCE_REPLAY_TEMPERATURE", 0.75)
        )
        self.value_experience_importance_beta = float(
            np.clip(
                getattr(config, "VALUE_EXPERIENCE_REPLAY_IMPORTANCE_BETA", 0.4),
                0.0,
                1.0,
            )
        )
        self.value_experience_importance_min = float(
            getattr(config, "VALUE_EXPERIENCE_REPLAY_IMPORTANCE_MIN", 0.25)
        )
        self.value_experience_importance_max = float(
            getattr(config, "VALUE_EXPERIENCE_REPLAY_IMPORTANCE_MAX", 4.0)
        )
        self.value_experience_query_utility_fraction = float(np.clip(
            getattr(
                config, "VALUE_EXPERIENCE_REPLAY_QUERY_UTILITY_FRACTION", 0.60
            ),
            0.0,
            1.0,
        ))
        self.value_experience_action_contrast_fraction = float(np.clip(
            getattr(
                config, "VALUE_EXPERIENCE_REPLAY_ACTION_CONTRAST_FRACTION", 0.15
            ),
            0.0,
            1.0,
        ))
        self.value_experience_action_contrast_weight = max(
            0.0,
            float(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_ACTION_CONTRAST_WEIGHT",
                    0.75,
                )
            ),
        )
        self.value_experience_transfer_enabled = bool(
            getattr(config, "VALUE_EXPERIENCE_REPLAY_TRANSFER_ENABLED", True)
        )
        self.value_experience_transfer_interval = max(
            1,
            int(
                getattr(config, "VALUE_EXPERIENCE_REPLAY_TRANSFER_INTERVAL", 16)
            ),
        )
        self.value_experience_transfer_groups = max(
            1,
            int(getattr(config, "VALUE_EXPERIENCE_REPLAY_TRANSFER_GROUPS", 4)),
        )
        self.value_experience_transfer_probe_size = max(
            1,
            int(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_TRANSFER_PROBE_SIZE",
                    64,
                )
            ),
        )
        self.value_experience_transfer_score_weight = max(
            0.0,
            float(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_TRANSFER_SCORE_WEIGHT",
                    0.35,
                )
            ),
        )
        self.value_experience_transfer_feedback_weight = max(
            0.0,
            float(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_TRANSFER_FEEDBACK_WEIGHT",
                    0.25,
                )
            ),
        )
        self.value_experience_mer_enabled = bool(
            getattr(config, "VALUE_EXPERIENCE_REPLAY_MER_ENABLED", False)
        ) and self.value_experience_transfer_enabled
        self.value_experience_mer_min_transfer_cosine = float(
            getattr(
                config,
                "VALUE_EXPERIENCE_REPLAY_MER_MIN_TRANSFER_COSINE",
                0.10,
            )
        )
        self.value_experience_mer_meta_step_size = max(
            0.0,
            float(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_MER_META_STEP_SIZE",
                    0.10,
                )
            ),
        )
        self.value_experience_mer_update_interval = max(
            self.value_experience_transfer_interval,
            int(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_MER_UPDATE_INTERVAL",
                    64,
                )
            ),
        )
        self.value_experience_mer_max_main_gradient_ratio = max(
            0.0,
            float(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_MER_MAX_MAIN_GRADIENT_RATIO",
                    0.05,
                )
            ),
        )
        self.value_experience_mer_max_adam_step_ratio = max(
            0.0,
            float(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_MER_MAX_ADAM_STEP_RATIO",
                    0.01,
                )
            ),
        )
        self.value_experience_mer_max_relative_parameter_step = max(
            0.0,
            float(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_MER_MAX_RELATIVE_PARAMETER_STEP",
                    1e-5,
                )
            ),
        )
        self.value_experience_mer_require_positive_query_improvement = bool(
            getattr(
                config,
                "VALUE_EXPERIENCE_REPLAY_MER_REQUIRE_POSITIVE_QUERY_IMPROVEMENT",
                True,
            )
        )
        self.value_experience_mer_min_query_improvement_ema = float(
            getattr(
                config,
                "VALUE_EXPERIENCE_REPLAY_MER_MIN_QUERY_IMPROVEMENT_EMA",
                0.0,
            )
        )
        self.value_experience_mer_query_improvement_ema_decay = float(
            np.clip(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_MER_QUERY_IMPROVEMENT_EMA_DECAY",
                    0.95,
                ),
                0.0,
                0.9999,
            )
        )
        self.value_experience_mer_acceleration_window = max(
            1,
            int(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_MER_ACCELERATION_WINDOW",
                    12,
                )
            ),
        )
        self.value_experience_mer_acceleration_center_tolerance = max(
            0.0,
            float(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_MER_ACCELERATION_CENTER_TOLERANCE_FRACTION",
                    0.20,
                )
            ),
        )
        self.value_experience_mer_acceleration_quantile = float(
            np.clip(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_MER_ACCELERATION_QUANTILE",
                    0.80,
                ),
                0.0,
                1.0,
            )
        )
        self.value_experience_mer_acceleration_quantile_tolerance = max(
            0.0,
            float(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_MER_ACCELERATION_QUANTILE_TOLERANCE_FRACTION",
                    0.35,
                )
            ),
        )
        self.value_experience_mer_acceleration_max_exceedance_fraction = float(
            np.clip(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_MER_ACCELERATION_MAX_EXCEEDANCE_FRACTION",
                    0.25,
                ),
                0.0,
                1.0,
            )
        )
        self._value_experience_rng = np.random.default_rng(replay_seed + 131_071)
        self.value_experience_replay_scorer = None
        self.value_experience_replay_optimizer = None
        self.value_experience_selection_interval = max(
            1,
            int(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_SELECTION_INTERVAL",
                    16,
                )
            ),
        )
        self._value_experience_last_selected_features = None
        self._value_experience_last_pre_td_error = None
        self._value_experience_last_query_features = None
        self._value_experience_last_baseline_features = None
        self._value_experience_last_query_pre_td_error = None
        self._value_experience_query_data = None
        self._value_experience_last_selected_group_ids = None
        self._value_experience_last_transfer_targets = None
        self._value_experience_importance_weights = None
        self._value_experience_pending_mer = None
        self._value_experience_candidate_reservoir = None
        self._value_experience_selection_count = 0
        self._value_experience_reservoir_refresh_count = 0
        self._value_experience_feedback_count = 0
        self._value_experience_selection_seconds_total = 0.0
        self._value_experience_reservoir_refresh_seconds_total = 0.0
        self._value_experience_feedback_seconds_total = 0.0
        self._value_experience_transfer_seconds_total = 0.0
        self._td_update_seconds_ema: Optional[float] = None
        self._predictive_ssl_update_seconds_ema: Optional[float] = None
        self._value_experience_transfer_measurement_count = 0
        self._value_experience_query_improvement_ema: Optional[float] = None
        self._value_experience_mer_attempt_count = 0
        self._value_experience_mer_accept_count = 0
        self._value_experience_mer_applied_gradient_norm_total = 0.0
        self._value_experience_pending_summaries = collections.deque(
            maxlen=int(
                getattr(
                    config,
                    "VALUE_EXPERIENCE_REPLAY_MAX_PENDING_SUMMARIES",
                    256,
                )
            )
        )
        self._value_experience_last_eval_financial = None
        self._value_experience_last_eval_normalized = None
        self._value_experience_last_eval_by_stratum: Dict[
            str, Tuple[float, float]
        ] = {}
        self._value_experience_feedback_baseline = 0.0
        self._last_value_experience_stats: Dict[str, float] = {}
        if self.verbose > 0 and self.basket_replay_stratify_enabled:
            print(
                "[BasketReplay] Curriculum-role mixed financial sampling enabled "
                f"(mixed={self.curriculum_mixed_replay_enabled}, "
                f"group_selection={self.basket_replay_group_selection}, "
                f"min_unique_rows_per_role="
                f"{self.curriculum_mixed_replay_min_unique_transitions_per_role}, "
                f"target_distinct_tasks_per_role="
                f"{self.curriculum_mixed_replay_target_distinct_tasks_per_role}, "
                f"max_basket_fraction={self.curriculum_mixed_replay_max_basket_fraction:.3f}, "
                f"without_replacement="
                f"{self.basket_replay_sample_without_replacement}, "
                f"min_completed_episodes="
                f"{self.basket_replay_min_completed_episodes}, "
                f"partial_unique_batch="
                f"{self.basket_replay_allow_partial_unique_batch}, "
                f"seed={replay_seed})"
            )

        # ─── N-step return blending ─────────────────────────────────────────────
        self.use_n_step_returns = use_n_step_returns
        self.n_step = n_step
        self.n_step_blend = n_step_blend
        self._nstep_deque: collections.deque = collections.deque()
        self._cached_action_low: Optional[torch.Tensor] = None
        self._cached_action_high: Optional[torch.Tensor] = None

        # ─── Proportional-exp sampled action distribution ─────────────────────
        proportional_exp_mode = str(proportional_exp_mode or "off").strip().lower()
        valid_proportional_exp_modes = {"off", "local_boltzmann", "boltzmann_disagreement"}
        if proportional_exp_mode not in valid_proportional_exp_modes:
            raise ValueError(
                f"Unknown proportional_exp_mode='{proportional_exp_mode}'. "
                f"Expected one of {sorted(valid_proportional_exp_modes)}."
            )
        self.proportional_exp_mode = proportional_exp_mode
        self.proportional_exp_candidates = max(1, int(proportional_exp_candidates))
        self.proportional_exp_alpha = max(1e-6, float(proportional_exp_alpha))
        self.proportional_exp_alpha_max = max(self.proportional_exp_alpha, float(proportional_exp_alpha_max))
        self.proportional_exp_disagreement_gain = max(0.0, float(proportional_exp_disagreement_gain))
        self.proportional_exp_ema_decay = float(np.clip(proportional_exp_ema_decay, 0.0, 0.99999))
        self.proportional_exp_eps = 1e-6
        cloud_space = str(proportional_exp_cloud_space or "portfolio_weights").strip().lower()
        valid_cloud_spaces = {"raw_actions", "portfolio_weights"}
        if cloud_space not in valid_cloud_spaces:
            raise ValueError(
                f"Unknown proportional_exp_cloud_space='{proportional_exp_cloud_space}'. "
                f"Expected one of {sorted(valid_cloud_spaces)}."
            )
        self.proportional_exp_cloud_space = cloud_space
        self.proportional_exp_budget_candidate_ratio = float(np.clip(proportional_exp_budget_candidate_ratio, 0.0, 1.0))

        # The actor's proposed portfolio must remain the value anchor. Nearby
        # budget and composition families are robustness corrections, not a
        # candidate-count vote that can reduce the center action to 1/K of the
        # objective. Keep the residual weights explicit and bounded so the
        # center coefficient can never become negative.
        self.proportional_exp_budget_residual_weight = float(
            proportional_exp_budget_residual_weight
        )
        self.proportional_exp_composition_residual_weight = float(
            proportional_exp_composition_residual_weight
        )
        if (
            self.proportional_exp_budget_residual_weight < 0.0
            or self.proportional_exp_composition_residual_weight < 0.0
            or (
                self.proportional_exp_budget_residual_weight
                + self.proportional_exp_composition_residual_weight
            ) > 1.0
        ):
            raise ValueError(
                "ProExp residual weights must be non-negative and sum to at most 1."
            )

        # Boltzmann temperature must live on the same numerical scale as the
        # candidate Q differences. The historical absolute alpha=0.05 was much
        # larger than the observed milliscale Q spread and made the candidate
        # weights almost uniform. Retain alpha as an absolute safety ceiling,
        # but derive the operative temperature from detached within-family Q
        # dispersion. Detachment prevents the temperature statistic from
        # becoming a second optimization route through the critic.
        self.proportional_exp_q_temperature_scale = float(
            proportional_exp_q_temperature_scale
        )
        self.proportional_exp_alpha_floor_ratio = float(
            proportional_exp_alpha_floor_ratio
        )
        if self.proportional_exp_q_temperature_scale <= 0.0:
            raise ValueError("proportional_exp_q_temperature_scale must be positive")
        if not 0.0 < self.proportional_exp_alpha_floor_ratio <= 1.0:
            raise ValueError(
                "proportional_exp_alpha_floor_ratio must be in (0, 1]"
            )

        base_std = float(max(self.target_policy_noise, 1e-4))
        self.proportional_exp_target_budget_std = float(
            base_std * 0.40 if proportional_exp_target_budget_std is None else proportional_exp_target_budget_std
        )
        self.proportional_exp_target_composition_std = float(
            base_std * 0.25 if proportional_exp_target_composition_std is None else proportional_exp_target_composition_std
        )
        self.proportional_exp_actor_budget_std = float(
            base_std * 0.25 if proportional_exp_actor_budget_std is None else proportional_exp_actor_budget_std
        )
        self.proportional_exp_actor_composition_std = float(
            base_std * 0.20 if proportional_exp_actor_composition_std is None else proportional_exp_actor_composition_std
        )

        self._proportional_exp_budget_disagreement_ema: Optional[torch.Tensor] = None
        self._proportional_exp_composition_disagreement_ema: Optional[torch.Tensor] = None
        self.use_proportional_exp = self.proportional_exp_mode != "off"
        if self.verbose > 0 and self.use_proportional_exp:
            print(
                f"[ProExp] Mode={self.proportional_exp_mode} | K={self.proportional_exp_candidates} | "
                f"alpha={self.proportional_exp_alpha:.4f} | alpha_max={self.proportional_exp_alpha_max:.4f} | "
                f"gain={self.proportional_exp_disagreement_gain:.4f} | ema_decay={self.proportional_exp_ema_decay:.5f} | "
                f"space={self.proportional_exp_cloud_space} | budget_ratio={self.proportional_exp_budget_candidate_ratio:.2f} | "
                f"center_residuals=({self.proportional_exp_budget_residual_weight:.2f}, "
                f"{self.proportional_exp_composition_residual_weight:.2f}) | "
                f"q_temp_scale={self.proportional_exp_q_temperature_scale:.2f}"
            )

        self.qwm_enabled = bool(qwm_enabled)
        if (
            self.earnmore_masked_reconstruction_enabled
            and bool(getattr(config, "FOUNDATION_LEARNING_ENABLED", False))
        ):
            raise ValueError(
                "EarnMore-TD3 is the foundation-free representation experiment; "
                "Foundation learning must be disabled."
            )
        if self.qwm_enabled and self.earnmore_masked_reconstruction_enabled:
            raise ValueError(
                "The EarnMore-TD3 experiment requires QWM to be disabled so "
                "masked representation learning is the only new mechanism."
            )
        if self.qwm_enabled:
            if not component_aux_supported:
                raise ValueError(
                    "QWM requires the variable-coin AttentionTD3Policy Dict contract."
                )
            if not self.use_proportional_exp:
                raise ValueError(
                    "QWM requires ProExp candidates; select local_boltzmann or "
                    "boltzmann_disagreement."
                )
        self.qwm_hidden_dim = int(
            getattr(config, "QWM_HIDDEN_DIM", 256)
            if qwm_hidden_dim is None else qwm_hidden_dim
        )
        self.qwm_max_normalized_residual = float(
            getattr(config, "QWM_MAX_NORMALIZED_RESIDUAL", 0.50)
            if qwm_max_normalized_residual is None
            else qwm_max_normalized_residual
        )
        self.qwm_latent_dim = max(1, int(
            getattr(config, "QWM_LATENT_DIM", 32)
            if qwm_latent_dim is None else qwm_latent_dim
        ))
        self.qwm_kl_weight = max(0.0, float(
            getattr(config, "QWM_KL_WEIGHT", 1e-3)
            if qwm_kl_weight is None else qwm_kl_weight
        ))
        self.qwm_kl_free_nats = max(0.0, float(
            getattr(config, "QWM_KL_FREE_NATS", 0.02)
            if qwm_kl_free_nats is None else qwm_kl_free_nats
        ))
        self.qwm_reward_target_scale = float(
            getattr(config, "QWM_REWARD_TARGET_SCALE", 100.0)
            if qwm_reward_target_scale is None else qwm_reward_target_scale
        )
        self.qwm_antithetic_samples = bool(
            getattr(config, "QWM_ANTITHETIC_SAMPLES", True)
            if qwm_antithetic_samples is None else qwm_antithetic_samples
        )
        self.qwm_prior_reconstruction_weight = max(0.0, float(
            getattr(config, "QWM_PRIOR_RECONSTRUCTION_WEIGHT", 1.0)
            if qwm_prior_reconstruction_weight is None
            else qwm_prior_reconstruction_weight
        ))
        self.qwm_learning_rate = float(
            getattr(config, "QWM_LEARNING_RATE", 3e-4)
            if qwm_learning_rate is None else qwm_learning_rate
        )
        self.qwm_weight_decay = float(
            getattr(config, "QWM_WEIGHT_DECAY", 1e-5)
            if qwm_weight_decay is None else qwm_weight_decay
        )
        self.qwm_batch_size = max(1, int(
            getattr(config, "QWM_BATCH_SIZE", 128)
            if qwm_batch_size is None else qwm_batch_size
        ))
        self.qwm_pretrain_replay_rows = max(1, int(
            getattr(config, "QWM_PRETRAIN_REPLAY_ROWS", 5_000)
            if qwm_pretrain_replay_rows is None
            else qwm_pretrain_replay_rows
        ))
        self.qwm_pretrain_min_unique_baskets = max(1, int(
            getattr(config, "QWM_PRETRAIN_MIN_UNIQUE_BASKETS", 1)
            if qwm_pretrain_min_unique_baskets is None
            else qwm_pretrain_min_unique_baskets
        ))
        self.qwm_replay_baskets_per_batch = max(1, int(
            getattr(config, "QWM_REPLAY_BASKETS_PER_BATCH", 16)
            if qwm_replay_baskets_per_batch is None
            else qwm_replay_baskets_per_batch
        ))
        self.qwm_proposal_local_scale = float(
            getattr(config, "QWM_PROPOSAL_LOCAL_SCALE", 0.04)
        )
        self.qwm_proposal_medium_scale = float(
            getattr(config, "QWM_PROPOSAL_MEDIUM_SCALE", 0.15)
        )
        self.qwm_proposal_global_cash_max = float(
            getattr(config, "QWM_PROPOSAL_GLOBAL_CASH_MAX", 0.35)
        )
        self.qwm_pretrain_updates = max(0, int(
            getattr(config, "QWM_PRETRAIN_UPDATES", 5_000)
            if qwm_pretrain_updates is None else qwm_pretrain_updates
        ))
        self.qwm_online_updates_per_td_step = max(0, int(
            getattr(config, "QWM_ONLINE_UPDATES_PER_TD_STEP", 1)
            if qwm_online_updates_per_td_step is None
            else qwm_online_updates_per_td_step
        ))
        self.qwm_state_loss_weight = float(
            getattr(config, "QWM_STATE_LOSS_WEIGHT", 1.0)
            if qwm_state_loss_weight is None else qwm_state_loss_weight
        )
        self.qwm_reward_loss_weight = float(
            getattr(config, "QWM_REWARD_LOSS_WEIGHT", 1.0)
            if qwm_reward_loss_weight is None else qwm_reward_loss_weight
        )
        self.qwm_max_grad_norm = float(
            getattr(config, "QWM_MAX_GRAD_NORM", 10.0)
            if qwm_max_grad_norm is None else qwm_max_grad_norm
        )
        self.qwm_initial_depth = max(1, int(
            getattr(config, "QWM_INITIAL_DEPTH", 2)
            if qwm_initial_depth is None else qwm_initial_depth
        ))
        self.qwm_max_depth = max(self.qwm_initial_depth, int(
            getattr(config, "QWM_MAX_DEPTH", 4)
            if qwm_max_depth is None else qwm_max_depth
        ))
        self.qwm_depth_three_updates = max(0, int(
            getattr(config, "QWM_DEPTH_THREE_UPDATES", 25_000)
            if qwm_depth_three_updates is None else qwm_depth_three_updates
        ))
        self.qwm_depth_four_updates = max(
            self.qwm_depth_three_updates,
            int(
                getattr(config, "QWM_DEPTH_FOUR_UPDATES", 100_000)
                if qwm_depth_four_updates is None else qwm_depth_four_updates
            ),
        )
        self.qwm_candidates = max(1, int(
            getattr(config, "QWM_CANDIDATES", self.proportional_exp_candidates)
            if qwm_candidates is None else qwm_candidates
        ))
        self.qwm_leaf_candidates = max(1, int(
            getattr(config, "QWM_LEAF_CANDIDATES", self.qwm_candidates)
            if qwm_leaf_candidates is None else qwm_leaf_candidates
        ))
        self.qwm_model_samples = max(1, int(
            getattr(config, "QWM_MODEL_SAMPLES", 1)
            if qwm_model_samples is None else qwm_model_samples
        ))
        self.qwm_beam_width = max(1, int(
            getattr(config, "QWM_BEAM_WIDTH", 2)
            if qwm_beam_width is None else qwm_beam_width
        ))
        self.qwm_search_discount = float(
            getattr(config, "QWM_SEARCH_DISCOUNT", 0.20)
            if qwm_search_discount is None else qwm_search_discount
        )
        if not 0.0 <= self.qwm_search_discount <= 1.0:
            raise ValueError("QWM search discount must be in [0, 1]")
        self.qwm_leaf_aggregation = str(
            getattr(config, "QWM_LEAF_AGGREGATION", "max")
            if qwm_leaf_aggregation is None else qwm_leaf_aggregation
        ).strip().lower()
        if self.qwm_leaf_aggregation not in {"max", "mean"}:
            raise ValueError("QWM leaf aggregation must be max or mean")
        self.qwm_successor_aggregation = str(
            getattr(config, "QWM_SUCCESSOR_AGGREGATION", "mean")
            if qwm_successor_aggregation is None
            else qwm_successor_aggregation
        ).strip().lower()
        if self.qwm_successor_aggregation not in {"max", "mean"}:
            raise ValueError("QWM successor aggregation must be max or mean")
        self.qwm_cuda_autocast = bool(
            getattr(config, "QWM_CUDA_AUTOCAST", True)
            if qwm_cuda_autocast is None else qwm_cuda_autocast
        )
        self.qwm_compile_regions = bool(
            getattr(config, "QWM_COMPILE_REGIONS", True)
            if qwm_compile_regions is None else qwm_compile_regions
        )
        self.qwm_online_search_enabled = bool(
            getattr(config, "QWM_ONLINE_SEARCH_ENABLED", True)
            if qwm_online_search_enabled is None else qwm_online_search_enabled
        )
        self.qwm_evaluation_search_enabled = bool(
            getattr(config, "QWM_EVALUATION_SEARCH_ENABLED", True)
            if qwm_evaluation_search_enabled is None
            else qwm_evaluation_search_enabled
        )
        self.qwm_crl_aware_scoring = bool(
            getattr(config, "QWM_CRL_AWARE_SCORING", True)
            if qwm_crl_aware_scoring is None else qwm_crl_aware_scoring
        )
        self.qwm_search_seed = int(
            getattr(config, "QWM_SEARCH_SEED", 1701)
            if qwm_search_seed is None else qwm_search_seed
        )
        self.qwm_schema_version = int(getattr(config, "QWM_SCHEMA_VERSION", 1))
        self.qwm_world_updates = int(getattr(self, "qwm_world_updates", 0))
        self.qwm_pretrain_complete = bool(
            getattr(self, "qwm_pretrain_complete", not self.qwm_enabled)
        )
        self.qwm_pretrain_armed = bool(
            getattr(self, "qwm_pretrain_armed", self.qwm_enabled)
        )
        self.qwm_ready = bool(getattr(self, "qwm_ready", False))
        self.qwm_loss_ema = getattr(self, "qwm_loss_ema", None)
        self._qwm_loss_ema_tensor = None
        self.qwm_pretrain_initial_loss = getattr(
            self, "qwm_pretrain_initial_loss", None
        )
        self.qwm_pretrain_final_loss = getattr(
            self, "qwm_pretrain_final_loss", None
        )
        self.qwm_pretrain_best_loss = getattr(
            self, "qwm_pretrain_best_loss", None
        )
        self.qwm_pretrain_relative_loss = getattr(
            self, "qwm_pretrain_relative_loss", None
        )
        self.qwm_log_dir = getattr(self, "qwm_log_dir", None)
        self.qwm_search_calls = int(getattr(self, "qwm_search_calls", 0))
        self.qwm_last_search_depth = int(
            getattr(self, "qwm_last_search_depth", self.qwm_initial_depth)
        )
        self.qwm_last_root_score_mean = float(
            getattr(self, "qwm_last_root_score_mean", 0.0)
        )
        self.qwm_last_root_score_std = float(
            getattr(self, "qwm_last_root_score_std", 0.0)
        )
        self.qwm_last_successor_state_std = float(
            getattr(self, "qwm_last_successor_state_std", 0.0)
        )
        self.qwm_last_successor_reward_std = float(
            getattr(self, "qwm_last_successor_reward_std", 0.0)
        )
        self._qwm_root_telemetry = None
        self._qwm_successor_telemetry = None
        self._qwm_last_root_family_ids = None
        self.qwm_world_model = None
        self.qwm_world_model_optimizer = None

        if _init_setup_model:
            self._setup_model()
            self._setup_custom_optimizers()
            # Build sigma index map after config.SCOLS is ready (requires config.initialize first)
            try:
                self._build_sigma_position_map()
            except Exception as exc:
                if self.verbose > 0:
                    print(f"[SigmaGating] Position map deferred: {exc}")

    def _setup_model(self) -> None:
        """Build TD3 and then the separately optimized replay utility model."""
        self._configure_cross_basket_relabeling()
        super()._setup_model()
        if self.replay_buffer is not None:
            self.replay_buffer.market_input_contract = self.market_input_contract
        self._setup_task_popart()
        self._setup_qwm()
        self._setup_value_experience_replay()

    def _configure_cross_basket_relabeling(self) -> None:
        """Size relabel replay storage and read the relabeling schedule."""
        observation_spaces = getattr(self.observation_space, "spaces", {})
        buffer_class = getattr(self, "replay_buffer_class", None)
        supported = bool(
            getattr(config, "CROSS_BASKET_RELABEL_ENABLED", False)
            and isinstance(self.observation_space, spaces.Dict)
            and all(
                key in observation_spaces
                for key in ("observations", "mask", "prev_action")
            )
            and observation_spaces["mask"].shape[0] == len(getattr(config, "COIN_POOL", ()))
            and isinstance(buffer_class, type)
            and issubclass(buffer_class, ComponentOutcomeReplayBuffer)
        )
        replay_kwargs = dict(getattr(self, "replay_buffer_kwargs", None) or {})
        if supported:
            replay_kwargs["relabel_market_width"] = int(
                config.PRIMARY_FEATURE_COUNT + config.COMPARATIVE_FEATURE_COUNT
            )
        else:
            replay_kwargs.pop("relabel_market_width", None)
        self.replay_buffer_kwargs = replay_kwargs
        self.cross_basket_relabel_enabled = supported
        self.cross_basket_relabel_fraction = float(
            np.clip(getattr(config, "CROSS_BASKET_RELABEL_FRACTION", 0.5), 0.0, 0.5)
        )
        self.cross_basket_relabel_one_swap_fraction = float(
            np.clip(getattr(config, "CROSS_BASKET_RELABEL_ONE_SWAP_FRACTION", 0.5), 0.0, 1.0)
        )
        self.cross_basket_relabel_policy_action_fraction = float(
            np.clip(
                getattr(config, "CROSS_BASKET_RELABEL_POLICY_ACTION_FRACTION", 0.5),
                0.0,
                1.0,
            )
        )
        self.cross_basket_relabel_policy_action_noise = max(
            0.0, float(getattr(config, "CROSS_BASKET_RELABEL_POLICY_ACTION_NOISE", 0.2))
        )
        self.cross_basket_relabel_min_position_weight = float(
            getattr(config, "CROSS_BASKET_RELABEL_MIN_POSITION_WEIGHT", 5e-5)
        )
        self.cross_basket_relabel_critic_rows = normalize_cross_basket_row_mode(
            getattr(config, "CROSS_BASKET_RELABEL_CRITIC_ROWS", "mixed"),
            default="mixed",
        )
        self.cross_basket_relabel_actor_crl_rows = normalize_cross_basket_row_mode(
            getattr(config, "CROSS_BASKET_RELABEL_ACTOR_CRL_ROWS", "original"),
            default="original",
        )
        self._last_cross_basket_relabel_source_index = None
        self._last_cross_basket_relabel_target_index = None
        if not isinstance(
            getattr(self, "_cross_basket_relabel_rng", None), np.random.Generator
        ):
            self._cross_basket_relabel_rng = np.random.default_rng(
                int(getattr(config, "CROSS_BASKET_RELABEL_SEED", 20260924))
            )
        self._cross_basket_relabel_name_rank = None
        if supported and self.verbose > 0:
            print(
                "[CrossBasketRelabel] exact same-minute basket relabeling enabled | "
                f"fraction={self.cross_basket_relabel_fraction:.2f} "
                f"one_swap={self.cross_basket_relabel_one_swap_fraction:.2f} "
                f"policy_actions={self.cross_basket_relabel_policy_action_fraction:.2f} "
                f"noise={self.cross_basket_relabel_policy_action_noise:.2f} "
                f"critic_rows={self.cross_basket_relabel_critic_rows} "
                f"actor_crl_rows={self.cross_basket_relabel_actor_crl_rows}"
            )

    def _setup_task_popart(self) -> None:
        if not self.task_popart_enabled:
            self.task_popart = None
            return
        if not isinstance(self.observation_space, spaces.Dict):
            raise TypeError("Task PopArt requires global-pool dictionary observations")
        mask_space = self.observation_space.spaces.get("mask")
        if mask_space is None or len(mask_space.shape) != 1:
            raise ValueError("Task PopArt requires a one-dimensional availability mask")
        task_width = int(mask_space.shape[0])
        if task_width != len(getattr(config, "COIN_POOL", ())):
            raise ValueError(
                "Task PopArt requires the mask to span the complete global pool"
            )
        self.task_popart = TaskConditionedPopArt(
            task_width,
            beta=self.task_popart_beta,
            min_std=self.task_popart_min_std,
            max_std=self.task_popart_max_std,
            prior_count=self.task_popart_prior_count,
        ).to(self.device)

    def load_replay_buffer(self, path, truncate_last_traj=True):
        previous = self.replay_buffer
        super().load_replay_buffer(path, truncate_last_traj=truncate_last_traj)
        received = getattr(self.replay_buffer, "market_input_contract", "selected_basket_v1")
        if received != self.market_input_contract:
            self.replay_buffer = previous
            raise ValueError(
                f"Replay market input contract mismatch: {received} != "
                f"{self.market_input_contract}. Collect fresh replay."
            )

    def _setup_qwm(self) -> None:
        """Create the independent real-transition model used by QWM search."""
        if not bool(getattr(self, "qwm_enabled", False)):
            self.qwm_world_model = None
            self.qwm_world_model_optimizer = None
            return
        self.qwm_world_model = PortfolioWorldModel(
            self.observation_space,
            action_dim=int(self.action_space.shape[0]),
            hidden_dim=self.qwm_hidden_dim,
            max_normalized_residual=self.qwm_max_normalized_residual,
            latent_dim=self.qwm_latent_dim,
            kl_weight=self.qwm_kl_weight,
            kl_free_nats=self.qwm_kl_free_nats,
            reward_target_scale=self.qwm_reward_target_scale,
            antithetic_samples=self.qwm_antithetic_samples,
            prior_reconstruction_weight=self.qwm_prior_reconstruction_weight,
        ).to(self.device)
        optimizer_kwargs = {}
        if (
            bool(getattr(config, "QWM_FUSED_OPTIMIZER", True))
            and torch.device(self.device).type == "cuda"
        ):
            optimizer_kwargs["fused"] = True
        self.qwm_world_model_optimizer = torch.optim.AdamW(
            self.qwm_world_model.parameters(),
            lr=self.qwm_learning_rate,
            weight_decay=self.qwm_weight_decay,
            **optimizer_kwargs,
        )
        if self.qwm_compile_regions:
            for region_name in (
                "token_encoder",
                "global_encoder",
                "action_token_encoder",
                "action_global_encoder",
                "prior_head",
                "posterior_token_encoder",
                "posterior_head",
                "reward_head",
            ):
                region = getattr(self.qwm_world_model, region_name, None)
                if isinstance(region, nn.Sequential):
                    _maybe_compile_module(
                        region,
                        f"PortfolioWorldModel.{region_name}",
                        mode_override=str(
                            getattr(config, "QWM_COMPILE_MODE", "reduce-overhead")
                        ),
                        dynamic_override=bool(
                            getattr(config, "QWM_COMPILE_DYNAMIC", False)
                        ),
                        fullgraph_override=bool(
                            getattr(config, "QWM_COMPILE_FULLGRAPH", True)
                        ),
                    )
            # Compile inference-only stochastic transition methods. Keep the
            # diagnostic-rich ``loss`` wrapper eager: its telemetry is outside
            # the optimization objective, whereas AOT multi-output backward
            # can still differentiate zero-cotangent diagnostics. The costly
            # neural subregions above remain compiled during training.
            for method_name in ("sample", "sample_action_candidates"):
                _maybe_compile_method(
                    self.qwm_world_model,
                    method_name,
                    "PortfolioWorldModel",
                    mode_override=str(
                        getattr(config, "QWM_COMPILE_MODE", "reduce-overhead")
                    ),
                    dynamic_override=bool(
                        getattr(config, "QWM_COMPILE_DYNAMIC", False)
                    ),
                    fullgraph=False,
                )
            # Do not replace the shared critic candidate evaluator with one
            # CUDA-Graph callable. QWM invokes it under ``inference_mode``, but
            # TD3/ProExp invokes the same method with autograd enabled during
            # actor training. Capturing the former contract and replaying it in
            # the latter path creates inference tensors that Adam cannot update
            # in place. The critic's expensive attention/value subregions stay
            # compiled by AttentionTD3Policy; only this cross-mode composite
            # wrapper remains eager.
        if self.verbose > 0:
            print(
                "[QWM] TD3 decision-time search enabled | "
                f"pretrain_rows={self.qwm_pretrain_replay_rows} | "
                f"pretrain_baskets={self.qwm_pretrain_min_unique_baskets} | "
                f"pretrain={self.qwm_pretrain_updates} real-replay updates | "
                f"online_ratio={self.qwm_online_updates_per_td_step}:1 | "
                f"depth={self.qwm_initial_depth}->{self.qwm_max_depth} | "
                f"N={self.qwm_candidates} | leaf={self.qwm_leaf_candidates} | "
                f"K={self.qwm_model_samples} stochastic successors | "
                f"latent={self.qwm_latent_dim} | "
                f"prior_supervision={self.qwm_prior_reconstruction_weight:.2f} | "
                f"agg_K={self.qwm_successor_aggregation} | "
                f"beam={self.qwm_beam_width} | lambda={self.qwm_search_discount:.2f}"
            )

    def qwm_pretrain(self) -> None:
        """Arm initial world-model fitting after ordinary real replay warmup."""
        if not self.qwm_enabled:
            return
        self.qwm_pretrain_armed = True
        if self.qwm_pretrain_updates == 0:
            self.qwm_pretrain_complete = True
            self.qwm_ready = True

    def _qwm_update_from_replay_data(
        self, replay_data
    ) -> Dict[str, torch.Tensor]:
        if self.qwm_world_model is None or self.qwm_world_model_optimizer is None:
            return {}
        if not isinstance(replay_data.observations, dict):
            raise TypeError("QWM requires dictionary replay observations")
        observations = {
            key: value.float() if value.is_floating_point() else value
            for key, value in replay_data.observations.items()
        }
        next_observations = {
            key: value.float() if value.is_floating_point() else value
            for key, value in replay_data.next_observations.items()
        }
        self.qwm_world_model.train()
        use_autocast = bool(
            self.qwm_cuda_autocast
            and torch.device(self.device).type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        if use_autocast:
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                loss_stats = self.qwm_world_model.loss(
                    observations,
                    replay_data.actions.float(),
                    replay_data.rewards.float(),
                    next_observations,
                    state_weight=self.qwm_state_loss_weight,
                    reward_weight=self.qwm_reward_loss_weight,
                )
        else:
            loss_stats = self.qwm_world_model.loss(
                observations,
                replay_data.actions.float(),
                replay_data.rewards.float(),
                next_observations,
                state_weight=self.qwm_state_loss_weight,
                reward_weight=self.qwm_reward_loss_weight,
            )
        loss = loss_stats["loss"]
        self.qwm_world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.qwm_world_model.parameters(),
            self.qwm_max_grad_norm,
            error_if_nonfinite=False,
        )
        _qwm_assert_finite(
            torch.stack(
                (
                    loss.detach().float(),
                    torch.as_tensor(gradient_norm).detach().float(),
                )
            ),
            "QWM produced a non-finite loss or gradient norm",
        )
        self.qwm_world_model_optimizer.step()
        self.qwm_world_updates += 1
        detached_loss = loss.detach().float()
        if self._qwm_loss_ema_tensor is None:
            if self.qwm_loss_ema is None:
                self._qwm_loss_ema_tensor = detached_loss.clone()
            else:
                self._qwm_loss_ema_tensor = detached_loss.new_tensor(
                    float(self.qwm_loss_ema)
                )
                self._qwm_loss_ema_tensor.mul_(0.99).add_(
                    detached_loss, alpha=0.01
                )
        else:
            self._qwm_loss_ema_tensor.mul_(0.99).add_(detached_loss, alpha=0.01)
        return {
            "loss": detached_loss,
            "reconstruction_loss": loss_stats["reconstruction_loss"].detach().float(),
            "posterior_reconstruction_loss": loss_stats[
                "posterior_reconstruction_loss"
            ].detach().float(),
            "prior_reconstruction_loss": loss_stats[
                "prior_reconstruction_loss"
            ].detach().float(),
            "state_loss": loss_stats["state_loss"].detach().float(),
            "reward_loss": loss_stats["reward_loss"].detach().float(),
            "prior_state_loss": loss_stats["prior_state_loss"].detach().float(),
            "prior_reward_loss": loss_stats["prior_reward_loss"].detach().float(),
            "kl_loss": loss_stats["kl_loss"].detach().float(),
            "observation_loss": loss_stats["observation_loss"].detach().float(),
            "local_history_loss": loss_stats["local_history_loss"].detach().float(),
            "episode_history_loss": loss_stats["episode_history_loss"].detach().float(),
            "context_loss": loss_stats["context_loss"].detach().float(),
            "prior_std_mean": loss_stats["prior_std_mean"].detach().float(),
            "posterior_std_mean": loss_stats["posterior_std_mean"].detach().float(),
            "latent_mean_gap": loss_stats["latent_mean_gap"].detach().float(),
            "gradient_norm": torch.as_tensor(gradient_norm).detach().float(),
        }

    @staticmethod
    def _materialize_qwm_stats(
        stats: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Transfer all QWM diagnostics with one device synchronization."""
        keys = [key for key in QWM_TRAINING_STAT_KEYS if key in stats]
        if not keys:
            return {}
        packed = torch.stack(
            [torch.as_tensor(stats[key]).detach().float() for key in keys]
        ).cpu().tolist()
        return {key: float(value) for key, value in zip(keys, packed)}

    def _qwm_available_basket_count(self) -> int:
        """Count distinct causal basket tasks currently represented in replay."""
        buffer = self.replay_buffer
        rebuild = getattr(buffer, "_rebuild_basket_position_index", None)
        indexed = getattr(buffer, "_basket_positions_by_key", None)
        if indexed is None and callable(rebuild):
            rebuild()
            indexed = getattr(buffer, "_basket_positions_by_key", None)
        if isinstance(indexed, dict):
            return sum(1 for positions in indexed.values() if positions)
        observations = getattr(buffer, "observations", None)
        if isinstance(observations, dict) and "basket_membership" in observations:
            upper = buffer.buffer_size if buffer.full else buffer.pos
            membership = np.asarray(observations["basket_membership"][:upper])
            if membership.ndim == 3 and membership.shape[1] == 1:
                membership = membership[:, 0, :]
            if membership.ndim == 2 and membership.shape[0] > 0:
                return int(np.unique(membership > 0.5, axis=0).shape[0])
        # Generic smoke environments have no basket provenance.  Treat their
        # single task as one basket so an explicit min=1 remains testable.
        return 1

    def _sample_qwm_replay_data(self, batch_size: int):
        """Sample a basket-balanced batch exclusively for world-model fitting.

        TD3 retains its configured Bellman replay distribution.  QWM receives
        a separate, uniformly basket-stratified view so long episodes from a
        few early 8-of-13 combinations cannot dominate transition learning.
        """
        buffer = self.replay_buffer
        sampler = getattr(buffer, "sample_hierarchical_batch_inds", None)
        indexed = getattr(buffer, "_basket_positions_by_key", {})
        available = sum(1 for positions in indexed.values() if positions)
        if callable(sampler) and getattr(buffer, "n_envs", 1) == 1 and available:
            try:
                batch_inds = sampler(
                    batch_size=int(batch_size),
                    baskets_per_batch=min(
                        self.qwm_replay_baskets_per_batch, available
                    ),
                    rng=self._basket_replay_rng,
                    sample_without_replacement=True,
                    group_selection="uniform",
                    task_probabilities=None,
                    min_completed_episodes=0,
                    uniform_mix=0.0,
                    allow_partial_unique_batch=True,
                )
                return buffer._get_samples(
                    np.asarray(batch_inds, dtype=np.int64),
                    env=self._vec_normalize_env,
                )
            except BasketReplayEvidenceUnavailable:
                pass
        return buffer.sample(int(batch_size), env=self._vec_normalize_env)

    def _run_qwm_pretrain(self) -> bool:
        """Fit only the world model; leave TD3/ProExp/CRL parameters untouched."""
        if not self.qwm_enabled or self.qwm_pretrain_complete:
            return True
        if not self.qwm_pretrain_armed:
            return False
        replay_slots = int(self.replay_buffer.size())
        replay_envs = max(1, int(getattr(self.replay_buffer, "n_envs", 1)))
        replay_size = replay_slots * replay_envs
        required_rows = max(
            1,
            int(self.learning_starts),
            self.qwm_pretrain_replay_rows,
        )
        available_baskets = self._qwm_available_basket_count()
        if (
            replay_size < required_rows
            or available_baskets < self.qwm_pretrain_min_unique_baskets
        ):
            return False
        if self.verbose > 0:
            print(
                "[QWM] Starting initial residual-model fit from observed replay "
                f"({replay_size} rows, {available_baskets} baskets, "
                f"{self.qwm_pretrain_updates} updates)."
            )
        started_at = time.perf_counter()
        pretrain_csv_path = None
        if self.qwm_log_dir:
            os.makedirs(self.qwm_log_dir, exist_ok=True)
            pretrain_csv_path = os.path.join(
                self.qwm_log_dir, "qwm_pretrain.csv"
            )
            if self.qwm_world_updates == 0 or not os.path.exists(
                pretrain_csv_path
            ):
                with open(pretrain_csv_path, "w", newline="") as stream:
                    csv.writer(stream).writerow(
                        (
                            "qwm_update", "loss", "reconstruction_loss",
                            "posterior_reconstruction_loss",
                            "prior_reconstruction_loss",
                            "state_loss", "reward_loss",
                            "prior_state_loss", "prior_reward_loss", "kl_loss",
                            "observation_loss", "local_history_loss",
                            "episode_history_loss", "context_loss",
                            "prior_std_mean", "posterior_std_mean",
                            "latent_mean_gap", "gradient_norm",
                            "replay_rows", "replay_baskets",
                            "elapsed_seconds", "device",
                        )
                    )
            if self.verbose > 0:
                print(f"[QWM] live pretrain metrics: {pretrain_csv_path}")

        def _write_pretrain_row(stats: Dict[str, float]) -> None:
            if pretrain_csv_path is None:
                return
            with open(pretrain_csv_path, "a", newline="") as stream:
                csv.writer(stream).writerow(
                    (
                        int(self.qwm_world_updates),
                        float(stats.get("loss", math.nan)),
                        float(stats.get("reconstruction_loss", math.nan)),
                        float(stats.get("posterior_reconstruction_loss", math.nan)),
                        float(stats.get("prior_reconstruction_loss", math.nan)),
                        float(stats.get("state_loss", math.nan)),
                        float(stats.get("reward_loss", math.nan)),
                        float(stats.get("prior_state_loss", math.nan)),
                        float(stats.get("prior_reward_loss", math.nan)),
                        float(stats.get("kl_loss", math.nan)),
                        float(stats.get("observation_loss", math.nan)),
                        float(stats.get("local_history_loss", math.nan)),
                        float(stats.get("episode_history_loss", math.nan)),
                        float(stats.get("context_loss", math.nan)),
                        float(stats.get("prior_std_mean", math.nan)),
                        float(stats.get("posterior_std_mean", math.nan)),
                        float(stats.get("latent_mean_gap", math.nan)),
                        float(stats.get("gradient_norm", math.nan)),
                        replay_size,
                        available_baskets,
                        float(time.perf_counter() - started_at),
                        str(self.device),
                    )
                )
        latest: Dict[str, torch.Tensor] = {}
        initial_loss_tensor = None
        best_loss_tensor = None
        for update_index in range(self.qwm_pretrain_updates):
            replay_data = self._sample_qwm_replay_data(self.qwm_batch_size)
            latest = self._qwm_update_from_replay_data(replay_data)
            current_loss = latest.get("loss")
            if current_loss is not None:
                current_loss = current_loss.detach().float()
                if initial_loss_tensor is None:
                    initial_loss_tensor = current_loss.clone()
                best_loss_tensor = (
                    current_loss.clone()
                    if best_loss_tensor is None
                    else torch.minimum(best_loss_tensor, current_loss)
                )
            report_due = bool(
                update_index == 0
                or (update_index + 1) % 500 == 0
                or update_index + 1 == self.qwm_pretrain_updates
            )
            if report_due:
                materialized = self._materialize_qwm_stats(latest)
                summary_tensors = torch.stack(
                    (
                        initial_loss_tensor,
                        current_loss,
                        best_loss_tensor,
                        current_loss / initial_loss_tensor.clamp(min=1e-12),
                    )
                ).cpu().tolist()
                (
                    self.qwm_pretrain_initial_loss,
                    self.qwm_pretrain_final_loss,
                    self.qwm_pretrain_best_loss,
                    self.qwm_pretrain_relative_loss,
                ) = (float(value) for value in summary_tensors)
                if self._qwm_loss_ema_tensor is not None:
                    self.qwm_loss_ema = float(
                        self._qwm_loss_ema_tensor.detach().cpu()
                    )
                _write_pretrain_row(materialized)
            if self.verbose > 0 and report_due:
                print(
                    f"[QWM] pretrain {update_index + 1}/{self.qwm_pretrain_updates} "
                    f"loss={materialized.get('loss', float('nan')):.6f} "
                    f"recon={materialized.get('reconstruction_loss', float('nan')):.6f} "
                    f"prior={materialized.get('prior_reconstruction_loss', float('nan')):.6f} "
                    f"state={materialized.get('state_loss', float('nan')):.6f} "
                    f"reward={materialized.get('reward_loss', float('nan')):.6f} "
                    f"kl={materialized.get('kl_loss', float('nan')):.6f} "
                    f"prior_std={materialized.get('prior_std_mean', float('nan')):.4f} "
                    f"obs={materialized.get('observation_loss', float('nan')):.6f} "
                    f"local={materialized.get('local_history_loss', float('nan')):.6f} "
                    f"episode={materialized.get('episode_history_loss', float('nan')):.6f} "
                    f"context={materialized.get('context_loss', float('nan')):.6f}"
                )
        self.qwm_pretrain_complete = True
        self.qwm_ready = True
        self.qwm_world_model.eval()
        if self.verbose > 0:
            print(
                "[QWM] Initial fit complete; search is now active and the "
                "world model will continue learning from real TD3 replay. "
                f"initial={float(self.qwm_pretrain_initial_loss or float('nan')):.6f} "
                f"final={float(self.qwm_pretrain_final_loss or float('nan')):.6f} "
                f"relative={float(self.qwm_pretrain_relative_loss or float('nan')):.4f}"
            )
        return True

    def _setup_value_experience_replay(self) -> None:
        if not bool(getattr(self, "value_experience_replay_enabled", False)):
            self.value_experience_replay_scorer = None
            self.value_experience_replay_optimizer = None
            return
        if not hasattr(self.critic, "value_experience_representation"):
            raise RuntimeError(
                "value-experience replay requires the Path-2 DeepSet critic"
            )
        # Masked mean/std moments describe state. The action side contains six
        # learned financial outcomes, four pooled behavioral allocation roles,
        # and six invariant portfolio statistics. Eight fresh live-Bellman
        # diagnostics are appended during every selection.
        (
            replay_state_width,
            replay_action_width,
        ) = self.critic.value_experience_representation_widths()
        input_dim = replay_state_width + replay_action_width + 8
        self.value_experience_replay_state_width = replay_state_width
        self.value_experience_replay_action_width = replay_action_width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                int(getattr(config, "BASKET_CURRICULUM_SEED", 0)) + 911
            )
            self.value_experience_replay_scorer = ValueExperienceReplayScorer(
                input_dim=input_dim,
                hidden_dim=int(
                    getattr(config, "VALUE_EXPERIENCE_REPLAY_HIDDEN_DIM", 128)
                ),
            ).to(self.device)
        self.value_experience_replay_optimizer = torch.optim.AdamW(
            self.value_experience_replay_scorer.parameters(),
            lr=float(
                getattr(config, "VALUE_EXPERIENCE_REPLAY_LEARNING_RATE", 1e-4)
            ),
            weight_decay=1e-5,
        )
        if self.verbose > 0:
            print(
                "[ValueExperienceReplay] Mixed FAM/AAM state-action sampler "
                f"enabled | candidates={self.value_experience_candidate_multiplier}x "
                f"| selection_interval={self.value_experience_selection_interval} "
                f"| heldout_query={self.value_experience_query_size} "
                f"(hard={self.value_experience_query_hard_fraction:.2f}) "
                f"| transfer_interval={self.value_experience_transfer_interval} "
                f"| uniform_floor={self.value_experience_uniform_fraction:.2f} "
                f"| descriptor_v={self.value_experience_replay_descriptor_version} "
                f"| state_dim={replay_state_width} "
                f"| action_dim={replay_action_width} | input_dim={input_dim}"
            )

    def _get_torch_save_params(self):
        state_dicts, tensor_names = super()._get_torch_save_params()
        if self.task_popart is not None:
            state_dicts = list(state_dicts) + ["task_popart"]
        if self.value_experience_replay_scorer is not None:
            state_dicts = list(state_dicts) + [
                "value_experience_replay_scorer",
                "value_experience_replay_optimizer",
            ]
        if self.qwm_world_model is not None:
            state_dicts = list(state_dicts) + [
                "qwm_world_model",
                "qwm_world_model_optimizer",
            ]
        return state_dicts, tensor_names

    def _detect_attention_policy(self, policy) -> bool:
        """
        Detect if the policy uses attention-based feature extraction.

        Args:
            policy: Policy class or string name
            
        Returns:
            bool: True if attention-based policy detected
        """
        if isinstance(policy, str):
            policy_name = policy.lower()
            return "attention" in policy_name
        else:
            # Check if it's an attention policy class
            policy_name = policy.__name__.lower() if hasattr(policy, '__name__') else str(policy).lower()
            attention_keywords = ["attention", "transformer", "variablecoin", "multihead", "coinattention"]
            return any(keyword in policy_name for keyword in attention_keywords)

    @torch.no_grad()
    def _accumulate_basket_td_errors(
        self,
        observations,
        current_q_values,
        target_q_values,
    ) -> None:
        """Accumulate detached scalar Bellman correction by basket membership."""
        if not isinstance(observations, dict):
            return
        memberships = observations.get("basket_membership")
        if memberships is None or not torch.is_tensor(memberships):
            return
        if len(current_q_values) == 0:
            return

        twin_absolute_error = torch.stack(
            [
                (current_q.detach() - target_q_values.detach()).abs().squeeze(-1)
                for current_q in current_q_values
            ],
            dim=0,
        ).mean(dim=0)
        binary_memberships = memberships.detach() > 0.5
        unique_memberships, inverse = torch.unique(
            binary_memberships,
            dim=0,
            return_inverse=True,
        )
        group_count = int(unique_memberships.shape[0])
        error_sums = torch.zeros(
            group_count,
            dtype=twin_absolute_error.dtype,
            device=twin_absolute_error.device,
        )
        error_counts = torch.zeros(
            group_count,
            dtype=twin_absolute_error.dtype,
            device=twin_absolute_error.device,
        )
        error_sums.scatter_add_(0, inverse, twin_absolute_error)
        error_counts.scatter_add_(
            0,
            inverse,
            torch.ones_like(twin_absolute_error),
        )
        membership_rows = unique_memberships.to("cpu").numpy()
        error_sums_cpu = error_sums.to("cpu").numpy()
        error_counts_cpu = error_counts.to("cpu").numpy()

        stats = getattr(self, "_basket_td_error_stats", None)
        if stats is None:
            stats = {}
            self._basket_td_error_stats = stats
        for membership_row, error_sum, error_count in zip(
            membership_rows,
            error_sums_cpu,
            error_counts_cpu,
        ):
            key = tuple(np.flatnonzero(membership_row).astype(int).tolist())
            accumulator = stats.setdefault(key, [0.0, 0.0])
            accumulator[0] += float(error_sum)
            accumulator[1] += float(error_count)

    def drain_basket_td_error_stats(self) -> Dict[Tuple[int, ...], Dict[str, float]]:
        """Return and clear detached per-basket Bellman diagnostics."""
        stats = getattr(self, "_basket_td_error_stats", {})
        drained = {
            tuple(key): {
                "absolute_td_error_mean": float(total / max(count, 1.0)),
                "sample_count": int(count),
            }
            for key, (total, count) in stats.items()
        }
        self._basket_td_error_stats = {}
        return drained

    # ════════════════════════════════════════════════════════════════════════════
    # SIGMA INPUT MODULATION + LOSS FOCUS
    # ════════════════════════════════════════════════════════════════════════════

    def _build_sigma_position_map(self) -> None:
        """
        Pre-compute flat-vector index triples for each (signal, sigma, scale_factor) feature.

        The flat observation for variable-portfolio mode is laid out as:
            [coin_0 local SCOLS..., coin_0 cost_basis,
             coin_0 comparative SCOLS..., coin_0 episode-relative features...,
             coin_1 local SCOLS..., coin_1 cost_basis,
             coin_1 comparative SCOLS..., coin_1 episode-relative features..., ...]

        Base features (vh/vl/vc + mu_h/mu_l/mu_c + volume_s/mu_vol + weightedAverage_s/mu_wa)
        use full gate stiffness.
        Rolling features (vh_roll_7, vh_roll_14, vh_roll_30, etc.) use gentler stiffness
        so long-horizon summaries retain influence when short-horizon signals get muted.
        Same sigma governor, different valve stiffness: gate_w = 1 - tanh(|σ| * scale * scale_factor).
        """
        SIGMA_GOVERNED_BASES = getattr(config, 'SIGMA_GOVERNED_BASES', [
            ('vh', 'vh_sigma'), ('vl', 'vl_sigma'), ('vc', 'vc_sigma'),
            ('mu_h', 'vh_sigma'), ('mu_l', 'vl_sigma'), ('mu_c', 'vc_sigma'),
            ('volume_s', 'volume_sigma'), ('quoteVolume_s', 'quoteVolume_sigma'),
            ('weightedAverage_s', 'wa_sigma'),
            ('mu_vol', 'volume_sigma'), ('mu_wa', 'wa_sigma'),
        ])
        ROLLING_WINDOWS = getattr(config, 'ROLLING_WINDOWS', ['7', '14', '30'])
        ROLL_SCALE_FACTORS = getattr(config, 'ROLL_SCALE_FACTORS', {'7': 0.7, '14': 0.5, '30': 0.3})

        if not hasattr(config, 'SCOLS') or not config.SCOLS:
            if self.verbose > 0:
                print("[SigmaGating] config.SCOLS not ready — map not built.")
            return

        scols = config.SCOLS
        n_feats = config.FEATURES_PER_COIN
        expected_n_feats = (
            len(scols)
            + 1
            + int(getattr(config, 'COMPARATIVE_FEATURE_COUNT', 0))
            + int(getattr(config, 'EPISODE_RELATIVE_FEATURE_COUNT', 0))
        )
        if n_feats != expected_n_feats:
            raise ValueError(
                "Sigma position map cannot use an ambiguous token stride: "
                f"FEATURES_PER_COIN={n_feats}, expected {expected_n_feats} "
                "from local SCOLS + cost basis + comparative SCOLS + "
                "selected-basket features."
            )
        n_coins = config.MAX_COINS

        signal_positions: List[int] = []
        sigma_positions:  List[int] = []
        scale_factors:    List[float] = []

        for sig_name, sigma_name in SIGMA_GOVERNED_BASES:
            if sigma_name not in scols:
                continue
            sigma_idx = scols.index(sigma_name)
            # Base feature: full stiffness (scale_factor = 1.0)
            if sig_name in scols:
                sig_idx = scols.index(sig_name)
                for coin_i in range(n_coins):
                    base = coin_i * n_feats
                    signal_positions.append(base + sig_idx)
                    sigma_positions.append(base + sigma_idx)
                    scale_factors.append(1.0)
            # Rolling features: gentler stiffness for longer windows
            for w in ROLLING_WINDOWS:
                roll_name = f'{sig_name}_roll_{w}'
                if roll_name in scols:
                    roll_idx = scols.index(roll_name)
                    sf = ROLL_SCALE_FACTORS[w]
                    for coin_i in range(n_coins):
                        base = coin_i * n_feats
                        signal_positions.append(base + roll_idx)
                        sigma_positions.append(base + sigma_idx)
                        scale_factors.append(sf)

        if signal_positions:
            self._sigma_signal_idx = torch.tensor(signal_positions, dtype=torch.long, device=self.device)
            self._sigma_gate_idx   = torch.tensor(sigma_positions,  dtype=torch.long, device=self.device)
            self._sigma_scale_factors = torch.tensor(scale_factors, dtype=torch.float32, device=self.device)
            unique_sigma_positions = sorted(set(sigma_positions))
            self._sigma_unique_idx = torch.tensor(unique_sigma_positions, dtype=torch.long, device=self.device)
            if self.verbose > 0:
                print(f"[SigmaGating] Built map: {len(signal_positions)} gated positions "
                      f"(base + roll_7/14/30), scale factors 1.0/0.7/0.5/0.3")
        else:
            self._sigma_unique_idx = None
            if self.verbose > 0:
                print("[SigmaGating] No matching sigma pairs found in config.SCOLS — gating disabled.")

    def set_basket_replay_task_probabilities(
        self,
        probabilities_by_membership_ids: Dict[Tuple[int, ...], float],
        *,
        role_probabilities_by_membership_ids: Optional[
            Dict[str, Dict[Tuple[int, ...], float]]
        ] = None,
        predictive_uniform_mix: Optional[float] = None,
    ) -> None:
        """Install the teacher's current task law without sharing its RNG.

        Financial replay follows the law directly. Predictive replay receives
        the same law blended with a stage-dependent uniform-over-seen component
        so broad representation learning cannot overrule foundation pacing.
        """
        resolved: Dict[bytes, float] = {}
        membership_space = (
            self.observation_space.spaces.get("basket_membership")
            if isinstance(self.observation_space, spaces.Dict)
            else None
        )
        membership_dim = int(
            membership_space.shape[0]
            if membership_space is not None
            else getattr(config, "BASKET_MEMBERSHIP_DIM", 0)
        )
        for membership_ids, probability in probabilities_by_membership_ids.items():
            row = np.zeros(membership_dim, dtype=np.float32)
            indices = np.asarray(tuple(membership_ids), dtype=np.int64)
            if np.any(indices < 0) or np.any(indices >= membership_dim):
                raise ValueError("teacher membership IDs exceed replay schema")
            row[indices] = 1.0
            resolved[_membership_key(row)] = max(float(probability), 0.0)
        total = float(sum(resolved.values()))
        if total <= 0.0:
            raise ValueError("teacher replay probabilities must have positive mass")
        self._basket_replay_teacher_probabilities = {
            key: value / total for key, value in resolved.items()
        }
        self._basket_replay_teacher_anchor_share = float(
            self._basket_replay_teacher_probabilities.get(
                self._basket_replay_anchor_key, 0.0
            )
        )
        # Keep the public/legacy field synchronized. Financial sampling replaces
        # this flat law with an evidence-aware role-preserving law immediately
        # before it asks replay for a batch; predictive sampling intentionally
        # continues to use the unfiltered teacher law.
        self._basket_replay_task_probabilities = dict(
            self._basket_replay_teacher_probabilities
        )
        resolved_roles: Dict[str, Dict[bytes, float]] = {}
        if role_probabilities_by_membership_ids:
            for role, role_probabilities in role_probabilities_by_membership_ids.items():
                resolved_role: Dict[bytes, float] = {}
                for membership_ids, probability in dict(role_probabilities).items():
                    row = np.zeros(membership_dim, dtype=np.float32)
                    indices = np.asarray(tuple(membership_ids), dtype=np.int64)
                    if np.any(indices < 0) or np.any(indices >= membership_dim):
                        raise ValueError("teacher role membership IDs exceed replay schema")
                    row[indices] = 1.0
                    key = _membership_key(row)
                    resolved_role[key] = resolved_role.get(key, 0.0) + max(
                        float(probability), 0.0
                    )
                if sum(resolved_role.values()) > 0.0:
                    resolved_roles[str(role)] = resolved_role
        if not resolved_roles:
            anchor_key = self._basket_replay_anchor_key
            resolved_roles = {
                "anchor": {
                    anchor_key: self._basket_replay_teacher_probabilities.get(
                        anchor_key, 0.0
                    )
                },
                "transfer": {
                    key: probability
                    for key, probability in self._basket_replay_teacher_probabilities.items()
                    if key != anchor_key
                },
            }
            resolved_roles = {
                role: probabilities
                for role, probabilities in resolved_roles.items()
                if sum(probabilities.values()) > 0.0
            }
        role_total = float(
            sum(sum(probabilities.values()) for probabilities in resolved_roles.values())
        )
        self._basket_replay_role_probabilities = {
            role: {
                key: float(probability) / role_total
                for key, probability in probabilities.items()
                if probability > 0.0
            }
            for role, probabilities in resolved_roles.items()
        }
        if predictive_uniform_mix is not None:
            self.predictive_ssl_replay_uniform_mix = float(
                np.clip(predictive_uniform_mix, 0.0, 1.0)
            )

    def _role_preserving_financial_replay_probabilities(self) -> Dict[bytes, float]:
        """Current teacher law after evidence gating, without role collapse."""
        teacher_probabilities = getattr(
            self, "_basket_replay_teacher_probabilities", None
        ) or getattr(self, "_basket_replay_task_probabilities", {})
        anchor_key = getattr(self, "_basket_replay_anchor_key", None)
        if anchor_key is None:
            # Compatibility for focused sampler shells and legacy checkpoints
            # that predate the explicit anchor key. Production instances always
            # install it during construction.
            return dict(teacher_probabilities)
        buf = self.replay_buffer
        completed = getattr(buf, "_basket_completed_episodes_by_key", None)
        if completed is None:
            rebuild = getattr(buf, "_rebuild_basket_position_index", None)
            if callable(rebuild):
                rebuild()
                completed = getattr(
                    buf, "_basket_completed_episodes_by_key", None
                )
        minimum = int(self.basket_replay_min_completed_episodes)
        mature_keys = {
            key
            for key, count in dict(completed or {}).items()
            if int(count) >= minimum
        }
        adjusted = role_preserving_teacher_probabilities(
            teacher_probabilities,
            mature_keys=mature_keys,
            anchor_key=anchor_key,
        )
        resolved = adjusted or dict(teacher_probabilities)
        self._basket_replay_financial_anchor_share = float(
            resolved.get(self._basket_replay_anchor_key, 0.0)
        )
        return resolved

    def _annotate_basket_replay_stats(
        self,
        diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        diagnostics = dict(diagnostics or {})
        membership_ids = tuple(diagnostics.get("membership_ids", ()))
        anchor_ids = set(self._basket_replay_anchor_membership_ids)
        selected_ids = set(int(index) for index in membership_ids)
        if anchor_ids and selected_ids:
            diagnostics["distance"] = int(len(anchor_ids - selected_ids))
            diagnostics["is_anchor"] = float(selected_ids == anchor_ids)
        else:
            diagnostics["distance"] = -1
            diagnostics["is_anchor"] = 0.0
        return diagnostics

    def _update_basket_replay_metric_ema(
        self,
        metric: str,
        value: float,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not np.isfinite(value):
            return
        diagnostics = diagnostics or self._last_basket_replay_sample_stats
        if int(diagnostics.get("distinct_tasks", 1)) > 1:
            role = "mixed"
        else:
            role = "anchor" if float(diagnostics.get("is_anchor", 0.0)) > 0.5 else "transfer"
        key = f"{role}_{metric}"
        previous = self._basket_replay_metric_ema.get(key)
        alpha = self.basket_replay_diagnostic_ema_alpha
        self._basket_replay_metric_ema[key] = float(
            value
            if previous is None
            else (1.0 - alpha) * float(previous) + alpha * value
        )

    @staticmethod
    def _robust_candidate_standardize(values: torch.Tensor) -> torch.Tensor:
        """Column-wise robust scaling inside one candidate set."""
        median = values.median(dim=0, keepdim=True).values
        deviation = (values - median).abs()
        mad = deviation.median(dim=0, keepdim=True).values
        return ((values - median) / (1.4826 * mad + 1e-4)).clamp(-6.0, 6.0)

    @torch.no_grad()
    def _value_experience_target_representation(self, replay_data):
        """Return the slow, target-network-only replay coordinate.

        This is the only selector quantity which may survive a financial critic
        update.  It contains no online-critic Q value, TD error, advantage, or
        query ranking.  The replay action is immutable data and the encoder is
        the Polyak target critic, so reusing this coordinate does not freeze the
        selector's view of the changing live Bellman surface.
        """
        observations = {
            key: value.float()
            for key, value in replay_data.observations.items()
        }
        observations_g = self._gate_observations_with_sigma(observations)
        actions = replay_data.actions.float()
        # The target critic is the replay key encoder.  It is already a
        # Polyak-smoothed copy of the financial critic, so a candidate keeps a
        # stable FAM/AAM coordinate while the live critic is being updated.
        # This is the same target-encoder separation used by SPR, but the keys
        # remain value-facing state-action coordinates rather than raw features.
        representation = self.critic_target.value_experience_representation(
            observations_g, actions
        )
        representation = F.normalize(
            torch.nan_to_num(representation), dim=-1, eps=1e-6
        )
        return representation.detach()

    @torch.no_grad()
    def _value_experience_candidate_features(
        self,
        replay_data,
        *,
        target_representation: Optional[torch.Tensor] = None,
    ):
        """Build fresh selector evidence for the current financial critic.

        ``target_representation`` may be cached because it is produced only by
        the Polyak target encoder.  Every quantity below which depends on the
        online critic is recomputed on every selection.  Current and previous
        allocations share one action-independent state encode per critic twin;
        this preserves the exact state-action critic while avoiding the two
        redundant FAM/AAM passes used by the former diagnostic path.
        """
        observations = {
            key: value.float()
            for key, value in replay_data.observations.items()
        }
        next_observations = {
            key: value.float()
            for key, value in replay_data.next_observations.items()
        }
        observations_g = self._gate_observations_with_sigma(observations)
        next_observations_g = self._gate_observations_with_sigma(
            next_observations
        )
        actions = replay_data.actions.float()
        representation = (
            self._value_experience_target_representation(replay_data)
            if target_representation is None
            else target_representation.to(actions.device)
        )

        previous_raw_action = self._previous_portfolio_to_raw_action(
            observations_g,
            dtype=actions.dtype,
            device=actions.device,
        )
        candidate_evaluator = getattr(
            self.critic, "evaluate_action_candidates", None
        )
        if previous_raw_action is not None and callable(candidate_evaluator):
            # [row, {replay action, previous allocation}, action dimension].
            # The existing encoded-candidate path evaluates both allocations
            # after one live state encode per twin instead of encoding the same
            # FAM/AAM state twice.
            action_candidates = torch.stack(
                (actions, previous_raw_action), dim=1
            )
            candidate_q_values = candidate_evaluator(
                observations_g, action_candidates
            )
            stacked_q = torch.stack(
                [values[:, 0] for values in candidate_q_values], dim=1
            )
            previous_q = torch.stack(
                [values[:, 1] for values in candidate_q_values], dim=1
            ).min(dim=1).values
        else:
            q_values = self.critic(observations_g, actions)
            stacked_q = torch.cat(q_values, dim=1)
            previous_q = self._detached_previous_action_q(
                observations_g,
                actions,
                use_twin_minimum=True,
            )
        conservative_q = stacked_q.min(dim=1, keepdim=True).values
        twin_disagreement = stacked_q.std(
            dim=1, keepdim=True, unbiased=False
        )
        next_actions = self.actor_target(
            self._actor_observation(next_observations_g, use_no_grad=True)
        )
        next_q = torch.min(
            torch.cat(
                self.critic_target(next_observations_g, next_actions), dim=1
            ),
            dim=1,
            keepdim=True,
        ).values
        bellman_target = replay_data.rewards.float() + (
            1.0 - replay_data.dones.float()
        ) * self.gamma * next_q
        absolute_td = (conservative_q - bellman_target).abs()
        if previous_q is None:
            advantage = torch.zeros_like(conservative_q)
        else:
            advantage = conservative_q - previous_q.reshape(-1, 1)
        action_mask = _build_action_mask_from_observations(
            observations,
            action_dim=actions.shape[1],
            device=actions.device,
            dtype=actions.dtype,
        )
        current_weights = normalize_raw_actions_to_portfolio_weights(
            actions, action_mask
        )
        previous_weights = observations.get("prev_action")
        if previous_weights is None:
            action_delta = torch.zeros_like(conservative_q)
        else:
            action_delta = (
                (current_weights - previous_weights.float()) * action_mask
            ).abs().sum(dim=1, keepdim=True)
        diagnostics = torch.cat(
            (
                replay_data.rewards.float(),
                replay_data.dones.float(),
                conservative_q,
                bellman_target,
                absolute_td,
                twin_disagreement,
                advantage,
                action_delta,
            ),
            dim=1,
        )
        diagnostics = torch.nan_to_num(
            self._robust_candidate_standardize(diagnostics)
        )
        feature_vector = torch.nan_to_num(
            torch.cat((representation, diagnostics), dim=1)
        ).detach()
        model_dim = int(self.critic.q_networks[0].token_encoder.model_dim)
        q_encoder = self.critic.q_networks[0].token_encoder
        state_width = int(
            getattr(
                self,
                "value_experience_replay_state_width",
                2
                * (
                    int(q_encoder.fam_asset_vector_dim)
                    + int(q_encoder.model_dim)
                ),
            )
        )
        if representation.shape[1] <= state_width:
            raise RuntimeError(
                "value-experience representation lost its state-action boundary"
            )
        return (
            feature_vector,
            absolute_td.squeeze(-1).detach(),
            representation[:, :state_width].detach(),
            representation[:, state_width:].detach(),
        )

    def _value_experience_reservoir_is_valid(
        self,
        reservoir: Optional[Dict[str, Any]],
        buffer,
        candidate_size: int,
    ) -> bool:
        """Return whether a cached target-only replay coordinate is reusable.

        Selection, live Q diagnostics, TD error, advantage, query construction,
        and scorer evaluation happen on every Bellman update.  Only the slow,
        no-gradient Polyak-target representation is amortized.  A cache expires
        after a short interval, when its requested size changes, or when
        circular replay has overwritten a referenced physical row.
        """
        if not reservoir or int(reservoir["candidate_size"]) != int(candidate_size):
            return False
        age = int(self._n_updates) - int(reservoir["built_update"])
        if age < 0 or age >= self.value_experience_candidate_refresh_interval:
            return False
        candidate_indices = np.asarray(
            reservoir["candidate_indices"], dtype=np.int64
        )
        upper = buffer.buffer_size if buffer.full else buffer.pos
        if candidate_indices.size != int(candidate_size):
            return False
        if candidate_indices.size and (
            int(candidate_indices.min()) < 0
            or int(candidate_indices.max()) >= int(upper)
        ):
            return False
        if buffer.full:
            if (
                buffer.optimize_memory_usage
                and np.any(candidate_indices == int(buffer.pos))
            ):
                return False
            elapsed_writes = max(
                0,
                int(self.num_timesteps)
                - int(reservoir.get("built_num_timesteps", self.num_timesteps)),
            )
            if elapsed_writes >= int(buffer.buffer_size):
                return False
            if elapsed_writes:
                overwritten = (
                    int(reservoir["built_buffer_pos"])
                    + np.arange(elapsed_writes, dtype=np.int64)
                ) % int(buffer.buffer_size)
                if np.intersect1d(
                    candidate_indices, overwritten, assume_unique=False
                ).size:
                    return False
        return True

    def _value_experience_candidate_reservoir_for_selection(
        self,
        buffer,
        candidate_size: int,
        available: np.ndarray,
    ) -> Tuple[Dict[str, Any], bool, float, Optional[Any]]:
        """Return cached target coordinates plus an optional just-read batch."""
        reservoir = self._value_experience_candidate_reservoir
        if self._value_experience_reservoir_is_valid(
            reservoir, buffer, candidate_size
        ):
            return reservoir, True, 0.0, None

        refresh_start = time.perf_counter()
        candidate_indices = np.asarray(
            self._value_experience_rng.choice(
                available,
                size=int(candidate_size),
                replace=False,
            ),
            dtype=np.int64,
        )
        candidate_data = buffer._get_samples(
            candidate_indices, env=self._vec_normalize_env
        )
        target_representation = self._value_experience_target_representation(
            candidate_data
        )
        refresh_seconds = time.perf_counter() - refresh_start
        reservoir = {
            "candidate_size": int(candidate_size),
            "candidate_indices": candidate_indices,
            # This is intentionally the sole cached neural quantity.  Do not
            # add online Q values, Bellman errors, advantages, or query rows.
            "target_representation": target_representation.detach(),
            "built_update": int(self._n_updates),
            "built_num_timesteps": int(self.num_timesteps),
            "built_buffer_pos": int(buffer.pos),
        }
        self._value_experience_candidate_reservoir = reservoir
        self._value_experience_reservoir_refresh_count += 1
        self._value_experience_reservoir_refresh_seconds_total += float(
            refresh_seconds
        )
        # Reuse the batch already transferred from replay on a cache miss.  It
        # is returned only to the current selection and is never retained by
        # the reservoir, keeping the eight-update cache small.
        return reservoir, False, float(refresh_seconds), candidate_data

    def _sample_value_experience_replay(self, batch_size: int):
        """Select a query-relevant, action-contrastive mixed replay batch.

        A uniformly drawn held-out query remains outside the TD update.  It
        states the current value problem in FAM/AAM coordinates.  Utility rows
        are selected because they match that problem; contrast rows keep the
        same observable state neighborhood while changing the allocation; and
        a residual uniform role prevents either mechanism from narrowing the
        universal Bellman operator into a hidden basket partition.
        """
        buffer = self.replay_buffer
        if (
            self.value_experience_replay_scorer is None
            or getattr(buffer, "n_envs", 1) != 1
            or not hasattr(buffer, "_get_samples")
        ):
            return None
        upper = buffer.buffer_size if buffer.full else buffer.pos
        requested_candidate_size = int(math.ceil(
            int(batch_size) * self.value_experience_candidate_multiplier
        ))
        candidate_size = min(int(upper), requested_candidate_size)
        query_count = min(
            self.value_experience_query_size,
            max(0, candidate_size - int(batch_size)),
        )
        if candidate_size < int(batch_size) or query_count <= 0:
            return None
        if buffer.optimize_memory_usage and buffer.full:
            available = np.delete(np.arange(buffer.buffer_size), buffer.pos)
        else:
            available = np.arange(upper, dtype=np.int64)
        if available.size < candidate_size:
            return None
        (
            reservoir,
            reservoir_cache_hit,
            reservoir_refresh_seconds,
            candidate_data,
        ) = (
            self._value_experience_candidate_reservoir_for_selection(
                buffer, candidate_size, available
            )
        )
        candidate_indices = reservoir["candidate_indices"]
        if candidate_data is None:
            candidate_data = buffer._get_samples(
                candidate_indices, env=self._vec_normalize_env
            )
        live_refresh_start = time.perf_counter()
        (
            candidate_features,
            pre_td_error,
            candidate_state,
            candidate_action,
        ) = self._value_experience_candidate_features(
            candidate_data,
            target_representation=reservoir["target_representation"],
        )
        live_refresh_seconds = time.perf_counter() - live_refresh_start
        candidate_positions = np.arange(candidate_size, dtype=np.int64)
        # The query is held out from this TD update.  Its first half asks about
        # unresolved value regions (large current *absolute* Bellman error);
        # its second half is uniformly drawn so transient market noise cannot
        # define the whole learning problem.  This is a query construction,
        # not a raw-TD-error sampling rule for the final training batch.
        hard_query_count = int(round(
            query_count * self.value_experience_query_hard_fraction
        ))
        hard_query_count = min(query_count, max(0, hard_query_count))
        hard_order = torch.argsort(pre_td_error, descending=True).detach().cpu().numpy()
        hard_positions = hard_order[:hard_query_count].astype(np.int64, copy=False)
        remaining_for_query = candidate_positions[
            ~np.isin(candidate_positions, hard_positions, assume_unique=True)
        ]
        uniform_query_count = query_count - hard_query_count
        uniform_positions = np.asarray(
            self._value_experience_rng.choice(
                remaining_for_query,
                size=uniform_query_count,
                replace=False,
            ) if uniform_query_count > 0 else np.empty(0, dtype=np.int64),
            dtype=np.int64,
        )
        query_positions = np.concatenate((hard_positions, uniform_positions))
        selectable_mask = np.ones(candidate_size, dtype=bool)
        selectable_mask[query_positions] = False
        selectable_positions = candidate_positions[selectable_mask]
        selectable_tensor = torch.as_tensor(
            selectable_positions,
            dtype=torch.long,
            device=candidate_features.device,
        )
        query_tensor = torch.as_tensor(
            query_positions,
            dtype=torch.long,
            device=candidate_features.device,
        )
        query_features = candidate_features.index_select(0, query_tensor)
        selectable_features = candidate_features.index_select(
            0, selectable_tensor
        )
        with torch.no_grad():
            utility_logits, transfer_logits = self.value_experience_replay_scorer(
                selectable_features,
                query_features,
                return_transfer=True,
            )
            logits = torch.nan_to_num(utility_logits + (
                self.value_experience_transfer_score_weight * transfer_logits
            ))
            utility_probabilities = value_experience_sampling_probabilities(
                logits,
                temperature=self.value_experience_temperature,
                uniform_fraction=self.value_experience_uniform_fraction,
            )
            selectable_state = F.normalize(
                candidate_state.index_select(0, selectable_tensor), dim=-1, eps=1e-6
            )
            selectable_action = F.normalize(
                candidate_action.index_select(0, selectable_tensor), dim=-1, eps=1e-6
            )
            query_state = F.normalize(
                candidate_state.index_select(0, query_tensor), dim=-1, eps=1e-6
            )
            query_action = F.normalize(
                candidate_action.index_select(0, query_tensor), dim=-1, eps=1e-6
            )
            # Match every candidate to an actual query row.  This preserves
            # multimodal regimes and creates a true same-neighborhood /
            # different-action contrast instead of contrasting with the mean
            # of several unrelated states and allocations.
            state_similarity_matrix = (
                selectable_state @ query_state.transpose(0, 1)
            )
            action_difference_matrix = 1.0 - (
                selectable_action @ query_action.transpose(0, 1)
            )
            pair_scores = state_similarity_matrix + (
                self.value_experience_action_contrast_weight
                * action_difference_matrix
            )
            contrast_logits, matched_query = pair_scores.max(dim=1)
            row_index = torch.arange(
                selectable_state.shape[0], device=selectable_state.device
            )
            state_similarity = state_similarity_matrix[row_index, matched_query]
            action_difference = action_difference_matrix[row_index, matched_query]
            contrast_probabilities = value_experience_sampling_probabilities(
                contrast_logits,
                temperature=self.value_experience_temperature,
                uniform_fraction=self.value_experience_uniform_fraction,
            )
        contrast_fraction = min(
            self.value_experience_action_contrast_fraction, 1.0
        )
        utility_fraction = min(
            self.value_experience_query_utility_fraction,
            1.0 - contrast_fraction,
        )
        uniform_fraction = max(0.0, 1.0 - utility_fraction - contrast_fraction)
        role_counts = {
            "utility": int(round(int(batch_size) * utility_fraction)),
            "contrast": int(round(int(batch_size) * contrast_fraction)),
        }
        role_counts["uniform"] = int(batch_size) - sum(role_counts.values())
        # Integer rounding must not move the batch out of its requested size.
        while role_counts["uniform"] < 0:
            role = "utility" if role_counts["utility"] >= role_counts["contrast"] else "contrast"
            role_counts[role] -= 1
            role_counts["uniform"] += 1
        role_probabilities = {
            "utility": utility_probabilities.detach().cpu().numpy(),
            "contrast": contrast_probabilities.detach().cpu().numpy(),
            "uniform": np.full(
                selectable_positions.size,
                1.0 / float(selectable_positions.size),
                dtype=np.float64,
            ),
        }
        selected_local_positions = []
        selected_group_ids = []
        available_local = np.arange(selectable_positions.size, dtype=np.int64)
        for group_id, role in enumerate(("utility", "contrast", "uniform")):
            count = min(int(role_counts[role]), int(available_local.size))
            if count <= 0:
                continue
            role_prob = role_probabilities[role][available_local]
            role_prob = role_prob / role_prob.sum()
            chosen = np.asarray(
                self._value_experience_rng.choice(
                    available_local,
                    size=count,
                    replace=False,
                    p=role_prob,
                ),
                dtype=np.int64,
            )
            selected_local_positions.extend(chosen.tolist())
            selected_group_ids.extend([group_id] * int(chosen.size))
            available_local = available_local[
                ~np.isin(available_local, chosen, assume_unique=True)
            ]
        if len(selected_local_positions) != int(batch_size):
            return None
        selected_local_positions = np.asarray(selected_local_positions, dtype=np.int64)
        selected_group_ids = np.asarray(selected_group_ids, dtype=np.int64)
        # Downstream transfer measurement must never recover sampler-role blocks
        # from row order.  Shuffle rows and their diagnostic role ids together.
        row_permutation = self._value_experience_rng.permutation(
            selected_local_positions.size
        )
        selected_local_positions = selected_local_positions[row_permutation]
        selected_group_ids = selected_group_ids[row_permutation]
        selected_positions = selectable_positions[selected_local_positions]
        selected_replay_indices = candidate_indices[selected_positions]
        selected_data = buffer._get_samples(
            selected_replay_indices, env=self._vec_normalize_env
        )
        # Importance correction uses the explicit mixture law.  Sequential
        # without-replacement drawing only improves diversity; this mixture is
        # the conservative marginal approximation used for the TD row weight.
        mixed_probabilities = (
            utility_fraction * utility_probabilities
            + contrast_fraction * contrast_probabilities
            + uniform_fraction * torch.full_like(
                utility_probabilities,
                1.0 / float(selectable_positions.size),
            )
        )
        selected_probabilities = mixed_probabilities[
            torch.as_tensor(
                selected_local_positions,
                dtype=torch.long,
                device=mixed_probabilities.device,
            )
        ]
        importance = (
            float(selectable_positions.size) * selected_probabilities
        ).pow(-self.value_experience_importance_beta)
        importance = importance / importance.mean().clamp(min=1e-8)
        importance = importance.clamp(
            self.value_experience_importance_min,
            self.value_experience_importance_max,
        ).detach()
        self._value_experience_importance_weights = importance
        selected_tensor = torch.as_tensor(
            selected_positions,
            dtype=torch.long,
            device=candidate_features.device,
        )
        self._value_experience_last_selected_features = (
            candidate_features.index_select(0, selected_tensor).detach()
        )
        self._value_experience_last_pre_td_error = (
            pre_td_error.index_select(0, selected_tensor).detach()
        )
        self._value_experience_last_query_features = query_features.detach()
        remaining_after_selection = np.setdiff1d(
            np.arange(selectable_positions.size, dtype=np.int64),
            selected_local_positions,
            assume_unique=False,
        )
        if remaining_after_selection.size:
            self._value_experience_last_baseline_features = (
                selectable_features.index_select(
                    0,
                    torch.as_tensor(
                        remaining_after_selection,
                        dtype=torch.long,
                        device=selectable_features.device,
                    ),
                ).detach()
            )
        else:
            self._value_experience_last_baseline_features = selectable_features.detach()
        self._value_experience_last_query_pre_td_error = pre_td_error.index_select(
            0, query_tensor
        ).detach()
        self._value_experience_query_data = buffer._get_samples(
            candidate_indices[query_positions], env=self._vec_normalize_env
        )
        self._value_experience_last_selected_group_ids = torch.as_tensor(
            selected_group_ids,
            dtype=torch.long,
            device=candidate_features.device,
        )
        self._value_experience_last_transfer_targets = None
        self._value_experience_pending_summaries.append(
            {
                "selected": self._value_experience_last_selected_features.mean(
                    dim=0
                ).cpu(),
                "candidate": self._value_experience_last_baseline_features.mean(
                    dim=0
                ).cpu(),
                "query": query_features.mean(dim=0).cpu(),
            }
        )
        self._last_replay_batch_inds = selected_replay_indices
        self._value_experience_selection_count += 1
        diagnostics_builder = getattr(buffer, "basket_sample_diagnostics", None)
        if callable(diagnostics_builder):
            self._last_basket_replay_sample_stats = (
                self._annotate_basket_replay_stats(
                    diagnostics_builder(
                        self._last_replay_batch_inds, int(batch_size)
                    )
                )
            )
        entropy = float(
            (-(mixed_probabilities * mixed_probabilities.clamp(min=1e-12).log()).sum())
            .detach()
            .item()
        )
        self._last_value_experience_stats = {
            "candidate_count": float(candidate_size),
            "eligible_candidate_count": float(selectable_positions.size),
            "selected_candidate_fraction": float(batch_size)
            / float(selectable_positions.size),
            "query_count": float(query_count),
            "query_hard_fraction": float(hard_query_count) / float(query_count),
            "probability_entropy": entropy,
            "effective_candidate_count": float(math.exp(entropy)),
            "score_std": float(logits.detach().std(unbiased=False).item()),
            "transfer_score_std": float(
                transfer_logits.detach().std(unbiased=False).item()
            ),
            "utility_role_fraction": float(role_counts["utility"]) / float(batch_size),
            "contrast_role_fraction": float(role_counts["contrast"]) / float(batch_size),
            "uniform_role_fraction": float(role_counts["uniform"]) / float(batch_size),
            "query_state_similarity": float(state_similarity.mean().item()),
            "contrast_action_difference": float(action_difference.mean().item()),
            "importance_min": float(importance.min().item()),
            "importance_max": float(importance.max().item()),
            "reservoir_cache_hit": float(reservoir_cache_hit),
            "reservoir_refresh_seconds": float(reservoir_refresh_seconds),
            "live_feature_refresh_seconds": float(live_refresh_seconds),
            # Zero is the enforced freshness age: these quantities were built
            # from the critic which will receive this exact TD update.
            "live_feature_age_updates": 0.0,
            "reservoir_refresh_interval": float(
                self.value_experience_candidate_refresh_interval
            ),
            "reservoir_refresh_fraction": float(
                self._value_experience_reservoir_refresh_count
            ) / float(max(self._value_experience_selection_count, 1)),
            "reservoir_refresh_seconds_mean": float(
                self._value_experience_reservoir_refresh_seconds_total
            ) / float(max(self._value_experience_reservoir_refresh_count, 1)),
            "reservoir_refresh_seconds_amortized": float(
                self._value_experience_reservoir_refresh_seconds_total
            ) / float(max(self._value_experience_selection_count, 1)),
        }
        return selected_data

    def _value_experience_weight_rows(self, values: torch.Tensor) -> torch.Tensor:
        weights = self._value_experience_importance_weights
        if (
            weights is None
            or values.ndim == 0
            or values.shape[0] != weights.shape[0]
        ):
            return values
        shape = (weights.shape[0],) + (1,) * (values.ndim - 1)
        return values * weights.to(values.device, values.dtype).view(shape)

    def _value_experience_transfer_td_loss(self, replay_data) -> torch.Tensor:
        """One-Q financial Bellman loss used only for transfer measurement.

        This deliberately uses the existing reward and target critic.  It is
        not a new policy objective and does not touch CRL, ProExp, or the twin
        critic update.  Its sole question is MER's question: would a gradient
        from these candidate transitions reduce financial Bellman error on a
        different, held-out mixed-replay query?
        """
        observations = {
            key: value.float() for key, value in replay_data.observations.items()
        }
        next_observations = {
            key: value.float()
            for key, value in replay_data.next_observations.items()
        }
        observations_g = self._gate_observations_with_sigma(observations)
        next_observations_g = self._gate_observations_with_sigma(
            next_observations
        )
        with torch.no_grad():
            next_actions = self.actor_target(
                self._actor_observation(next_observations_g, use_no_grad=True)
            )
            next_q = torch.min(
                torch.cat(
                    self.critic_target(next_observations_g, next_actions), dim=1
                ),
                dim=1,
            ).values
            target = replay_data.rewards.float().squeeze(-1) + (
                1.0 - replay_data.dones.float().squeeze(-1)
            ) * self.gamma * next_q
        q1 = self.critic.q1_forward(
            observations_g, replay_data.actions.float()
        ).squeeze(-1)
        return F.smooth_l1_loss(q1, target, reduction="mean")

    def _value_experience_transfer_parameters(self):
        """The identity-neutral FAM/AAM boundary used for MER diagnostics."""
        encoder = self.critic.q_networks[0].token_encoder
        names = (
            "feature_attention",
            "fam_asset_embedding",
            "fam_to_model_projection",
            "transformer",
            "market_attention",
        )
        parameters = [
            parameter
            for name, parameter in encoder.named_parameters()
            if parameter.requires_grad and any(token in name for token in names)
        ]
        # FAM can be disabled in an architecture-ablation.  Its shared market
        # stem is then still the correct value-equivalence boundary.
        if not parameters:
            parameters = [
                parameter
                for parameter in encoder.parameters()
                if parameter.requires_grad
            ]
        return parameters

    @staticmethod
    def _gradient_cosine(
        left: Tuple[Optional[torch.Tensor], ...],
        right: Tuple[Optional[torch.Tensor], ...],
    ) -> float:
        dot = None
        left_norm = None
        right_norm = None
        for left_gradient, right_gradient in zip(left, right):
            if left_gradient is None or right_gradient is None:
                continue
            product = (left_gradient.detach() * right_gradient.detach()).sum()
            square_left = left_gradient.detach().square().sum()
            square_right = right_gradient.detach().square().sum()
            dot = product if dot is None else dot + product
            left_norm = square_left if left_norm is None else left_norm + square_left
            right_norm = square_right if right_norm is None else right_norm + square_right
        if dot is None or left_norm is None or right_norm is None:
            return 0.0
        return float(
            (dot / (left_norm.sqrt() * right_norm.sqrt() + 1e-12)).item()
        )

    def _value_experience_mer_constraint_ready(self) -> bool:
        """Require stable acceleration control before meta-adapting Q1.

        The Reptile displacement never directly mutates the actor or either
        dual variable.  It still changes the critic that will guide the next
        constrained actor step, so during CRL training it is allowed only when
        recent raw acceleration costs form the same robust threshold band used
        by the curriculum readiness gate.  It is intentionally not a demand
        for a zero violation or frozen multiplier.
        """
        if not bool(self.use_crl):
            return True
        history = list(getattr(self, "acceleration_constraint_history", ()))
        window = int(self.value_experience_mer_acceleration_window)
        if len(history) < window:
            return False
        costs = np.asarray(history[-window:], dtype=np.float64)
        if not np.all(np.isfinite(costs)):
            return False
        threshold = max(float(self.acceleration_constraint_threshold), 1e-8)
        center_ready = abs(float(np.mean(costs)) - threshold) <= (
            self.value_experience_mer_acceleration_center_tolerance * threshold
        )
        upper_ready = float(
            np.quantile(costs, self.value_experience_mer_acceleration_quantile)
        ) <= (
            1.0 + self.value_experience_mer_acceleration_quantile_tolerance
        ) * threshold
        exceedance = float(
            np.mean(
                costs
                > (
                    1.0
                    + self.value_experience_mer_acceleration_quantile_tolerance
                )
                * threshold
            )
        )
        return bool(
            center_ready
            and upper_ready
            and exceedance
            <= self.value_experience_mer_acceleration_max_exceedance_fraction
        )

    def _apply_pending_value_experience_mer(self) -> Optional[float]:
        """Merge only empirically supported group transfer before Adam.

        The held-out query is an evaluator, not an auxiliary training batch.
        It qualifies behavioral group gradients and supplies a projection
        reference, but its own hard-biased gradient is never added to Q1.  The
        accepted group mean must also agree with the ordinary TD gradient and
        pass raw-gradient plus predicted-Adam-step trust regions.
        """
        pending = self._value_experience_pending_mer
        self._value_experience_pending_mer = None
        if pending is None:
            return None
        self._last_value_experience_stats.update(
            mer_candidate_cosine=float(pending["cosine"]),
            mer_update_interval=float(self.value_experience_mer_update_interval),
            mer_update_due=float(
                self._n_updates % self.value_experience_mer_update_interval == 0
            ),
        )
        if self._n_updates % self.value_experience_mer_update_interval != 0:
            return None
        self._value_experience_mer_attempt_count += 1
        for rejection_name in (
            "mer_rejected_disabled",
            "mer_rejected_query_trend",
            "mer_rejected_crl_unready",
            "mer_rejected_transfer_threshold",
            "mer_rejected_parameter_mismatch",
            "mer_rejected_zero_gradient",
            "mer_rejected_adam_trust_region",
        ):
            self._last_value_experience_stats[rejection_name] = 0.0
        self._last_value_experience_stats.update(
            mer_attempted=1.0,
            mer_accepted=0.0,
            mer_accept_fraction=float(self._value_experience_mer_accept_count)
            / float(max(self._value_experience_mer_attempt_count, 1)),
        )
        if not self.value_experience_mer_enabled:
            self._last_value_experience_stats["mer_rejected_disabled"] = 1.0
            return None
        query_improvement_ema = self._value_experience_query_improvement_ema
        self._last_value_experience_stats[
            "mer_query_improvement_ema"
        ] = float(query_improvement_ema or 0.0)
        if (
            self.value_experience_mer_require_positive_query_improvement
            and (
                query_improvement_ema is None
                or query_improvement_ema
                <= self.value_experience_mer_min_query_improvement_ema
            )
        ):
            self._last_value_experience_stats[
                "mer_rejected_query_trend"
            ] = 1.0
            return None
        if not self._value_experience_mer_constraint_ready():
            self._last_value_experience_stats["mer_rejected_crl_unready"] = 1.0
            return None
        parameters = self._value_experience_transfer_parameters()
        query_gradients = pending["query_gradients"]
        all_group_gradients = pending["group_gradients"]
        group_cosines = tuple(
            pending.get(
                "group_cosines",
                (float(pending["cosine"]),) * len(all_group_gradients),
            )
        )
        qualified_groups = tuple(
            group
            for cosine, group in zip(group_cosines, all_group_gradients)
            if float(cosine) >= self.value_experience_mer_min_transfer_cosine
        )
        self._last_value_experience_stats.update(
            mer_qualified_group_count=float(len(qualified_groups)),
            mer_group_qualification_fraction=float(len(qualified_groups))
            / float(max(len(all_group_gradients), 1)),
        )
        if not qualified_groups:
            self._last_value_experience_stats[
                "mer_rejected_transfer_threshold"
            ] = 1.0
            return None
        if (
            len(parameters) != len(query_gradients)
            or any(len(parameters) != len(group) for group in qualified_groups)
        ):
            self._last_value_experience_stats["mer_rejected_parameter_mismatch"] = 1.0
            return None
        main_gradients = tuple(
            None if parameter.grad is None else parameter.grad.detach()
            for parameter in parameters
        )
        mean_group_gradients = []
        for parameter_index, parameter in enumerate(parameters):
            if (
                query_gradients[parameter_index] is None
                or parameter.grad is None
            ):
                mean_group_gradients.append(None)
                continue
            parts = [
                group[parameter_index].to(parameter.device, parameter.dtype)
                for group in qualified_groups
                if group[parameter_index] is not None
            ]
            mean_group_gradients.append(
                None if not parts else torch.stack(parts, dim=0).mean(dim=0)
            )
        meta_gradients = transfer_compatible_gradient_sequence(
            tuple(mean_group_gradients), tuple(query_gradients)
        )
        meta_gradients = transfer_compatible_gradient_sequence(
            meta_gradients, main_gradients
        )
        main_parts = [
            main.reshape(-1)
            for main, meta in zip(main_gradients, meta_gradients)
            if main is not None and meta is not None
        ]
        meta_parts = [
            meta.detach().reshape(-1)
            for main, meta in zip(main_gradients, meta_gradients)
            if main is not None and meta is not None
        ]
        if not meta_parts:
            self._last_value_experience_stats["mer_rejected_zero_gradient"] = 1.0
            return None
        main_norm = torch.cat(main_parts).norm()
        meta_norm = torch.cat(meta_parts).norm()
        if (
            not torch.isfinite(main_norm)
            or not torch.isfinite(meta_norm)
            or float(meta_norm.item()) <= 0.0
        ):
            self._last_value_experience_stats["mer_rejected_zero_gradient"] = 1.0
            return None
        ratio_cap = self.value_experience_mer_max_main_gradient_ratio
        norm_scale = ratio_cap * main_norm / meta_norm.clamp_min(1e-12)
        requested_scale = min(
            self.value_experience_mer_meta_step_size,
            float(norm_scale.detach().item()),
        )
        if not np.isfinite(requested_scale) or requested_scale <= 0.0:
            self._last_value_experience_stats["mer_rejected_zero_gradient"] = 1.0
            return None
        scale, adam_step_ratio, relative_parameter_step = (
            optimizer_aware_meta_gradient_scale(
                self.critic.optimizer,
                tuple(parameters),
                main_gradients,
                meta_gradients,
                requested_scale,
                max_adam_step_ratio=(
                    self.value_experience_mer_max_adam_step_ratio
                ),
                max_relative_parameter_step=(
                    self.value_experience_mer_max_relative_parameter_step
                ),
            )
        )
        if not np.isfinite(scale) or scale <= 0.0:
            self._last_value_experience_stats[
                "mer_rejected_adam_trust_region"
            ] = 1.0
            return None
        with torch.no_grad():
            for parameter, meta_gradient in zip(parameters, meta_gradients):
                if parameter.grad is not None and meta_gradient is not None:
                    parameter.grad.add_(meta_gradient, alpha=scale)
        applied_norm = float(meta_norm.detach().item()) * scale
        applied_ratio = applied_norm / max(float(main_norm.detach().item()), 1e-12)
        self._value_experience_mer_accept_count += 1
        self._value_experience_mer_applied_gradient_norm_total += applied_norm
        self._last_value_experience_stats.update(
            mer_meta_gradient_norm=float(meta_norm.detach().item()),
            mer_main_gradient_norm=float(main_norm.detach().item()),
            mer_applied_gradient_norm=applied_norm,
            mer_applied_gradient_ratio=applied_ratio,
            mer_requested_scale=float(requested_scale),
            mer_applied_scale=float(scale),
            mer_adam_step_ratio=float(adam_step_ratio),
            mer_relative_parameter_step=float(relative_parameter_step),
            mer_accepted=1.0,
            mer_accept_fraction=float(self._value_experience_mer_accept_count)
            / float(max(self._value_experience_mer_attempt_count, 1)),
            mer_applied_gradient_norm_mean=float(
                self._value_experience_mer_applied_gradient_norm_total
            )
            / float(max(self._value_experience_mer_accept_count, 1)),
        )
        return applied_norm

    def _measure_value_experience_transfer(self, replay_data) -> Optional[float]:
        """Label replay groups and retain one candidate for a safe MER trial.

        Full per-row meta-gradients would recreate the August-24 throughput
        failure. Instead, every configured interval, a small number of
        query-conditioned behavioral neighborhoods from the actual TD batch
        are compared with one probe in the shared FAM/AAM parameter subspace.
        Positive cosine means that a
        candidate group's financial update supports the held-out problem;
        negative cosine is interference.  Every cosine supervises the replay
        scorer.  Detached group/query gradients and their individual cosines
        are retained as a pending proposal.  The merge path later admits only
        groups above the transfer threshold and can act only when realized
        held-out improvement, cadence, CRL, and optimizer trust gates all pass.
        """
        if (
            not self.value_experience_transfer_enabled
            or self._n_updates % self.value_experience_transfer_interval != 0
            or self._value_experience_query_data is None
            or self._value_experience_last_selected_features is None
        ):
            return None
        parameters = self._value_experience_transfer_parameters()
        if not parameters:
            return None
        query_data = self._value_experience_query_data
        probe_size = min(
            int(query_data.actions.shape[0]),
            self.value_experience_transfer_probe_size,
        )
        if probe_size <= 0:
            return None
        if probe_size < int(query_data.actions.shape[0]):
            probe_data = self._slice_replay_data(
                query_data, np.arange(probe_size, dtype=np.int64)
            )
        else:
            probe_data = query_data
        probe_loss = self._value_experience_transfer_td_loss(probe_data)
        probe_gradient = torch.autograd.grad(
            probe_loss,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        row_count = int(replay_data.actions.shape[0])
        group_rows = build_query_conditioned_transfer_groups(
            self._value_experience_last_selected_features,
            self._value_experience_last_query_features,
            self.value_experience_transfer_groups,
        )
        targets = torch.zeros(
            row_count,
            dtype=torch.float32,
            device=self.device,
        )
        cosines = []
        detached_group_gradients = []
        for rows in group_rows:
            if rows.size == 0:
                continue
            group_data = self._slice_replay_data(replay_data, rows)
            group_loss = self._value_experience_transfer_td_loss(group_data)
            group_gradient = torch.autograd.grad(
                group_loss,
                parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            cosine = self._gradient_cosine(group_gradient, probe_gradient)
            targets[torch.as_tensor(rows, device=self.device)] = cosine
            cosines.append(cosine)
            detached_group_gradients.append(
                tuple(
                    None if gradient is None else gradient.detach().clone()
                    for gradient in group_gradient
                )
            )
        if not cosines:
            return None
        self._value_experience_last_transfer_targets = targets.detach()
        mean_cosine = float(np.mean(cosines))
        self._last_value_experience_stats.update(
            transfer_gradient_cosine=mean_cosine,
            transfer_positive_fraction=float(np.mean(np.asarray(cosines) > 0.0)),
            transfer_group_count=float(len(cosines)),
        )
        self._value_experience_pending_mer = (
            {
                "cosine": float(max(cosines)),
                "group_cosines": tuple(float(value) for value in cosines),
                "group_gradients": tuple(detached_group_gradients),
                "query_gradients": tuple(
                    None
                    if gradient is None
                    else gradient.detach().clone()
                    for gradient in probe_gradient
                ),
            }
            if self.value_experience_mer_enabled
            else None
        )
        self._last_value_experience_stats["mer_eligible"] = float(
            self._value_experience_pending_mer is not None
        )
        return mean_cosine

    def _train_value_experience_from_bellman(self, replay_data) -> Optional[float]:
        features = self._value_experience_last_selected_features
        baseline_features = self._value_experience_last_baseline_features
        old_error = self._value_experience_last_pre_td_error
        query_features = self._value_experience_last_query_features
        if (
            self.value_experience_replay_scorer is None
            or features is None
            or old_error is None
            or query_features is None
        ):
            return None
        with torch.no_grad():
            observations = {
                key: value.float()
                for key, value in replay_data.observations.items()
            }
            next_observations = {
                key: value.float()
                for key, value in replay_data.next_observations.items()
            }
            observations_g = self._gate_observations_with_sigma(observations)
            next_observations_g = self._gate_observations_with_sigma(
                next_observations
            )
            actions = replay_data.actions.float()
            new_q = torch.min(
                torch.cat(self.critic(observations_g, actions), dim=1),
                dim=1,
            ).values
            next_actions = self.actor_target(
                self._actor_observation(next_observations_g, use_no_grad=True)
            )
            next_q = torch.min(
                torch.cat(
                    self.critic_target(next_observations_g, next_actions), dim=1
                ),
                dim=1,
            ).values
            target = replay_data.rewards.float().squeeze(-1) + (
                1.0 - replay_data.dones.float().squeeze(-1)
            ) * self.gamma * next_q
            new_error = (new_q - target).abs()
            improvement = old_error.to(new_error.device) - new_error
            scale = improvement.abs().median().clamp(min=1e-6)
            target_utility = (improvement / scale).clamp(-3.0, 3.0)
        predictions, transfer_predictions = self.value_experience_replay_scorer(
            features,
            query_features,
            return_transfer=True,
        )
        loss = F.smooth_l1_loss(predictions, target_utility.detach())
        loss = loss * float(
            getattr(
                config,
                "VALUE_EXPERIENCE_REPLAY_BELLMAN_FEEDBACK_WEIGHT",
                0.25,
            )
        )
        transfer_targets = self._value_experience_last_transfer_targets
        if (
            transfer_targets is not None
            and transfer_targets.shape[0] == transfer_predictions.shape[0]
        ):
            transfer_loss = F.smooth_l1_loss(
                transfer_predictions,
                transfer_targets.to(
                    transfer_predictions.device, transfer_predictions.dtype
                ),
            ) * self.value_experience_transfer_feedback_weight
            loss = loss + transfer_loss
            self._last_value_experience_stats["transfer_feedback_loss"] = float(
                transfer_loss.detach().item()
            )
        query_data = self._value_experience_query_data
        query_old_error = self._value_experience_last_query_pre_td_error
        if query_data is not None and query_old_error is not None:
            with torch.no_grad():
                _, query_new_error, _, _ = self._value_experience_candidate_features(
                    query_data
                )
                probe_improvement = (
                    query_old_error.to(query_new_error.device).mean()
                    - query_new_error.mean()
                )
            self._last_value_experience_stats["query_probe_td_improvement"] = float(
                probe_improvement.item()
            )
            if baseline_features is not None:
                old_query_scale = query_old_error.to(
                    probe_improvement.device
                ).abs().mean().clamp(min=1e-6)
                relative_improvement = (
                    probe_improvement / old_query_scale
                ).clamp(-3.0, 3.0)
                relative_improvement_value = float(
                    relative_improvement.detach().item()
                )
                if self._value_experience_query_improvement_ema is None:
                    self._value_experience_query_improvement_ema = (
                        relative_improvement_value
                    )
                else:
                    decay = self.value_experience_mer_query_improvement_ema_decay
                    self._value_experience_query_improvement_ema = (
                        decay * self._value_experience_query_improvement_ema
                        + (1.0 - decay) * relative_improvement_value
                    )
                selected_preference_score = self.value_experience_replay_scorer(
                    features, query_features
                ).mean()
                baseline_preference_score = self.value_experience_replay_scorer(
                    baseline_features, query_features
                ).mean()
                preference = (
                    selected_preference_score - baseline_preference_score
                )
                direction = torch.sign(relative_improvement.detach())
                query_preference_loss = (
                    F.softplus(-direction * preference)
                    * relative_improvement.detach().abs()
                    * float(
                        getattr(
                            config,
                            "VALUE_EXPERIENCE_REPLAY_QUERY_FEEDBACK_WEIGHT",
                            1.0,
                        )
                    )
                )
                loss = loss + query_preference_loss
                self._last_value_experience_stats.update(
                    query_relative_improvement=relative_improvement_value,
                    query_relative_improvement_ema=float(
                        self._value_experience_query_improvement_ema
                    ),
                    query_preference=float(preference.detach().item()),
                    query_preference_loss=float(
                        query_preference_loss.detach().item()
                    ),
                )
        self.value_experience_replay_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.value_experience_replay_scorer.parameters(), 1.0
        )
        self.value_experience_replay_optimizer.step()
        return float(loss.detach().item())

    def update_value_experience_replay_from_evaluation(
        self,
        financial_return: float,
        opportunity_normalized_return: float,
        evaluation_strata: Optional[List[Dict[str, float]]] = None,
    ) -> Optional[float]:
        """Ground the replay policy in measured curriculum-matched outcomes.

        Bellman improvement supplies dense local feedback.  This delayed update
        supplies the missing outer-loop fact from ERO/NERS: did the replay
        choices made since the last evaluation improve actual financial
        performance on the task distribution currently being taught?
        """
        financial_return = float(financial_return)
        opportunity_normalized_return = float(opportunity_normalized_return)
        current_by_stratum: Dict[str, Tuple[float, float]] = {}
        if evaluation_strata:
            grouped: Dict[str, List[Tuple[float, float]]] = collections.defaultdict(list)
            for row in evaluation_strata:
                key = str(row.get("stratum", "unknown"))
                grouped[key].append(
                    (
                        float(row["financial_return"]),
                        float(row["opportunity_normalized_return"]),
                    )
                )
            current_by_stratum = {
                key: (
                    float(np.mean([item[0] for item in values])),
                    float(np.mean([item[1] for item in values])),
                )
                for key, values in grouped.items()
            }
        if self._value_experience_last_eval_financial is None:
            self._value_experience_last_eval_financial = financial_return
            self._value_experience_last_eval_normalized = (
                opportunity_normalized_return
            )
            self._value_experience_last_eval_by_stratum = current_by_stratum
            self._value_experience_pending_summaries.clear()
            return None
        matched = sorted(
            set(current_by_stratum).intersection(
                self._value_experience_last_eval_by_stratum
            )
        )
        if matched:
            stratum_feedback = []
            for key in matched:
                current_financial, current_normalized = current_by_stratum[key]
                previous_financial, previous_normalized = (
                    self._value_experience_last_eval_by_stratum[key]
                )
                stratum_feedback.append(
                    0.5
                    * (
                        (current_financial - previous_financial)
                        / max(abs(previous_financial), 0.005)
                        + (current_normalized - previous_normalized)
                        / max(abs(previous_normalized), 0.05)
                    )
                )
            feedback = float(np.mean(stratum_feedback))
        else:
            financial_scale = max(
                abs(float(self._value_experience_last_eval_financial)), 0.005
            )
            normalized_scale = max(
                abs(float(self._value_experience_last_eval_normalized)), 0.05
            )
            feedback = 0.5 * (
                (financial_return - self._value_experience_last_eval_financial)
                / financial_scale
                + (
                    opportunity_normalized_return
                    - self._value_experience_last_eval_normalized
                )
                / normalized_scale
            )
        clip = float(
            getattr(config, "VALUE_EXPERIENCE_REPLAY_FEEDBACK_CLIP", 3.0)
        )
        feedback = float(np.clip(feedback, -clip, clip))
        advantage = feedback - float(self._value_experience_feedback_baseline)
        self._value_experience_feedback_baseline = (
            0.9 * float(self._value_experience_feedback_baseline)
            + 0.1 * feedback
        )
        self._value_experience_last_eval_financial = financial_return
        self._value_experience_last_eval_normalized = (
            opportunity_normalized_return
        )
        self._value_experience_last_eval_by_stratum = current_by_stratum
        if not self._value_experience_pending_summaries:
            return None
        selected = torch.stack(
            [item["selected"] for item in self._value_experience_pending_summaries]
        ).to(self.device)
        candidates = torch.stack(
            [item["candidate"] for item in self._value_experience_pending_summaries]
        ).to(self.device)
        queries = torch.stack(
            [item.get("query", item["candidate"])
             for item in self._value_experience_pending_summaries]
        ).to(self.device)
        selected_scores = self.value_experience_replay_scorer(selected, queries)
        candidate_scores = self.value_experience_replay_scorer(candidates, queries)
        preference = selected_scores - candidate_scores
        direction = preference.new_tensor(float(np.sign(advantage)))
        magnitude = min(abs(float(advantage)), clip)
        loss = F.softplus(-direction * preference).mean() * magnitude
        loss = loss * float(
            getattr(
                config,
                "VALUE_EXPERIENCE_REPLAY_EXTERNAL_FEEDBACK_WEIGHT",
                1.0,
            )
        )
        self.value_experience_replay_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.value_experience_replay_scorer.parameters(), 1.0
        )
        self.value_experience_replay_optimizer.step()
        self._value_experience_pending_summaries.clear()
        self._last_value_experience_stats.update(
            external_feedback=feedback,
            external_advantage=float(advantage),
            external_loss=float(loss.detach().item()),
            external_matched_strata=float(len(matched)),
        )
        return float(loss.detach().item())

    def _sample_replay_data(
        self,
        batch_size: int,
        *,
        baskets_per_batch: Optional[int] = None,
        group_selection: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        evidence_gated: bool = True,
        track_diagnostics: bool = True,
        uniform_mix: float = 0.0,
    ):
        """Sample financial role mixtures or predictive basket blocks."""
        if evidence_gated:
            self._value_experience_importance_weights = None
            self._value_experience_last_selected_features = None
            self._value_experience_last_pre_td_error = None
            self._value_experience_last_query_features = None
            self._value_experience_last_baseline_features = None
            self._value_experience_last_query_pre_td_error = None
            self._value_experience_query_data = None
            self._value_experience_last_selected_group_ids = None
            self._value_experience_last_transfer_targets = None
        buf = self.replay_buffer
        env = self._vec_normalize_env
        resolved_baskets = (
            max(1, int(getattr(config, "BASKET_REPLAY_BASKETS_PER_BATCH", 8)))
            if baskets_per_batch is None
            else max(1, int(baskets_per_batch))
        )
        resolved_group_selection = (
            self.basket_replay_group_selection
            if group_selection is None
            else str(group_selection).strip().lower()
        )
        teacher_probabilities = getattr(
            self, "_basket_replay_teacher_probabilities", None
        ) or getattr(self, "_basket_replay_task_probabilities", {})
        mixed_replay_enabled = bool(
            getattr(
                self,
                "curriculum_mixed_replay_enabled",
                getattr(config, "CURRICULUM_MIXED_REPLAY_ENABLED", False),
            )
        )
        replay_task_probabilities = (
            self._role_preserving_financial_replay_probabilities()
            if evidence_gated
            and not mixed_replay_enabled
            and resolved_group_selection in {"teacher", "teacher_uniform"}
            else teacher_probabilities
        )
        if evidence_gated:
            self._basket_replay_task_probabilities = dict(
                replay_task_probabilities
            )
        resolved_rng = self._basket_replay_rng if rng is None else rng
        if not self.basket_replay_stratify_enabled:
            if (
                evidence_gated
                and bool(getattr(self, "value_experience_replay_enabled", False))
                and (
                    int(getattr(self, "_n_updates", 0)) <= 1
                    or int(getattr(self, "_n_updates", 0))
                    % self.value_experience_selection_interval
                    == 0
                )
            ):
                value_selection_start = time.perf_counter()
                selected = self._sample_value_experience_replay(batch_size)
                value_selection_seconds = (
                    time.perf_counter() - value_selection_start
                )
                self._last_value_experience_stats["selection_seconds"] = float(
                    value_selection_seconds
                )
                if selected is not None:
                    self._value_experience_selection_seconds_total = float(
                        getattr(
                            self,
                            "_value_experience_selection_seconds_total",
                            0.0,
                        )
                    ) + float(value_selection_seconds)
                    selection_count = max(
                        int(
                            getattr(
                                self,
                                "_value_experience_selection_count",
                                1,
                            )
                        ),
                        1,
                    )
                    selection_seconds_mean = float(
                        self._value_experience_selection_seconds_total
                    ) / float(selection_count)
                    self._last_value_experience_stats.update(
                        selection_seconds_mean=selection_seconds_mean,
                        # Retain the old telemetry name for log consumers.  A
                        # cumulative mean is stable even though each new
                        # selection replaces the per-selection stats mapping.
                        selection_seconds_ema=selection_seconds_mean,
                    )
                    self._last_value_experience_stats.update(
                        selection_applied=1.0,
                        selection_interval=float(
                            self.value_experience_selection_interval
                        ),
                    )
                    return selected
            if evidence_gated and bool(
                getattr(self, "value_experience_replay_enabled", False)
            ):
                self._last_value_experience_stats.update(
                    selection_applied=0.0,
                    selection_interval=float(
                        self.value_experience_selection_interval
                    ),
                )
            sample_with_indices = getattr(buf, "sample_with_batch_inds", None)
            if callable(sample_with_indices):
                replay_data, batch_inds = sample_with_indices(
                    batch_size, env=env
                )
                if track_diagnostics:
                    self._last_replay_batch_inds = np.asarray(
                        batch_inds, dtype=np.int64
                    )
                    diagnostics_builder = getattr(
                        buf, "basket_sample_diagnostics", None
                    )
                    if callable(diagnostics_builder):
                        self._last_basket_replay_sample_stats = (
                            self._annotate_basket_replay_stats(
                                diagnostics_builder(batch_inds, batch_size)
                            )
                        )
                return replay_data
            if track_diagnostics:
                self._last_replay_batch_inds = None
            return buf.sample(batch_size, env=env)
        if getattr(buf, "n_envs", 1) != 1:
            if track_diagnostics:
                self._last_replay_batch_inds = None
            return buf.sample(batch_size, env=env)
        if evidence_gated and mixed_replay_enabled:
            mixed_sampler = getattr(
                buf, "sample_curriculum_mixed_batch_inds", None
            )
            # Generic SB3 smoke environments and ablations that disable the
            # component replay buffer have no curriculum provenance. They
            # retain ordinary replay rather than fabricating roles. Production
            # Path 2 training always has the typed replay buffer and teacher
            # role law installed before learning starts.
            role_probabilities = getattr(
                self, "_basket_replay_role_probabilities", {}
            )
            if not role_probabilities and replay_task_probabilities:
                # Compatibility for callers that install only a flat teacher
                # law. This is still one mixed universal-MDP update; ``teacher``
                # is a provenance-neutral role, not an exact-basket minibatch.
                role_probabilities = {
                    "teacher": dict(replay_task_probabilities)
                }
            indexed_positions = getattr(buf, "_basket_positions_by_key", {})
            if (
                not callable(mixed_sampler)
                or not role_probabilities
                or not indexed_positions
            ):
                if track_diagnostics:
                    self._last_replay_batch_inds = None
                return buf.sample(batch_size, env=env)
            try:
                batch_inds, mixed_diagnostics = mixed_sampler(
                    batch_size=batch_size,
                    role_task_probabilities=role_probabilities,
                    rng=resolved_rng,
                    min_unique_transitions_per_role=(
                        getattr(
                            self,
                            "curriculum_mixed_replay_min_unique_transitions_per_role",
                            int(
                                getattr(
                                    config,
                                    "CURRICULUM_MIXED_REPLAY_MIN_UNIQUE_TRANSITIONS_PER_ROLE",
                                    1,
                                )
                            ),
                        )
                    ),
                    target_distinct_tasks_per_role=(
                        getattr(
                            self,
                            "curriculum_mixed_replay_target_distinct_tasks_per_role",
                            int(
                                getattr(
                                    config,
                                    "CURRICULUM_MIXED_REPLAY_TARGET_DISTINCT_TASKS_PER_ROLE",
                                    1,
                                )
                            ),
                        )
                    ),
                    max_basket_fraction=(
                        getattr(
                            self,
                            "curriculum_mixed_replay_max_basket_fraction",
                            float(
                                getattr(
                                    config,
                                    "CURRICULUM_MIXED_REPLAY_MAX_BASKET_FRACTION",
                                    1.0,
                                )
                            ),
                        )
                    ),
                )
                if track_diagnostics:
                    self._last_replay_batch_inds = np.asarray(
                        batch_inds, dtype=np.int64
                    )
            except BasketReplayEvidenceUnavailable as error:
                raise RuntimeError(
                    "Financial Bellman replay has no unique curriculum rows. "
                    "Increase learning_starts or verify replay metadata."
                ) from error
            if track_diagnostics:
                basket_diagnostics = getattr(buf, "basket_sample_diagnostics")(
                    batch_inds, batch_size
                )
                basket_diagnostics.update(mixed_diagnostics)
                role_counts = dict(mixed_diagnostics.get("role_counts", {}))
                anchor_rows = int(role_counts.get("anchor", 0))
                basket_diagnostics["is_anchor"] = float(
                    anchor_rows / max(int(batch_inds.size), 1)
                )
                basket_diagnostics["anchor_fraction"] = float(
                    anchor_rows / max(int(batch_inds.size), 1)
                )
                basket_diagnostics["distance"] = -1
                self._last_basket_replay_sample_stats = basket_diagnostics
                self._basket_replay_financial_anchor_share = float(
                    basket_diagnostics["anchor_fraction"]
                )
            return buf._get_samples(batch_inds, env=env)
        # Reuse the incremental position index; balanced replay must not scan
        # the entire replay membership array on every financial update.
        indexed_sampler = getattr(buf, "sample_hierarchical_batch_inds", None)
        if callable(indexed_sampler):
            try:
                batch_inds = indexed_sampler(
                    batch_size=batch_size,
                    baskets_per_batch=resolved_baskets,
                    rng=resolved_rng,
                    sample_without_replacement=(
                        self.basket_replay_sample_without_replacement
                    ),
                    group_selection=resolved_group_selection,
                    task_probabilities=(
                        replay_task_probabilities
                        if resolved_group_selection
                        in {"teacher", "teacher_uniform"}
                        else None
                    ),
                    min_completed_episodes=(
                        self.basket_replay_min_completed_episodes
                        if evidence_gated
                        else 0
                    ),
                    uniform_mix=uniform_mix,
                    allow_partial_unique_batch=(
                        self.basket_replay_allow_partial_unique_batch
                    ),
                )
                if track_diagnostics:
                    self._last_replay_batch_inds = np.asarray(
                        batch_inds, dtype=np.int64
                    )
                    diagnostics_builder = getattr(
                        buf, "basket_sample_diagnostics", None
                    )
                    diagnostics = (
                        diagnostics_builder(batch_inds, batch_size)
                        if callable(diagnostics_builder)
                        else {}
                    )
                    self._last_basket_replay_sample_stats = (
                        self._annotate_basket_replay_stats(diagnostics)
                    )
                return buf._get_samples(batch_inds, env=env)
            except BasketReplayEvidenceUnavailable as error:
                if evidence_gated:
                    raise RuntimeError(
                        "Financial Bellman replay has no completed basket "
                        "trajectory. Increase learning_starts or verify terminal "
                        "episode metadata."
                    ) from error
                # Prediction can still learn from ordinary transitions in a
                # generic environment that does not expose curriculum keys.
                return buf.sample(batch_size, env=env)
        observations = getattr(buf, "observations", None)
        if not isinstance(observations, dict) or "basket_membership" not in observations:
            return buf.sample(batch_size, env=env)
        upper = buf.buffer_size if buf.full else buf.pos
        if upper < 1:
            return buf.sample(batch_size, env=env)
        mem = np.asarray(observations["basket_membership"][:upper])
        if mem.ndim == 3:
            if mem.shape[1] != 1:
                # Multi-env stratification needs joint (pos, env) keys; fall back.
                return buf.sample(batch_size, env=env)
            mem2d = mem[:, 0, :]
        elif mem.ndim == 2:
            mem2d = mem
        else:
            return buf.sample(batch_size, env=env)
        available_positions = np.arange(upper, dtype=np.int64)
        completed_by_key: Dict[bytes, int] = {}
        if evidence_gated:
            dones = np.asarray(buf.dones[:upper])
            if dones.ndim == 2:
                done_rows = dones[:, 0]
            else:
                done_rows = dones.reshape(-1)
            row_keys = [_membership_key(row) for row in mem2d]
            for key, done_value in zip(row_keys, done_rows):
                if bool(done_value):
                    completed_by_key[key] = int(
                        completed_by_key.get(key, 0)
                    ) + 1
            eligible_positions = np.asarray(
                [
                    position
                    for position, key in enumerate(row_keys)
                    if int(completed_by_key.get(key, 0))
                    >= self.basket_replay_min_completed_episodes
                ],
                dtype=np.int64,
            )
            if eligible_positions.size == 0:
                raise RuntimeError(
                    "Financial Bellman replay has no completed basket "
                    "trajectory. Increase learning_starts or verify terminal "
                    "episode metadata."
                )
            available_positions = eligible_positions
        sampling_memberships = mem2d[available_positions]
        local_batch_inds = sample_basket_hierarchical_batch_inds(
            sampling_memberships,
            batch_size=batch_size,
            baskets_per_batch=resolved_baskets,
            rng=resolved_rng,
            sample_without_replacement=(
                self.basket_replay_sample_without_replacement
            ),
            group_selection=resolved_group_selection,
            task_probabilities=(
                replay_task_probabilities
                if resolved_group_selection in {"teacher", "teacher_uniform"}
                else None
            ),
            uniform_mix=uniform_mix,
            allow_partial_unique_batch=(
                self.basket_replay_allow_partial_unique_batch
            ),
        )
        batch_inds = available_positions[local_batch_inds]
        if track_diagnostics:
            selected_row = mem2d[int(batch_inds[0])]
            selected_key = _membership_key(selected_row)
            selected_membership_ids = tuple(
                int(index) for index in np.flatnonzero(selected_row > 0.5)
            )
            all_row_keys = [_membership_key(row) for row in mem2d]
            stored_rows = sum(key == selected_key for key in all_row_keys)
            sampled_rows = int(len(batch_inds))
            unique_rows = int(np.unique(batch_inds).size)
            diagnostics = {
                "membership_ids": selected_membership_ids,
                "task_fingerprint": int.from_bytes(
                    selected_key[:4], "little"
                ),
                "stored_rows": int(stored_rows),
                "sampled_rows": sampled_rows,
                "unique_rows": unique_rows,
                "completed_episodes": int(
                    completed_by_key.get(selected_key, 0)
                ),
                "full_batch_ready": float(stored_rows >= batch_size),
                "partial_unique_batch": float(sampled_rows < batch_size),
                "unique_fraction": float(
                    unique_rows / max(sampled_rows, 1)
                ),
                "requested_fraction": float(
                    sampled_rows / max(int(batch_size), 1)
                ),
            }
            self._last_basket_replay_sample_stats = (
                self._annotate_basket_replay_stats(diagnostics)
            )
        return buf._get_samples(batch_inds, env=env)

    def _cross_basket_name_rank(self, device: torch.device) -> torch.Tensor:
        """Rank of each global slot's ticker in the environment's row order."""
        rank = getattr(self, "_cross_basket_relabel_name_rank", None)
        if rank is None or rank.device != torch.device(device):
            tickers = list(getattr(config, "COIN_POOL", ()))
            ordering = sorted(range(len(tickers)), key=lambda slot: tickers[slot])
            values = np.empty(len(tickers), dtype=np.int64)
            values[np.asarray(ordering, dtype=np.int64)] = np.arange(len(tickers))
            rank = torch.as_tensor(values, device=device)
            self._cross_basket_relabel_name_rank = rank
        return rank

    def _apply_cross_basket_relabeling(
        self,
        replay_data,
        stats_sink: Optional[List[Dict[str, float]]] = None,
    ):
        """Replace the tail of a TD3 batch with same-minute relabeled copies.

        Row ``j`` of the first ``R`` rows is the source of relabeled row
        ``B - R + j``, so each relabel sits beside the real row it came from.
        Rows whose source lacks relabel metadata keep their original sample.
        """
        self._last_cross_basket_relabel_source_index = None
        self._last_cross_basket_relabel_target_index = None
        if not getattr(self, "cross_basket_relabel_enabled", False):
            return replay_data
        if getattr(self, "value_experience_replay_enabled", False):
            # Value-experience replay attaches row-aligned importance weights
            # and feedback to the sampled positions; replaced rows would
            # inherit them.
            return replay_data
        fraction = float(getattr(self, "cross_basket_relabel_fraction", 0.0))
        buffer = self.replay_buffer
        if fraction <= 0.0 or not getattr(buffer, "relabel_enabled", False):
            return replay_data
        if not isinstance(replay_data.observations, dict) or not hasattr(
            replay_data, "component_outcomes"
        ):
            return replay_data
        batch_size = int(replay_data.actions.shape[0])
        batch_inds = getattr(self, "_last_replay_batch_inds", None)
        env_indices = getattr(buffer, "_last_sample_env_indices", None)
        if (
            batch_inds is None
            or env_indices is None
            or len(batch_inds) != batch_size
            or len(env_indices) != batch_size
        ):
            return replay_data
        relabel_count = min(int(round(fraction * batch_size)), batch_size // 2)
        if relabel_count < 1:
            return replay_data

        rng = self._cross_basket_relabel_rng
        sources = np.arange(relabel_count, dtype=np.int64)
        destinations = np.arange(batch_size - relabel_count, batch_size, dtype=np.int64)
        metadata = buffer.cross_basket_relabel_metadata(
            np.asarray(batch_inds, dtype=np.int64)[sources],
            np.asarray(env_indices, dtype=np.int64)[sources],
        )
        source_masks = (
            replay_data.observations["mask"][:relabel_count].detach().cpu().numpy() > 0.5
        )
        eligible = metadata["price_valid"] & metadata["valid"][:, None]
        new_masks, changed, swapped = sample_cross_basket_relabel_masks(
            source_masks,
            eligible,
            self.cross_basket_relabel_one_swap_fraction,
            rng,
        )
        rows = np.flatnonzero(changed & metadata["valid"])
        stats = {
            "requested_rows": float(relabel_count),
            "applied_rows": float(rows.size),
            "valid_source_fraction": float(np.mean(metadata["valid"])),
        }
        if rows.size == 0:
            if stats_sink is not None:
                stats_sink.append(stats)
            return replay_data

        device = replay_data.actions.device
        source_index = torch.as_tensor(rows, device=device, dtype=torch.long)
        target_index = torch.as_tensor(destinations[rows], device=device, dtype=torch.long)
        observations = {
            key: value.index_select(0, source_index)
            for key, value in replay_data.observations.items()
        }
        next_observations = {
            key: value.index_select(0, source_index)
            for key, value in replay_data.next_observations.items()
        }
        actions = replay_data.actions.index_select(0, source_index).float()
        new_mask = torch.as_tensor(new_masks[rows], device=device, dtype=torch.float32)
        market = torch.as_tensor(metadata["market"][rows], device=device, dtype=torch.float32)
        next_market = torch.as_tensor(
            metadata["next_market"][rows], device=device, dtype=torch.float32
        )
        asset_returns = torch.as_tensor(
            metadata["asset_returns"][rows], device=device, dtype=torch.float32
        )
        price_valid = torch.as_tensor(
            metadata["price_valid"][rows], device=device, dtype=torch.bool
        )
        partial_counts = torch.as_tensor(
            metadata["next_partial_count"][rows], device=device, dtype=torch.long
        )
        layout = dict(
            features_per_coin=int(config.FEATURES_PER_COIN),
            primary_count=int(config.PRIMARY_FEATURE_COUNT),
            comparative_offset=int(config.COMPARATIVE_FEATURE_OFFSET),
            comparative_count=int(config.COMPARATIVE_FEATURE_COUNT),
        )
        relabeled_observations = relabel_cross_basket_observations(
            observations, new_mask, market, **layout
        )

        relabeled_actions = actions.clone()
        policy_rows = np.flatnonzero(
            rng.random(rows.size) < self.cross_basket_relabel_policy_action_fraction
        )
        if policy_rows.size:
            policy_index = torch.as_tensor(policy_rows, device=device, dtype=torch.long)
            policy_observations = {
                key: value.index_select(0, policy_index).float()
                for key, value in relabeled_observations.items()
            }
            with torch.no_grad():
                proposal = self.actor(
                    self._actor_observation(
                        self._gate_observations_with_sigma(policy_observations)
                    )
                ).float()
            noise = torch.as_tensor(
                rng.normal(
                    0.0,
                    self.cross_basket_relabel_policy_action_noise,
                    size=tuple(proposal.shape),
                ),
                device=device,
                dtype=proposal.dtype,
            )
            low, high = self._get_action_bounds()
            relabeled_actions[policy_index] = torch.max(
                torch.min(proposal + noise, high), low
            )

        component_kind = str(
            getattr(buffer, "component_outcome_target_kind", "scaled_unit_asset_return")
        )
        (
            relabeled_next_observations,
            rewards,
            component_outcomes,
            component_mask,
            executed_weights,
        ) = relabel_cross_basket_outcomes(
            relabeled_observations,
            next_observations,
            relabeled_actions,
            next_market,
            asset_returns,
            price_valid,
            partial_counts,
            self._cross_basket_name_rank(device),
            reward_scale=float(getattr(config, "FINANCIAL_REWARD_SCALE", 10.0)),
            cost_basis_index=int(config.COST_BASIS_TOKEN_INDEX),
            cost_basis_scale=float(config.COST_BASIS_FEATURE_SCALE),
            cost_basis_bounds=(-15.0, 50.0),
            min_position_weight=self.cross_basket_relabel_min_position_weight,
            unit_return_targets=component_kind == "scaled_unit_asset_return",
            component_target_clip=float(
                getattr(config, "COMPONENT_OUTCOME_TARGET_CLIP", 5.0)
            ),
            **layout,
        )

        def scatter(original: torch.Tensor, relabeled: torch.Tensor) -> torch.Tensor:
            updated = original.clone()
            updated[target_index] = relabeled.to(original.dtype)
            return updated

        new_observations = {
            key: scatter(value, relabeled_observations[key])
            for key, value in replay_data.observations.items()
        }
        new_next_observations = {
            key: scatter(value, relabeled_next_observations[key])
            for key, value in replay_data.next_observations.items()
        }
        source_rewards = replay_data.rewards.index_select(0, source_index).float()
        relabeled_replay = replay_data._replace(
            observations=new_observations,
            actions=scatter(replay_data.actions, relabeled_actions),
            next_observations=new_next_observations,
            dones=scatter(
                replay_data.dones, replay_data.dones.index_select(0, source_index)
            ),
            rewards=scatter(replay_data.rewards, rewards),
            component_outcomes=scatter(
                replay_data.component_outcomes, component_outcomes
            ),
            component_outcome_mask=scatter(
                replay_data.component_outcome_mask, component_mask
            ),
        )
        if stats_sink is not None:
            old_active = observations["mask"] > 0.5
            new_active = new_mask > 0.5
            stats.update(
                batch_fraction=float(rows.size / max(batch_size, 1)),
                one_swap_fraction=float(np.mean(swapped[rows])),
                policy_action_fraction=float(policy_rows.size / rows.size),
                added_assets_mean=float((new_active & ~old_active).sum(dim=1).float().mean().item()),
                reward_mean=float(rewards.mean().item()),
                source_reward_mean=float(source_rewards.mean().item()),
                reward_abs_gap_mean=float((rewards - source_rewards).abs().mean().item()),
                cash_weight_mean=float(executed_weights[:, 0].mean().item()),
            )
            stats_sink.append(stats)
        self._last_cross_basket_relabel_source_index = source_index
        self._last_cross_basket_relabel_target_index = target_index
        return relabeled_replay

    def _prepare_actor_crl_tensors(
        self,
        actor_replay_data,
        critic_replay_data,
        critic_obs_float,
        critic_obs_float_g,
        critic_actions_float,
        critic_next_actions_center,
    ):
        """Rebuild the actor/CRL inputs from the allowed replay batch.

        When the actor consumes the pre-relabel batch, ProExp, the previous-
        portfolio reference, the target actor's next action, masks, and CRL
        all have to be recomputed from those observations.  Reusing the
        critic's mixed-batch tensors would leave the projected book in the
        control problem.
        """
        if actor_replay_data is critic_replay_data:
            return (
                critic_obs_float,
                critic_obs_float_g,
                critic_actions_float,
                critic_next_actions_center,
                False,
            )
        actor_obs_float = replay_observations_as_float(
            actor_replay_data.observations
        )
        actor_obs_float_g = self._gate_observations_with_sigma(actor_obs_float)
        actor_actions_float = actor_replay_data.actions.float()
        with torch.no_grad():
            actor_next_obs_float = replay_observations_as_float(
                actor_replay_data.next_observations
            )
            actor_next_obs_float_g = self._gate_observations_with_sigma(
                actor_next_obs_float
            )
            actor_next_actions_center = self.actor_target(
                self._actor_observation(
                    actor_next_obs_float_g, use_no_grad=True
                )
            )
        return (
            actor_obs_float,
            actor_obs_float_g,
            actor_actions_float,
            actor_next_actions_center,
            True,
        )

    def _index_observation_rows(self, observations, row_index: torch.Tensor):
        if not torch.is_tensor(row_index) or row_index.numel() == 0:
            return None
        if isinstance(observations, dict):
            return {
                key: value.index_select(0, row_index.to(device=value.device))
                for key, value in observations.items()
            }
        return observations.index_select(0, row_index.to(device=observations.device))

    def _detached_centered_actor_advantage(self, obs_float_g):
        """No-grad Q(pi(s)) - Q(w_prev) for the XBR consumer-split log."""
        if not isinstance(obs_float_g, dict):
            return None
        sample = next(iter(obs_float_g.values()), None)
        if not torch.is_tensor(sample) or sample.shape[0] == 0:
            return None
        cuda_devices = []
        if sample.is_cuda:
            cuda_devices = [
                sample.device.index
                if sample.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            with torch.no_grad():
                actor_actions = self.actor(
                    self._actor_observation(obs_float_g, use_no_grad=True)
                )
                previous_raw_action = self._previous_portfolio_to_raw_action(
                    obs_float_g,
                    dtype=actor_actions.dtype,
                    device=actor_actions.device,
                )
                if previous_raw_action is None:
                    return None
                if self.use_proportional_exp:
                    current_q = torch.min(
                        torch.cat(self.critic(obs_float_g, actor_actions), dim=1),
                        dim=1,
                    ).values
                    previous_q = torch.min(
                        torch.cat(
                            self.critic(obs_float_g, previous_raw_action), dim=1
                        ),
                        dim=1,
                    ).values
                else:
                    current_q = self.critic.q1_forward(
                        obs_float_g, actor_actions
                    ).squeeze(-1)
                    previous_q = self.critic.q1_forward(
                        obs_float_g, previous_raw_action
                    ).squeeze(-1)
                return (current_q - previous_q).detach()

    def _cross_basket_actor_advantage_split(
        self,
        actor_action_advantage,
        actor_obs_float,
        mixed_obs_float_g,
        trained_on_original: bool,
    ):
        """Mean centered advantage on real rows vs rewritten tail rows."""
        real_value = None
        relabeled_value = None
        target_index = getattr(self, "_last_cross_basket_relabel_target_index", None)
        source_index = getattr(self, "_last_cross_basket_relabel_source_index", None)
        if actor_action_advantage is None:
            return real_value, relabeled_value
        if trained_on_original:
            real_value = float(
                self._basket_balanced_replay_mean(
                    actor_action_advantage.detach(),
                    actor_obs_float,
                ).item()
            )
            tail_obs = self._index_observation_rows(
                mixed_obs_float_g, target_index
            )
            if tail_obs is not None:
                tail_advantage = self._detached_centered_actor_advantage(tail_obs)
                if tail_advantage is not None:
                    relabeled_value = float(tail_advantage.mean().item())
            return real_value, relabeled_value
        if source_index is not None and source_index.numel() > 0:
            real_value = float(
                actor_action_advantage.detach()
                .index_select(0, source_index.to(actor_action_advantage.device))
                .mean()
                .item()
            )
        if target_index is not None and target_index.numel() > 0:
            relabeled_value = float(
                actor_action_advantage.detach()
                .index_select(0, target_index.to(actor_action_advantage.device))
                .mean()
                .item()
            )
        return real_value, relabeled_value

    def _sample_predictive_replay_data(self, batch_size: int):
        """Return a broad batch and its physical rows, independent of Bellman replay."""
        if not self.predictive_ssl_separate_replay_enabled:
            replay_data = self._sample_replay_data(batch_size)
            self._last_predictive_env_inds = getattr(
                self.replay_buffer, "_last_sample_env_indices", None
            )
            return replay_data, getattr(self, "_last_replay_batch_inds", None)
        numpy_state = np.random.get_state()
        financial_batch_inds = getattr(self, "_last_replay_batch_inds", None)
        financial_stats = dict(self._last_basket_replay_sample_stats)
        try:
            # Path 2's predictive learner is deliberately broader than the
            # query-selected Bellman batch. Draw its physical rows directly
            # with a private RNG so no mutable ``_last_replay_batch_inds``
            # hand-off can pair an SPR trajectory with another sample. The
            # stratified legacy ablation still uses its specialized sampler.
            buffer = self.replay_buffer
            replay_envs = int(getattr(buffer, "n_envs", 1))
            if (
                not self.basket_replay_stratify_enabled
                and hasattr(buffer, "_get_samples")
                and (
                    replay_envs == 1
                    or isinstance(buffer, ComponentOutcomeReplayBuffer)
                )
            ):
                upper = buffer.buffer_size if buffer.full else buffer.pos
                if upper <= 0:
                    raise ValueError(
                        "cannot sample predictive replay from an empty buffer"
                    )
                if buffer.optimize_memory_usage and buffer.full:
                    predictive_batch_inds = (
                        self._predictive_replay_rng.integers(
                            1, buffer.buffer_size, size=int(batch_size)
                        )
                        + buffer.pos
                    ) % buffer.buffer_size
                else:
                    predictive_batch_inds = self._predictive_replay_rng.integers(
                        0, upper, size=int(batch_size)
                    )
                predictive_batch_inds = np.asarray(
                    predictive_batch_inds, dtype=np.int64
                )
                if replay_envs == 1:
                    predictive_data = buffer._get_samples(
                        predictive_batch_inds, env=self._vec_normalize_env
                    )
                    self._last_predictive_env_inds = np.zeros(
                        len(predictive_batch_inds), dtype=np.int64
                    )
                else:
                    self._last_predictive_env_inds = np.asarray(
                        self._predictive_replay_rng.integers(
                            0, replay_envs, size=int(batch_size)
                        ),
                        dtype=np.int64,
                    )
                    predictive_data = buffer._get_samples(
                        predictive_batch_inds,
                        env=self._vec_normalize_env,
                        env_indices=self._last_predictive_env_inds,
                    )
                return predictive_data, predictive_batch_inds
            replay_data = self._sample_replay_data(
                batch_size,
                baskets_per_batch=self.predictive_ssl_replay_baskets_per_batch,
                group_selection=self.predictive_ssl_replay_group_selection,
                rng=self._predictive_replay_rng,
                evidence_gated=False,
                track_diagnostics=True,
                uniform_mix=self.predictive_ssl_replay_uniform_mix,
            )
            predictive_batch_inds = getattr(
                self, "_last_replay_batch_inds", None
            )
            self._last_predictive_env_inds = getattr(
                buffer, "_last_sample_env_indices", None
            )
            return replay_data, predictive_batch_inds
        finally:
            self._last_replay_batch_inds = financial_batch_inds
            self._last_basket_replay_sample_stats = financial_stats
            # Base replay buffers still use NumPy's global RNG to select an
            # environment index. The production path has one environment, but
            # even drawing that guaranteed zero would otherwise perturb later
            # control randomness merely because SSL was enabled.
            np.random.set_state(numpy_state)

    @staticmethod
    def _slice_replay_data(replay_data, row_indices: np.ndarray):
        """Select rows from an SB3 replay sample without changing its type."""
        index = torch.as_tensor(
            np.asarray(row_indices, dtype=np.int64),
            device=replay_data.actions.device,
            dtype=torch.long,
        )
        values = []
        for field in replay_data._fields:
            value = getattr(replay_data, field)
            if isinstance(value, dict):
                values.append(
                    {
                        key: tensor.index_select(0, index)
                        for key, tensor in value.items()
                    }
                )
            elif torch.is_tensor(value):
                values.append(value.index_select(0, index))
            else:
                values.append(value)
        return replay_data.__class__(*values)

    def _predictive_sequence_from_financial_batch(
        self,
        replay_data,
        *,
        start_batch_inds: Optional[np.ndarray] = None,
        maximum_batch_size: Optional[int] = None,
        require_component_outcomes: bool = False,
        start_env_inds: Optional[np.ndarray] = None,
    ):
        """Build aligned SPR sequences from an explicitly sampled replay batch."""
        batch_size = int(replay_data.actions.shape[0])
        configured_batch_size = self.predictive_ssl_sequence_batch_size
        if maximum_batch_size is not None:
            configured_batch_size = min(
                configured_batch_size, max(1, int(maximum_batch_size))
            )
        sequence_batch_size = min(batch_size, configured_batch_size)
        if sequence_batch_size < batch_size:
            selected_rows = np.asarray(
                self._predictive_replay_rng.choice(
                    batch_size, size=sequence_batch_size, replace=False
                ),
                dtype=np.int64,
            )
        else:
            selected_rows = np.arange(batch_size, dtype=np.int64)
        selected_replay = (
            replay_data
            if sequence_batch_size == batch_size
            else self._slice_replay_data(replay_data, selected_rows)
        )

        if start_batch_inds is None:
            start_batch_inds = getattr(self, "_last_replay_batch_inds", None)
        sequence_builder = getattr(
            self.replay_buffer, "get_predictive_sequences", None
        )
        if start_batch_inds is not None and callable(sequence_builder):
            selected_starts = np.asarray(start_batch_inds, dtype=np.int64)[
                selected_rows
            ]
            # For a one-step SSL target, the sampled replay transition already
            # contains the exact action, reward, next observation, and component
            # outcomes. Re-gathering every large observation field from NumPy
            # and copying it to CUDA a second time is redundant. Raw done is
            # still read from storage so timeout truncation matches the general
            # sequence builder rather than SB3's Bellman timeout masking.
            if (
                self.predictive_ssl_horizon == 1
                and not self.foundation_learning_enabled
                and isinstance(self.replay_buffer, ComponentOutcomeReplayBuffer)
                and (
                    self.replay_buffer.n_envs == 1
                    or (
                        start_env_inds is not None
                        and len(start_env_inds) == len(start_batch_inds)
                    )
                )
            ):
                selected_envs = (
                    np.zeros(len(selected_starts), dtype=np.int64)
                    if self.replay_buffer.n_envs == 1
                    else np.asarray(start_env_inds, dtype=np.int64)[selected_rows]
                )
                raw_dones = torch.as_tensor(
                    self.replay_buffer.dones[selected_starts, selected_envs].copy(),
                    device=selected_replay.actions.device,
                    dtype=selected_replay.dones.dtype,
                ).reshape(-1, 1)
                valid_steps = raw_dones < 0.5
                future_observations = {
                    key: torch.where(
                        valid_steps.reshape(
                            -1, *([1] * (next_value.ndim - 1))
                        ),
                        next_value,
                        selected_replay.observations[key],
                    ).unsqueeze(1)
                    for key, next_value in selected_replay.next_observations.items()
                }
                return selected_replay, PredictiveSequenceReplayBufferSamples(
                    actions=selected_replay.actions.unsqueeze(1),
                    rewards=selected_replay.rewards.unsqueeze(1),
                    dones=raw_dones.unsqueeze(1),
                    future_observations=future_observations,
                    component_outcomes=(
                        selected_replay.component_outcomes.unsqueeze(1)
                    ),
                    component_outcome_masks=(
                        selected_replay.component_outcome_mask.unsqueeze(1)
                    ),
                    outcome_valid_steps=torch.ones_like(
                        valid_steps, dtype=torch.bool
                    ),
                    valid_steps=valid_steps,
                )
            sequence = sequence_builder(
                selected_starts,
                max(
                    self.predictive_ssl_horizon,
                    self.foundation_max_horizon
                    if self.foundation_learning_enabled
                    else 1,
                ),
                env=self._vec_normalize_env,
            )
            # The sequence must be anchored on exactly the same actions and
            # first future states as the financial sample subset.
            if not torch.allclose(
                sequence.actions[:, 0], selected_replay.actions
            ):
                raise RuntimeError(
                    "SPR sequence actions are not aligned with sampled replay "
                    f"rows={selected_starts.tolist()} max_abs_diff="
                    f"{float((sequence.actions[:, 0] - selected_replay.actions).abs().max().item()):.6g}"
                )
            if not torch.allclose(
                sequence.rewards[:, 0], selected_replay.rewards
            ):
                raise RuntimeError(
                    "value-model rewards are not aligned with financial replay"
                )
            # Sequence truncation uses raw episode boundaries, including time
            # limits; SB3's Bellman sample masks timeouts out of ``dones``.
            # Raw sequence terminals must therefore be a superset rather than
            # necessarily bit-identical.
            if not torch.all(
                sequence.dones[:, 0] >= selected_replay.dones
            ):
                raise RuntimeError(
                    "value-model terminals are not aligned with financial replay"
                )
            return selected_replay, sequence

        # Generic SB3 smoke buffers do not expose physical replay positions.
        # Preserve a valid one-step auxiliary path rather than inventing
        # adjacency that cannot be verified.
        if require_component_outcomes:
            raise RuntimeError(
                "Live foundation rehearsal requires chronological component "
                "outcomes, but the active replay buffer does not implement "
                "get_predictive_sequences(). Use ComponentOutcomeReplayBuffer "
                "and keep FOUNDATION_SEQUENCE_REPLAY_ENABLED=True."
            )
        future_observations = {
            key: value.unsqueeze(1)
            for key, value in selected_replay.next_observations.items()
        }
        outcome_assets = int(selected_replay.observations["mask"].shape[-1])
        outcome_dim = len(
            tuple(getattr(config, "COMPONENT_OUTCOME_HORIZONS", (1,)))
        )
        empty_component_outcomes = torch.zeros(
            (
                sequence_batch_size,
                1,
                outcome_assets,
                outcome_dim,
            ),
            device=selected_replay.actions.device,
            dtype=selected_replay.rewards.dtype,
        )
        return selected_replay, PredictiveSequenceReplayBufferSamples(
            actions=selected_replay.actions.unsqueeze(1),
            rewards=selected_replay.rewards.unsqueeze(1),
            dones=selected_replay.dones.unsqueeze(1),
            future_observations=future_observations,
            component_outcomes=empty_component_outcomes,
            component_outcome_masks=torch.zeros_like(
                empty_component_outcomes, dtype=torch.bool
            ),
            outcome_valid_steps=torch.ones_like(
                selected_replay.dones.view(-1, 1), dtype=torch.bool
            ),
            valid_steps=(selected_replay.dones.view(-1, 1) < 0.5),
        )

    @staticmethod
    def _basket_balanced_replay_mean(
        values: torch.Tensor,
        observations,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bool(getattr(config, "CURRICULUM_MIXED_REPLAY_ENABLED", False)):
            if values.ndim == 0:
                return values
            scalar_values = values.reshape(values.shape[0], -1)
            if scalar_values.shape[1] != 1:
                raise ValueError(
                    "mixed replay objectives require one scalar per row"
                )
            scalar_values = scalar_values[:, 0]
            if valid_mask is None:
                return scalar_values.mean()
            valid = valid_mask.reshape(valid_mask.shape[0], -1)
            if valid.shape[1] != 1:
                raise ValueError("valid_mask must provide one value per row")
            valid = valid[:, 0].bool()
            return (
                scalar_values[valid].mean()
                if valid.any()
                else scalar_values.sum() * 0.0
            )
        memberships = None
        if isinstance(observations, dict):
            candidate = observations.get("basket_membership")
            if candidate is None and bool(getattr(config, "EARNMORE_GLOBAL_POOL_ENABLED", False)):
                # A global mask has stable asset coordinates and names the
                # exact task even when membership is private replay metadata.
                candidate = observations.get("mask")
            if torch.is_tensor(candidate):
                memberships = candidate.float()
        return basket_balanced_mean(values, memberships, valid_mask)

    def _gate_observations_with_sigma(self, obs_float):
        """
        Apply sigma-based input modulation to an observation tensor (or dict).

        Modes:
        - passthrough: return input unchanged (Option B default).
        - dampen_input: legacy Option A dampening:
            gated_signal = signal * (1 - tanh(|sigma| * scale * scale_factor))

        In dampen_input mode, base features use scale_factor=1.0 and rolling
        features use gentler factors (0.7/0.5/0.3 for roll_7/14/30).

        When sigma ≈ 0  →  gate ≈ 1.0  →  signal passes through unchanged.
        When sigma is large  →  gate → eps (10% floor)  →  signal faintly present.
        Floor prevents full blind: high sigma can coincide with high opportunity.

        Works with both:
          - Flat tensor:  shape (batch, obs_dim)
          - Dict obs:     keys 'observations', 'asset_ids', 'mask', 'prev_action',
                          'prev_mask', 'basket_context', 'basket_membership'
                          — only 'observations' is gated; asset_ids, basket_context,
                          and basket_membership pass through unchanged
                          (identity residuals + UVFA g(B) + privileged z(B)).

        The map is built lazily on the first call if not already initialised.
        """
        if not self.use_sigma_gating:
            return obs_float
        if self.sigma_input_mode == "passthrough":
            return obs_float

        # Lazy initialisation (e.g. if config was not ready at __init__ time)
        if self._sigma_signal_idx is None:
            try:
                self._build_sigma_position_map()
            except Exception:
                pass
        if self._sigma_signal_idx is None:
            return obs_float

        def _gate_flat(flat: torch.Tensor) -> torch.Tensor:
            dev = flat.device
            sig_idx    = self._sigma_gate_idx.to(dev)
            signal_idx = self._sigma_signal_idx.to(dev)
            obs_dim    = flat.shape[-1]
            valid      = (sig_idx < obs_dim) & (signal_idx < obs_dim)
            if not valid.any():
                return flat
            gated = flat.clone()
            sigma_vals = flat[:, sig_idx[valid]].abs()
            # Per-position scale: base features use full stiffness, roll_7/14/30 use gentler
            scale_fac = self._sigma_scale_factors[valid].to(dev)
            effective_scale = self.sigma_gate_scale * scale_fac
            gate = 1.0 - torch.tanh(sigma_vals * effective_scale)
            # Floor: never fully blind the critic; high sigma can coincide with high opportunity
            eps = 0.10  # 10% minimum signal passthrough
            gate = eps + (1.0 - eps) * gate
            gated[:, signal_idx[valid]] = flat[:, signal_idx[valid]] * gate
            return gated

        if isinstance(obs_float, dict):
            if 'observations' not in obs_float:
                return obs_float
            result = dict(obs_float)
            result['observations'] = _gate_flat(obs_float['observations'])
            return result
        else:
            return _gate_flat(obs_float)

    def _extract_flat_observations(self, obs_float):
        """Return flat observation tensor from either dict-observation or flat tensor."""
        if isinstance(obs_float, dict):
            return obs_float.get('observations', None)
        return obs_float

    def _actor_observation(self, obs_float, use_no_grad=False):
        """
        Return the observation to pass to the actor (and actor_target).

        Default SB3 TD3 Actor (stable_baselines3.td3.policies.Actor) does
        extract_features(obs, self.features_extractor) inside forward(), so it
        expects the raw observation (dict or tensor). We pass obs_float as-is
        in that case. Custom actors (e.g. one that expects pre-extracted
        features) are handled by running the policy's feature extractor here
        and passing the resulting tensor.

        use_no_grad: if True, run feature extraction under no_grad (for target-actor calls).
        """
        actor_module = self.actor_target if use_no_grad else self.actor
        expects_precomputed_features = bool(getattr(actor_module, "expects_precomputed_features", False))

        if (
            expects_precomputed_features
            and isinstance(obs_float, dict)
            and getattr(self.policy, "features_extractor", None) is not None
        ):
            if use_no_grad:
                with torch.no_grad():
                    return self.policy.features_extractor(obs_float)
            return self.policy.features_extractor(obs_float)
        return obs_float

    def _compute_sigma_loss_weights(self, obs_float):
        """
        Compute per-sample critic loss weights for Option B sigma-aware focus.

        weight_i = clamp(1 + alpha * (sigma_intensity_i / (mean_sigma + eps))^p, 1, w_max)
        """
        flat = self._extract_flat_observations(obs_float)
        if flat is None or not torch.is_tensor(flat):
            return None, None
        if not self.sigma_loss_focus_enabled:
            return torch.ones(flat.shape[0], device=flat.device, dtype=flat.dtype), None

        # Lazy build map if needed.
        if self._sigma_unique_idx is None:
            try:
                self._build_sigma_position_map()
            except Exception:
                pass
        if self._sigma_unique_idx is None:
            return torch.ones(flat.shape[0], device=flat.device, dtype=flat.dtype), None

        idx = self._sigma_unique_idx.to(flat.device)
        valid = idx < flat.shape[-1]
        if not valid.any():
            return torch.ones(flat.shape[0], device=flat.device, dtype=flat.dtype), None

        with torch.no_grad():
            sigma_vals = flat[:, idx[valid]].abs()
            if config.LOCAL_CAUSAL_INPUTS_ENABLED:
                # Compare economic sigma magnitudes, not the nonlinear input
                # coordinates. Clipping remains the observation's safety bound.
                columns = (idx[valid] % int(config.FEATURES_PER_COIN)).cpu().numpy()
                references = config.local_primary_references(config.SCOLS)
                if np.all(columns < len(references)):
                    units = torch.as_tensor(references[columns], device=flat.device, dtype=flat.dtype)
                    sigma_vals = torch.sinh(sigma_vals) * units
            sigma_intensity = sigma_vals.mean(dim=1)
            denom = sigma_intensity.mean() + self.sigma_loss_focus_eps
            normalized = sigma_intensity / denom
            weights = 1.0 + self.sigma_loss_focus_alpha * torch.pow(normalized, self.sigma_loss_focus_power)
            weights = torch.clamp(weights, min=1.0, max=self.sigma_loss_focus_max_weight)

        return weights.to(flat.dtype), sigma_intensity

    # ════════════════════════════════════════════════════════════════════════════
    # N-STEP RETURN BUFFERING
    # ════════════════════════════════════════════════════════════════════════════

    def _store_transition(
        self,
        replay_buffer,
        buffer_action: np.ndarray,
        new_obs,
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list,
    ) -> None:
        """
        Override SB3's _store_transition to accumulate n-step returns.

        A rolling deque collects (obs_t, a_t, r_t, obs_{t+1}, done_t) tuples.
        When the deque reaches n_step entries, the oldest is popped and stored in
        the replay buffer with a blended reward:

            r_blended = n_step_blend  * (r_0 + γ*r_1 + γ²*r_2)
                      + (1-n_step_blend) * r_0

        On episode termination (any done=True), remaining partial sequences are
        flushed immediately so no transitions are lost.

        Falls back to standard 1-step behaviour when use_n_step_returns=False.
        """
        if not self.use_n_step_returns or self.n_step <= 1:
            super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
            return

        def _copy_obs(x):
            if isinstance(x, dict):
                return {k: np.array(v) for k, v in x.items()}
            return np.array(x)

        cur_obs = _copy_obs(self._last_obs)
        new_obs_copy = _copy_obs(new_obs)

        self._nstep_deque.append((
            cur_obs,
            np.array(buffer_action),
            np.array(reward, dtype=np.float32),
            new_obs_copy,
            np.array(dones),
            infos,
        ))

        if len(self._nstep_deque) >= self.n_step:
            self._commit_nstep(replay_buffer)

        if np.any(dones):
            while len(self._nstep_deque) > 0:
                self._commit_nstep(replay_buffer)

    def _commit_nstep(self, replay_buffer) -> None:
        """
        Pop the oldest entry from the n-step deque and store a blended
        (1-step + n-step compound) transition directly into the replay buffer.
        """
        if len(self._nstep_deque) == 0:
            return

        entries = list(self._nstep_deque)
        # Unpack all fields from the FIRST entry, including first_new_obs = s_{t+1}.
        # CRITICAL: we must store s_{t+1} (not s_{t+n}) as next_obs in the replay buffer.
        # Storing s_{t+n} makes the CRL velocity constraint span n minutes instead of 1,
        # immediately and permanently violating the 1-minute-calibrated threshold.
        # The multi-step reward signal is carried by blended_r; the bootstrap and the
        # CRL three-point finite-difference both work correctly off the 1-step gap.
        first_obs, first_act, first_r, first_new_obs, first_done, first_infos = entries[0]

        compound_r    = np.zeros_like(first_r, dtype=np.float32)
        episode_done  = first_done.copy()
        episode_infos = first_infos

        for k, (_, _, r_k, _, done_k, info_k) in enumerate(entries):
            compound_r   += (self.gamma ** k) * r_k.astype(np.float32)
            episode_done  = done_k
            episode_infos = info_k
            if np.any(done_k):
                break

        blended_r = (
            self.n_step_blend * compound_r
            + (1.0 - self.n_step_blend) * first_r.astype(np.float32)
        )

        self._nstep_deque.popleft()

        def _squeeze(x, i=0):
            """Remove the n_envs=1 leading dimension for replay_buffer.add()."""
            if isinstance(x, dict):
                out = {}
                for k, v in x.items():
                    arr = np.array(v)
                    out[k] = arr[i] if arr.ndim > 1 else arr
                return out
            arr = np.array(x)
            return arr[i] if arr.ndim > 1 else arr

        r_scalar    = float(blended_r.flat[0])
        done_scalar = bool(np.array(episode_done).flat[0])
        episode_info = (
            dict(episode_infos[0])
            if isinstance(episode_infos, list)
            else dict(episode_infos)
        )
        first_info = (
            first_infos[0] if isinstance(first_infos, list) else first_infos
        )
        # The replayed state and next state belong to the first one-step edge,
        # so its component label must remain aligned even when the scalar reward
        # is compounded over later edges. Keep terminal/time-limit metadata from
        # the final edge while restoring only the retrospective component fields.
        for component_key in (
            "component_outcomes",
            "component_outcome_mask",
            "curriculum_role",
        ):
            if component_key in first_info:
                episode_info[component_key] = first_info[component_key]
        infos_list = [episode_info]

        replay_buffer.add(
            _squeeze(first_obs),
            _squeeze(first_new_obs),   # s_{t+1}: 1-step gap, correct for CRL and bootstrap
            _squeeze(first_act),
            r_scalar,
            done_scalar,
            infos_list,
        )

    def _create_schedule(self, lr: Union[float, Schedule]) -> Schedule:
        """Helper to create a schedule from a float."""
        if isinstance(lr, (float, int)):
            return lambda progress: lr
        return lr

    def _setup_custom_optimizers(self) -> None:
        """Create actor, critic, and lambda optimizers with separate learning rates."""
        self.actor_lr_schedule = self._create_schedule(self.actor_learning_rate)
        self.critic_lr_schedule = self._create_schedule(self.critic_learning_rate)

        # Assume Adam optimizer if not specified
        optimizer_class = self.policy_kwargs.get("optimizer_class", torch.optim.Adam)

        # Create optimizers with the initial learning rate
        self.actor.optimizer = optimizer_class(
            self.actor.parameters(), lr=self.actor_lr_schedule(1.0)
        )
        self.critic.optimizer = optimizer_class(
            self.critic.parameters(), lr=self.critic_lr_schedule(1.0)
        )
        
        # Note: No optimizer needed for λ - we use direct updates

    def _get_action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache action-space bounds on the model device."""
        if self._cached_action_low is None or self._cached_action_high is None:
            self._cached_action_low = torch.as_tensor(
                self.action_space.low,
                device=self.device,
                dtype=torch.float32,
            )
            self._cached_action_high = torch.as_tensor(
                self.action_space.high,
                device=self.device,
                dtype=torch.float32,
            )
        return self._cached_action_low, self._cached_action_high

    def _previous_portfolio_to_raw_action(
        self,
        observations,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Encode replay's normalized previous allocation as critic raw logits."""
        if not isinstance(observations, dict) or "prev_action" not in observations:
            return None
        previous_weights = observations["prev_action"].to(
            device=device,
            dtype=dtype,
        )
        action_mask = _build_action_mask_from_observations(
            observations,
            action_dim=previous_weights.shape[1],
            device=device,
            dtype=dtype,
        )
        cash_weight = previous_weights[:, 0:1].clamp(0.0, 1.0)
        asset_mask = action_mask[:, 1:]
        asset_weights = previous_weights[:, 1:] * asset_mask
        asset_total = asset_weights.sum(dim=1, keepdim=True)
        relative_assets = torch.where(
            asset_total > 1e-8,
            asset_weights / asset_total.clamp(min=1e-8),
            torch.zeros_like(asset_weights),
        )
        raw_cash = 2.0 * cash_weight - 1.0
        raw_assets = torch.where(
            asset_mask > 0.0,
            2.0 * relative_assets - 1.0,
            torch.full_like(relative_assets, -1.0),
        )
        low, high = self._get_action_bounds()
        return torch.max(
            torch.min(torch.cat([raw_cash, raw_assets], dim=1), high),
            low,
        )

    def _detached_previous_action_q(
        self,
        observations,
        actor_actions: torch.Tensor,
        use_twin_minimum: bool,
    ) -> Optional[torch.Tensor]:
        """Value the previous allocation for a detached diagnostic baseline."""
        previous_raw_action = self._previous_portfolio_to_raw_action(
            observations,
            dtype=actor_actions.dtype,
            device=actor_actions.device,
        )
        if previous_raw_action is None:
            return None
        cuda_devices = []
        if actor_actions.is_cuda:
            cuda_devices = [
                actor_actions.device.index
                if actor_actions.device.index is not None
                else torch.cuda.current_device()
            ]
        # Preserve RNG so enabling the diagnostic cannot move later dropout,
        # candidate sampling, or CRL randomness.
        with torch.random.fork_rng(devices=cuda_devices):
            with torch.no_grad():
                if use_twin_minimum:
                    previous_q_values = self.critic(
                        observations,
                        previous_raw_action,
                    )
                    previous_q = torch.min(
                        torch.cat(previous_q_values, dim=1),
                        dim=1,
                    ).values
                else:
                    previous_q = self.critic.q1_forward(
                        observations,
                        previous_raw_action,
                    ).squeeze(-1)
        return previous_q.detach()

    @staticmethod
    def _gradient_l2_norm(gradients) -> torch.Tensor:
        squared_terms = [
            gradient.detach().float().pow(2).sum()
            for gradient in gradients
            if gradient is not None
        ]
        if not squared_terms:
            return torch.tensor(0.0)
        return torch.sqrt(torch.stack(squared_terms).sum())

    @staticmethod
    def _masked_cross_sectional_correlation(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> Optional[float]:
        """Average within-basket Pearson correlation over valid samples/horizons."""
        with torch.no_grad():
            counts = mask.sum(dim=1).clamp(min=1.0)
            prediction_mean = (predictions * mask).sum(dim=1) / counts
            target_mean = (targets * mask).sum(dim=1) / counts
            prediction_centered = (
                predictions - prediction_mean.unsqueeze(1)
            ) * mask
            target_centered = (
                targets - target_mean.unsqueeze(1)
            ) * mask
            covariance = (
                prediction_centered * target_centered
            ).sum(dim=1)
            prediction_energy = prediction_centered.pow(2).sum(dim=1)
            target_energy = target_centered.pow(2).sum(dim=1)
            denominator = torch.sqrt(
                prediction_energy * target_energy
            )
            valid = (
                (counts >= 2.0)
                & torch.isfinite(denominator)
                & (denominator > 1e-12)
            )
            if not valid.any():
                return None
            correlations = covariance[valid] / denominator[valid]
            return float(correlations.mean().item())

    def _calibrate_component_outcome_gradient_scale(
        self,
        td_loss: torch.Tensor,
        component_loss: torch.Tensor,
    ) -> float:
        """Bound auxiliary pressure as a fraction of shared encoder TD pressure."""
        should_measure = (
            self._component_outcome_gradient_scale is None
            or self._n_updates <= 512
            or self._n_updates
            % self.component_outcome_gradient_calibration_interval
            == 0
        )
        if not should_measure:
            return float(self._component_outcome_gradient_scale)

        encoder_parameters = tuple(self.critic.component_encoder_parameters())
        td_gradients = torch.autograd.grad(
            td_loss,
            encoder_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        component_gradients = torch.autograd.grad(
            component_loss,
            encoder_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        # Calibrate at the parameters both objectives actually reach.  The
        # broader list also contains task/Q-only projections whose TD gradient
        # says nothing about the strength of the asset teacher.
        shared_pairs = [
            (td_gradient, component_gradient)
            for td_gradient, component_gradient in zip(
                td_gradients, component_gradients
            )
            if td_gradient is not None and component_gradient is not None
        ]
        if not shared_pairs:
            raise RuntimeError(
                "Asset supervision and Bellman TD have no shared encoder gradient"
            )
        td_norm = self._gradient_l2_norm(
            tuple(pair[0] for pair in shared_pairs)
        )
        component_norm = self._gradient_l2_norm(
            tuple(pair[1] for pair in shared_pairs)
        )
        raw_scale = (
            self.component_outcome_target_gradient_ratio
            * float(td_norm.item())
            / max(float(component_norm.item()), 1e-12)
        )
        bounded_scale = float(
            np.clip(
                raw_scale,
                self.component_outcome_gradient_scale_min,
                self.component_outcome_gradient_scale_max,
            )
        )
        # A coefficient floor is not permission to exceed the measured shared
        # gradient ratio. This matters after removing PopArt: tiny raw-Huber TD
        # gradients may require a scale below the historical 1e-4 floor.
        bounded_scale = min(raw_scale, bounded_scale)
        if self._component_outcome_gradient_scale is None:
            smoothed_scale = bounded_scale
        else:
            decay = float(
                np.clip(self.component_outcome_gradient_scale_ema, 0.0, 0.9999)
            )
            # Smooth increases in teacher pressure, but reduce it immediately
            # when the measured raw gradient grows.  A symmetric EMA can hold
            # yesterday's large coefficient for many updates and overshoot the
            # target ratio by orders of magnitude after the forecast wakes up.
            smoothed_scale = min(
                bounded_scale,
                decay * self._component_outcome_gradient_scale
                + (1.0 - decay) * bounded_scale,
            )
        self._component_outcome_gradient_scale = float(smoothed_scale)
        self._component_outcome_td_gradient_norm = float(td_norm.item())
        self._component_outcome_aux_gradient_norm = float(
            component_norm.item()
        )
        self._component_outcome_gradient_measurement_update = self._n_updates
        return float(self._component_outcome_gradient_scale)

    def _expand_observations_for_candidates(self, obs_float, num_candidates: int):
        """
        Repeat a batch of observations K times so critics can score [batch, K, action_dim] candidates.
        """
        if num_candidates <= 1:
            return obs_float

        def _expand_tensor(tensor: torch.Tensor) -> torch.Tensor:
            batch_size = tensor.shape[0]
            expanded = tensor.unsqueeze(1).expand(batch_size, num_candidates, *tensor.shape[1:])
            return expanded.reshape(batch_size * num_candidates, *tensor.shape[1:])

        if isinstance(obs_float, dict):
            return {
                key: _expand_tensor(value) if torch.is_tensor(value) else value
                for key, value in obs_float.items()
            }
        if torch.is_tensor(obs_float):
            return _expand_tensor(obs_float)
        raise TypeError(f"Unsupported observation type for candidate expansion: {type(obs_float)}")

    def _split_candidate_family_counts(self, num_candidates: int) -> Tuple[int, int]:
        """
        Split the non-center candidates into budget and composition families.

        The split is intentionally biased toward composition candidates so the actor
        can still discover faster cross-asset reshuffles even while budget moves remain
        comparatively conservative.
        """
        non_center = max(0, int(num_candidates) - 1)
        if non_center == 0:
            return 0, 0
        if non_center == 1:
            return 0, 1

        n_budget = int(round(non_center * self.proportional_exp_budget_candidate_ratio))
        n_budget = max(1, min(non_center - 1, n_budget))
        n_composition = non_center - n_budget
        return n_budget, n_composition

    def _sample_local_candidate_actions(
        self,
        center_actions: torch.Tensor,
        candidate_observations,
        num_candidates: int,
        low: torch.Tensor,
        high: torch.Tensor,
        budget_std: float,
        composition_std: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a local action cloud around the center action.

        Candidate 0 is always the center action. The remaining candidates are split
        between budget-only moves and composition-only moves so the learned basin is
        evaluated in economically meaningful portfolio directions.
        """
        center_actions = center_actions.float()
        batch_size, action_dim = center_actions.shape
        num_candidates = max(1, int(num_candidates))
        action_mask = _build_action_mask_from_observations(
            candidate_observations,
            action_dim=action_dim,
            device=center_actions.device,
            dtype=center_actions.dtype,
        )

        candidates = center_actions.unsqueeze(1).expand(batch_size, num_candidates, action_dim).clone()
        family_ids = center_actions.new_zeros((num_candidates,), dtype=torch.long)
        if num_candidates == 1:
            return candidates, family_ids

        n_budget, n_composition = self._split_candidate_family_counts(num_candidates)
        next_idx = 1
        center_weights = normalize_raw_actions_to_portfolio_weights(center_actions, action_mask)
        center_cash = center_weights[:, 0:1]
        center_equity = (1.0 - center_cash).clamp(min=0.0)
        center_relative = _safe_relative_asset_weights(center_weights, action_mask, eps=self.proportional_exp_eps)
        asset_mask = action_mask[:, 1:]

        if self.proportional_exp_cloud_space == "raw_actions":
            if n_budget > 0:
                budget_noise = torch.randn(
                    batch_size,
                    n_budget,
                    1,
                    device=center_actions.device,
                    dtype=center_actions.dtype,
                ) * float(budget_std)
                budget_noise = budget_noise.clamp(-float(self.target_noise_clip), float(self.target_noise_clip))
                budget_candidates = center_actions.unsqueeze(1).expand(batch_size, n_budget, action_dim).clone()
                budget_candidates[:, :, 0:1] = center_actions[:, None, 0:1] + budget_noise
                candidates[:, next_idx:next_idx + n_budget, :] = budget_candidates
                family_ids[next_idx:next_idx + n_budget] = 1
                next_idx += n_budget

            if n_composition > 0:
                composition_noise = torch.randn(
                    batch_size,
                    n_composition,
                    action_dim - 1,
                    device=center_actions.device,
                    dtype=center_actions.dtype,
                ) * float(composition_std)
                composition_noise = composition_noise.clamp(-float(self.target_noise_clip), float(self.target_noise_clip))
                composition_candidates = center_actions.unsqueeze(1).expand(batch_size, n_composition, action_dim).clone()
                composition_candidates[:, :, 1:] = center_actions[:, None, 1:] + composition_noise
                candidates[:, next_idx:next_idx + n_composition, :] = composition_candidates
                family_ids[next_idx:next_idx + n_composition] = 2
            low_view = low.to(device=center_actions.device, dtype=center_actions.dtype).view(1, 1, -1)
            high_view = high.to(device=center_actions.device, dtype=center_actions.dtype).view(1, 1, -1)
            candidates = torch.max(torch.min(candidates, high_view), low_view)
            return candidates, family_ids

        candidate_weights = center_weights.unsqueeze(1).expand(batch_size, num_candidates, action_dim).clone()

        if n_budget > 0:
            budget_delta = torch.randn(
                batch_size,
                n_budget,
                1,
                device=center_actions.device,
                dtype=center_actions.dtype,
            ) * float(budget_std)
            budget_delta = budget_delta.clamp(-2.5 * float(max(budget_std, self.proportional_exp_eps)),
                                              2.5 * float(max(budget_std, self.proportional_exp_eps)))
            budget_cash = (center_cash.unsqueeze(1) + budget_delta).clamp(0.0, 1.0)
            budget_equity = 1.0 - budget_cash
            budget_assets = center_relative.unsqueeze(1) * budget_equity
            budget_weights = center_weights.unsqueeze(1).expand(batch_size, n_budget, action_dim).clone()
            budget_weights[:, :, 0:1] = budget_cash
            budget_weights[:, :, 1:] = budget_assets
            candidate_weights[:, next_idx:next_idx + n_budget, :] = budget_weights
            family_ids[next_idx:next_idx + n_budget] = 1
            next_idx += n_budget

        if n_composition > 0:
            composition_noise = torch.randn(
                batch_size,
                n_composition,
                action_dim - 1,
                device=center_actions.device,
                dtype=center_actions.dtype,
            ) * float(composition_std)
            composition_noise = composition_noise.clamp(
                -2.5 * float(max(composition_std, self.proportional_exp_eps)),
                2.5 * float(max(composition_std, self.proportional_exp_eps)),
            )
            raw_relative = center_relative.unsqueeze(1) + (composition_noise * asset_mask.unsqueeze(1))
            raw_relative = torch.clamp(raw_relative, min=0.0)
            raw_relative = raw_relative * asset_mask.unsqueeze(1)
            relative_sum = raw_relative.sum(dim=2, keepdim=True)
            inactive_or_empty = relative_sum.squeeze(-1) <= self.proportional_exp_eps
            if inactive_or_empty.any():
                uniform_relative = asset_mask.unsqueeze(1) / asset_mask.sum(dim=1, keepdim=True).clamp(min=1.0).unsqueeze(1)
                raw_relative = torch.where(
                    inactive_or_empty.unsqueeze(-1),
                    uniform_relative,
                    raw_relative,
                )
                relative_sum = raw_relative.sum(dim=2, keepdim=True)
            composition_relative = raw_relative / relative_sum.clamp(min=self.proportional_exp_eps)
            composition_assets = composition_relative * center_equity.unsqueeze(1)
            composition_weights = center_weights.unsqueeze(1).expand(batch_size, n_composition, action_dim).clone()
            composition_weights[:, :, 1:] = composition_assets
            candidate_weights[:, next_idx:next_idx + n_composition, :] = composition_weights
            family_ids[next_idx:next_idx + n_composition] = 2

        flat_weights = candidate_weights.reshape(batch_size * num_candidates, action_dim)
        flat_mask = action_mask.unsqueeze(1).expand(batch_size, num_candidates, action_dim).reshape(batch_size * num_candidates, action_dim)
        flat_candidates = _portfolio_weights_to_raw_actions(flat_weights, flat_mask, eps=self.proportional_exp_eps)
        low_view = low.to(device=center_actions.device, dtype=center_actions.dtype).view(1, 1, -1)
        high_view = high.to(device=center_actions.device, dtype=center_actions.dtype).view(1, 1, -1)
        candidates = flat_candidates.reshape(batch_size, num_candidates, action_dim)
        candidates = torch.max(torch.min(candidates, high_view), low_view)
        return candidates, family_ids

    def _evaluate_candidate_critics(self, critic_module, obs_float, candidate_actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Evaluate critics on a [batch, K, action_dim] candidate set and return Q values shaped [batch, K].
        """
        encoded_candidate_evaluator = getattr(
            critic_module,
            "evaluate_action_candidates",
            None,
        )
        if callable(encoded_candidate_evaluator):
            return tuple(
                encoded_candidate_evaluator(obs_float, candidate_actions)
            )
        batch_size, num_candidates, action_dim = candidate_actions.shape
        expanded_obs = self._expand_observations_for_candidates(obs_float, num_candidates)
        flat_actions = candidate_actions.reshape(batch_size * num_candidates, action_dim)
        q_outputs = critic_module(expanded_obs, flat_actions)
        if not isinstance(q_outputs, (tuple, list)):
            q_outputs = (q_outputs,)
        return tuple(q.reshape(batch_size, num_candidates, -1).squeeze(-1) for q in q_outputs)

    def _update_proportional_exp_disagreement_ema(self, disagreement_mean: torch.Tensor, channel: str) -> torch.Tensor:
        """
        Maintain an EMA baseline of mean critic disagreement for the disagreement-aware alpha mode.
        """
        batch_mean = disagreement_mean.detach().mean()
        if channel == "budget":
            current_baseline = self._proportional_exp_budget_disagreement_ema
        elif channel == "composition":
            current_baseline = self._proportional_exp_composition_disagreement_ema
        else:
            raise ValueError(f"Unknown proportional-exp disagreement channel: {channel}")

        if current_baseline is None:
            updated = batch_mean
        else:
            decay = self.proportional_exp_ema_decay
            updated = (decay * current_baseline) + ((1.0 - decay) * batch_mean)

        if channel == "budget":
            self._proportional_exp_budget_disagreement_ema = updated
        else:
            self._proportional_exp_composition_disagreement_ema = updated
        return updated

    def _get_proportional_exp_alpha(
        self,
        disagreement_mean: torch.Tensor,
        candidate_q_values: torch.Tensor,
        channel: str,
        update_ema: bool = False,
    ) -> torch.Tensor:
        """
        Return a detached, Q-scale-aware effective alpha for each sample.

        ``proportional_exp_alpha`` is an absolute safety ceiling, not a fixed
        temperature floor. The operative temperature follows the within-family
        Q dispersion, so multiplying every Q by a positive constant preserves
        the Boltzmann weights until the explicit safety bounds are reached.
        """
        detached_q = candidate_q_values.detach()
        detached_disagreement = disagreement_mean.detach()
        q_spread = detached_q.std(dim=1, unbiased=False)
        absolute_floor = max(
            self.proportional_exp_eps,
            self.proportional_exp_alpha
            * self.proportional_exp_alpha_floor_ratio,
        )
        base_alpha = torch.clamp(
            q_spread * self.proportional_exp_q_temperature_scale,
            min=absolute_floor,
            max=self.proportional_exp_alpha,
        )
        alpha = base_alpha
        if self.proportional_exp_mode != "boltzmann_disagreement":
            return alpha

        if update_ema:
            baseline = self._update_proportional_exp_disagreement_ema(
                detached_disagreement,
                channel=channel,
            )
        elif channel == "budget" and self._proportional_exp_budget_disagreement_ema is not None:
            baseline = self._proportional_exp_budget_disagreement_ema
        elif channel == "composition" and self._proportional_exp_composition_disagreement_ema is not None:
            baseline = self._proportional_exp_composition_disagreement_ema
        else:
            baseline = detached_disagreement.mean()

        baseline_scalar = float(torch.clamp(baseline.detach(), min=self.proportional_exp_eps).item())
        ratio = detached_disagreement / (
            baseline_scalar + self.proportional_exp_eps
        )
        alpha = base_alpha * (
            1.0
            + self.proportional_exp_disagreement_gain
            * torch.relu(ratio - 1.0)
        )
        return torch.clamp(
            alpha,
            min=absolute_floor,
            max=self.proportional_exp_alpha_max,
        )

    def _compose_center_anchored_proportional_exp_value(
        self,
        center_value: torch.Tensor,
        budget_soft_value: Optional[torch.Tensor],
        composition_soft_value: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Keep the proposed action primary and add bounded basin residuals."""
        value = center_value
        if budget_soft_value is not None:
            value = value + self.proportional_exp_budget_residual_weight * (
                budget_soft_value - center_value
            )
        if composition_soft_value is not None:
            value = value + self.proportional_exp_composition_residual_weight * (
                composition_soft_value - center_value
            )
        return value

    def _compute_soft_value(self, candidate_q_values: torch.Tensor, alpha: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Compute alpha * logmeanexp(Q / alpha) over the candidate dimension.
        """
        if not torch.is_tensor(alpha):
            alpha = candidate_q_values.new_full((candidate_q_values.shape[0],), float(alpha))
        alpha = torch.clamp(alpha.to(candidate_q_values.dtype), min=self.proportional_exp_eps)
        scaled_q = candidate_q_values / alpha.unsqueeze(-1)
        return alpha * (torch.logsumexp(scaled_q, dim=1) - math.log(candidate_q_values.shape[1]))

    def _compute_proportional_exp_statistics(
        self,
        critic_module,
        obs_float,
        candidate_actions: torch.Tensor,
        candidate_family_ids: torch.Tensor,
        update_disagreement_ema: bool = False,
        precomputed_q_values: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate a split candidate set and compute a hierarchical local soft value.

        The center action, budget cloud, and composition cloud are evaluated separately.
        This keeps the proportional-exp basin aligned with the portfolio semantics of the
        hierarchical action space instead of forcing one mixed uncertainty measure to
        control both gross-exposure and relative-allocation moves.
        """
        q_values = (
            self._evaluate_candidate_critics(
                critic_module, obs_float, candidate_actions
            )
            if precomputed_q_values is None
            else precomputed_q_values
        )
        q_stack = torch.stack(q_values, dim=0)
        min_q = torch.min(q_stack, dim=0).values
        if q_stack.shape[0] >= 2:
            disagreement = (q_stack[0] - q_stack[1]).abs()
        else:
            disagreement = torch.zeros_like(min_q)

        center_mask = candidate_family_ids == 0
        budget_mask = candidate_family_ids == 1
        composition_mask = candidate_family_ids == 2
        center_index = int(torch.nonzero(center_mask, as_tuple=False)[0].item())
        center_q = min_q[:, center_index]

        component_weights = [1.0]

        budget_q = min_q[:, budget_mask] if budget_mask.any() else None
        composition_q = min_q[:, composition_mask] if composition_mask.any() else None
        budget_disagreement_mean = disagreement[:, budget_mask].mean(dim=1) if budget_mask.any() else None
        composition_disagreement_mean = disagreement[:, composition_mask].mean(dim=1) if composition_mask.any() else None

        budget_alpha = None
        composition_alpha = None
        budget_soft_value = center_q.new_zeros(center_q.shape)
        composition_soft_value = center_q.new_zeros(center_q.shape)
        budget_entropy = center_q.new_zeros(center_q.shape)
        composition_entropy = center_q.new_zeros(center_q.shape)
        budget_candidate_weights = None
        composition_candidate_weights = None

        if budget_q is not None:
            budget_alpha = self._get_proportional_exp_alpha(
                budget_disagreement_mean,
                budget_q,
                channel="budget",
                update_ema=update_disagreement_ema,
            )
            budget_soft_value = self._compute_soft_value(budget_q, budget_alpha)
            budget_candidate_weights = torch.softmax(budget_q / budget_alpha.unsqueeze(-1), dim=1)
            budget_entropy = -(
                budget_candidate_weights * torch.log(budget_candidate_weights.clamp(min=1e-8))
            ).sum(dim=1)
            component_weights.append(float(budget_q.shape[1]))

        if composition_q is not None:
            composition_alpha = self._get_proportional_exp_alpha(
                composition_disagreement_mean,
                composition_q,
                channel="composition",
                update_ema=update_disagreement_ema,
            )
            composition_soft_value = self._compute_soft_value(composition_q, composition_alpha)
            composition_candidate_weights = torch.softmax(
                composition_q / composition_alpha.unsqueeze(-1),
                dim=1,
            )
            composition_entropy = -(
                composition_candidate_weights * torch.log(composition_candidate_weights.clamp(min=1e-8))
            ).sum(dim=1)
            component_weights.append(float(composition_q.shape[1]))

        total_weight = sum(component_weights)
        soft_value = self._compose_center_anchored_proportional_exp_value(
            center_value=center_q,
            budget_soft_value=(budget_soft_value if budget_q is not None else None),
            composition_soft_value=(
                composition_soft_value if composition_q is not None else None
            ),
        )

        weighted_entropy = center_q.new_zeros(center_q.shape)
        weighted_alpha = center_q.new_full(center_q.shape, fill_value=self.proportional_exp_alpha)
        weighted_disagreement = disagreement.mean(dim=1)
        non_center_weight = max(total_weight - 1.0, 1.0)

        if budget_q is not None or composition_q is not None:
            entropy_terms = []
            alpha_terms = []
            disagreement_terms = []
            if budget_q is not None and budget_alpha is not None and budget_disagreement_mean is not None:
                entropy_terms.append(float(budget_q.shape[1]) * budget_entropy)
                alpha_terms.append(float(budget_q.shape[1]) * budget_alpha)
                disagreement_terms.append(float(budget_q.shape[1]) * budget_disagreement_mean)
            if composition_q is not None and composition_alpha is not None and composition_disagreement_mean is not None:
                entropy_terms.append(float(composition_q.shape[1]) * composition_entropy)
                alpha_terms.append(float(composition_q.shape[1]) * composition_alpha)
                disagreement_terms.append(float(composition_q.shape[1]) * composition_disagreement_mean)
            if entropy_terms:
                weighted_entropy = sum(entropy_terms) / non_center_weight
            if alpha_terms:
                weighted_alpha = sum(alpha_terms) / non_center_weight
            if disagreement_terms:
                weighted_disagreement = sum(disagreement_terms) / non_center_weight

        return {
            "q_values": q_values,
            "min_q": min_q,
            "disagreement_mean": weighted_disagreement,
            "alpha": weighted_alpha,
            "weights": None,
            "entropy": weighted_entropy,
            "soft_value": soft_value,
            "center_q": center_q,
            "budget_alpha": budget_alpha,
            "composition_alpha": composition_alpha,
            "budget_disagreement_mean": budget_disagreement_mean,
            "composition_disagreement_mean": composition_disagreement_mean,
            "budget_entropy": budget_entropy,
            "composition_entropy": composition_entropy,
            "budget_soft_value": budget_soft_value,
            "composition_soft_value": composition_soft_value,
            "budget_candidate_weights": budget_candidate_weights,
            "composition_candidate_weights": composition_candidate_weights,
        }

    def _qwm_current_depth(self) -> int:
        return qwm_search_depth(
            self.qwm_world_updates,
            initial_depth=self.qwm_initial_depth,
            maximum_depth=self.qwm_max_depth,
            depth_three_updates=self.qwm_depth_three_updates,
            depth_four_updates=self.qwm_depth_four_updates,
        )

    def _sample_qwm_candidate_actions(
        self,
        center_actions: torch.Tensor,
        observations: Dict[str, torch.Tensor],
        count: int,
        low: torch.Tensor,
        high: torch.Tensor,
        *,
        repeat_noise_groups: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct a heterogeneous QWM proposal set in portfolio space.

        The local ProExp cloud remains the actor-training distribution.  Tree
        search needs broader counterfactual coverage: an actor center, inertia,
        two structural anchors, multi-scale perturbations, and a global simplex
        draw.  This is proposal diversity, not foundation supervision.
        """
        center_actions = center_actions.float()
        batch_size, action_dim = center_actions.shape
        count = max(1, int(count))
        action_mask = _build_action_mask_from_observations(
            observations,
            action_dim=action_dim,
            device=center_actions.device,
            dtype=center_actions.dtype,
        )
        asset_mask = action_mask[:, 1:]
        center_weights = normalize_raw_actions_to_portfolio_weights(
            center_actions, action_mask
        )
        previous = observations["prev_action"].float()
        previous = previous * action_mask
        previous = previous / previous.sum(dim=1, keepdim=True).clamp(min=1e-6)
        equal = center_weights.new_zeros((batch_size, action_dim))
        equal[:, 1:] = asset_mask / asset_mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)
        cash = center_weights.new_zeros((batch_size, action_dim))
        cash[:, 0] = 1.0

        repeat_noise_groups = max(1, int(repeat_noise_groups))
        if batch_size % repeat_noise_groups != 0:
            raise ValueError("QWM proposal noise groups must divide the batch")

        def _random(*shape_tail: int) -> torch.Tensor:
            base_batch = batch_size // repeat_noise_groups
            sample = torch.rand(
                base_batch,
                *shape_tail,
                device=center_actions.device,
                dtype=center_actions.dtype,
            )
            if repeat_noise_groups > 1:
                sample = sample.repeat(
                    repeat_noise_groups, *([1] * len(shape_tail))
                )
            return sample

        center_cash = center_weights[:, :1]
        center_equity = (1.0 - center_cash).clamp(min=0.0)
        center_relative = _safe_relative_asset_weights(
            center_weights, action_mask, eps=self.proportional_exp_eps
        )

        def _composition(scale: float) -> torch.Tensor:
            perturbation = (
                2.0 * _random(action_dim - 1) - 1.0
            ) * float(scale)
            relative = (center_relative + perturbation * asset_mask).clamp(min=0.0)
            relative = relative * asset_mask
            fallback = asset_mask / asset_mask.sum(
                dim=1, keepdim=True
            ).clamp(min=1.0)
            relative = torch.where(
                (relative.sum(dim=1, keepdim=True) <= self.proportional_exp_eps),
                fallback,
                relative,
            )
            relative = relative / relative.sum(
                dim=1, keepdim=True
            ).clamp(min=self.proportional_exp_eps)
            result = center_weights.clone()
            result[:, 1:] = relative * center_equity
            return result

        local_budget = center_weights.clone()
        budget_delta = (
            2.0 * _random(1) - 1.0
        ) * self.qwm_proposal_local_scale
        local_budget[:, :1] = (center_cash + budget_delta).clamp(0.0, 1.0)
        local_budget[:, 1:] = center_relative * (1.0 - local_budget[:, :1])

        global_weights = center_weights.new_zeros((batch_size, action_dim))
        global_cash = _random(1) * self.qwm_proposal_global_cash_max
        global_logits = -torch.log(_random(action_dim - 1).clamp(min=1e-6))
        global_logits = global_logits * asset_mask
        global_relative = global_logits / global_logits.sum(
            dim=1, keepdim=True
        ).clamp(min=1e-6)
        global_weights[:, :1] = global_cash
        global_weights[:, 1:] = global_relative * (1.0 - global_cash)

        weight_candidates = [
            center_weights,
            previous,
            equal,
            cash,
            local_budget,
            _composition(self.qwm_proposal_local_scale),
            _composition(self.qwm_proposal_medium_scale),
            global_weights,
        ]
        while len(weight_candidates) < count:
            scale = (
                self.qwm_proposal_local_scale
                if len(weight_candidates) % 2 == 0
                else self.qwm_proposal_medium_scale
            )
            weight_candidates.append(_composition(scale))
        weights = torch.stack(weight_candidates[:count], dim=1)
        flat_mask = action_mask.unsqueeze(1).expand(
            -1, count, -1
        ).reshape(batch_size * count, action_dim)
        raw = _portfolio_weights_to_raw_actions(
            weights.reshape(batch_size * count, action_dim),
            flat_mask,
            eps=self.proportional_exp_eps,
        ).reshape(batch_size, count, action_dim)
        low_view = low.to(raw).view(1, 1, -1)
        high_view = high.to(raw).view(1, 1, -1)
        raw = torch.max(torch.min(raw, high_view), low_view)
        family_ids = raw.new_full((count,), 2, dtype=torch.long)
        family_ids[0] = 0
        if count > 4:
            family_ids[4] = 1
        return raw, family_ids

    def _qwm_propose_actions(
        self,
        observations: Dict[str, torch.Tensor],
        count: int,
        *,
        deterministic_candidates: bool,
        proposal_index: int,
        deterministic_groups: int = 1,
    ) -> torch.Tensor:
        gated_observations = self._gate_observations_with_sigma(observations)
        _qwm_mark_compiled_step(
            self.actor, "forward", gated_observations["mask"]
        )
        center = self.actor(self._actor_observation(gated_observations))
        low, high = self._get_action_bounds()

        def _sample() -> torch.Tensor:
            candidates, family_ids = self._sample_qwm_candidate_actions(
                center,
                gated_observations,
                count,
                low,
                high,
                repeat_noise_groups=(
                    deterministic_groups if deterministic_candidates else 1
                ),
            )
            if proposal_index == 0:
                self._qwm_last_root_family_ids = family_ids
            return candidates

        if not deterministic_candidates:
            return _sample()
        cuda_devices = []
        if center.is_cuda:
            cuda_devices = [
                center.device.index
                if center.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.qwm_search_seed + int(proposal_index))
            return _sample()

    def _qwm_crl_candidate_penalty(
        self,
        observations: Dict[str, torch.Tensor],
        candidate_actions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, count, action_dim = candidate_actions.shape
        if not (
            self.qwm_crl_aware_scoring
            and self.use_crl
            and "prev_action" in observations
        ):
            return candidate_actions.new_zeros((batch_size, count))
        action_mask = _build_action_mask_from_observations(
            observations,
            action_dim=action_dim,
            device=candidate_actions.device,
            dtype=candidate_actions.dtype,
        )
        expanded_mask = action_mask.unsqueeze(1).expand(-1, count, -1)
        flat_weights = normalize_raw_actions_to_portfolio_weights(
            candidate_actions.reshape(batch_size * count, action_dim),
            expanded_mask.reshape(batch_size * count, action_dim),
        )
        weights = flat_weights.reshape(batch_size, count, action_dim)
        previous = observations["prev_action"].float().unsqueeze(1)
        velocity_cost = (
            (weights - previous).square() * expanded_mask
        ).sum(dim=-1)
        penalty = torch.clamp(self.lambda_param_velocity.detach(), min=0.0) * velocity_cost

        older = observations.get("_qwm_older_action")
        if (
            older is not None
            and self._n_updates >= self.lambda_warmup_steps
        ):
            older = older.float().unsqueeze(1)
            acceleration_cost = (
                (weights - 2.0 * previous + older).square() * expanded_mask
            ).sum(dim=-1)
            penalty = penalty + (
                torch.clamp(self.lambda_param_acceleration.detach(), min=0.0)
                * acceleration_cost
            )
        return penalty

    def _qwm_candidate_values(
        self,
        observations: Dict[str, torch.Tensor],
        candidate_actions: torch.Tensor,
        *,
        return_q_values: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        gated_observations = self._gate_observations_with_sigma(observations)
        q_values = tuple(
            self._evaluate_candidate_critics(
                self.critic, gated_observations, candidate_actions
            )
        )
        q_stack = torch.stack(q_values, dim=0)
        twin_minimum = q_stack.min(dim=0).values
        values = twin_minimum - self._qwm_crl_candidate_penalty(
            observations, candidate_actions
        )
        if return_q_values:
            return values, q_values
        return values

    def _qwm_transition(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        model_samples: int,
        *,
        generator: Optional[torch.Generator] = None,
        latent_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if self.qwm_world_model is None:
            raise RuntimeError("QWM transition requested without a world model")
        self.qwm_world_model.eval()
        use_autocast = bool(
            self.qwm_cuda_autocast
            and actions.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        if use_autocast:
            _qwm_mark_compiled_step(
                self.qwm_world_model, "sample", actions
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                next_observations, predicted_rewards = self.qwm_world_model.sample(
                    observations,
                    actions,
                    int(model_samples),
                    generator=generator,
                    latent_noise=latent_noise,
                )
        else:
            _qwm_mark_compiled_step(
                self.qwm_world_model, "sample", actions
            )
            next_observations, predicted_rewards = self.qwm_world_model.sample(
                observations,
                actions,
                int(model_samples),
                generator=generator,
                latent_noise=latent_noise,
            )

        return self._qwm_finalize_transition(
            observations,
            actions,
            next_observations,
            predicted_rewards,
            int(model_samples),
        )

    def _qwm_transition_candidates(
        self,
        observations: Dict[str, torch.Tensor],
        candidate_actions: torch.Tensor,
        model_samples: int,
        *,
        generator: Optional[torch.Generator] = None,
        latent_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Expand root actions after one shared state preparation."""
        if self.qwm_world_model is None:
            raise RuntimeError("QWM transition requested without a world model")
        self.qwm_world_model.eval()
        use_autocast = bool(
            self.qwm_cuda_autocast
            and candidate_actions.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        if use_autocast:
            _qwm_mark_compiled_step(
                self.qwm_world_model,
                "sample_action_candidates",
                candidate_actions,
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                next_observations, predicted_rewards = (
                    self.qwm_world_model.sample_action_candidates(
                        observations,
                        candidate_actions,
                        int(model_samples),
                        generator=generator,
                        latent_noise=latent_noise,
                    )
                )
        else:
            _qwm_mark_compiled_step(
                self.qwm_world_model,
                "sample_action_candidates",
                candidate_actions,
            )
            next_observations, predicted_rewards = (
                self.qwm_world_model.sample_action_candidates(
                    observations,
                    candidate_actions,
                    int(model_samples),
                    generator=generator,
                    latent_noise=latent_noise,
                )
            )
        flat_actions = candidate_actions.reshape(
            -1, candidate_actions.shape[-1]
        )
        repeated_observations = _qwm_repeat_observations(
            observations, candidate_actions.shape[1]
        )
        return self._qwm_finalize_transition(
            repeated_observations,
            flat_actions,
            next_observations,
            predicted_rewards,
            int(model_samples),
        )

    def _qwm_finalize_transition(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_observations: Dict[str, torch.Tensor],
        predicted_rewards: torch.Tensor,
        model_samples: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Own compiled outputs, restore dtype, and retain device telemetry."""

        # The compiled world-model methods use CUDA Graphs in reduce-overhead
        # mode. Their outputs live in replayable static buffers. The tree must
        # own a copy before actor/critic/world-model graph replay can begin.
        next_observations, predicted_rewards = (
            _qwm_materialize_compiled_transition(
                next_observations, predicted_rewards
            )
        )

        samples = int(model_samples)
        if samples > 1 and "observations" in next_observations:
            normalized_tokens = self.qwm_world_model._normalized_observation_tokens(
                next_observations["observations"]
            ).reshape(
                actions.shape[0],
                samples,
                self.qwm_world_model.max_coins,
                self.qwm_world_model.features_per_coin,
            )
            active_mask = observations["mask"].float()[:, None, :, None]
            scenario_std = normalized_tokens.std(dim=1, unbiased=False)
            state_std = (scenario_std * active_mask[:, 0]).sum() / (
                active_mask[:, 0].sum()
                * self.qwm_world_model.features_per_coin
            ).clamp(min=1.0)
            reward_std = predicted_rewards.std(
                dim=1, unbiased=False
            ).mean()
            self._qwm_successor_telemetry = torch.stack(
                (state_std.detach().float(), reward_std.detach().float())
            )
        else:
            self._qwm_successor_telemetry = actions.new_zeros(2).float()
        return next_observations, predicted_rewards

    @torch.inference_mode()
    def _qwm_tree_search(
        self,
        observations: Dict[str, torch.Tensor],
        *,
        online_sampling: bool,
    ) -> torch.Tensor:
        if not self.qwm_ready or self.qwm_world_model is None:
            raise RuntimeError("QWM search requested before world-model pretraining")
        proposal_index = 0
        logical_batch_size = int(observations["mask"].shape[0])
        model_generator = None
        if not online_sampling:
            generator_device = (
                observations["mask"].device
                if observations["mask"].device.type == "cuda"
                else torch.device("cpu")
            )
            model_generator = torch.Generator(device=generator_device)
            model_generator.manual_seed(self.qwm_search_seed + 100_003)

        def deterministic_latent_noise(row_count, samples):
            if online_sampling:
                return None
            if not hasattr(self.qwm_world_model, "_sample_latent_noise"):
                return None
            if row_count % logical_batch_size != 0:
                raise ValueError(
                    "QWM deterministic rows must divide into logical searches"
                )
            rows_per_search = row_count // logical_batch_size
            latent_dtype = (
                torch.bfloat16
                if self.qwm_cuda_autocast
                and observations["mask"].device.type == "cuda"
                and torch.cuda.is_bf16_supported()
                else next(self.qwm_world_model.parameters()).dtype
            )
            base_noise = self.qwm_world_model._sample_latent_noise(
                rows_per_search,
                int(samples),
                device=observations["mask"].device,
                dtype=latent_dtype,
                generator=model_generator,
            )
            return base_noise.repeat(
                logical_batch_size, 1, 1
            )

        def propose(state, count):
            nonlocal proposal_index
            actions = self._qwm_propose_actions(
                state,
                count,
                deterministic_candidates=not online_sampling,
                proposal_index=proposal_index,
                deterministic_groups=(
                    logical_batch_size if not online_sampling else 1
                ),
            )
            proposal_index += 1
            return actions

        def transition(state, actions, samples):
            return self._qwm_transition(
                state,
                actions,
                samples,
                generator=(model_generator if online_sampling else None),
                latent_noise=deterministic_latent_noise(
                    actions.shape[0], samples
                ),
            )

        def transition_candidates(state, actions, samples):
            shared_candidate_noise = None
            if not online_sampling and hasattr(
                self.qwm_world_model, "_sample_latent_noise"
            ):
                latent_dtype = (
                    torch.bfloat16
                    if self.qwm_cuda_autocast
                    and observations["mask"].device.type == "cuda"
                    and torch.cuda.is_bf16_supported()
                    else next(self.qwm_world_model.parameters()).dtype
                )
                # Shape [logical state, K, latent]. The world model broadcasts
                # each scenario across all N actions, so candidate comparisons
                # use common random market futures.
                shared_candidate_noise = self.qwm_world_model._sample_latent_noise(
                    actions.shape[0],
                    int(samples),
                    device=observations["mask"].device,
                    dtype=latent_dtype,
                    generator=model_generator,
                )
            return self._qwm_transition_candidates(
                state,
                actions,
                samples,
                generator=(model_generator if online_sampling else None),
                latent_noise=shared_candidate_noise,
            )

        root_q_values = None
        candidate_call_index = 0

        def score_candidates(state, actions):
            nonlocal root_q_values, candidate_call_index
            if (
                candidate_call_index == 0
                and hasattr(
                    getattr(self, "critic", None),
                    "evaluate_action_candidates",
                )
            ):
                values, root_q_values = self._qwm_candidate_values(
                    state, actions, return_q_values=True
                )
            else:
                values = self._qwm_candidate_values(state, actions)
            candidate_call_index += 1
            return values

        depth = self._qwm_current_depth()
        root_actions, root_scores = qwm_tree_search(
            observations,
            propose_actions=propose,
            candidate_values=score_candidates,
            transition=transition,
            depth=depth,
            candidates=self.qwm_candidates,
            leaf_candidates=self.qwm_leaf_candidates,
            model_samples=self.qwm_model_samples,
            beam_width=self.qwm_beam_width,
            search_discount=self.qwm_search_discount,
            leaf_aggregation=self.qwm_leaf_aggregation,
            successor_aggregation=self.qwm_successor_aggregation,
            transition_candidates=(
                transition_candidates
                if hasattr(
                    self.qwm_world_model, "sample_action_candidates"
                )
                else None
            ),
        )
        if online_sampling:
            gated_observations = self._gate_observations_with_sigma(observations)
            root_family_ids = self._qwm_last_root_family_ids
            if (
                root_family_ids is None
                or root_family_ids.shape[0] != root_actions.shape[1]
            ):
                root_family_ids = root_actions.new_full(
                    (root_actions.shape[1],), 2, dtype=torch.long
                )
                root_family_ids[0] = 0
            root_statistics = self._compute_proportional_exp_statistics(
                self.critic,
                gated_observations,
                root_actions,
                root_family_ids,
                update_disagreement_ema=False,
                precomputed_q_values=root_q_values,
            )
            alpha = root_statistics["alpha"]
            probabilities = torch.softmax(
                root_scores / alpha.unsqueeze(-1), dim=1
            )
            selected = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        else:
            selected = root_scores.argmax(dim=1)
        # Count logical searched states, not Python invocations. Batched
        # collection/evaluation must remain directly comparable with the
        # historical one-state-per-call telemetry.
        self.qwm_search_calls += int(root_actions.shape[0])
        self.qwm_last_search_depth = depth
        selected_scores = root_scores.gather(1, selected.unsqueeze(1)).squeeze(1)
        selected_actions = root_actions[
            torch.arange(root_actions.shape[0], device=root_actions.device),
            selected,
        ]
        action_mask = _build_action_mask_from_observations(
            observations,
            action_dim=root_actions.shape[-1],
            device=root_actions.device,
            dtype=root_actions.dtype,
        )
        flat_mask = action_mask.unsqueeze(1).expand(
            -1, root_actions.shape[1], -1
        ).reshape(-1, root_actions.shape[-1])
        root_weights = normalize_raw_actions_to_portfolio_weights(
            root_actions.reshape(-1, root_actions.shape[-1]), flat_mask
        ).reshape_as(root_actions)
        selected_weights = root_weights[
            torch.arange(root_actions.shape[0], device=root_actions.device),
            selected,
        ]
        selected_distance = (
            (selected_weights - root_weights[:, 0]) * action_mask
        ).abs().sum(dim=1).mean()
        center_selected = (selected == 0).float().mean()
        score_gain = (selected_scores - root_scores[:, 0]).mean()
        if root_q_values is not None:
            direct_values = torch.stack(root_q_values, dim=0).min(dim=0).values
            direct_values = direct_values - self._qwm_crl_candidate_penalty(
                observations, root_actions
            )
            direct_argmax = direct_values.argmax(dim=1)
            tree_q_disagreement = (selected != direct_argmax).float().mean()
            model_contribution_std = (
                root_scores - direct_values
            ).std(unbiased=False)
        else:
            tree_q_disagreement = root_scores.new_zeros(())
            model_contribution_std = root_scores.new_zeros(())
        self._qwm_root_telemetry = torch.stack(
            (
                root_scores.mean().detach().float(),
                root_scores.std(unbiased=False).detach().float(),
                center_selected.detach().float(),
                score_gain.detach().float(),
                selected_distance.detach().float(),
                tree_q_disagreement.detach().float(),
                model_contribution_std.detach().float(),
            )
        )
        return selected_actions

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        """Use the same QWM controller for backtests once its model is ready."""
        if not (
            self.qwm_enabled
            and self.qwm_ready
            and self.qwm_evaluation_search_enabled
            and isinstance(observation, dict)
        ):
            return super().predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=deterministic,
            )
        self.policy.set_training_mode(False)
        observation_tensor, vectorized = self.policy.obs_to_tensor(observation)
        selected = self._qwm_tree_search(
            observation_tensor,
            online_sampling=not deterministic,
        )
        actions = selected.detach().cpu().numpy()
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        if not vectorized:
            actions = actions.squeeze(axis=0)
        return actions, state

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search before env.step(); store only the resulting real transition."""
        if not (
            self.qwm_enabled
            and self.qwm_ready
            and self.qwm_online_search_enabled
            and self.num_timesteps >= learning_starts
            and isinstance(self._last_obs, dict)
        ):
            return super()._sample_action(
                learning_starts,
                action_noise=action_noise,
                n_envs=n_envs,
            )
        observation_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
        unscaled_action = self._qwm_tree_search(
            observation_tensor,
            online_sampling=True,
        ).detach().cpu().numpy()
        unscaled_action = np.clip(
            unscaled_action, self.action_space.low, self.action_space.high
        )
        # Do not add OU noise after search: that would execute a different
        # action from the one the world model and critics selected.  Stochastic
        # ProExp candidates plus Boltzmann root selection provide exploration.
        buffer_action = self.policy.scale_action(unscaled_action)
        buffer_action = np.clip(buffer_action, -1.0, 1.0)
        action = self.policy.unscale_action(buffer_action)
        return action, buffer_action

    def _update_learning_rate(self, optimizers) -> None:
        """
        Update learning rates according to schedules.
        This is called automatically by the training loop.
        """
        if self.overall_total_timesteps is not None:
            # When using a custom training loop, progress must be calculated manually.
            progress = 1.0 - self.num_timesteps / self.overall_total_timesteps
        else:
            # Default SB3 behavior.
            progress = self._current_progress_remaining

        # Update actor learning rate
        new_actor_lr = self.actor_lr_schedule(progress)
        for param_group in self.actor.optimizer.param_groups:
            param_group["lr"] = new_actor_lr

        # Update critic learning rate
        new_critic_lr = self.critic_lr_schedule(progress)
        for param_group in self.critic.optimizer.param_groups:
            param_group["lr"] = new_critic_lr

        # Log the learning rates for monitoring
        self.logger.record("train/actor_lr", new_actor_lr)
        self.logger.record("train/critic_lr", new_critic_lr)

        # Log CRL-specific metrics
        if self.use_crl:
            current_lambda_vel = self.lambda_param_velocity.item()
            current_lambda_accel = self.lambda_param_acceleration.item()
            self.logger.record("train/lambda_velocity", current_lambda_vel)
            self.logger.record("train/lambda_acceleration", current_lambda_accel)
            if len(self.velocity_constraint_history) > 0:
                self.logger.record("train/avg_velocity_cost", self.velocity_constraint_history[-1])
                self.logger.record("train/velocity_threshold", self.velocity_constraint_threshold)
            if len(self.acceleration_constraint_history) > 0:
                self.logger.record("train/avg_acceleration_cost", self.acceleration_constraint_history[-1])
                self.logger.record("train/acceleration_threshold", self.acceleration_constraint_threshold)

    def _update_lambdas(self, velocity_violation: torch.Tensor, acceleration_violation: Optional[torch.Tensor]) -> None:
        """
        Update Lagrange multipliers λ using a stable, asymmetric additive update rule.
        This avoids the upward creep of exponential updates while preventing collapse to zero.

        ✅ BIDIRECTIONAL LEARNING PHILOSOPHY:
        Both velocity and acceleration lambdas can increase or decrease based on violation signals
        throughout all training phases. The dead-band holds lambda constant (no forced decay) to allow
        the model to re-explore and learn optimal constraints after curriculum phases complete.

        The curriculum progression is managed by the coupling factor that gates the acceleration penalty,
        NOT by artificially weakening the velocity lambda through decay. This allows velocity constraints
        to naturally find their optimal learned threshold rather than being permanently suppressed.
        """
        if not self.use_crl:
            return

        with torch.no_grad():
            # --- General Helper for the Asymmetric Additive Update ---
            def _update_single_lambda(lambda_param, violation, lambda_lr):
                if violation is None:
                    return

                # Initialize EMA tracker if it doesn't exist
                is_velocity = (lambda_param is self.lambda_param_velocity)


                if is_velocity and self._viol_ema_v is None:
                    self._viol_ema_v = torch.zeros_like(violation)
                elif not is_velocity and self._viol_ema_a is None:
                    self._viol_ema_a = torch.zeros_like(violation)
                
                viol_ema = self._viol_ema_v if is_velocity else self._viol_ema_a
                # ✅ ACCELERATION RESPONSIVENESS: Use higher beta (less smoothing) for acceleration
                # to reduce lag and allow faster convergence to threshold
                beta = 0.6 if not is_velocity else 0.4  # Acceleration: 0.6 (more responsive), Velocity: 0.4 (stable)
                viol_ema.mul_(1 - beta).add_(beta * violation)

                # Parameters for the dead-band
                threshold = self.velocity_constraint_threshold if is_velocity else self.acceleration_constraint_threshold
                # ✅ ASYMMETRIC DEAD-BAND FOR ACCELERATION: Allow fine-tuning when cost is below threshold
                # - Velocity: Symmetric dead-band (5% tolerance) for stability
                # - Acceleration: Asymmetric dead-band - no dead-band when violation is negative
                #   (cost < threshold), allowing lambda to decrease and cost to rise back to threshold.
                #   This prevents the asymptotic "approach but never reach" behavior.
                if is_velocity:
                    dead_band_width = 0.05 * float(threshold)  # Symmetric 5% tolerance zone
                else:
                    # Acceleration: Only dead-band on positive side (when cost > threshold)
                    # No dead-band on negative side (when cost < threshold) to allow convergence
                    dead_band_width_positive = 0.05 * float(threshold)  # Dead-band when cost > threshold
                    dead_band_width_negative = 0.0  # No dead-band when cost < threshold (allow fine-tuning)
                lam_min = 1e-18
                lam_max_velocity = 1e5  # Very loose upper bound for velocity lambda

                # ✅ ROBUSTNESS FIX: Use a linear multiplicative update instead of exponential to prevent blowup.
                # This is more stable as it avoids the sensitivity of the exp() function.
                max_update_factor_velocity = 1.015 # Max 1.5% increase per step
                min_update_factor_velocity = 0.985 # Max 1.5% decrease per step
                # ✅ ADAPTIVE ACCELERATION UPDATES: Scale update rate based on violation magnitude.
                # When cost is well above threshold (large positive violation), allow larger updates
                # to converge faster. When close to threshold, use smaller updates for stability.
                # Base rate: 0.2% per step (2x previous), can scale up to 0.5% for large violations.
                max_update_factor_acceleration = 1.005 # Max 0.5% increase per step (5x previous for faster convergence)
                min_update_factor_acceleration = 0.998 # Max 0.2% decrease per step (2x previous for faster convergence)

                max_update = max_update_factor_velocity if is_velocity else max_update_factor_acceleration
                min_update = min_update_factor_velocity if is_velocity else min_update_factor_acceleration
                
                # ✅ ADAPTIVE UPDATE SCALING: Scale update magnitude based on violation size for acceleration.
                # When violation is large (cost well above threshold), use larger updates to converge faster.
                # This addresses the asymptotic approach issue where small updates can't overcome the gap.
                # Calculate adaptive learning rate BEFORE clamping the violation signal.
                if not is_velocity:
                    # For acceleration, scale update rate by violation magnitude (use raw EMA, not clamped)
                    # Large violations (> 2x threshold) get 2x update rate, medium (1-2x) get 1.5x, small (< 1x) get 1x
                    violation_magnitude = torch.abs(viol_ema)
                    threshold_abs = float(threshold)
                    if violation_magnitude > 2.0 * threshold_abs:
                        adaptive_lambda_lr = lambda_lr * 2.0  # Large violation: 2x update rate
                    elif violation_magnitude > threshold_abs:
                        adaptive_lambda_lr = lambda_lr * 1.5  # Medium violation: 1.5x update rate
                    else:
                        adaptive_lambda_lr = lambda_lr  # Small violation: normal update rate
                else:
                    adaptive_lambda_lr = lambda_lr  # Velocity: no adaptive scaling
                
                # ✅ INSIGHT FROM BATCH.PY: Clamp the violation signal to tame the exp() update for stability.
                v = viol_ema.clamp(-1.0, 1.0)

                # ✅ WARM-UP STRUCTURE: During warm-up, only allow lambdas to increase.
                if self._n_updates < self.lambda_warmup_steps:
                    if is_velocity:
                        # Velocity: symmetric dead-band
                        if v > dead_band_width:
                            update_factor = 1.0 + lambda_lr * (v - dead_band_width)
                            update_factor = torch.clamp(update_factor, 1.0, max_update)
                            lambda_param.mul_(update_factor)
                    else:
                        # Acceleration: no dead-band during warm-up (always increase if positive)
                        if v > 0.0:
                            update_factor = 1.0 + adaptive_lambda_lr * v #adaptive_lambda_lr
                            update_factor = torch.clamp(update_factor, 1.0, max_update)
                            lambda_param.mul_(update_factor)
                else:
                    # ✅ ASYMMETRIC UPDATE LOGIC: Different behavior for velocity vs acceleration
                    if is_velocity:
                        # Velocity: symmetric dead-band logic with bidirectional learning
                        if v > dead_band_width:
                            update_factor = 1.0 + lambda_lr * (v - dead_band_width)
                            update_factor = torch.clamp(update_factor, 1.0, max_update)
                            lambda_param.mul_(update_factor)
                        elif v < -dead_band_width:
                            update_factor = 1.0 + lambda_lr * (v + dead_band_width)
                            update_factor = torch.clamp(update_factor, min_update, 1.0)
                            lambda_param.mul_(update_factor)
                        else:
                            # ✅ BIDIRECTIONAL LEARNING: Dead-band holds lambda constant instead of decay.
                            # This allows velocity constraint to re-learn optimal threshold after acceleration
                            # refinement completes, rather than being permanently suppressed by accumulated decay.
                            # The curriculum progression is already managed by the coupling factor that gates
                            # acceleration penalty, so we don't need to artificially weaken velocity lambda.
                            # lambda_param.mul_(self.lambda_decay_factor)                            
                            pass  # Hold lambda constant - no decay, no update
                    else:
                        # Acceleration: asymmetric dead-band - no dead-band when violation is negative
                        if v > dead_band_width_positive:
                            # Violation is positive (cost > threshold), increase lambda
                            # Use adaptive learning rate for faster convergence when violation is large
                            update_factor = 1.0 + adaptive_lambda_lr * (v - dead_band_width_positive) #adaptive_lambda_lr
                            update_factor = torch.clamp(update_factor, 1.0, max_update)
                            lambda_param.mul_(update_factor)
                        elif v < -dead_band_width_negative:
                            # Violation is negative (cost < threshold), decrease lambda
                            # No dead-band here - allow fine-tuning to bring cost back to threshold
                            # This prevents the asymptotic "approach but never reach" behavior
                            update_factor = 1.0 + adaptive_lambda_lr * (v + dead_band_width_negative) #adaptive_lambda_lr
                            update_factor = torch.clamp(update_factor, min_update, 1.0)
                            lambda_param.mul_(update_factor)
                        else:
                            # Very small violation (within dead-band on positive side or near zero)
                            # Hold lambda constant to allow cost to stabilize at threshold
                            # This creates a stable fixed point at the threshold
                            pass  # No update - lambda holds constant
                # ✅ DIFFERENTIATED CLAMPING:
                # - Velocity lambda can grow quite large without destabilising training.
                # - Acceleration lambda is tied to the *reference* scale of velocity lambda
                #   (its historical peak) and to a global ceiling, so it can never dominate
                #   the loss by orders of magnitude, but it also does not vanish when
                #   λ_velocity decays late in training.
                if is_velocity:
                    lambda_param.clamp_(lam_min, lam_max_velocity)
                    # Track the largest velocity lambda we have seen so far for use as
                    # a stable reference when capping λ_acceleration.
                    current_vel = float(lambda_param.item())
                    if current_vel > self.max_velocity_lambda_reference:
                        self.max_velocity_lambda_reference = current_vel
                else:
                    # Use the larger of the current and reference velocity lambdas as the
                    # basis for the ratio cap. This keeps the acceleration cap anchored
                    # to the strongest phase of velocity control, even if λ_velocity
                    # later decays.
                    vel_ref = max(
                        float(self.lambda_param_velocity.item()),
                        self.max_velocity_lambda_reference,
                    )
                    max_accel_from_ratio = vel_ref * self.max_accel_to_velocity_lambda_ratio
                    # ✅ RATIO CAP ONLY: No global ceiling. The ratio-based bound is sufficient
                    # to prevent explosions (keeps acceleration proportional to velocity) while
                    # allowing lambda to reach empirically necessary levels (0.25-0.30).
                    max_accel_lambda_tensor = lambda_param.new_tensor(max_accel_from_ratio)
                    lambda_param.clamp_(lam_min, max_accel_lambda_tensor)

            # --- Update Both Lambdas Using the Helper ---
            _update_single_lambda(self.lambda_param_velocity, velocity_violation, self.velocity_lambda_lr)
            # ✅ STAGED WARM-UP: Only update acceleration lambda after the warmup period
            if acceleration_violation is not None and self._n_updates >= self.lambda_warmup_steps:
            # ⚡️ if acceleration_violation is not None:
                _update_single_lambda(self.lambda_param_acceleration, acceleration_violation, self.acceleration_lambda_lr)

        # Track for monitoring
        self.velocity_constraint_history.append(velocity_violation.item() + self.velocity_constraint_threshold)
        self.velocity_lambda_history.append(self.lambda_param_velocity.item())
        if acceleration_violation is not None:
            self.acceleration_constraint_history.append(acceleration_violation.item() + self.acceleration_constraint_threshold)
            self.acceleration_lambda_history.append(self.lambda_param_acceleration.item())
        
        # Keep only recent history for efficiency
        if len(self.velocity_constraint_history) > 1000:
            self.velocity_constraint_history = self.velocity_constraint_history[-500:]
            self.velocity_lambda_history = self.velocity_lambda_history[-500:]
        if len(self.acceleration_constraint_history) > 1000:
            self.acceleration_constraint_history = self.acceleration_constraint_history[-500:]
            self.acceleration_lambda_history = self.acceleration_lambda_history[-500:]
        
        if self.verbose > 1 and self._n_updates % 100 == 0:
            log_msg = f"🎯 CRL Update: vel_viol={velocity_violation.item():.4f}, λ_vel={self.lambda_param_velocity.item():.4f}"
            if acceleration_violation is not None:
                log_msg += f" | accel_viol={acceleration_violation.item():.4f}, λ_accel={self.lambda_param_acceleration.item():.4f}"
            print(log_msg)

    def foundation_warmup(self) -> bool:
        """Warm the live actor and critics before their first financial update.

        On the production 8-of-N path the public call samples the indexed full
        market panel immediately after model construction.  Tiny ablations that
        do not contain an out-of-basket peer retain a replay-backed fallback.
        """
        self.foundation_warmup_requested = True
        if not self.foundation_learning_enabled or self.foundation_warmup_complete:
            return bool(self.foundation_warmup_complete)
        if self._foundation_market_sampler is None:
            try:
                from enviorment import FoundationMarketViewSampler

                self._foundation_market_sampler = FoundationMarketViewSampler(
                    seed=int(getattr(config, "BASKET_CURRICULUM_SEED", 0)) + 4049
                )
                if self.foundation_sampler_rng_state is not None:
                    self._foundation_market_sampler.rng.bit_generator.state = (
                        self.foundation_sampler_rng_state
                    )
            except ValueError as exc:
                if self.verbose > 0:
                    print(f"[Foundation] panel sampler unavailable: {exc}")
        replay_size = int(self.replay_buffer.size())
        minimum_rows = self.foundation_max_horizon + 1
        if self._foundation_market_sampler is None and replay_size < minimum_rows:
            if self.verbose > 0:
                print(
                    "[Foundation] Warmup armed; waiting for causal replay "
                    f"rows={replay_size}/{minimum_rows}."
                )
            return False

        batch_size = int(getattr(config, "FOUNDATION_BATCH_SIZE", 128))
        if self._foundation_market_sampler is None:
            batch_size = min(batch_size, replay_size)

        def _sample_foundation_batch(sampled_override=None):
            if self._foundation_market_sampler is not None:
                sampled = (
                    self._foundation_market_sampler.sample(batch_size)
                    if sampled_override is None
                    else sampled_override
                )
                observation_keys = (
                    "observations", "asset_ids", "mask", "prev_action",
                    "prev_mask", "basket_context", "basket_membership",
                    "local_feature_history", "episode_feature_history",
                    "feature_history_mask",
                )
                observations = {
                    key: torch.as_tensor(sampled[key], device=self.device)
                    for key in observation_keys
                }
                peer_observations = {
                    key: torch.as_tensor(
                        sampled[
                            key
                            if key in {
                                "local_feature_history",
                                "episode_feature_history",
                            }
                            else f"peer_{key}"
                        ],
                        device=self.device,
                    )
                    for key in observation_keys
                }
                peer_observations["local_feature_history"] = torch.as_tensor(
                    sampled["peer_local_feature_history"], device=self.device
                )
                peer_observations["episode_feature_history"] = torch.as_tensor(
                    sampled["peer_episode_feature_history"], device=self.device
                )
                future_returns = torch.as_tensor(
                    sampled["future_unit_returns"],
                    device=self.device,
                    dtype=torch.float32,
                )
                future_valid = torch.as_tensor(
                    sampled["future_valid"],
                    device=self.device,
                    dtype=torch.bool,
                )
                observations = {
                    key: value.float() if key != "asset_ids" else value.long()
                    for key, value in observations.items()
                }
                peer_observations = {
                    key: value.float() if key != "asset_ids" else value.long()
                    for key, value in peer_observations.items()
                }
                original_local = observations["local_feature_history"].float()
                coordinate_mask = torch.rand_like(original_local) < float(
                    getattr(config, "FOUNDATION_LOCAL_MASK_PROBABILITY", 0.15)
                )
                observations["foundation_local_history_target"] = original_local
                observations[
                    "foundation_local_reconstruction_mask"
                ] = coordinate_mask
                observations["local_feature_history"] = original_local.masked_fill(
                    coordinate_mask, 0.0
                )
                with torch.no_grad():
                    actor_proposal = self.actor(
                        self._actor_observation(observations)
                    )
                    candidates = build_foundation_candidates(
                        observations, actor_proposal
                    )
                    targets, utilities, _, _ = build_foundation_action_targets(
                        observations,
                        candidates,
                        future_returns.unsqueeze(-1)
                        * float(getattr(config, "FINANCIAL_REWARD_SCALE", 10.0)),
                        future_valid.unsqueeze(-1),
                    )
                return (
                    observations, peer_observations, candidates, targets,
                    utilities, future_returns, future_valid,
                )
            replay_data, starts = self._sample_predictive_replay_data(batch_size)
            selected, sequence = self._predictive_sequence_from_financial_batch(
                replay_data,
                start_batch_inds=starts,
                require_component_outcomes=True,
            )
            observations = {
                key: value.float()
                for key, value in selected.observations.items()
            }
            observations = self._gate_observations_with_sigma(observations)
            original_local = observations["local_feature_history"].float()
            coordinate_mask = (
                torch.rand_like(original_local)
                < float(
                    getattr(config, "FOUNDATION_LOCAL_MASK_PROBABILITY", 0.15)
                )
            )
            coordinate_mask &= (
                observations["feature_history_mask"]
                .bool()
                .unsqueeze(1)
                .unsqueeze(-1)
            )
            coordinate_mask &= (
                observations["mask"].bool().unsqueeze(-1).unsqueeze(-1)
            )
            observations["foundation_local_history_target"] = original_local
            observations[
                "foundation_local_reconstruction_mask"
            ] = coordinate_mask
            observations["local_feature_history"] = original_local.masked_fill(
                coordinate_mask, 0.0
            )
            with torch.no_grad():
                actor_proposal = self.actor(
                    self._actor_observation(observations)
                )
                candidates = build_foundation_candidates(
                    observations, actor_proposal
                )
                targets, utilities, future_returns, future_valid = (
                    build_foundation_action_targets(
                        observations,
                        candidates,
                        sequence.component_outcomes[
                            :, : self.foundation_max_horizon
                        ],
                        sequence.component_outcome_masks[
                            :, : self.foundation_max_horizon
                        ],
                    )
                )
            return (
                observations,
                None,
                candidates,
                targets,
                utilities,
                future_returns,
                future_valid,
            )

        self.policy.set_training_mode(True)
        best_probe = float(self.foundation_best_probe)
        stale_probes = int(self.foundation_stale_probes)
        last_actor_loss = math.nan
        last_critic_loss = math.nan
        started_at = time.perf_counter()
        warmup_csv_path = None
        if self.foundation_log_dir:
            os.makedirs(self.foundation_log_dir, exist_ok=True)
            warmup_csv_path = os.path.join(
                self.foundation_log_dir, "foundation_warmup.csv"
            )
            if self.foundation_warmup_updates == 0 or not os.path.exists(
                warmup_csv_path
            ):
                with open(warmup_csv_path, "w", newline="") as stream:
                    csv.writer(stream).writerow(
                        (
                            "foundation_update", "actor_loss", "critic_loss",
                            "representation_loss", "action_outcome_loss",
                            "q_rank_loss",
                            "actor_regret", "probe_loss", "stale_probes",
                            "sample_wait_seconds", "critic_update_seconds",
                            "actor_update_seconds", "elapsed_seconds",
                            "device",
                        )
                    )
            if self.verbose > 0:
                print(f"[Foundation] live metrics: {warmup_csv_path}")

        def _write_warmup_row(
            *, probe_loss=math.nan, sample_wait=math.nan,
            critic_seconds=math.nan, actor_seconds=math.nan,
            actor_stats=None, critic_stats=None,
        ):
            if warmup_csv_path is None:
                return
            actor_stats = actor_stats or {}
            critic_stats = critic_stats or {}
            with open(warmup_csv_path, "a", newline="") as stream:
                csv.writer(stream).writerow(
                    (
                        int(self.foundation_warmup_updates),
                        float(last_actor_loss), float(last_critic_loss),
                        float(
                            critic_stats.get(
                                "representation_loss", torch.tensor(math.nan)
                            ).detach().item()
                        ),
                        float(
                            critic_stats.get(
                                "action_loss", torch.tensor(math.nan)
                            ).detach().item()
                        ),
                        float(
                            critic_stats.get(
                                "q_rank_loss", torch.tensor(math.nan)
                            ).detach().item()
                        ),
                        float(
                            actor_stats.get(
                                "actor_regret", torch.tensor(math.nan)
                            ).detach().item()
                        ),
                        float(probe_loss), int(stale_probes),
                        float(sample_wait), float(critic_seconds),
                        float(actor_seconds),
                        float(time.perf_counter() - started_at), str(self.device),
                    )
                )

        executor = None
        pending_sample = None
        cached_foundation_batch = None
        cached_batch_reuses = 0
        if (
            self._foundation_market_sampler is not None
            and self.foundation_prefetch_enabled
        ):
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="foundation-sampler"
            )
            pending_sample = executor.submit(
                self._foundation_market_sampler.sample, batch_size
            )

        def _next_foundation_batch(*, reuse=True):
            nonlocal pending_sample, cached_foundation_batch, cached_batch_reuses
            if (
                reuse
                and cached_foundation_batch is not None
                and cached_batch_reuses < self.foundation_batch_reuse_updates
            ):
                cached_batch_reuses += 1
                return cached_foundation_batch, 0.0
            wait_started = time.perf_counter()
            if pending_sample is None:
                result = _sample_foundation_batch()
            else:
                sampled = pending_sample.result()
                pending_sample = executor.submit(
                    self._foundation_market_sampler.sample, batch_size
                )
                result = _sample_foundation_batch(sampled)
            elapsed = time.perf_counter() - wait_started
            if reuse:
                cached_foundation_batch = result
                cached_batch_reuses = 1
            return result, elapsed

        if self.verbose > 0:
            print(
                "[Foundation] live GPU warmup starting | "
                f"batch={batch_size} | min_updates="
                f"{self.foundation_warmup_min_updates} | max_updates="
                f"{self.foundation_warmup_max_updates} | "
                f"batch_reuse={self.foundation_batch_reuse_updates} | "
                f"prefetch={executor is not None} | device={self.device}"
            )
        probe_batch, _ = _next_foundation_batch(reuse=False)
        while self.foundation_warmup_updates < self.foundation_warmup_max_updates:
            (
                observations,
                peer_observations,
                candidates,
                targets,
                utilities,
                future_returns,
                future_valid,
            ), sample_wait_seconds = _next_foundation_batch()
            critic_started = time.perf_counter()
            critic_stats = self.critic.foundation_loss(
                observations,
                future_returns,
                future_valid,
                candidates,
                targets,
                peer_observations=peer_observations,
                subsample_candidates=bool(
                    getattr(
                        config,
                        "FOUNDATION_CRITIC_CANDIDATE_SUBSAMPLE",
                        True,
                    )
                ),
            )
            self.critic.optimizer.zero_grad(set_to_none=True)
            critic_stats["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
            self.critic.optimizer.step()
            critic_update_seconds = time.perf_counter() - critic_started

            actor_started = time.perf_counter()
            actor_stats = self.actor.foundation_loss(
                self._actor_observation(observations),
                future_returns,
                future_valid,
                candidates,
                utilities,
                peer_observations=peer_observations,
            )
            self.actor.optimizer.zero_grad(set_to_none=True)
            actor_stats["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
            self.actor.optimizer.step()
            actor_update_seconds = time.perf_counter() - actor_started
            self.foundation_warmup_updates += 1
            last_actor_loss = float(actor_stats["loss"].detach().item())
            last_critic_loss = float(critic_stats["loss"].detach().item())

            probe_value = math.nan
            if self.foundation_warmup_updates % self.foundation_probe_interval == 0:
                with torch.no_grad():
                    (
                        probe_observations,
                        probe_peer_observations,
                        probe_candidates,
                        probe_targets,
                        probe_utilities,
                        probe_returns,
                        probe_valid,
                    ) = probe_batch
                    probe_actor = self.actor.foundation_loss(
                        self._actor_observation(probe_observations),
                        probe_returns,
                        probe_valid,
                        probe_candidates,
                        probe_utilities,
                        peer_observations=probe_peer_observations,
                    )["loss"]
                    probe_critic = self.critic.foundation_loss(
                        probe_observations,
                        probe_returns,
                        probe_valid,
                        probe_candidates,
                        probe_targets,
                        peer_observations=probe_peer_observations,
                        subsample_candidates=bool(
                            getattr(
                                config,
                                "FOUNDATION_CRITIC_CANDIDATE_SUBSAMPLE",
                                True,
                            )
                        ),
                    )["loss"]
                    probe = 0.5 * float(
                        (probe_actor + probe_critic).detach().item()
                    )
                relative_gain = (
                    (best_probe - probe) / max(abs(best_probe), 1e-8)
                    if math.isfinite(best_probe)
                    else math.inf
                )
                if relative_gain >= self.foundation_min_relative_improvement:
                    best_probe = probe
                    stale_probes = 0
                else:
                    stale_probes += 1
                probe_value = probe
                if self.verbose > 0:
                    print(
                        f"[Foundation] update={self.foundation_warmup_updates} "
                        f"actor={last_actor_loss:.6f} critic={last_critic_loss:.6f} "
                        f"probe={probe:.6f} stale={stale_probes}/"
                        f"{self.foundation_probe_patience}"
                    )
                if (
                    self.foundation_warmup_updates
                    >= self.foundation_warmup_min_updates
                    and stale_probes >= self.foundation_probe_patience
                ):
                    _write_warmup_row(
                        probe_loss=probe_value,
                        sample_wait=sample_wait_seconds,
                        critic_seconds=critic_update_seconds,
                        actor_seconds=actor_update_seconds,
                        actor_stats=actor_stats,
                        critic_stats=critic_stats,
                    )
                    break

            if self.foundation_warmup_updates % self.foundation_log_interval == 0:
                _write_warmup_row(
                    probe_loss=probe_value,
                    sample_wait=sample_wait_seconds,
                    critic_seconds=critic_update_seconds,
                    actor_seconds=actor_update_seconds,
                    actor_stats=actor_stats,
                    critic_stats=critic_stats,
                )
                if self.verbose > 0:
                    print(
                        f"[Foundation] update={self.foundation_warmup_updates} "
                        f"actor={last_actor_loss:.6f} "
                        f"critic={last_critic_loss:.6f} "
                        f"sample_wait={sample_wait_seconds:.3f}s "
                        f"gpu_update="
                        f"{critic_update_seconds + actor_update_seconds:.3f}s"
                    )

        # Always persist the terminal update, including tiny verification runs
        # whose maximum update count is below the normal logging interval.
        if (
            self.foundation_warmup_updates > 0
            and self.foundation_warmup_updates % self.foundation_log_interval != 0
        ):
            _write_warmup_row(
                probe_loss=probe_value,
                sample_wait=sample_wait_seconds,
                critic_seconds=critic_update_seconds,
                actor_seconds=actor_update_seconds,
                actor_stats=actor_stats,
                critic_stats=critic_stats,
            )

        if executor is not None:
            if pending_sample is not None:
                pending_sample.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        if self._foundation_market_sampler is not None:
            self.foundation_sampler_rng_state = (
                self._foundation_market_sampler.rng.bit_generator.state
            )

        self.foundation_best_probe = best_probe
        self.foundation_stale_probes = stale_probes
        polyak_update(
            self.actor.parameters(), self.actor_target.parameters(), 1.0
        )
        polyak_update(
            self.critic.parameters(), self.critic_target.parameters(), 1.0
        )
        # Same parameters, fresh TD3 moments: no network transplant and no
        # frozen handoff.
        self.actor.optimizer.state.clear()
        self.critic.optimizer.state.clear()
        self.foundation_warmup_complete = True
        self.foundation_warmup_requested = False
        if self.verbose > 0:
            print(
                f"[Foundation] live warmup complete after "
                f"{self.foundation_warmup_updates} updates in "
                f"{time.perf_counter() - started_at:.1f}s; all parameters live."
            )
        return True

    def _predictive_ssl_update_due(self) -> bool:
        """Whether the expensive multi-step auxiliary is due this TD update."""
        return bool(
            self.predictive_ssl_enabled
            and self.predictive_ssl_online_enabled
            and self._n_updates % self.predictive_ssl_update_interval == 0
        )

    def _foundation_online_update_due(self) -> bool:
        """Whether post-warmup foundation rehearsal is due this TD update."""
        return bool(
            self.foundation_learning_enabled
            and self.foundation_warmup_complete
            and self.foundation_online_rehearsal_enabled
            and self._n_updates % self.foundation_online_update_interval == 0
        )

    def _earnmore_reconstruction_update_due(self) -> bool:
        """Whether the whole-stock auxiliary is due this TD update."""
        return bool(
            self.earnmore_masked_reconstruction_enabled
            and self._n_updates % self.earnmore_update_interval == 0
        )

    def _earnmore_masked_reconstruction_update(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Train twin-critic reconstruction without moving the live actor.

        The actor changes only in its delayed financial/CRL step, with guarded
        predictive SSL combined there. An encoder-only actor reconstruction
        step would still change its decisions after the constraint-aware step.
        Critic reconstruction uses the existing critic optimizer; action and
        scalar-Q heads receive no reconstruction gradient.
        """
        if not self._earnmore_reconstruction_update_due():
            return None
        if not (
            isinstance(observations, dict)
            and hasattr(self.actor, "prepare_earnmore_reconstruction_inputs")
            and hasattr(self.critic, "earnmore_masked_reconstruction_loss")
        ):
            raise RuntimeError(
                "EarnMore-TD3 requires the variable-coin AttentionTD3Policy"
            )

        real_asset_mask = observations["mask"].float()
        sampling_mask = real_asset_mask
        if self.actor.token_encoder.pool_level_enabled:
            sampling_mask = observations["pool_history_valid"].bool().any(dim=-1).float()
        synthetic_mask, _ = self.actor.token_encoder.sample_earnmore_asset_mask(
            sampling_mask
        )
        if self.actor.token_encoder.pool_level_enabled:
            use_real = torch.rand((real_asset_mask.shape[0], 1), device=real_asset_mask.device) < config.EARNMORE_REAL_POOL_MASK_FRACTION
            synthetic_mask = torch.where(use_real, ~real_asset_mask.bool() & sampling_mask.bool(), synthetic_mask)

        # This parser only reshapes immutable tensors; it does not encode them
        # or construct an actor gradient graph. Both critics reuse its result.
        prepared_inputs = self.actor.prepare_earnmore_reconstruction_inputs(
            observations
        )

        self.critic.optimizer.zero_grad(set_to_none=True)
        critic_stats = self.critic.earnmore_masked_reconstruction_loss(
            observations,
            synthetic_mask=synthetic_mask,
            prepared_inputs=prepared_inputs,
        )
        critic_stats["loss"].backward()
        critic_parameters = tuple(
            parameter
            for q_network in self.critic.q_networks
            for parameter in q_network.token_encoder.parameters()
            if parameter.requires_grad
        )
        critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
            critic_parameters, self.earnmore_max_grad_norm
        )
        self.critic.optimizer.step()

        # Target synchronization belongs to the ordinary full-network delayed
        # TD3 update. Do not advance encoder targets independently of Q heads.

        self.earnmore_updates += 1
        detached_critic_loss = critic_stats["loss"].detach().float()
        return {
            "loss": detached_critic_loss,
            "actor_loss": detached_critic_loss.new_zeros(()),
            "critic_loss": detached_critic_loss,
            "masked_asset_fraction": critic_stats[
                "masked_asset_fraction"
            ].detach().float(),
            "sampled_mask_ratio": critic_stats[
                "sampled_mask_ratio"
            ].detach().float(),
            "masked_local_coordinate_fraction": critic_stats[
                "masked_local_coordinate_fraction"
            ].detach().float(),
            "masked_assets_per_sample": critic_stats[
                "masked_assets_per_sample"
            ].detach().float(),
            "visible_assets_per_sample": critic_stats[
                "visible_assets_per_sample"
            ].detach().float(),
            "actor_gradient_norm": detached_critic_loss.new_zeros(()),
            "critic_gradient_norm": torch.as_tensor(
                critic_gradient_norm
            ).detach().float(),
        }

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Override the default training loop to use:
        1. Huber loss for the critic
        2. Adaptive Lagrange multiplier for action smoothness (CRL)
        3. Custom learning rate updates
        """
        warmup_completed_this_call = False
        if self.foundation_learning_enabled and not self.foundation_warmup_complete:
            warmup_was_complete = bool(self.foundation_warmup_complete)
            self.foundation_warmup()
            if not self.foundation_warmup_complete:
                return
            warmup_completed_this_call = bool(
                self.foundation_warmup_complete and not warmup_was_complete
            )
        if self.qwm_enabled and not self.qwm_pretrain_complete:
            # TD3 must not sit idle while QWM waits for broad 8-of-13 replay
            # coverage. Search remains disabled until the initial model fit,
            # but ordinary real-transition learning continues immediately.
            self._run_qwm_pretrain()
        cuda_memory_telemetry = bool(
            getattr(config, "FOUNDATION_CUDA_MEMORY_TELEMETRY", True)
            and torch.cuda.is_available()
            and torch.device(self.device).type == "cuda"
        )
        if (
            warmup_completed_this_call
            and cuda_memory_telemetry
            and bool(
                getattr(
                    config,
                    "FOUNDATION_RELEASE_CUDA_CACHE_AFTER_WARMUP",
                    True,
                )
            )
        ):
            # Warmup-local graphs and prefetched batches are out of scope now.
            # A one-time handoff release prevents their cached high watermark
            # from masquerading as TD3's live requirement in nvidia-smi.
            torch.cuda.empty_cache()
        if cuda_memory_telemetry:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        train_started_at = time.perf_counter()

        actor_losses, actor_total_losses, critic_losses, critic_total_losses = [], [], [], []
        actor_q_losses = []
        actor_raw_q_losses = []
        actor_action_advantages = []
        actor_popart_advantages = []
        xbasket_actor_advantage_real = []
        xbasket_actor_advantage_relabeled = []
        xbasket_actor_batch_rows = []
        actor_popart_scale_means = []
        actor_popart_scale_maxes = []
        actor_velocity_penalty_terms = []
        actor_acceleration_penalty_terms = []
        actor_smooth_losses = []
        critic_losses_unweighted = []
        critic_weighted_td_losses = []
        actor_advantage_abs_maxima = []
        actor_policy_step_l2_means, actor_policy_step_l2_maxima = [], []
        bellman_extrema = []
        critic_popart_losses = []
        popart_target_means = []
        popart_target_sigmas = []
        popart_target_sigma_mins = []
        popart_target_sigma_maxes = []
        popart_task_reliabilities = []
        popart_distinct_tasks = []
        popart_critic_gradient_directions = []
        popart_actor_gradient_directions = []
        velocity_costs_log, acceleration_costs_log = [], []
        sigma_intensity_means, sigma_intensity_stds = [], []
        sigma_weight_means, sigma_weight_maxes = [], []
        proexp_alpha_means, proexp_disagreement_means = [], []
        proexp_candidate_entropies, proexp_soft_value_means = [], []
        proexp_center_q_means, proexp_soft_gap_means = [], []
        proexp_budget_alpha_means, proexp_composition_alpha_means = [], []
        proexp_budget_disagreement_means, proexp_composition_disagreement_means = [], []
        proexp_budget_entropies, proexp_composition_entropies = [], []
        component_outcome_losses, component_outcome_weighted_losses = [], []
        component_outcome_maes, component_outcome_valid_fractions = [], []
        basket_relative_losses, asset_teacher_real_rows = [], []
        local_financial_losses, local_financial_maes = [], []
        local_financial_context_abs, local_financial_base_abs = [], []
        local_financial_residual_abs = []
        local_financial_actor_replay_advantages = []
        local_financial_total_aux_losses = []
        component_outcome_gradient_scales = []
        component_outcome_gradient_measurements = []
        component_outcome_correlations = []
        component_outcome_target_abs_means = []
        component_outcome_target_clip_fractions = []
        predictive_critic_losses, predictive_actor_losses = [], []
        predictive_ssl_update_seconds: List[float] = []
        qwm_losses, qwm_reconstruction_losses = [], []
        qwm_posterior_reconstruction_losses = []
        qwm_prior_reconstruction_losses = []
        qwm_state_losses, qwm_reward_losses, qwm_kl_losses = [], [], []
        qwm_prior_state_losses, qwm_prior_reward_losses = [], []
        qwm_observation_losses, qwm_local_history_losses = [], []
        qwm_episode_history_losses, qwm_context_losses = [], []
        qwm_prior_stds, qwm_posterior_stds, qwm_latent_mean_gaps = [], [], []
        qwm_gradient_norms = []
        foundation_critic_losses, foundation_actor_losses = [], []
        foundation_action_losses, foundation_representation_losses = [], []
        foundation_q_rank_losses = []
        foundation_actor_regrets = []
        foundation_candidate_evaluation_fractions = []
        earnmore_losses, earnmore_actor_losses, earnmore_critic_losses = [], [], []
        earnmore_update_seconds: List[float] = []
        earnmore_masked_fractions, earnmore_sampled_ratios = [], []
        earnmore_coordinate_fractions = []
        earnmore_masked_counts, earnmore_visible_counts = [], []
        earnmore_actor_gradient_norms, earnmore_critic_gradient_norms = [], []
        predictive_feature_losses, predictive_market_losses = [], []
        predictive_relation_losses, predictive_fused_losses = [], []
        predictive_downstream_losses = []
        predictive_reward_losses = []
        predictive_continuation_value_losses = []
        predictive_bellman_equivalence_losses = []
        predictive_expectile_value_losses = []
        predictive_state_value_means = []
        predictive_advantage_means = []
        predictive_advantage_positive_fractions = []
        predictive_advantage_weight_mins = []
        predictive_advantage_weight_maxes = []
        predictive_routing_entropies = []
        predictive_expert_min_usages, predictive_expert_max_usages = [], []
        predictive_valid_fractions, predictive_control_gates = [], []
        predictive_valid_horizon_fractions = []
        predictive_horizon_loss_means = []
        predictive_per_horizon_losses: Dict[int, List[float]] = (
            collections.defaultdict(list)
        )
        predictive_per_horizon_value_losses: Dict[int, List[float]] = (
            collections.defaultdict(list)
        )
        predictive_control_projection_norms = []
        predictive_replay_unique_keys = []
        predictive_replay_anchor_fractions = []
        predictive_replay_mean_distances = []
        predictive_critic_gradient_stats = []
        predictive_actor_gradient_stats = []
        basket_replay_sample_stats: List[Dict[str, Any]] = []
        basket_diag_accum: Dict[str, List[float]] = {}
        cross_basket_relabel_stats: List[Dict[str, float]] = []
        
        for _ in range(gradient_steps):
            self._n_updates += 1
            foundation_this_update = self._foundation_online_update_due()
            # Complete foundation rehearsal replaces (rather than stacks on)
            # the historical predictive auxiliary at its cadence.
            predictive_ssl_this_update = bool(
                self._predictive_ssl_update_due() and not foundation_this_update
            )
            predictive_ssl_seconds = 0.0
            auxiliary_this_update = bool(
                predictive_ssl_this_update or foundation_this_update
            )
            replay_data = self._sample_replay_data(batch_size)
            bellman_source_replay_data = replay_data
            bellman_env_inds = getattr(
                self.replay_buffer, "_last_sample_env_indices", None
            )
            replay_data = self._apply_cross_basket_relabeling(
                replay_data, cross_basket_relabel_stats
            )
            mixed_replay_data = replay_data
            if getattr(self, "cross_basket_relabel_critic_rows", "mixed") == "original":
                replay_data = bellman_source_replay_data
            actor_crl_replay_data = select_cross_basket_consumer_replay(
                mixed_replay_data,
                bellman_source_replay_data,
                getattr(self, "cross_basket_relabel_actor_crl_rows", "original"),
                default="original",
            )
            if self.qwm_enabled:
                # Learn from every real TD3-era transition even while search
                # waits for the broader pretrain coverage gate. This preserves
                # the one-world-update-per-TD-update invariant throughout the
                # accumulation phase; readiness controls use, not learning.
                for _qwm_update in range(self.qwm_online_updates_per_td_step):
                    qwm_replay_data = self._sample_qwm_replay_data(
                        self.qwm_batch_size
                    )
                    qwm_stats = self._qwm_update_from_replay_data(
                        qwm_replay_data
                    )
                    if qwm_stats:
                        qwm_losses.append(qwm_stats["loss"])
                        qwm_reconstruction_losses.append(
                            qwm_stats["reconstruction_loss"]
                        )
                        qwm_posterior_reconstruction_losses.append(
                            qwm_stats["posterior_reconstruction_loss"]
                        )
                        qwm_prior_reconstruction_losses.append(
                            qwm_stats["prior_reconstruction_loss"]
                        )
                        qwm_state_losses.append(qwm_stats["state_loss"])
                        qwm_reward_losses.append(qwm_stats["reward_loss"])
                        qwm_prior_state_losses.append(
                            qwm_stats["prior_state_loss"]
                        )
                        qwm_prior_reward_losses.append(
                            qwm_stats["prior_reward_loss"]
                        )
                        qwm_kl_losses.append(qwm_stats["kl_loss"])
                        qwm_observation_losses.append(
                            qwm_stats["observation_loss"]
                        )
                        qwm_local_history_losses.append(
                            qwm_stats["local_history_loss"]
                        )
                        qwm_episode_history_losses.append(
                            qwm_stats["episode_history_loss"]
                        )
                        qwm_context_losses.append(qwm_stats["context_loss"])
                        qwm_prior_stds.append(qwm_stats["prior_std_mean"])
                        qwm_posterior_stds.append(
                            qwm_stats["posterior_std_mean"]
                        )
                        qwm_latent_mean_gaps.append(
                            qwm_stats["latent_mean_gap"]
                        )
                        qwm_gradient_norms.append(qwm_stats["gradient_norm"])
            current_replay_stats = dict(
                self._last_basket_replay_sample_stats
            )
            if current_replay_stats:
                basket_replay_sample_stats.append(current_replay_stats)
            predictive_sequence = None
            foundation_sequence = None
            if auxiliary_this_update:
                if self.predictive_ssl_separate_replay_enabled:
                    # The one-step SSL learner consumes only its configured
                    # sequence minibatch. Sampling a full TD3-sized batch and
                    # then discarding the excess rows transfers large history
                    # tensors to the device for no learning benefit.
                    predictive_sample_size = (
                        batch_size
                        if foundation_this_update
                        else min(batch_size, self.predictive_ssl_sequence_batch_size)
                    )
                    (
                        predictive_source_data,
                        predictive_source_inds,
                    ) = self._sample_predictive_replay_data(
                        predictive_sample_size
                    )
                    predictive_source_env_inds = getattr(
                        self, "_last_predictive_env_inds", None
                    )
                else:
                    predictive_source_data = bellman_source_replay_data
                    predictive_source_inds = getattr(
                        self, "_last_replay_batch_inds", None
                    )
                    predictive_source_env_inds = bellman_env_inds
                (
                    predictive_replay_data,
                    predictive_sequence,
                ) = self._predictive_sequence_from_financial_batch(
                    predictive_source_data,
                    start_batch_inds=predictive_source_inds,
                    maximum_batch_size=(
                        self.foundation_online_batch_size
                        if foundation_this_update
                        else None
                    ),
                    require_component_outcomes=foundation_this_update,
                    start_env_inds=predictive_source_env_inds,
                )
                foundation_sequence = predictive_sequence
            else:
                predictive_replay_data = replay_data

            with torch.no_grad():
                # ✅ DICTIONARY OBSERVATION FIX: Handle both dict and array observations
                if isinstance(replay_data.next_observations, dict):
                    # For dictionary observations (variable portfolio), get batch size from 'observations' key
                    next_obs_float = {k: v.float() for k, v in replay_data.next_observations.items()}
                    batch_size_ = next_obs_float['observations'].shape[0]
                else:
                    # For array observations (fixed portfolio)
                    next_obs_float = replay_data.next_observations.float()
                    batch_size_ = next_obs_float.shape[0]
                
                action_dim = self.action_space.shape[0]
                device = self.device

                # Clamp to the true action space bounds instead of hardcoded [-1, 1]
                low, high = self._get_action_bounds()

                # ── Sigma input modulation (legacy dampen_input or passthrough) ──
                next_obs_float_g = self._gate_observations_with_sigma(next_obs_float)
                next_actor_obs = self._actor_observation(next_obs_float_g, use_no_grad=True)
                next_actions_center = self.actor_target(next_actor_obs)
                if self.use_proportional_exp:
                    next_candidates, next_candidate_family_ids = self._sample_local_candidate_actions(
                        next_actions_center,
                        next_obs_float_g,
                        self.proportional_exp_candidates,
                        low,
                        high,
                        budget_std=self.proportional_exp_target_budget_std,
                        composition_std=self.proportional_exp_target_composition_std,
                    )
                    target_stats = self._compute_proportional_exp_statistics(
                        self.critic_target,
                        next_obs_float_g,
                        next_candidates,
                        next_candidate_family_ids,
                        update_disagreement_ema=True,
                    )
                    target_q_values = target_stats["soft_value"].unsqueeze(-1)
                else:
                    noise = torch.normal(
                        mean=0.0,
                        std=self.target_policy_noise,
                        size=(batch_size_, action_dim),
                        device=device
                    ).clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_actions_raw = next_actions_center + noise
                    next_actions = torch.max(torch.min(next_actions_raw, high), low)
                    target_q_values = torch.cat(self.critic_target(next_obs_float_g, next_actions), dim=1)
                    target_q_values, _ = torch.min(target_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards.float() + (1 - replay_data.dones.float()) * self.gamma * target_q_values

                popart_task_ids = None
                normalized_target_q_values = None
                popart_target_mean = None
                popart_target_sigma = None
                popart_task_reliability = None
                if self.task_popart is not None:
                    popart_task_ids = self.task_popart.task_ids_from_observations(
                        next_obs_float_g
                    )
                    # PopArt statistics are supervision state, never a source
                    # of gradient.  Updating them cannot move a raw Q output.
                    self.task_popart.update(target_q_values, popart_task_ids)
                    (
                        normalized_target_q_values,
                        popart_target_mean,
                        popart_target_sigma,
                        popart_task_reliability,
                    ) = self.task_popart.normalize(
                        target_q_values, popart_task_ids
                    )
                    popart_target_means.append(
                        float(popart_target_mean.mean().item())
                    )
                    popart_target_sigmas.append(
                        float(popart_target_sigma.mean().item())
                    )
                    popart_target_sigma_mins.append(
                        float(popart_target_sigma.min().item())
                    )
                    popart_target_sigma_maxes.append(
                        float(popart_target_sigma.max().item())
                    )
                    popart_task_reliabilities.append(
                        float(popart_task_reliability.mean().item())
                    )
                    popart_distinct_tasks.append(
                        float(torch.unique(popart_task_ids).numel())
                    )

            # ✅ CRITIC UPDATE: Use Huber loss for robustness
            if isinstance(replay_data.observations, dict):
                obs_float = {k: v.float() for k, v in replay_data.observations.items()}
            else:
                obs_float = replay_data.observations.float()
            actions_float = replay_data.actions.float()

            # ── Sigma input modulation for critic ────────────────────────────────
            obs_float_g = self._gate_observations_with_sigma(obs_float)

            predictive_obs_float_g = obs_float_g
            predictive_next_obs_float_g = next_obs_float_g
            predictive_actions_float = actions_float
            predictive_future_actions = actions_float.unsqueeze(1)
            predictive_future_obs_float_g = {
                key: value.unsqueeze(1)
                for key, value in next_obs_float_g.items()
            } if isinstance(next_obs_float_g, dict) else None
            predictive_future_valid_steps = (
                replay_data.dones.view(-1, 1) < 0.5
            )
            predictive_future_outcome_valid_steps = torch.ones_like(
                predictive_future_valid_steps, dtype=torch.bool
            )
            predictive_future_rewards = replay_data.rewards.float().unsqueeze(1)
            predictive_future_dones = replay_data.dones.float().unsqueeze(1)
            predictive_future_continuation_values = None
            predictive_future_next_actions = next_actions_center.unsqueeze(1)
            predictive_next_actions_center = next_actions_center
            predictive_dones = replay_data.dones.float()
            if predictive_ssl_this_update:
                if not (
                    isinstance(predictive_replay_data.observations, dict)
                    and predictive_sequence is not None
                ):
                    raise RuntimeError(
                        "Multi-step predictive replay requires dictionary observations."
                    )
                predictive_obs_float = {
                    key: value.float()
                    for key, value in predictive_replay_data.observations.items()
                }
                predictive_future_obs_float = {
                    key: value[:, : self.predictive_ssl_horizon].float()
                    for key, value in predictive_sequence.future_observations.items()
                }
                predictive_obs_float_g = self._gate_observations_with_sigma(
                    predictive_obs_float
                )
                predictive_future_actions = predictive_sequence.actions[
                    :, : self.predictive_ssl_horizon
                ].float()
                predictive_future_valid_steps = (
                    predictive_sequence.valid_steps[
                        :, : self.predictive_ssl_horizon
                    ].bool()
                )
                predictive_future_outcome_valid_steps = (
                    predictive_sequence.outcome_valid_steps[
                        :, : self.predictive_ssl_horizon
                    ].bool()
                )
                predictive_future_rewards = predictive_sequence.rewards[
                    :, : self.predictive_ssl_horizon
                ].float()
                predictive_future_dones = predictive_sequence.dones[
                    :, : self.predictive_ssl_horizon
                ].float()
                predictive_actions_float = predictive_future_actions[:, 0]
                predictive_dones = (
                    ~predictive_future_valid_steps[:, 0]
                ).float().unsqueeze(-1)
                predictive_batch_size, predictive_horizon = (
                    predictive_future_actions.shape[:2]
                )
                flattened_predictive_future = {
                    key: value.reshape(
                        predictive_batch_size * predictive_horizon,
                        *value.shape[2:],
                    )
                    for key, value in predictive_future_obs_float.items()
                }
                flattened_predictive_future_g = (
                    self._gate_observations_with_sigma(
                        flattened_predictive_future
                    )
                )
                predictive_future_obs_float_g = {
                    key: value.reshape(
                        predictive_batch_size,
                        predictive_horizon,
                        *value.shape[1:],
                    )
                    for key, value in flattened_predictive_future_g.items()
                }
                predictive_next_obs_float_g = {
                    key: value[:, 0]
                    for key, value in predictive_future_obs_float_g.items()
                }
                predictive_cuda_devices = []
                if predictive_actions_float.is_cuda:
                    predictive_cuda_devices = [
                        predictive_actions_float.device.index
                        if predictive_actions_float.device.index is not None
                        else torch.cuda.current_device()
                    ]
                if bool(getattr(config, "VALUE_EQUIVALENT_MODEL_ENABLED", True)):
                    # Broad representation learning has its own replay RNG and
                    # may not move the global control dropout/candidate RNG.
                    with torch.random.fork_rng(devices=predictive_cuda_devices):
                        with torch.no_grad():
                            flat_future_next_actions = self.actor_target(
                                self._actor_observation(
                                    flattened_predictive_future_g,
                                    use_no_grad=True,
                                )
                            )
                            predictive_future_next_actions = (
                                flat_future_next_actions.reshape(
                                    predictive_batch_size,
                                    predictive_horizon,
                                    -1,
                                )
                            )
                            flat_future_target_q = torch.min(
                                torch.cat(
                                    self.critic_target(
                                        flattened_predictive_future_g,
                                        flat_future_next_actions,
                                    ),
                                    dim=1,
                                ),
                                dim=1,
                                keepdim=True,
                            ).values
                            predictive_future_continuation_values = (
                                flat_future_target_q.reshape(
                                    predictive_batch_size,
                                    predictive_horizon,
                                    1,
                                )
                            )
                            predictive_next_actions_center = (
                                predictive_future_next_actions[:, 0]
                            )
                else:
                    # State/latent SPR never reads next-action or value targets
                    # while value-equivalent learning is disabled. Keep a
                    # shape-correct placeholder for the critic's legacy API.
                    predictive_future_next_actions = predictive_future_actions
                    predictive_next_actions_center = predictive_actions_float
                predictive_membership = predictive_obs_float.get(
                    "basket_membership"
                )
                if torch.is_tensor(predictive_membership):
                    predictive_replay_unique_keys.append(
                        float(
                            torch.unique(
                                predictive_membership.detach(), dim=0
                            ).shape[0]
                        )
                    )
                    membership_bool = predictive_membership.detach() > 0.5
                    membership_dim = int(membership_bool.shape[-1])
                    anchor_bool = torch.zeros(
                        membership_dim,
                        dtype=torch.bool,
                        device=membership_bool.device,
                    )
                    valid_anchor_ids = [
                        index
                        for index in self._basket_replay_anchor_membership_ids
                        if 0 <= int(index) < membership_dim
                    ]
                    if valid_anchor_ids:
                        anchor_bool[valid_anchor_ids] = True
                        predictive_replay_anchor_fractions.append(
                            float(
                                (membership_bool == anchor_bool)
                                .all(dim=-1)
                                .float()
                                .mean()
                                .item()
                            )
                        )
                        predictive_replay_mean_distances.append(
                            float(
                                (anchor_bool & ~membership_bool)
                                .sum(dim=-1)
                                .float()
                                .mean()
                                .item()
                            )
                        )

            foundation_observations = None
            foundation_candidates = None
            foundation_targets = None
            foundation_utilities = None
            foundation_future_returns = None
            foundation_future_valid = None
            if foundation_this_update:
                if (
                    foundation_sequence is None
                    or not isinstance(predictive_replay_data.observations, dict)
                ):
                    raise RuntimeError(
                        "Continuous foundation rehearsal requires aligned "
                        "dictionary replay sequences."
                    )
                # ``_predictive_sequence_from_financial_batch`` may reduce the
                # broad financial batch (512 in production) to the configured
                # sequence batch (128).  The returned ``predictive_replay_data``
                # and ``foundation_sequence`` own the same selected row set.
                # Building candidates from ``predictive_obs_float_g`` here used
                # the unreduced financial batch whenever foundation rehearsal
                # replaced predictive SSL, producing 512 candidate rows beside
                # 128 future-outcome rows at the first online foundation update.
                foundation_observations = self._gate_observations_with_sigma(
                    {
                        key: value.float()
                        for key, value in (
                            predictive_replay_data.observations.items()
                        )
                    }
                )
                original_local_history = foundation_observations[
                    "local_feature_history"
                ].float()
                coordinate_mask = (
                    torch.rand_like(original_local_history)
                    < float(
                        getattr(
                            config,
                            "FOUNDATION_LOCAL_MASK_PROBABILITY",
                            0.15,
                        )
                    )
                )
                history_valid = foundation_observations[
                    "feature_history_mask"
                ].bool().unsqueeze(1).unsqueeze(-1)
                asset_valid = foundation_observations["mask"].bool().unsqueeze(
                    -1
                ).unsqueeze(-1)
                coordinate_mask &= history_valid & asset_valid
                foundation_observations[
                    "foundation_local_history_target"
                ] = original_local_history
                foundation_observations[
                    "foundation_local_reconstruction_mask"
                ] = coordinate_mask
                foundation_observations["local_feature_history"] = (
                    original_local_history.masked_fill(coordinate_mask, 0.0)
                )
                with torch.no_grad():
                    foundation_actor_proposal = self.actor(
                        self._actor_observation(foundation_observations)
                    )
                    foundation_candidates = build_foundation_candidates(
                        foundation_observations,
                        foundation_actor_proposal,
                    )
                    (
                        foundation_targets,
                        foundation_utilities,
                        foundation_future_returns,
                        foundation_future_valid,
                    ) = build_foundation_action_targets(
                        foundation_observations,
                        foundation_candidates,
                        foundation_sequence.component_outcomes[
                            :, : self.foundation_max_horizon
                        ],
                        foundation_sequence.component_outcome_masks[
                            :, : self.foundation_max_horizon
                        ],
                    )

            if (
                self.basket_context_diagnostics_enabled
                and isinstance(obs_float, dict)
                and (
                    self._n_updates
                    % self.basket_context_diagnostics_interval
                    == 0
                    or self._n_updates
                    % self.basket_context_diagnostics_sensitivity_interval
                    == 0
                )
            ):
                try:
                    include_sensitivity = (
                        self._n_updates
                        % self.basket_context_diagnostics_sensitivity_interval
                        == 0
                    )
                    obs_for_diag = dict(obs_float)
                    if "asset_ids" in obs_for_diag and torch.is_tensor(
                        obs_for_diag["asset_ids"]
                    ):
                        obs_for_diag["asset_ids"] = obs_for_diag["asset_ids"].long()
                    basket_stats = collect_basket_amortization_diagnostics(
                        obs_for_diag,
                        actor=self.actor,
                        critic=self.critic,
                        actions=actions_float,
                        include_sensitivity=include_sensitivity,
                        sensitivity_delta=self.basket_context_diagnostics_sensitivity_delta,
                    )
                    for key, value in basket_stats.items():
                        basket_diag_accum.setdefault(key, []).append(float(value))
                except Exception as diag_exc:
                    if self.verbose > 0 and self._n_updates % 500 == 0:
                        print(f"[BasketDiag] skipped: {diag_exc}")

            component_predictions = None
            local_financial_predictions = None
            local_financial_corrections = None
            if self.component_outcome_aux_enabled:
                if not hasattr(replay_data, "component_outcomes"):
                    raise RuntimeError(
                        "Component-outcome supervision is enabled, but sampled "
                        "replay data does not contain component metadata."
                    )
                local_handoff = bool(
                    getattr(config, "LOCAL_FINANCIAL_HANDOFF_ENABLED", False)
                )
                component_result = self.critic.forward_with_component_outcomes(
                    obs_float_g,
                    actions_float,
                    return_financial_decomposition=local_handoff,
                )
                current_q_values, component_predictions = component_result[:2]
                if local_handoff:
                    local_financial_predictions = component_result[2]
                    local_financial_corrections = component_result[3]
                    with torch.no_grad():
                        local_financial_base_abs.append(
                            float(torch.stack(component_result[4]).abs().mean().item())
                        )
                        local_financial_residual_abs.append(
                            float(torch.stack(component_result[5]).abs().mean().item())
                        )
            else:
                current_q_values = self.critic(obs_float_g, actions_float)
            self._accumulate_basket_td_errors(
                obs_float,
                current_q_values,
                target_q_values,
            )
            sample_weights, sigma_intensity = self._compute_sigma_loss_weights(obs_float)
            replay_importance = self._value_experience_importance_weights
            if (
                replay_importance is not None
                and replay_importance.shape[0] == actions_float.shape[0]
            ):
                replay_importance = replay_importance.to(
                    actions_float.device, actions_float.dtype
                )
                sample_weights = (
                    replay_importance
                    if sample_weights is None
                    else sample_weights * replay_importance
                )
            critic_loss_terms = []
            critic_loss_terms_unweighted = []
            popart_gradient_per_sample = None
            for critic_index, current_q in enumerate(current_q_values):
                raw_per_sample_huber = F.smooth_l1_loss(
                    current_q, target_q_values, reduction="none"
                ).squeeze(-1)
                critic_loss_terms_unweighted.append(
                    self._basket_balanced_replay_mean(
                        raw_per_sample_huber,
                        obs_float,
                    )
                )
                if self.task_popart is not None:
                    if (
                        popart_task_ids is None
                        or normalized_target_q_values is None
                        or popart_target_mean is None
                        or popart_target_sigma is None
                    ):
                        raise RuntimeError("PopArt target statistics were not prepared")
                    normalized_current_q = (
                        current_q - popart_target_mean.unsqueeze(-1)
                    ) / popart_target_sigma.unsqueeze(-1)
                    per_sample_huber = F.smooth_l1_loss(
                        normalized_current_q,
                        normalized_target_q_values,
                        reduction="none",
                    ).squeeze(-1)
                else:
                    per_sample_huber = raw_per_sample_huber
                if sample_weights is not None:
                    optimized_per_sample = per_sample_huber * sample_weights
                else:
                    optimized_per_sample = per_sample_huber
                critic_loss_terms.append(
                    self._basket_balanced_replay_mean(
                        optimized_per_sample,
                        obs_float,
                    )
                )
                if critic_index == 0:
                    popart_gradient_per_sample = optimized_per_sample
            critic_loss_unweighted = sum(critic_loss_terms_unweighted)
            critic_loss = sum(critic_loss_terms)
            # Preserve batch extremes that aggregate Huber curves conceal.
            # These reuse already-computed tensors, with no extra Q/encoder
            # evaluation. One row can explain a large error even if reward and
            # prediction means remain ordinary.
            with torch.no_grad():
                bellman_extrema.append(torch.stack((
                    torch.stack([value.abs().amax() for value in current_q_values]).amax(),
                    target_q_values.abs().amax(),
                    torch.stack([(value - target_q_values).abs().amax()
                                 for value in current_q_values]).amax(),
                    replay_data.rewards.abs().amax(),
                )).float().cpu().tolist())
            if self.task_popart is not None:
                critic_popart_losses.append(float(critic_loss.detach().item()))
                if (
                    self._n_updates % self.task_popart_gradient_interval == 0
                    and popart_gradient_per_sample is not None
                    and popart_task_ids is not None
                ):
                    direction_stats = measure_task_gradient_directions(
                        popart_gradient_per_sample,
                        popart_task_ids,
                        tuple(self._value_experience_transfer_parameters()),
                        max_tasks=self.task_popart_gradient_max_tasks,
                        min_samples=self.task_popart_gradient_min_samples,
                    )
                    if direction_stats is not None:
                        popart_critic_gradient_directions.append(direction_stats)
            # Bellman TD owns long-horizon Q. Absolute and basket-relative
            # asset labels teach the real source minutes. With XBR disabled,
            # the teacher reuses the same original-row critic encoding.
            critic_td_loss = critic_loss
            critic_weighted_td_losses.append(float(critic_td_loss.detach().item()))
            if self.component_outcome_aux_enabled:
                set_decision_rule = bool(
                    getattr(
                        self.critic.q_networks[0],
                        "basket_conditioned_allocation_enabled", False,
                    )
                )
                use_real_asset_teacher = bool(
                    local_financial_predictions is not None
                    and replay_data is not bellman_source_replay_data
                )
                if use_real_asset_teacher:
                    real_obs_float = {
                        key: value.float()
                        for key, value in bellman_source_replay_data.observations.items()
                    }
                    real_obs_float_g = self._gate_observations_with_sigma(
                        real_obs_float
                    )
                    (
                        auxiliary_predictions,
                        auxiliary_local_predictions,
                        auxiliary_context_corrections,
                    ) = self.critic.forward_component_forecasts(
                        real_obs_float_g
                    )
                    auxiliary_replay_data = bellman_source_replay_data
                    auxiliary_observations = real_obs_float
                else:
                    auxiliary_predictions = component_predictions
                    auxiliary_local_predictions = local_financial_predictions
                    auxiliary_context_corrections = local_financial_corrections
                    auxiliary_replay_data = replay_data
                    auxiliary_observations = obs_float
                component_targets = auxiliary_replay_data.component_outcomes.float()
                component_mask = auxiliary_replay_data.component_outcome_mask.float()
                if isinstance(auxiliary_observations, dict) and "mask" in auxiliary_observations:
                    component_mask = component_mask * auxiliary_observations[
                        "mask"
                    ].float().unsqueeze(-1)
                asset_teacher_real_rows.append(
                    float(component_targets.shape[0])
                    if use_real_asset_teacher else 0.0
                )
                valid_count = component_mask.sum()
                return_reference = (
                    self.component_outcome_loss_return_reference
                    if set_decision_rule else 1.0
                )
                component_loss_terms = []
                component_final_loss_terms = []
                component_absolute_errors = []
                for critic_index, predictions in enumerate(auxiliary_predictions):
                    per_component_huber = F.smooth_l1_loss(
                        predictions / return_reference,
                        component_targets / return_reference,
                        reduction="none",
                        beta=self.component_outcome_huber_beta,
                    )
                    absolute_error = (
                        predictions - component_targets
                    ).abs()
                    if auxiliary_local_predictions is not None:
                        local_predictions = auxiliary_local_predictions[critic_index]
                        local_huber = F.smooth_l1_loss(
                            local_predictions / return_reference,
                            component_targets / return_reference,
                            reduction="none",
                            beta=self.component_outcome_huber_beta,
                        )
                        local_error = (local_predictions - component_targets).abs()
                        local_financial_context_abs.append(
                            float(
                                (
                                    auxiliary_context_corrections[critic_index].abs()
                                    * component_mask
                                ).sum().detach().item()
                                / max(float(valid_count.detach().item()), 1.0)
                            )
                        )
                    if valid_count.item() > 0:
                        per_sample_valid = component_mask.sum(dim=(1, 2))
                        per_sample_component_loss = (
                            (per_component_huber * component_mask).sum(dim=(1, 2))
                            / per_sample_valid.clamp(min=1.0)
                        )
                        per_sample_absolute_error = (
                            (absolute_error * component_mask).sum(dim=(1, 2))
                            / per_sample_valid.clamp(min=1.0)
                        )
                        valid_samples = per_sample_valid > 0
                        final_loss = self._basket_balanced_replay_mean(
                            per_sample_component_loss, auxiliary_observations, valid_samples
                        )
                        component_final_loss_terms.append(final_loss)
                        if auxiliary_local_predictions is not None:
                            local_per_sample_loss = (
                                (local_huber * component_mask).sum(dim=(1, 2))
                                / per_sample_valid.clamp(min=1.0)
                            )
                            local_per_sample_mae = (
                                (local_error * component_mask).sum(dim=(1, 2))
                                / per_sample_valid.clamp(min=1.0)
                            )
                            local_loss = self._basket_balanced_replay_mean(
                                local_per_sample_loss, auxiliary_observations, valid_samples
                            )
                            local_financial_losses.append(float(local_loss.detach().item()))
                            local_financial_maes.append(
                                float(self._basket_balanced_replay_mean(
                                    local_per_sample_mae, auxiliary_observations, valid_samples
                                ).detach().item())
                            )
                            final_loss = final_loss + float(
                                getattr(config, "LOCAL_FINANCIAL_LOCAL_LOSS_WEIGHT", 1.0)
                            ) * local_loss
                        component_loss_terms.append(final_loss)
                        component_absolute_errors.append(
                            self._basket_balanced_replay_mean(
                                per_sample_absolute_error,
                                auxiliary_observations,
                                valid_samples,
                            )
                        )
                    else:
                        # Keep the graph well formed while producing no update.
                        component_loss_terms.append(predictions.sum() * 0.0)
                        component_final_loss_terms.append(predictions.sum() * 0.0)
                        component_absolute_errors.append(
                            predictions.detach().sum() * 0.0
                        )
                component_loss = sum(component_loss_terms)
                component_final_loss = sum(component_final_loss_terms)
                if set_decision_rule and self.basket_relative_aux_weight > 0.0:
                    mixed_targets = replay_data.component_outcomes.float()
                    mixed_mask = replay_data.component_outcome_mask.float()
                    mixed_mask = mixed_mask * obs_float["mask"].unsqueeze(-1)
                    source_index = self._last_cross_basket_relabel_source_index
                    target_index = self._last_cross_basket_relabel_target_index
                    paired_index = (
                        torch.cat((source_index, target_index), dim=0)
                        if source_index is not None and target_index is not None
                        else None
                    )
                    relative_observations = (
                        self._index_observation_rows(obs_float, paired_index)
                        if paired_index is not None else obs_float
                    )
                    relative_terms = []
                    for predictions in component_predictions:
                        relative_per_row, relative_valid = (
                            basket_relative_forecast_loss(
                                predictions, mixed_targets, mixed_mask,
                                return_reference=return_reference,
                                huber_beta=self.component_outcome_huber_beta,
                            )
                        )
                        if paired_index is not None:
                            relative_per_row = relative_per_row.index_select(
                                0, paired_index
                            )
                            relative_valid = relative_valid.index_select(
                                0, paired_index
                            )
                        relative_terms.append(self._basket_balanced_replay_mean(
                            relative_per_row, relative_observations, relative_valid
                        ))
                    relative_loss = sum(relative_terms)
                    component_loss = (
                        component_loss
                        + self.basket_relative_aux_weight * relative_loss
                    )
                    basket_relative_losses.append(float(relative_loss.detach().item()))
                component_scale = (
                    self._calibrate_component_outcome_gradient_scale(
                        critic_td_loss,
                        component_loss,
                    )
                    if valid_count.item() > 0
                    else 0.0
                )
                weighted_component_loss = component_scale * component_loss
                critic_loss = critic_loss + weighted_component_loss
                component_outcome_losses.append(component_final_loss.item())
                component_outcome_weighted_losses.append(
                    float(weighted_component_loss.detach().item())
                )
                if local_financial_predictions is not None:
                    local_financial_total_aux_losses.append(
                        float(component_loss.detach().item())
                    )
                component_outcome_maes.append(
                    float(
                        torch.stack(component_absolute_errors).mean().item()
                    )
                )
                component_outcome_valid_fractions.append(
                    float(component_mask.mean().item())
                )
                component_outcome_gradient_scales.append(component_scale)
                if (
                    valid_count.item() > 0
                    and getattr(self, '_component_outcome_gradient_measurement_update', None)
                    == self._n_updates
                ):
                    td_norm = float(self._component_outcome_td_gradient_norm)
                    auxiliary_norm = float(self._component_outcome_aux_gradient_norm)
                    component_outcome_gradient_measurements.append((
                        td_norm, auxiliary_norm,
                        component_scale * auxiliary_norm / max(td_norm, 1e-12),
                    ))
                correlations = [
                    self._masked_cross_sectional_correlation(
                        predictions,
                        component_targets,
                        component_mask,
                    )
                    for predictions in auxiliary_predictions
                ]
                correlations = [
                    value for value in correlations if value is not None
                ]
                if correlations:
                    component_outcome_correlations.append(
                        float(np.mean(correlations))
                    )
                if valid_count.item() > 0:
                    valid_targets = component_targets[
                        component_mask.bool()
                    ]
                    component_outcome_target_abs_means.append(
                        float(valid_targets.abs().mean().item())
                    )
                    target_clip = float(
                        getattr(config, "COMPONENT_OUTCOME_TARGET_CLIP", 5.0)
                    )
                    component_outcome_target_clip_fractions.append(
                        float(
                            (valid_targets.abs() >= target_clip - 1e-6)
                            .float()
                            .mean()
                            .item()
                        )
                    )
            critic_report_loss = critic_loss
            critic_guard_parameters = None
            critic_main_gradients = None
            if (
                auxiliary_this_update
                and self.predictive_ssl_gradient_guard_enabled
            ):
                # Finish and free the original-row financial graph before building
                # the independent foundation graph.  The former implementation
                # kept both activation trees alive and only separated their
                # gradients afterward, creating the worst TD3 CUDA peak.
                critic_guard_parameters = tuple(
                    parameter
                    for parameter in self.critic.parameters()
                    if parameter.requires_grad
                )
                critic_main_gradients = torch.autograd.grad(
                    critic_report_loss,
                    critic_guard_parameters,
                    allow_unused=True,
                )
            if predictive_ssl_this_update:
                if not (
                    isinstance(obs_float_g, dict)
                    and isinstance(next_obs_float_g, dict)
                    and hasattr(self.critic, "predictive_representation_loss")
                ):
                    raise RuntimeError(
                        "Predictive representation learning requires the "
                        "variable-coin DeepSet dict actor/critic."
                    )
                predictive_ssl_started_at = time.perf_counter()
                predictive_cuda_devices = []
                if predictive_actions_float.is_cuda:
                    predictive_cuda_devices = [
                        predictive_actions_float.device.index
                        if predictive_actions_float.device.index is not None
                        else torch.cuda.current_device()
                    ]
                with torch.random.fork_rng(devices=predictive_cuda_devices):
                    predictive_q_targets = None
                    if bool(getattr(config, "VALUE_EQUIVALENT_MODEL_ENABLED", True)):
                        with torch.no_grad():
                            predictive_q_targets = torch.min(
                                torch.cat(
                                    self.critic(
                                        predictive_obs_float_g,
                                        predictive_actions_float,
                                    ),
                                    dim=1,
                                ),
                                dim=1,
                                keepdim=True,
                            ).values
                    predictive_stats = self.critic.predictive_representation_loss(
                        predictive_obs_float_g,
                        predictive_next_obs_float_g,
                        predictive_actions_float,
                        predictive_next_actions_center,
                        self.critic_target,
                        dones=predictive_dones,
                        future_observations=predictive_future_obs_float_g,
                        future_actions=predictive_future_actions,
                        future_next_actions=predictive_future_next_actions,
                        future_valid_steps=predictive_future_valid_steps,
                        future_outcome_valid_steps=(
                            predictive_future_outcome_valid_steps
                        ),
                        future_rewards=predictive_future_rewards,
                        future_dones=predictive_future_dones,
                        future_continuation_value_targets=(
                            predictive_future_continuation_values
                        ),
                        current_q_targets=predictive_q_targets,
                        gamma=self.gamma,
                        horizon_discount=self.predictive_ssl_horizon_discount,
                    )
                critic_auxiliary_loss = predictive_stats["loss"]
                if foundation_this_update:
                    foundation_critic_stats = self.critic.foundation_loss(
                        foundation_observations,
                        foundation_future_returns,
                        foundation_future_valid,
                        foundation_candidates,
                        foundation_targets,
                        subsample_candidates=bool(
                            getattr(
                                config,
                                "FOUNDATION_CRITIC_CANDIDATE_SUBSAMPLE",
                                True,
                            )
                        ),
                    )
                    critic_auxiliary_loss = 0.5 * (
                        critic_auxiliary_loss
                        + foundation_critic_stats["loss"]
                    )
                    foundation_critic_losses.append(
                        float(
                            foundation_critic_stats["loss"].detach().item()
                        )
                    )
                    foundation_action_losses.append(
                        float(
                            foundation_critic_stats["action_loss"]
                            .detach()
                            .item()
                        )
                    )
                    foundation_q_rank_losses.append(
                        float(
                            foundation_critic_stats["q_rank_loss"]
                            .detach()
                            .item()
                        )
                    )
                    foundation_representation_losses.append(
                        float(
                            foundation_critic_stats["representation_loss"]
                            .detach()
                            .item()
                        )
                    )
                    foundation_candidate_evaluation_fractions.append(
                        float(
                            foundation_critic_stats[
                                "candidate_evaluation_fraction"
                            ].detach().item()
                        )
                    )
                weighted_predictive_loss = (
                    self.predictive_ssl_critic_loss_weight
                    * critic_auxiliary_loss
                )
                critic_loss = critic_loss + weighted_predictive_loss
                # A separate .item() for every metric forces a CUDA/CPU
                # synchronization barrier for each scalar. Transfer the same
                # telemetry in one compact readback per predictive update.
                predictive_metric_destinations = (
                    ("loss", predictive_critic_losses),
                    ("feature_loss", predictive_feature_losses),
                    ("market_loss", predictive_market_losses),
                    ("relation_loss", predictive_relation_losses),
                    ("fused_loss", predictive_fused_losses),
                    ("downstream_loss", predictive_downstream_losses),
                    ("reward_loss", predictive_reward_losses),
                    (
                        "continuation_value_loss",
                        predictive_continuation_value_losses,
                    ),
                    (
                        "bellman_equivalence_loss",
                        predictive_bellman_equivalence_losses,
                    ),
                    ("expectile_value_loss", predictive_expectile_value_losses),
                    ("state_value_mean", predictive_state_value_means),
                    ("advantage_mean", predictive_advantage_means),
                    (
                        "advantage_positive_fraction",
                        predictive_advantage_positive_fractions,
                    ),
                    ("advantage_weight_min", predictive_advantage_weight_mins),
                    ("advantage_weight_max", predictive_advantage_weight_maxes),
                    ("routing_entropy", predictive_routing_entropies),
                    ("expert_min_usage", predictive_expert_min_usages),
                    ("expert_max_usage", predictive_expert_max_usages),
                    ("valid_fraction", predictive_valid_fractions),
                    (
                        "valid_horizon_fraction",
                        predictive_valid_horizon_fractions,
                    ),
                    ("horizon_loss_mean", predictive_horizon_loss_means),
                    ("control_gate", predictive_control_gates),
                    (
                        "control_projection_norm",
                        predictive_control_projection_norms,
                    ),
                )
                for horizon_index in range(1, self.predictive_ssl_horizon + 1):
                    metric_name = f"horizon_{horizon_index}_loss"
                    if metric_name in predictive_stats:
                        predictive_metric_destinations += (
                            (metric_name, predictive_per_horizon_losses[horizon_index]),
                        )
                    value_metric_name = f"value_horizon_{horizon_index}_loss"
                    if value_metric_name in predictive_stats:
                        predictive_metric_destinations += (
                            (
                                value_metric_name,
                                predictive_per_horizon_value_losses[horizon_index],
                            ),
                        )
                predictive_metric_values = torch.stack(
                    [
                        predictive_stats[name].detach().reshape(())
                        for name, _ in predictive_metric_destinations
                    ]
                ).tolist()
                for (_, destination), value in zip(
                    predictive_metric_destinations, predictive_metric_values
                ):
                    destination.append(float(value))
                predictive_ssl_seconds += (
                    time.perf_counter() - predictive_ssl_started_at
                )
            elif foundation_this_update:
                if not isinstance(foundation_observations, dict):
                    raise RuntimeError(
                        "Foundation rehearsal requires dictionary observations"
                    )
                foundation_critic_stats = self.critic.foundation_loss(
                    foundation_observations,
                    foundation_future_returns,
                    foundation_future_valid,
                    foundation_candidates,
                    foundation_targets,
                    subsample_candidates=bool(
                        getattr(
                            config,
                            "FOUNDATION_CRITIC_CANDIDATE_SUBSAMPLE",
                            True,
                        )
                    ),
                )
                critic_auxiliary_loss = foundation_critic_stats["loss"]
                weighted_predictive_loss = (
                    self.predictive_ssl_critic_loss_weight
                    * critic_auxiliary_loss
                )
                critic_loss = critic_loss + weighted_predictive_loss
                foundation_critic_losses.append(
                    float(critic_auxiliary_loss.detach().item())
                )
                foundation_action_losses.append(
                    float(
                        foundation_critic_stats["action_loss"].detach().item()
                    )
                )
                foundation_q_rank_losses.append(
                    float(
                        foundation_critic_stats["q_rank_loss"]
                        .detach()
                        .item()
                    )
                )
                foundation_representation_losses.append(
                    float(
                        foundation_critic_stats["representation_loss"]
                        .detach()
                        .item()
                    )
                )
                foundation_candidate_evaluation_fractions.append(
                    float(
                        foundation_critic_stats[
                            "candidate_evaluation_fraction"
                        ].detach().item()
                    )
                )
            critic_losses.append(critic_report_loss.item())
            critic_total_losses.append(critic_loss.item())
            critic_losses_unweighted.append(critic_loss_unweighted.item())
            self._update_basket_replay_metric_ema(
                "critic_td_loss",
                float(critic_loss_unweighted.detach().item()),
                current_replay_stats,
            )
            if sigma_intensity is not None and sample_weights is not None:
                sigma_intensity_means.append(float(sigma_intensity.mean().item()))
                sigma_intensity_stds.append(float(sigma_intensity.std(unbiased=False).item()))
                sigma_weight_means.append(float(sample_weights.mean().item()))
                sigma_weight_maxes.append(float(sample_weights.max().item()))

            # Periodic held-out probe gradients label whether selected
            # behavioral groups transfer through the shared FAM/AAM value
            # surface.  Dense labels supervise the scorer; the much sparser,
            # separately gated MER cadence may reuse a qualified group proposal.
            transfer_measurement_start = time.perf_counter()
            transfer_cosine = self._measure_value_experience_transfer(
                replay_data
            )
            transfer_measurement_seconds = (
                time.perf_counter() - transfer_measurement_start
            )
            if transfer_cosine is not None:
                self._value_experience_transfer_measurement_count += 1
                self._value_experience_transfer_seconds_total += float(
                    transfer_measurement_seconds
                )
                self._last_value_experience_stats.update(
                    transfer_seconds=float(transfer_measurement_seconds),
                    transfer_seconds_mean=float(
                        self._value_experience_transfer_seconds_total
                    )
                    / float(
                        max(
                            self._value_experience_transfer_measurement_count,
                            1,
                        )
                    ),
                    transfer_seconds_amortized=float(
                        self._value_experience_transfer_seconds_total
                    )
                    / float(max(self._value_experience_selection_count, 1)),
                )
            self.critic.optimizer.zero_grad(set_to_none=True)
            if (
                auxiliary_this_update
                and self.predictive_ssl_gradient_guard_enabled
            ):
                critic_gradient_stats = apply_bounded_auxiliary_gradients(
                    critic_report_loss,
                    weighted_predictive_loss,
                    critic_guard_parameters,
                    max_shared_ratio=(
                        float(
                            getattr(
                                config,
                                "FOUNDATION_CRITIC_SHARED_GRADIENT_MAX_RATIO",
                                self.predictive_ssl_critic_shared_gradient_max_ratio,
                            )
                        )
                        if foundation_this_update
                        else self.predictive_ssl_critic_shared_gradient_max_ratio
                    ),
                    project_conflicts=self.predictive_ssl_project_conflicts,
                    scale_by_positive_cosine=(
                        self.predictive_ssl_scale_shared_by_positive_cosine
                    ),
                    alignment_cosine_ema=(
                        self._predictive_ssl_critic_cosine_ema
                    ),
                    alignment_ema_decay=(
                        self.predictive_ssl_gradient_cosine_ema_decay
                    ),
                    epsilon=self.predictive_ssl_gradient_epsilon,
                    max_grad_norm=self.max_grad_norm,
                    main_gradients=critic_main_gradients,
                )
                predictive_critic_gradient_stats.append(
                    critic_gradient_stats
                )
                self._predictive_ssl_critic_cosine_ema = float(
                    critic_gradient_stats["smoothed_cosine"]
                )
            else:
                critic_loss.backward()
            if not (
                auxiliary_this_update
                and self.predictive_ssl_gradient_guard_enabled
            ):
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
            # Merge the CRL-gated, transfer-compatible first-order meta-gradient
            # before Adam so optimizer moments include every accepted update.
            self._apply_pending_value_experience_mer()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
            self.critic.optimizer.step()
            value_experience_feedback_due = (
                self._value_experience_last_selected_features is not None
                and self._n_updates % self.value_experience_feedback_interval
                == 0
            )
            value_experience_bellman_loss = None
            if value_experience_feedback_due:
                feedback_start = time.perf_counter()
                value_experience_bellman_loss = (
                    self._train_value_experience_from_bellman(replay_data)
                )
                feedback_seconds = time.perf_counter() - feedback_start
                if value_experience_bellman_loss is not None:
                    self._value_experience_feedback_count += 1
                    self._value_experience_feedback_seconds_total += float(
                        feedback_seconds
                    )
                    self._last_value_experience_stats.update(
                        feedback_seconds=float(feedback_seconds),
                        feedback_seconds_mean=float(
                            self._value_experience_feedback_seconds_total
                        )
                        / float(max(self._value_experience_feedback_count, 1)),
                        feedback_seconds_amortized=float(
                            self._value_experience_feedback_seconds_total
                        )
                        / float(max(self._value_experience_selection_count, 1)),
                    )
            self._last_value_experience_stats.update(
                feedback_applied=float(
                    value_experience_bellman_loss is not None
                ),
                feedback_interval=float(
                    self.value_experience_feedback_interval
                ),
                feedback_fraction=float(
                    self._value_experience_feedback_count
                ) / float(max(self._value_experience_selection_count, 1)),
            )
            if value_experience_bellman_loss is not None:
                self._last_value_experience_stats["bellman_feedback_loss"] = (
                    float(value_experience_bellman_loss)
                )

            # ✅ ACTOR UPDATE: Apply CRL or fixed penalty
            if self._n_updates % self.policy_delay == 0:
                (
                    actor_obs_float,
                    actor_obs_float_g,
                    actor_actions_float,
                    actor_next_actions_center,
                    actor_uses_original_replay,
                ) = self._prepare_actor_crl_tensors(
                    actor_crl_replay_data,
                    replay_data,
                    obs_float,
                    obs_float_g,
                    actions_float,
                    next_actions_center,
                )
                xbasket_actor_batch_rows.append(
                    float(actor_crl_replay_data.actions.shape[0])
                )
                # ── Sigma input modulation for actor ──────────────────────────────
                actor_obs = self._actor_observation(actor_obs_float_g)

                policy_step_diagnostic_due = (
                    self._n_updates % self.crl_policy_step_diagnostic_interval == 0
                )
                actor_step_cpu_rng = None
                actor_step_cuda_rng = None
                actor_step_cuda_devices = []
                if policy_step_diagnostic_due:
                    actor_step_cpu_rng = torch.get_rng_state()
                    if self.device.type == 'cuda':
                        actor_step_cuda_devices = [
                            self.device.index if self.device.index is not None
                            else torch.cuda.current_device()
                        ]
                        actor_step_cuda_rng = torch.cuda.get_rng_state(actor_step_cuda_devices[0])
                a_cur = self.actor(actor_obs)  # raw logits / weights from policy
                previous_raw_action = None
                actor_previous_q = None
                if (
                    self.actor_action_advantage_diagnostic_enabled
                    or self.task_popart is not None
                ):
                    previous_raw_action = self._previous_portfolio_to_raw_action(
                        actor_obs_float_g,
                        dtype=a_cur.dtype,
                        device=a_cur.device,
                    )
                # The actor needs dQ/da, never dQ/dθ_critic. Marking critic
                # weights constant during this forward leaves the action
                # Jacobian unchanged while preventing a second, unused critic
                # parameter-gradient graph on every delayed policy update.
                critic_parameter_grad_states = tuple(
                    parameter.requires_grad
                    for parameter in self.critic.parameters()
                )
                for parameter in self.critic.parameters():
                    parameter.requires_grad_(False)
                if self.use_proportional_exp:
                    actor_candidates, actor_candidate_family_ids = self._sample_local_candidate_actions(
                        a_cur,
                        actor_obs_float_g,
                        self.proportional_exp_candidates,
                        low,
                        high,
                        budget_std=self.proportional_exp_actor_budget_std,
                        composition_std=self.proportional_exp_actor_composition_std,
                    )
                    actor_candidate_q_values = None
                    if previous_raw_action is not None:
                        # The detached diagnostic baseline shares the same
                        # market state as the ProExp cloud. Only its action is
                        # appended to the already-shared value-head pass.
                        combined_candidates = torch.cat(
                            (
                                actor_candidates,
                                previous_raw_action.detach().unsqueeze(1),
                            ),
                            dim=1,
                        )
                        combined_q_values = self._evaluate_candidate_critics(
                            self.critic,
                            actor_obs_float_g,
                            combined_candidates,
                        )
                        actor_candidate_q_values = tuple(
                            values[:, :-1] for values in combined_q_values
                        )
                        actor_previous_q = torch.min(
                            torch.stack(
                                [values[:, -1] for values in combined_q_values],
                                dim=0,
                            ),
                            dim=0,
                        ).values.detach()
                    actor_stats = self._compute_proportional_exp_statistics(
                        self.critic,
                        actor_obs_float_g,
                        actor_candidates,
                        actor_candidate_family_ids,
                        update_disagreement_ema=False,
                        precomputed_q_values=actor_candidate_q_values,
                    )
                    actor_objective_value = actor_stats["soft_value"]
                    proexp_alpha_means.append(float(actor_stats["alpha"].mean().item()))
                    proexp_disagreement_means.append(float(actor_stats["disagreement_mean"].mean().item()))
                    proexp_candidate_entropies.append(float(actor_stats["entropy"].mean().item()))
                    proexp_soft_value_means.append(float(actor_stats["soft_value"].mean().item()))
                    proexp_center_q_means.append(float(actor_stats["center_q"].mean().item()))
                    proexp_soft_gap_means.append(float((actor_stats["soft_value"] - actor_stats["center_q"]).mean().item()))
                    if actor_stats["budget_alpha"] is not None:
                        proexp_budget_alpha_means.append(float(actor_stats["budget_alpha"].mean().item()))
                    if actor_stats["composition_alpha"] is not None:
                        proexp_composition_alpha_means.append(float(actor_stats["composition_alpha"].mean().item()))
                    if actor_stats["budget_disagreement_mean"] is not None:
                        proexp_budget_disagreement_means.append(float(actor_stats["budget_disagreement_mean"].mean().item()))
                    if actor_stats["composition_disagreement_mean"] is not None:
                        proexp_composition_disagreement_means.append(float(actor_stats["composition_disagreement_mean"].mean().item()))
                    proexp_budget_entropies.append(float(actor_stats["budget_entropy"].mean().item()))
                    proexp_composition_entropies.append(float(actor_stats["composition_entropy"].mean().item()))
                else:
                    actor_objective_value = self.critic.q1_forward(
                        actor_obs_float_g,
                        a_cur,
                    ).squeeze(-1)
                for parameter, requires_grad in zip(
                    self.critic.parameters(), critic_parameter_grad_states
                ):
                    parameter.requires_grad_(requires_grad)

                raw_actor_q_loss = -self._basket_balanced_replay_mean(
                    self._value_experience_weight_rows(actor_objective_value),
                    actor_obs_float,
                )
                actor_q_loss = raw_actor_q_loss
                actor_action_advantage = None
                actor_popart_per_sample_loss = None
                if (
                    self.actor_action_advantage_diagnostic_enabled
                    or self.task_popart is not None
                ):
                    previous_action_q = actor_previous_q
                    if previous_action_q is None and previous_raw_action is not None:
                        previous_action_q = self._detached_previous_action_q(
                            actor_obs_float_g,
                            a_cur,
                            use_twin_minimum=self.use_proportional_exp,
                        )
                    if previous_action_q is not None:
                        actor_action_advantage = (
                            actor_objective_value - previous_action_q
                        )
                    elif bool(
                        getattr(
                            config,
                            "CRITIC_CENTERED_ADVANTAGE_ENABLED",
                            False,
                        )
                    ):
                        raise RuntimeError(
                            "centered actor diagnostic could not value the "
                            "current-portfolio reference action"
                        )

                if self.task_popart is not None:
                    if actor_action_advantage is None:
                        raise RuntimeError(
                            "Task PopArt requires the current-portfolio-centered "
                            "actor advantage"
                        )
                    actor_popart_task_ids = (
                        self.task_popart.task_ids_from_observations(actor_obs_float_g)
                    )
                    (
                        actor_popart_advantage,
                        actor_popart_scale,
                        _,
                    ) = self.task_popart.balanced_advantage(
                        actor_action_advantage,
                        actor_popart_task_ids,
                    )
                    # This is the PopArt policy update.  Dividing by each
                    # task's sigma equalizes basket authority; multiplying by
                    # one shared global sigma preserves the established
                    # financial-to-CRL unit scale for every basket equally.
                    actor_popart_per_sample_loss = -self._value_experience_weight_rows(
                        actor_popart_advantage
                    )
                    actor_q_loss = self._basket_balanced_replay_mean(
                        actor_popart_per_sample_loss,
                        actor_obs_float,
                    )
                    actor_popart_advantages.append(
                        float(actor_popart_advantage.detach().mean().item())
                    )
                    actor_popart_scale_means.append(
                        float(actor_popart_scale.detach().mean().item())
                    )
                    actor_popart_scale_maxes.append(
                        float(actor_popart_scale.detach().max().item())
                    )

                # Calculate action smoothness constraint based on portfolio weights, not raw outputs
                # ✅ CRITICAL STABILITY FIX: Use TARGET ACTOR for next action to filter high-frequency noise.
                # The current actor is jittery and creates phantom "acceleration" violations from noise,
                # causing λ_accel to explode. The target network provides a stable, slowly-evolving
                # reference that represents the true policy trajectory, not training noise.
                with torch.no_grad():
                    # Target parameters are Polyak-updated only after this
                    # block.  When the actor uses the original batch, this
                    # next action is the target actor on those real next
                    # observations, not the mixed-batch Bellman action.
                    a_next = actor_next_actions_center.detach()
                
                # Compute mask for active dimensions FIRST (needed for all weight calculations)

                # Get prev_action which are the previous step's weights (w_{t-1})
                # IMPORTANT: prev_action from env is already normalized, but we need to apply the same mask
                # to ensure consistent handling of inactive dimensions across all weight tensors
                w_prev = None
                if isinstance(actor_crl_replay_data.observations, dict) and "prev_action" in actor_crl_replay_data.observations:
                    # prev_action from the buffer are already normalized weights from last_target_allocation
                    w_prev_raw = actor_crl_replay_data.observations["prev_action"].to(a_cur.device)
                    # Store raw for potential use, but we'll mask it properly after action_mask is computed
                    w_prev = w_prev_raw


                action_mask = _build_action_mask_from_observations(
                    actor_crl_replay_data.observations,
                    action_dim=a_cur.shape[1],
                    device=a_cur.device,
                    dtype=a_cur.dtype,
                )

                active_dims = action_mask.sum(dim=1).clamp(min=1.0)

                # ── WEIGHT-SPACE PENALTY (matches MLP path exactly) ───────────────
                # Both penalty cost AND violation are computed in normalised portfolio
                # weight space, identical to the flat SimpleMlpTD3Policy CRL.
                # The gradient of the penalty flows through
                # normalize_raw_actions_to_portfolio_weights so the CRL Jacobian is
                # rank-N (same as MLP).  Approach 1 already gave the Q-gradient full
                # rank (N+1) by feeding raw logits into the critic, so the combined
                # gradient structure is:  full-rank Q + rank-N CRL — the same geometry
                # that makes the MLP curriculum work.
                w_cur = normalize_raw_actions_to_portfolio_weights(a_cur, action_mask)
                with torch.no_grad():
                    w_next = normalize_raw_actions_to_portfolio_weights(a_next, action_mask)
                    if (
                        local_financial_predictions is not None
                        and actor_crl_replay_data is replay_data
                    ):
                        replay_weights = normalize_raw_actions_to_portfolio_weights(
                            actor_actions_float.detach(), action_mask
                        )
                        forecast = component_predictions[0].detach().squeeze(-1)
                        base_advantage = (
                            (w_cur.detach()[:, 1:] - replay_weights[:, 1:])
                            * forecast * action_mask[:, 1:]
                        ).sum(dim=1)
                        local_financial_actor_replay_advantages.append(
                            float(base_advantage.abs().mean().item())
                        )

                # --- 1. & 2. Constraint Calculation ---
                actor_velocity_penalty_cost = torch.tensor(0.0, device=self.device)
                batch_velocity_violation = torch.tensor(0.0, device=self.device)
                actor_acceleration_penalty_cost = torch.tensor(0.0, device=self.device)
                batch_acceleration_violation = None

                if w_prev is not None:
                    with torch.no_grad():
                        w_prev_masked = w_prev * action_mask if w_prev is not None else None
                        historical_asset_mask = (w_prev_masked > 1e-9).float() if w_prev_masked is not None else action_mask.new_zeros(action_mask.shape)
                        shared_mask = historical_asset_mask * action_mask
                        has_shared_space = shared_mask.sum(dim=1, keepdim=True) > 1e-8
                        shared_active_dims = shared_mask.sum(dim=1).clamp(min=1.0)

                    use_shared_subspace = has_shared_space.any()
                    if use_shared_subspace:
                        shared_ratio = shared_active_dims.mean() / active_dims.mean()
                        if shared_ratio < 0.3:
                            use_shared_subspace = False
                            if self.verbose > 1:
                                print(f"⚠️ Shared subspace too small ({shared_ratio:.2%} of current space). Using velocity-only mode.")

                    if use_shared_subspace:
                        velocity_diff_sq = (w_next - w_cur).pow(2) * shared_mask
                        per_sample_velocity_costs = self._value_experience_weight_rows(
                            velocity_diff_sq.sum(dim=1)
                        )
                        per_sample_velocity_thresholds = shared_active_dims * self.velocity_constraint_threshold
                        per_sample_velocity_violations = per_sample_velocity_costs - self._value_experience_weight_rows(per_sample_velocity_thresholds)

                        batch_velocity_violation = self._basket_balanced_replay_mean(
                            per_sample_velocity_violations,
                            actor_crl_replay_data.observations,
                            has_shared_space.squeeze(-1),
                        )
                        actor_velocity_penalty_cost = self._basket_balanced_replay_mean(
                            per_sample_velocity_costs,
                            actor_crl_replay_data.observations,
                            has_shared_space.squeeze(-1),
                        )
                        velocity_costs_log.append(actor_velocity_penalty_cost.item())

                        with torch.no_grad():
                            w_prev_shared = w_prev * shared_mask
                            w_next_shared = w_next * shared_mask

                        w_cur_shared = w_cur * shared_mask

                        acceleration_diff_sq = (w_next_shared - 2 * w_cur_shared + w_prev_shared).pow(2) * shared_mask

                        per_sample_acceleration_costs = self._value_experience_weight_rows(
                            acceleration_diff_sq.sum(dim=1)
                        )
                        per_sample_acceleration_thresholds = shared_active_dims * self.acceleration_constraint_threshold
                        per_sample_acceleration_violations = per_sample_acceleration_costs - self._value_experience_weight_rows(per_sample_acceleration_thresholds)

                        batch_acceleration_violation = self._basket_balanced_replay_mean(
                            per_sample_acceleration_violations,
                            actor_crl_replay_data.observations,
                            has_shared_space.squeeze(-1),
                        )
                        actor_acceleration_penalty_cost = self._basket_balanced_replay_mean(
                            per_sample_acceleration_costs,
                            actor_crl_replay_data.observations,
                            has_shared_space.squeeze(-1),
                        )
                        acceleration_costs_log.append(actor_acceleration_penalty_cost.item())
                    else:
                        velocity_diff_sq = (w_next - w_cur).pow(2) * action_mask
                        per_sample_velocity_costs = self._value_experience_weight_rows(
                            velocity_diff_sq.sum(dim=1)
                        )
                        per_sample_velocity_thresholds = active_dims * self.velocity_constraint_threshold
                        per_sample_velocity_violations = per_sample_velocity_costs - self._value_experience_weight_rows(per_sample_velocity_thresholds)
                        batch_velocity_violation = self._basket_balanced_replay_mean(
                            per_sample_velocity_violations,
                            actor_crl_replay_data.observations,
                        )
                        actor_velocity_penalty_cost = self._basket_balanced_replay_mean(
                            per_sample_velocity_costs,
                            actor_crl_replay_data.observations,
                        )
                        velocity_costs_log.append(actor_velocity_penalty_cost.item())
                        batch_acceleration_violation = None
                        actor_acceleration_penalty_cost = torch.tensor(0.0, device=self.device)
                else:
                    velocity_diff_sq = (w_next - w_cur).pow(2) * action_mask
                    per_sample_velocity_costs = self._value_experience_weight_rows(
                        velocity_diff_sq.sum(dim=1)
                    )
                    per_sample_velocity_thresholds = active_dims * self.velocity_constraint_threshold
                    per_sample_velocity_violations = per_sample_velocity_costs - self._value_experience_weight_rows(per_sample_velocity_thresholds)
                    batch_velocity_violation = self._basket_balanced_replay_mean(
                        per_sample_velocity_violations,
                        actor_crl_replay_data.observations,
                    )
                    actor_velocity_penalty_cost = self._basket_balanced_replay_mean(
                        per_sample_velocity_costs,
                        actor_crl_replay_data.observations,
                    )
                    velocity_costs_log.append(actor_velocity_penalty_cost.item())

                # --- 3. Combine Penalties ---
                smooth_loss = torch.tensor(0.0, device=self.device)
                velocity_penalty_term = torch.tensor(0.0, device=self.device)
                acceleration_penalty_term = torch.tensor(0.0, device=self.device)
                if self.use_crl:
                    # Velocity Penalty
                    current_lambda_velocity = torch.clamp(self.lambda_param_velocity, min=0.0)
                    velocity_penalty_term = current_lambda_velocity * actor_velocity_penalty_cost
                    smooth_loss += velocity_penalty_term
                    
                   
                    with torch.no_grad():

                        coupling_k = 25.0  # orginally 50.0 then 15.0 now should be 25.0 
                        
                        
                        # ✅ TELEONOMY FIX: Switch from ReLU to Sigmoid for smooth, bounded coupling.
                        # ReLU caused "sharp inactive learning" (dead zones where coupling = 1.0) and
                        # potential explosions (unbounded growth).
                        # Sigmoid maps the velocity violation to a smooth (0, 1) range.
                        # We use a scaled sigmoid: 2.0 * sigmoid(x) so it can range from 0 to 2.
                        # This allows the coupling to boost acceleration learning when velocity is high,
                        # but smoothly saturates instead of exploding. 
                        
                        # coupling_factor = torch.exp(-coupling_k * F.relu(batch_velocity_violation)) # OLD ReLU

                        # New Sigmoid Logic:
                        # 1. Scale violation to be sensitive to small changes (k=50 is good)
                        # 2. Sigmoid squash -> (0, 1)
                        # 3. Invert logic: High violation should REDUCE coupling (shut off acceleration constraint)
                        #    Low violation should INCREASE coupling (enable acceleration constraint)
                        
                        scaled_violation = coupling_k * batch_velocity_violation
                        # We want:
                        # High Velocity Violation -> Coupling -> 0 (Focus on Velocity)
                        # Low/Negative Violation -> Coupling -> 1 (Allow Acceleration Refinement)
                        
                        # Sigmoid(x) goes 0->1 as x goes -inf->inf.
                        # We want the opposite response to violation.
                        # coupling_factor = 1.0 - torch.sigmoid(scaled_violation)
                        
                        # Actually, let's keep the exponential decay logic but make the input bounded by Tanh
                        # to prevent the "explosion" part, while Tanh's linearity near 0 preserves gradient.
                        # Tanh(ReLU(x)) is still 0 for x<0.
                        # Softplus is a smooth ReLU: log(1 + exp(x)). It is always positive and smooth.
                        
                        # Let's go with the user's request: SIGMOID-like behavior.
                        # Standard logistic function for "gating":
                        # Gate = 1 / (1 + exp(k * violation))
                        # If violation is HIGH (positive), exp is huge, Gate -> 0. (Correct: decouple acceleration)
                        # If violation is LOW (negative), exp is tiny, Gate -> 1. (Correct: couple acceleration)
                        
                        coupling_factor = torch.sigmoid(-coupling_k * batch_velocity_violation)
                        
                       
                    # Log the coupling factor for monitoring
                    self.logger.record("train/crl_coupling_factor", coupling_factor.item())
                    # ✅ LOG RAW VIOLATION: Track the raw acceleration violation used for lambda updates
                    # This helps monitor whether acceleration lambda is learning from the true signal
                    if w_prev is not None and self._n_updates >= self.lambda_warmup_steps and batch_acceleration_violation is not None:
                        self.logger.record("train/raw_acceleration_violation", batch_acceleration_violation.item())
                        # Also log what the gated violation would have been (for comparison)
                        min_lambda_learning_gate = 0.05
                        dynamic_gate = min_lambda_learning_gate + (1.0 - min_lambda_learning_gate) * coupling_factor
                        gated_violation = batch_acceleration_violation * dynamic_gate
                        self.logger.record("train/gated_acceleration_violation", gated_violation.item())
                        self.logger.record("train/dynamic_gate", dynamic_gate.item())

                    # Acceleration Penalty is only active after the warmup period and is scaled by the coupling factor
                    if w_prev is not None and self._n_updates >= self.lambda_warmup_steps:
                        current_lambda_acceleration = torch.clamp(self.lambda_param_acceleration, min=0.0)
                        # ✅ CURRICULUM LEARNING FIX: Apply acceleration penalty, modulated by the coupling factor.
                        # When velocity is violated (coupling ≈ 0), acceleration penalty is weak (actor focuses on velocity).
                        # When velocity is satisfied (coupling ≈ 1), acceleration penalty is full strength (refinement phase).
                        acceleration_penalty_term = coupling_factor * current_lambda_acceleration * actor_acceleration_penalty_cost
                        smooth_loss += acceleration_penalty_term

                        # ✅ CRITICAL FIX: Use RAW violation for lambda updates, NOT gated violation.
                        # This breaks the catch-22 where acceleration lambda couldn't learn when velocity was violated.
                        # 
                        # Rationale:
                        # - Lambda updates need the TRUE violation signal to learn properly
                        # - The coupling factor gates the PENALTY (actor loss), not the lambda learning signal
                        # - This allows acceleration lambda to grow even during exploration phase (when velocity is violated)
                        # - Once velocity is satisfied and coupling → 1, the penalty becomes active and lambda is already learned
                        # 
                        # This maintains curriculum learning (penalty is gated) while allowing lambda to learn continuously.
                        raw_acceleration_violation = batch_acceleration_violation
                    else:
                        raw_acceleration_violation = None                    

                    # ✅ SEPARATE SIGNALS: Raw violation for lambda learning, gated penalty for actor loss
                    # This allows acceleration lambda to learn and reach threshold even when velocity is satisfied
                    self._update_lambdas(batch_velocity_violation, raw_acceleration_violation)
                else:
                    # Fallback to fixed penalty on velocity only
                    velocity_penalty_term = self.action_reg_coef * actor_velocity_penalty_cost
                    smooth_loss = velocity_penalty_term

                actor_control_loss = actor_q_loss + smooth_loss
                actor_loss = actor_control_loss
                if (
                    self.task_popart is not None
                    and actor_popart_per_sample_loss is not None
                    and self._n_updates % self.task_popart_gradient_interval == 0
                ):
                    actor_direction_stats = measure_task_gradient_directions(
                        actor_popart_per_sample_loss,
                        actor_popart_task_ids,
                        tuple(
                            parameter
                            for parameter in self.actor.parameters()
                            if parameter.requires_grad
                        ),
                        max_tasks=self.task_popart_gradient_max_tasks,
                        min_samples=self.task_popart_gradient_min_samples,
                    )
                    if actor_direction_stats is not None:
                        popart_actor_gradient_directions.append(
                            actor_direction_stats
                        )
                actor_guard_parameters = None
                actor_main_gradients = None
                if (
                    auxiliary_this_update
                    and self.predictive_ssl_gradient_guard_enabled
                ):
                    # As with the critic, materialize the financial gradient
                    # first so its large action/Q graph can be released before
                    # constructing actor-side foundation rehearsal.
                    actor_guard_parameters = tuple(
                        parameter
                        for parameter in self.actor.parameters()
                        if parameter.requires_grad
                    )
                    actor_main_gradients = torch.autograd.grad(
                        actor_control_loss,
                        actor_guard_parameters,
                        allow_unused=True,
                    )
                if predictive_ssl_this_update:
                    if not (
                        isinstance(predictive_obs_float_g, dict)
                        and isinstance(predictive_next_obs_float_g, dict)
                        and hasattr(self.actor, "predictive_representation_loss")
                    ):
                        raise RuntimeError(
                            "Actor predictive learning requires the variable-"
                            "coin DeepSet dict policy."
                        )
                    predictive_ssl_started_at = time.perf_counter()
                    predictive_actor_obs = self._actor_observation(
                        predictive_obs_float_g
                    )
                    next_actor_obs_predictive = self._actor_observation(
                        predictive_next_obs_float_g
                    )
                    predictive_cuda_devices = []
                    if predictive_actions_float.is_cuda:
                        predictive_cuda_devices = [
                            predictive_actions_float.device.index
                            if predictive_actions_float.device.index is not None
                            else torch.cuda.current_device()
                        ]
                    with torch.random.fork_rng(devices=predictive_cuda_devices):
                        actor_predictive_stats = (
                            self.actor.predictive_representation_loss(
                                predictive_actor_obs,
                                next_actor_obs_predictive,
                                predictive_actions_float,
                                self.actor_target,
                                dones=predictive_dones,
                                future_observations=(
                                    predictive_future_obs_float_g
                                ),
                                future_actions=predictive_future_actions,
                                future_valid_steps=(
                                    predictive_future_valid_steps
                                ),
                                future_outcome_valid_steps=(
                                    predictive_future_outcome_valid_steps
                                ),
                                future_rewards=predictive_future_rewards,
                                future_dones=predictive_future_dones,
                                future_continuation_value_targets=(
                                    predictive_future_continuation_values
                                ),
                                current_q_targets=predictive_q_targets,
                                gamma=self.gamma,
                                horizon_discount=(
                                    self.predictive_ssl_horizon_discount
                                ),
                            )
                        )
                    actor_auxiliary_loss = actor_predictive_stats["loss"]
                    if foundation_this_update:
                        foundation_actor_stats = self.actor.foundation_loss(
                            self._actor_observation(foundation_observations),
                            foundation_future_returns,
                            foundation_future_valid,
                            foundation_candidates,
                            foundation_utilities,
                        )
                        actor_auxiliary_loss = 0.5 * (
                            actor_auxiliary_loss
                            + foundation_actor_stats["loss"]
                        )
                        foundation_actor_losses.append(
                            float(
                                foundation_actor_stats["loss"]
                                .detach()
                                .item()
                            )
                        )
                        foundation_actor_regrets.append(
                            float(
                                foundation_actor_stats["actor_regret"]
                                .detach()
                                .item()
                            )
                        )
                    actor_loss = actor_loss + (
                        self.predictive_ssl_actor_loss_weight
                        * actor_auxiliary_loss
                    )
                    predictive_actor_losses.append(
                        float(
                            actor_predictive_stats["loss"].detach().item()
                        )
                    )
                    predictive_ssl_seconds += (
                        time.perf_counter() - predictive_ssl_started_at
                    )
                elif foundation_this_update:
                    foundation_actor_stats = self.actor.foundation_loss(
                        self._actor_observation(foundation_observations),
                        foundation_future_returns,
                        foundation_future_valid,
                        foundation_candidates,
                        foundation_utilities,
                    )
                    actor_auxiliary_loss = foundation_actor_stats["loss"]
                    actor_loss = actor_loss + (
                        self.predictive_ssl_actor_loss_weight
                        * actor_auxiliary_loss
                    )
                    foundation_actor_losses.append(
                        float(actor_auxiliary_loss.detach().item())
                    )
                    foundation_actor_regrets.append(
                        float(
                            foundation_actor_stats["actor_regret"]
                            .detach()
                            .item()
                        )
                    )
                actor_q_losses.append(actor_q_loss.item())
                actor_raw_q_losses.append(raw_actor_q_loss.item())
                if actor_action_advantage is not None:
                    actor_advantage_abs_maxima.append(
                        float(actor_action_advantage.detach().abs().amax().item())
                    )
                    actor_action_advantages.append(
                        float(
                            self._basket_balanced_replay_mean(
                                actor_action_advantage.detach(),
                                actor_obs_float,
                            ).item()
                        )
                    )
                actor_velocity_penalty_terms.append(velocity_penalty_term.item())
                actor_acceleration_penalty_terms.append(acceleration_penalty_term.item())
                actor_smooth_losses.append(smooth_loss.item())
                actor_losses.append(actor_control_loss.item())
                actor_total_losses.append(actor_loss.item())
                self._update_basket_replay_metric_ema(
                    "actor_loss",
                    float(actor_control_loss.detach().item()),
                    current_replay_stats,
                )
                self._update_basket_replay_metric_ema(
                    "actor_q_loss",
                    float(actor_q_loss.detach().item()),
                    current_replay_stats,
                )

                self.actor.optimizer.zero_grad(set_to_none=True)
                if (
                    auxiliary_this_update
                    and self.predictive_ssl_gradient_guard_enabled
                ):
                    actor_gradient_stats = apply_bounded_auxiliary_gradients(
                        actor_control_loss,
                        (
                            self.predictive_ssl_actor_loss_weight
                            * actor_auxiliary_loss
                        ),
                        actor_guard_parameters,
                        max_shared_ratio=(
                            float(
                                getattr(
                                    config,
                                    "FOUNDATION_ACTOR_SHARED_GRADIENT_MAX_RATIO",
                                    self.predictive_ssl_actor_shared_gradient_max_ratio,
                                )
                            )
                            if foundation_this_update
                            else self.predictive_ssl_actor_shared_gradient_max_ratio
                        ),
                        project_conflicts=(
                            self.predictive_ssl_project_conflicts
                        ),
                        scale_by_positive_cosine=(
                            self.predictive_ssl_scale_shared_by_positive_cosine
                        ),
                        alignment_cosine_ema=(
                            self._predictive_ssl_actor_cosine_ema
                        ),
                        alignment_ema_decay=(
                            self.predictive_ssl_gradient_cosine_ema_decay
                        ),
                        epsilon=self.predictive_ssl_gradient_epsilon,
                        max_grad_norm=self.max_grad_norm,
                        main_gradients=actor_main_gradients,
                    )
                    predictive_actor_gradient_stats.append(
                        actor_gradient_stats
                    )
                    self._predictive_ssl_actor_cosine_ema = float(
                        actor_gradient_stats["smoothed_cosine"]
                    )
                else:
                    actor_loss.backward()
                if not (
                    auxiliary_this_update
                    and self.predictive_ssl_gradient_guard_enabled
                ):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                self.actor.optimizer.step()
                if policy_step_diagnostic_due:
                    # Same states and same stochastic masks before/after the
                    # optimizer: measure actual policy movement separately
                    # from changing replay composition. Restore global RNG
                    # afterward so instrumentation cannot alter training.
                    with torch.random.fork_rng(devices=actor_step_cuda_devices), torch.no_grad():
                        torch.set_rng_state(actor_step_cpu_rng)
                        if actor_step_cuda_rng is not None:
                            torch.cuda.set_rng_state(actor_step_cuda_rng, actor_step_cuda_devices[0])
                        weights_after_step = normalize_raw_actions_to_portfolio_weights(
                            self.actor(actor_obs), action_mask
                        )
                        policy_step_norm = (weights_after_step - w_cur.detach()).norm(dim=1)
                        actor_policy_step_l2_means.append(float(policy_step_norm.mean().item()))
                        actor_policy_step_l2_maxima.append(float(policy_step_norm.max().item()))
                if actor_action_advantage is not None:
                    mixed_obs_float_g = (
                        obs_float_g
                        if mixed_replay_data is replay_data
                        else self._gate_observations_with_sigma(
                            replay_observations_as_float(
                                mixed_replay_data.observations
                            )
                        )
                    )
                    real_adv, relabeled_adv = self._cross_basket_actor_advantage_split(
                        actor_action_advantage,
                        actor_obs_float,
                        mixed_obs_float_g,
                        actor_uses_original_replay,
                    )
                    if real_adv is not None:
                        xbasket_actor_advantage_real.append(real_adv)
                    if relabeled_adv is not None:
                        xbasket_actor_advantage_relabeled.append(relabeled_adv)

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

            # Disabled in the coherent financial profile. Bypass observation
            # conversion and encoding as well as the independent optimizer.
            earnmore_stats = None
            if self._earnmore_reconstruction_update_due():
                earnmore_started_at = time.perf_counter()
                earnmore_stats = self._earnmore_masked_reconstruction_update(
                    replay_observations_as_float(
                        bellman_source_replay_data.observations
                    )
                )
            if earnmore_stats is not None:
                earnmore_update_seconds.append(
                    time.perf_counter() - earnmore_started_at
                )
                earnmore_losses.append(earnmore_stats["loss"])
                earnmore_actor_losses.append(earnmore_stats["actor_loss"])
                earnmore_critic_losses.append(earnmore_stats["critic_loss"])
                earnmore_masked_fractions.append(
                    earnmore_stats["masked_asset_fraction"]
                )
                earnmore_sampled_ratios.append(
                    earnmore_stats["sampled_mask_ratio"]
                )
                earnmore_coordinate_fractions.append(
                    earnmore_stats["masked_local_coordinate_fraction"]
                )
                earnmore_masked_counts.append(
                    earnmore_stats["masked_assets_per_sample"]
                )
                earnmore_visible_counts.append(
                    earnmore_stats["visible_assets_per_sample"]
                )
                earnmore_actor_gradient_norms.append(
                    earnmore_stats["actor_gradient_norm"]
                )
                earnmore_critic_gradient_norms.append(
                    earnmore_stats["critic_gradient_norm"]
                )
            if predictive_ssl_this_update:
                predictive_ssl_update_seconds.append(predictive_ssl_seconds)

        # Logging
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record(
            "train/xbasket_relabel_enabled",
            float(getattr(self, "cross_basket_relabel_enabled", False)),
        )
        self.logger.record(
            "train/xbasket_relabel_critic_mixed",
            float(
                getattr(self, "cross_basket_relabel_critic_rows", "mixed")
                == "mixed"
            ),
        )
        self.logger.record(
            "train/xbasket_relabel_actor_crl_original",
            float(
                getattr(self, "cross_basket_relabel_actor_crl_rows", "original")
                == "original"
            ),
        )
        if xbasket_actor_batch_rows:
            self.logger.record(
                "train/xbasket_relabel_actor_batch_rows",
                float(np.mean(xbasket_actor_batch_rows)),
            )
        if xbasket_actor_advantage_real:
            self.logger.record(
                "train/xbasket_relabel_actor_advantage_real",
                float(np.mean(xbasket_actor_advantage_real)),
            )
        if xbasket_actor_advantage_relabeled:
            self.logger.record(
                "train/xbasket_relabel_actor_advantage_relabeled",
                float(np.mean(xbasket_actor_advantage_relabeled)),
            )
        if cross_basket_relabel_stats:
            for stat_key in (
                "requested_rows",
                "applied_rows",
                "valid_source_fraction",
                "batch_fraction",
                "one_swap_fraction",
                "policy_action_fraction",
                "added_assets_mean",
                "reward_mean",
                "source_reward_mean",
                "reward_abs_gap_mean",
                "cash_weight_mean",
            ):
                values = [
                    entry[stat_key]
                    for entry in cross_basket_relabel_stats
                    if stat_key in entry
                ]
                if values:
                    self.logger.record(
                        f"train/xbasket_relabel_{stat_key}", float(np.mean(values))
                    )
        self.logger.record(
            "train/earnmore_masked_reconstruction_enabled",
            float(self.earnmore_masked_reconstruction_enabled),
        )
        self.logger.record(
            "train/earnmore_schema_version",
            float(self.earnmore_schema_version),
        )
        self.logger.record("train/earnmore_actor_reconstruction_enabled", 0.0)
        self.logger.record(
            "train/earnmore_updates", float(self.earnmore_updates)
        )
        self.logger.record(
            "train/earnmore_update_interval",
            float(self.earnmore_update_interval),
        )
        self.logger.record(
            "train/earnmore_realized_update_fraction",
            float(len(earnmore_losses)) / float(max(gradient_steps, 1)),
        )
        if earnmore_update_seconds:
            self.logger.record(
                "train/earnmore_update_seconds",
                float(np.mean(earnmore_update_seconds)),
            )
            self.logger.record(
                "train/earnmore_amortized_seconds_per_td_update",
                float(np.sum(earnmore_update_seconds))
                / float(max(gradient_steps, 1)),
            )
        earnmore_series = (
            ("earnmore_reconstruction_loss", earnmore_losses),
            ("earnmore_actor_reconstruction_loss", earnmore_actor_losses),
            ("earnmore_critic_reconstruction_loss", earnmore_critic_losses),
            ("earnmore_masked_asset_fraction", earnmore_masked_fractions),
            ("earnmore_sampled_mask_ratio", earnmore_sampled_ratios),
            (
                "earnmore_masked_local_coordinate_fraction",
                earnmore_coordinate_fractions,
            ),
            ("earnmore_masked_assets_per_sample", earnmore_masked_counts),
            ("earnmore_visible_assets_per_sample", earnmore_visible_counts),
            ("earnmore_actor_gradient_norm", earnmore_actor_gradient_norms),
            ("earnmore_critic_gradient_norm", earnmore_critic_gradient_norms),
        )
        earnmore_names = []
        earnmore_values = []
        for metric_name, metric_values in earnmore_series:
            if metric_values:
                earnmore_names.append(metric_name)
                earnmore_values.append(torch.stack(metric_values).mean())
        if earnmore_values:
            materialized = torch.stack(earnmore_values).detach().float().cpu().tolist()
            for metric_name, metric_value in zip(earnmore_names, materialized):
                self.logger.record(f"train/{metric_name}", float(metric_value))
        self.logger.record("train/qwm_enabled", float(self.qwm_enabled))
        self.logger.record("train/qwm_ready", float(self.qwm_ready))
        self.logger.record("train/qwm_world_updates", float(self.qwm_world_updates))
        self.logger.record("train/qwm_search_depth", float(self._qwm_current_depth()))
        self.logger.record("train/qwm_search_calls", float(self.qwm_search_calls))
        if self.qwm_enabled:
            self.logger.record(
                "train/qwm_replay_available_baskets",
                float(self._qwm_available_basket_count()),
            )
        qwm_device_metrics = []
        qwm_device_metric_names = []
        if self._qwm_root_telemetry is not None:
            qwm_device_metrics.extend(self._qwm_root_telemetry.unbind())
            qwm_device_metric_names.extend((
                "qwm_root_score_mean",
                "qwm_root_score_std",
                "qwm_center_selected_fraction",
                "qwm_selected_score_gain",
                "qwm_selected_weight_l1_distance",
                "qwm_tree_q_choice_disagreement",
                "qwm_model_contribution_std",
            ))
        if self._qwm_successor_telemetry is not None:
            qwm_device_metrics.extend(self._qwm_successor_telemetry.unbind())
            qwm_device_metric_names.extend((
                "qwm_successor_state_std", "qwm_successor_reward_std"
            ))
        qwm_training_series = (
            ("qwm_loss", qwm_losses),
            ("qwm_reconstruction_loss", qwm_reconstruction_losses),
            (
                "qwm_posterior_reconstruction_loss",
                qwm_posterior_reconstruction_losses,
            ),
            ("qwm_prior_reconstruction_loss", qwm_prior_reconstruction_losses),
            ("qwm_state_loss", qwm_state_losses),
            ("qwm_reward_loss", qwm_reward_losses),
            ("qwm_prior_state_loss", qwm_prior_state_losses),
            ("qwm_prior_reward_loss", qwm_prior_reward_losses),
            ("qwm_kl_loss", qwm_kl_losses),
            ("qwm_observation_loss", qwm_observation_losses),
            ("qwm_local_history_loss", qwm_local_history_losses),
            ("qwm_episode_history_loss", qwm_episode_history_losses),
            ("qwm_context_loss", qwm_context_losses),
            ("qwm_prior_std_mean", qwm_prior_stds),
            ("qwm_posterior_std_mean", qwm_posterior_stds),
            ("qwm_latent_mean_gap", qwm_latent_mean_gaps),
            ("qwm_gradient_norm", qwm_gradient_norms),
        )
        for metric_name, metric_values in qwm_training_series:
            if metric_values:
                qwm_device_metrics.append(torch.stack(metric_values).mean())
                qwm_device_metric_names.append(metric_name)
        if self._qwm_loss_ema_tensor is not None:
            qwm_device_metrics.append(self._qwm_loss_ema_tensor)
            qwm_device_metric_names.append("qwm_loss_ema")
        if qwm_device_metrics:
            qwm_materialized_values = torch.stack(
                [value.detach().float() for value in qwm_device_metrics]
            ).cpu().tolist()
            qwm_materialized = {
                name: float(value)
                for name, value in zip(
                    qwm_device_metric_names, qwm_materialized_values
                )
            }
            self.qwm_last_root_score_mean = qwm_materialized.get(
                "qwm_root_score_mean", self.qwm_last_root_score_mean
            )
            self.qwm_last_root_score_std = qwm_materialized.get(
                "qwm_root_score_std", self.qwm_last_root_score_std
            )
            self.qwm_last_successor_state_std = qwm_materialized.get(
                "qwm_successor_state_std", self.qwm_last_successor_state_std
            )
            self.qwm_last_successor_reward_std = qwm_materialized.get(
                "qwm_successor_reward_std", self.qwm_last_successor_reward_std
            )
            self.qwm_loss_ema = qwm_materialized.get(
                "qwm_loss_ema", self.qwm_loss_ema
            )
            for metric_name, metric_value in qwm_materialized.items():
                self.logger.record(f"train/{metric_name}", metric_value)
        self.logger.record(
            "train/qwm_root_score_mean", self.qwm_last_root_score_mean
        )
        self.logger.record(
            "train/qwm_root_score_std", self.qwm_last_root_score_std
        )
        self.logger.record(
            "train/qwm_successor_state_std",
            self.qwm_last_successor_state_std,
        )
        self.logger.record(
            "train/qwm_successor_reward_std",
            self.qwm_last_successor_reward_std,
        )
        if self.qwm_loss_ema is not None:
            self.logger.record("train/qwm_loss_ema", float(self.qwm_loss_ema))
        for metric_name, metric_value in (
            ("qwm_pretrain_initial_loss", self.qwm_pretrain_initial_loss),
            ("qwm_pretrain_final_loss", self.qwm_pretrain_final_loss),
            ("qwm_pretrain_best_loss", self.qwm_pretrain_best_loss),
            ("qwm_pretrain_relative_loss", self.qwm_pretrain_relative_loss),
        ):
            if metric_value is not None:
                self.logger.record(f"train/{metric_name}", float(metric_value))
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/actor_total_loss", np.mean(actor_total_losses))
            self.logger.record("train/actor_q_loss", np.mean(actor_q_losses))
            self.logger.record(
                "train/actor_raw_q_loss",
                np.mean(actor_raw_q_losses),
            )
            if actor_action_advantages:
                self.logger.record(
                    "train/actor_action_advantage",
                    np.mean(actor_action_advantages),
                )
            if actor_popart_advantages:
                self.logger.record(
                    "train/actor_popart_advantage",
                    np.mean(actor_popart_advantages),
                )
                self.logger.record(
                    "train/actor_popart_scale_mean",
                    np.mean(actor_popart_scale_means),
                )
                self.logger.record(
                    "train/actor_popart_scale_max",
                    np.max(actor_popart_scale_maxes),
                )
            self.logger.record(
                "train/actor_action_advantage_diagnostic_enabled",
                float(self.actor_action_advantage_diagnostic_enabled),
            )
            self.logger.record("train/actor_velocity_penalty", np.mean(actor_velocity_penalty_terms))
            self.logger.record("train/actor_acceleration_penalty", np.mean(actor_acceleration_penalty_terms))
            self.logger.record("train/actor_smooth_loss", np.mean(actor_smooth_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record('train/gamma', float(self.gamma))
        self.logger.record('train/earnmore_reconstruction_enabled',
                           float(self.earnmore_masked_reconstruction_enabled))
        self.logger.record('train/critic_encoder_extra_target_refresh_enabled', 0.0)
        self.logger.record("train/critic_total_loss", np.mean(critic_total_losses))
        if len(critic_losses_unweighted) > 0:
            self.logger.record("train/critic_loss_unweighted", np.mean(critic_losses_unweighted))
        if critic_weighted_td_losses:
            self.logger.record('train/critic_td_weighted_loss', np.mean(critic_weighted_td_losses))
        if actor_advantage_abs_maxima:
            self.logger.record('train/actor_action_advantage_abs_max', np.max(actor_advantage_abs_maxima))
        if actor_policy_step_l2_means:
            self.logger.record('train/actor_policy_step_weight_l2_mean', np.mean(actor_policy_step_l2_means))
            self.logger.record('train/actor_policy_step_weight_l2_max', np.max(actor_policy_step_l2_maxima))
        if bellman_extrema:
            for name, maximum in zip(
                ("critic_q_abs_max", "bellman_target_abs_max",
                 "bellman_error_abs_max", "replay_reward_abs_max"),
                np.max(np.asarray(bellman_extrema), axis=0),
            ):
                self.logger.record(f"train/{name}", float(maximum))
        self.logger.record(
            "train/task_popart_enabled",
            float(self.task_popart is not None),
        )
        if self.task_popart is not None and popart_target_sigmas:
            global_popart_mean, global_popart_sigma = (
                self.task_popart.global_statistics(
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self.logger.record(
                "train/critic_popart_loss",
                float(np.mean(critic_popart_losses)),
            )
            self.logger.record(
                "train/popart_target_mean",
                float(np.mean(popart_target_means)),
            )
            self.logger.record(
                "train/popart_target_sigma_mean",
                float(np.mean(popart_target_sigmas)),
            )
            self.logger.record(
                "train/popart_target_sigma_min",
                float(np.min(popart_target_sigma_mins)),
            )
            self.logger.record(
                "train/popart_target_sigma_max",
                float(np.max(popart_target_sigma_maxes)),
            )
            self.logger.record(
                "train/popart_task_reliability_mean",
                float(np.mean(popart_task_reliabilities)),
            )
            self.logger.record(
                "train/popart_distinct_tasks_per_batch",
                float(np.mean(popart_distinct_tasks)),
            )
            self.logger.record(
                "train/popart_global_mean",
                float(global_popart_mean.item()),
            )
            self.logger.record(
                "train/popart_global_sigma",
                float(global_popart_sigma.item()),
            )
            self.logger.record(
                "train/popart_statistics_updates",
                float(self.task_popart.statistics_updates.item()),
            )
        for prefix, measurements in (
            ("popart_critic_task_gradient", popart_critic_gradient_directions),
            ("popart_actor_task_gradient", popart_actor_gradient_directions),
        ):
            if not measurements:
                continue
            for metric_name in measurements[0]:
                self.logger.record(
                    f"train/{prefix}_{metric_name}",
                    float(
                        np.mean(
                            [measurement[metric_name] for measurement in measurements]
                        )
                    ),
                )
        if basket_replay_sample_stats:
            numeric_replay_fields = (
                "task_fingerprint",
                "stored_rows",
                "sampled_rows",
                "effective_batch_size",
                "unique_rows",
                "completed_episodes",
                "full_batch_ready",
                "partial_unique_batch",
                "unique_fraction",
                "requested_fraction",
                "distance",
                "is_anchor",
                "anchor_fraction",
                "distinct_tasks",
                "minimum_task_sampled_rows",
                "minimum_task_stored_rows",
                "minimum_task_completed_episodes",
                "dominant_task_fraction",
                "maximum_task_fraction",
            )
            for field in numeric_replay_fields:
                values = [
                    float(stats[field])
                    for stats in basket_replay_sample_stats
                    if field in stats
                ]
                if values:
                    self.logger.record(
                        f"train/bellman_replay_{field}",
                        float(np.min(values) if field.startswith('minimum_task_')
                              else np.mean(values)),
                    )
            replay_roles = sorted(
                {
                    str(role)
                    for stats in basket_replay_sample_stats
                    for role in dict(stats.get("role_counts", {}))
                }
            )
            for role in replay_roles:
                fractions = []
                for stats in basket_replay_sample_stats:
                    sampled_rows = max(int(stats.get("sampled_rows", 0)), 1)
                    role_count = int(
                        dict(stats.get("role_counts", {})).get(role, 0)
                    )
                    fractions.append(role_count / sampled_rows)
                safe_role = "".join(
                    character if character.isalnum() else "_"
                    for character in role.lower()
                ).strip("_")
                self.logger.record(
                    f"train/br_role_{safe_role}_share",
                    float(np.mean(fractions)),
                )
        for metric, value in sorted(self._basket_replay_metric_ema.items()):
            self.logger.record(
                f"train/bellman_{metric}_ema",
                float(value),
            )
        self.logger.record(
            "train/predictive_replay_uniform_mix",
            float(self.predictive_ssl_replay_uniform_mix),
        )
        self.logger.record(
            "train/bellman_teacher_anchor_share",
            float(self._basket_replay_teacher_anchor_share),
        )
        self.logger.record(
            "train/bellman_role_preserved_anchor_share",
            float(self._basket_replay_financial_anchor_share),
        )
        self.logger.record(
            "train/predictive_ssl_enabled",
            float(self.predictive_ssl_enabled),
        )
        self.logger.record(
            "train/predictive_ssl_online_enabled",
            float(self.predictive_ssl_online_enabled),
        )
        self.logger.record(
            "train/predictive_ssl_update_interval",
            float(self.predictive_ssl_update_interval),
        )
        self.logger.record(
            "train/predictive_ssl_realized_update_fraction",
            float(len(predictive_critic_losses)) / float(max(gradient_steps, 1)),
        )
        if predictive_ssl_update_seconds:
            predictive_ssl_mean_seconds = float(
                np.mean(predictive_ssl_update_seconds)
            )
            if self._predictive_ssl_update_seconds_ema is None:
                self._predictive_ssl_update_seconds_ema = (
                    predictive_ssl_mean_seconds
                )
            else:
                self._predictive_ssl_update_seconds_ema = (
                    0.95 * self._predictive_ssl_update_seconds_ema
                    + 0.05 * predictive_ssl_mean_seconds
                )
            self.logger.record(
                "train/predictive_ssl_update_seconds",
                predictive_ssl_mean_seconds,
            )
            self.logger.record(
                "train/predictive_ssl_update_seconds_ema",
                float(self._predictive_ssl_update_seconds_ema),
            )
        self.logger.record(
            "train/foundation_schema_version",
            float(self.foundation_schema_version),
        )
        self.logger.record(
            "train/value_experience_descriptor_version",
            float(self.value_experience_replay_descriptor_version),
        )
        if self.value_experience_replay_scorer is not None:
            self.logger.record(
                "train/value_experience_action_width",
                float(self.value_experience_replay_action_width),
            )
        self.logger.record(
            "train/foundation_warmup_complete",
            float(self.foundation_warmup_complete),
        )
        self.logger.record(
            "train/foundation_warmup_updates",
            float(self.foundation_warmup_updates),
        )
        self.logger.record(
            "train/foundation_online_update_interval",
            float(self.foundation_online_update_interval),
        )
        self.logger.record(
            "train/foundation_online_rehearsal_enabled",
            float(self.foundation_online_rehearsal_enabled),
        )
        self.logger.record(
            "train/foundation_sequence_replay_enabled",
            float(self.foundation_sequence_replay_enabled),
        )
        self.logger.record(
            "train/foundation_online_batch_size",
            float(self.foundation_online_batch_size),
        )
        if foundation_critic_losses:
            self.logger.record(
                "train/foundation_critic_loss",
                float(np.mean(foundation_critic_losses)),
            )
            self.logger.record(
                "train/foundation_action_outcome_loss",
                float(np.mean(foundation_action_losses)),
            )
            self.logger.record(
                "train/foundation_q_rank_loss",
                float(np.mean(foundation_q_rank_losses)),
            )
            self.logger.record(
                "train/foundation_representation_loss",
                float(np.mean(foundation_representation_losses)),
            )
            self.logger.record(
                "train/foundation_candidate_evaluation_fraction",
                float(np.mean(foundation_candidate_evaluation_fractions)),
            )
        if foundation_actor_losses:
            self.logger.record(
                "train/foundation_actor_loss",
                float(np.mean(foundation_actor_losses)),
            )
            self.logger.record(
                "train/foundation_actor_regret",
                float(np.mean(foundation_actor_regrets)),
            )
        for metric, value in sorted(self._last_value_experience_stats.items()):
            self.logger.record(
                f"replay/ve_{metric}",
                float(value),
            )
        if predictive_critic_losses:
            if predictive_replay_unique_keys:
                self.logger.record(
                    "train/predictive_replay_unique_keys",
                    np.mean(predictive_replay_unique_keys),
                )
            if predictive_replay_anchor_fractions:
                self.logger.record(
                    "train/predictive_replay_anchor_fraction",
                    np.mean(predictive_replay_anchor_fractions),
                )
            if predictive_replay_mean_distances:
                self.logger.record(
                    "train/predictive_replay_mean_distance",
                    np.mean(predictive_replay_mean_distances),
                )
            self.logger.record(
                "train/predictive_critic_loss",
                np.mean(predictive_critic_losses),
            )
            self.logger.record(
                "train/predictive_critic_weighted_loss",
                self.predictive_ssl_critic_loss_weight
                * np.mean(predictive_critic_losses),
            )
            self.logger.record(
                "train/predictive_feature_loss",
                np.mean(predictive_feature_losses),
            )
            self.logger.record(
                "train/predictive_market_latent_loss",
                np.mean(predictive_market_losses),
            )
            self.logger.record(
                "train/predictive_relation_latent_loss",
                np.mean(predictive_relation_losses),
            )
            self.logger.record(
                "train/predictive_fused_latent_loss",
                np.mean(predictive_fused_losses),
            )
            self.logger.record(
                "train/predictive_downstream_latent_loss",
                np.mean(predictive_downstream_losses),
            )
            self.logger.record(
                "train/ve_reward_loss", np.mean(predictive_reward_losses)
            )
            self.logger.record(
                "train/ve_continuation_loss",
                np.mean(predictive_continuation_value_losses),
            )
            self.logger.record(
                "train/ve_bellman_loss",
                np.mean(predictive_bellman_equivalence_losses),
            )
            self.logger.record(
                "train/ve_expectile_loss",
                np.mean(predictive_expectile_value_losses),
            )
            self.logger.record(
                "train/ve_state_value",
                np.mean(predictive_state_value_means),
            )
            self.logger.record(
                "train/ve_advantage", np.mean(predictive_advantage_means)
            )
            self.logger.record(
                "train/ve_advantage_positive",
                np.mean(predictive_advantage_positive_fractions),
            )
            self.logger.record(
                "train/ve_weight_min",
                np.mean(predictive_advantage_weight_mins),
            )
            self.logger.record(
                "train/ve_weight_max",
                np.mean(predictive_advantage_weight_maxes),
            )
            self.logger.record(
                "train/predictive_routing_entropy",
                np.mean(predictive_routing_entropies),
            )
            self.logger.record(
                "train/predictive_expert_min_usage",
                np.mean(predictive_expert_min_usages),
            )
            self.logger.record(
                "train/predictive_expert_max_usage",
                np.mean(predictive_expert_max_usages),
            )
            self.logger.record(
                "train/predictive_valid_transition_fraction",
                np.mean(predictive_valid_fractions),
            )
            self.logger.record(
                "train/predictive_valid_horizon_fraction",
                np.mean(predictive_valid_horizon_fractions),
            )
            self.logger.record(
                "train/predictive_horizon_loss_mean",
                np.mean(predictive_horizon_loss_means),
            )
            self.logger.record(
                "train/predictive_horizon",
                float(self.predictive_ssl_horizon),
            )
            for horizon_index, values in sorted(
                predictive_per_horizon_losses.items()
            ):
                self.logger.record(
                    f"train/predictive_h{horizon_index}_loss",
                    float(np.mean(values)),
                )
            for horizon_index, values in sorted(
                predictive_per_horizon_value_losses.items()
            ):
                self.logger.record(
                    f"train/ve_h{horizon_index}_loss",
                    float(np.mean(values)),
                )
            self.logger.record(
                "train/predictive_control_gate",
                np.mean(predictive_control_gates),
            )
            self.logger.record(
                "train/predictive_control_projection_norm",
                np.mean(predictive_control_projection_norms),
            )
        if predictive_actor_losses:
            self.logger.record(
                "train/predictive_actor_loss",
                np.mean(predictive_actor_losses),
            )
            self.logger.record(
                "train/predictive_actor_weighted_loss",
                self.predictive_ssl_actor_loss_weight
                * np.mean(predictive_actor_losses),
            )
        for owner, records in (
            ("critic", predictive_critic_gradient_stats),
            ("actor", predictive_actor_gradient_stats),
        ):
            if not records:
                continue
            for metric in (
                "main_norm",
                "auxiliary_shared_raw_norm",
                "auxiliary_shared_projected_norm",
                "auxiliary_private_norm",
                "auxiliary_scale",
                "effective_shared_ratio",
                "cosine",
                "smoothed_cosine",
                "alignment_authority",
                "conflict",
                "control_clip_scale",
                "private_clip_scale",
            ):
                self.logger.record(
                    f"train/predictive_{owner}_gradient_{metric}",
                    float(np.mean([record[metric] for record in records])),
                )
        self.logger.record(
            "train/predictive_gradient_guard_enabled",
            float(self.predictive_ssl_gradient_guard_enabled),
        )
        self.logger.record(
            "train/predictive_critic_shared_gradient_ratio_cap",
            float(self.predictive_ssl_critic_shared_gradient_max_ratio),
        )
        self.logger.record(
            "train/predictive_actor_shared_gradient_ratio_cap",
            float(self.predictive_ssl_actor_shared_gradient_max_ratio),
        )
        self.logger.record(
            "train/predictive_ssl_horizon",
            float(self.predictive_ssl_horizon),
        )
        self.logger.record(
            "train/predictive_ssl_sequence_batch_size",
            float(self.predictive_ssl_sequence_batch_size),
        )
        if len(sigma_intensity_means) > 0:
            self.logger.record("train/sigma_intensity_mean", np.mean(sigma_intensity_means))
            self.logger.record("train/sigma_intensity_std", np.mean(sigma_intensity_stds))
            self.logger.record("train/sigma_weight_mean", np.mean(sigma_weight_means))
            self.logger.record("train/sigma_weight_max", np.mean(sigma_weight_maxes))
        if len(proexp_alpha_means) > 0:
            self.logger.record("train/proexp_alpha_mean", np.mean(proexp_alpha_means))
            self.logger.record("train/proexp_disagreement_mean", np.mean(proexp_disagreement_means))
            self.logger.record("train/proexp_candidate_entropy", np.mean(proexp_candidate_entropies))
            self.logger.record("train/proexp_soft_value_mean", np.mean(proexp_soft_value_means))
            self.logger.record("train/proexp_center_q_mean", np.mean(proexp_center_q_means))
            self.logger.record("train/proexp_soft_gap_mean", np.mean(proexp_soft_gap_means))
        if len(proexp_budget_alpha_means) > 0:
            self.logger.record("train/proexp_budget_alpha_mean", np.mean(proexp_budget_alpha_means))
        if len(proexp_composition_alpha_means) > 0:
            self.logger.record("train/proexp_composition_alpha_mean", np.mean(proexp_composition_alpha_means))
        if len(proexp_budget_disagreement_means) > 0:
            self.logger.record("train/proexp_budget_disagreement_mean", np.mean(proexp_budget_disagreement_means))
        if len(proexp_composition_disagreement_means) > 0:
            self.logger.record("train/proexp_composition_disagreement_mean", np.mean(proexp_composition_disagreement_means))
        if len(proexp_budget_entropies) > 0:
            self.logger.record("train/proexp_budget_entropy", np.mean(proexp_budget_entropies))
        if len(proexp_composition_entropies) > 0:
            self.logger.record("train/proexp_composition_entropy", np.mean(proexp_composition_entropies))
        self.logger.record(
            "train/centered_advantage_enabled",
            float(
                getattr(
                    self.critic.q_networks[0],
                    "centered_advantage_enabled",
                    False,
                )
            ),
        )
        self.logger.record(
            "train/critic_direct_q_bellman_enabled",
            float(getattr(self.critic.q_networks[0], "bellman_composition", "separate_v")
                  == "direct_q"),
        )
        self.logger.record(
            "train/critic_separate_v_bellman_enabled",
            float(getattr(self.critic.q_networks[0], "centered_bellman_enabled", False)),
        )
        if len(component_outcome_losses) > 0:
            self.logger.record(
                "train/component_outcome_unit_return_targets",
                float(
                    getattr(
                        self.critic.q_networks[0],
                        "component_outcome_unit_return_targets",
                        False,
                    )
                ),
            )
            self.logger.record(
                "train/component_outcome_loss",
                np.mean(component_outcome_losses),
            )
            self.logger.record(
                "train/component_outcome_weighted_loss",
                np.mean(component_outcome_weighted_losses),
            )
            self.logger.record(
                "train/component_outcome_mae",
                np.mean(component_outcome_maes),
            )
            self.logger.record(
                "train/component_outcome_valid_fraction",
                np.mean(component_outcome_valid_fractions),
            )
            self.logger.record(
                "train/component_outcome_gradient_scale",
                np.mean(component_outcome_gradient_scales),
            )
            self.logger.record(
                "train/asset_teacher_real_rows",
                np.mean(asset_teacher_real_rows),
            )
            if basket_relative_losses:
                self.logger.record(
                    "train/basket_relative_forecast_loss",
                    np.mean(basket_relative_losses),
                )
            if local_financial_losses:
                self.logger.record(
                    "train/local_financial_loss", np.mean(local_financial_losses)
                )
                self.logger.record(
                    "train/local_financial_mae", np.mean(local_financial_maes)
                )
                self.logger.record(
                    "train/local_financial_total_aux_loss",
                    np.mean(local_financial_total_aux_losses),
                )
            if local_financial_context_abs:
                self.logger.record(
                    "train/local_financial_context_abs",
                    np.mean(local_financial_context_abs),
                )
            if local_financial_base_abs:
                self.logger.record(
                    "train/local_financial_q_base_abs",
                    np.mean(local_financial_base_abs),
                )
                self.logger.record(
                    "train/local_financial_q_residual_abs",
                    np.mean(local_financial_residual_abs),
                )
            if local_financial_actor_replay_advantages:
                self.logger.record(
                    "train/local_financial_actor_replay_advantage_abs",
                    np.mean(local_financial_actor_replay_advantages),
                )
            if component_outcome_correlations:
                self.logger.record(
                    "train/component_outcome_cross_sectional_correlation",
                    np.mean(component_outcome_correlations),
                )
            if component_outcome_target_abs_means:
                self.logger.record(
                    "train/component_outcome_target_abs_mean",
                    np.mean(component_outcome_target_abs_means),
                )
            if component_outcome_target_clip_fractions:
                self.logger.record(
                    "train/component_outcome_target_clip_fraction",
                    np.mean(component_outcome_target_clip_fractions),
                )
            if component_outcome_gradient_measurements:
                measured = np.asarray(component_outcome_gradient_measurements)
                self.logger.record(
                    "train/component_outcome_td_encoder_grad_norm",
                    float(measured[:, 0].mean()),
                )
                self.logger.record(
                    "train/component_outcome_aux_encoder_grad_norm",
                    float(measured[:, 1].mean()),
                )
                self.logger.record(
                    "train/component_aux_td_grad_ratio",
                    float(measured[:, 2].mean()),
                )
                self.logger.record(
                    "train/component_aux_td_grad_ratio_max",
                    float(measured[:, 2].max()),
                )
                self.logger.record(
                    "train/component_gradient_measurement_fraction",
                    len(measured) / max(len(component_outcome_gradient_scales), 1),
                )
        bridge_networks = [
            q_network
            for q_network in getattr(self.critic, "q_networks", ())
            if bool(
                getattr(
                    q_network,
                    "component_forecast_bridge_enabled",
                    False,
                )
            )
        ]
        if bridge_networks:
            self.logger.record(
                "train/component_forecast_bridge_local_only",
                float(
                    all(
                        getattr(
                            q_network,
                            "component_forecast_bridge_local_only",
                            False,
                        )
                        for q_network in bridge_networks
                    )
                ),
            )
            self.logger.record(
                "train/component_forecast_bridge_weight_norm",
                float(
                    np.mean(
                        [
                            q_network.component_forecast_bridge_projection
                            .weight.detach().norm().item()
                            for q_network in bridge_networks
                        ]
                    )
                ),
            )
            self.logger.record(
                "train/component_forecast_bridge_effective_gate",
                float(
                    np.mean(
                        [
                            q_network.component_forecast_bridge_max_abs
                            * torch.sigmoid(
                                q_network.component_forecast_bridge_gate_logit
                                .detach()
                            ).item()
                            for q_network in bridge_networks
                        ]
                    )
                ),
            )
        
        # CRL-specific logging
        if self.use_crl:
            if len(velocity_costs_log) > 0:
                self.logger.record("train/avg_velocity_cost", np.mean(velocity_costs_log))
            if len(acceleration_costs_log) > 0:
                self.logger.record("train/avg_acceleration_cost", np.mean(acceleration_costs_log))

        if basket_diag_accum:
            for key, values in basket_diag_accum.items():
                if values:
                    self.logger.record(f"train/{key}", float(np.mean(values)))
        td_update_seconds = (
            float(time.perf_counter() - train_started_at)
            / float(max(1, gradient_steps))
        )
        if self._td_update_seconds_ema is None:
            self._td_update_seconds_ema = td_update_seconds
        else:
            self._td_update_seconds_ema = (
                0.95 * self._td_update_seconds_ema
                + 0.05 * td_update_seconds
            )
        self.logger.record("train/td_update_seconds", td_update_seconds)
        self.logger.record(
            "train/td_update_seconds_ema",
            float(self._td_update_seconds_ema),
        )
        if cuda_memory_telemetry:
            bytes_per_gib = float(1024 ** 3)
            self.logger.record(
                "train/cuda_memory_allocated_gib",
                float(torch.cuda.memory_allocated(self.device)) / bytes_per_gib,
            )
            self.logger.record(
                "train/cuda_memory_reserved_gib",
                float(torch.cuda.memory_reserved(self.device)) / bytes_per_gib,
            )
            self.logger.record(
                "train/cuda_peak_memory_allocated_gib",
                float(torch.cuda.max_memory_allocated(self.device))
                / bytes_per_gib,
            )
            self.logger.record(
                "train/cuda_peak_memory_reserved_gib",
                float(torch.cuda.max_memory_reserved(self.device))
                / bytes_per_gib,
            )

    def get_crl_stats(self) -> Dict[str, float]:
        """Get CRL-specific statistics for monitoring."""
        if not self.use_crl:
            return {"crl_enabled": False}
        
        stats = {
            "crl_enabled": True,
            "crl_schema": "dual_velocity_acceleration",
            "current_lambda_velocity": self.lambda_param_velocity.item(),
            "velocity_constraint_threshold": self.velocity_constraint_threshold,
            "current_lambda_acceleration": self.lambda_param_acceleration.item(),
            "acceleration_constraint_threshold": self.acceleration_constraint_threshold,
            "is_attention_policy": self.is_attention_policy,
            "auto_adjust_enabled": self.auto_adjust_for_attention,
            "attention_exploration_bonus": getattr(self, 'attention_exploration_bonus', 0.0),
            "velocity_lambda_lr": self.velocity_lambda_lr,
            "acceleration_lambda_lr": self.acceleration_lambda_lr,
            "current_lambda": self.lambda_param_velocity.item(),
            "current_lambda_total": self.lambda_param_velocity.item() + self.lambda_param_acceleration.item(),
            "constraint_threshold": self.velocity_constraint_threshold + self.acceleration_constraint_threshold,

        }
        
        velocity_recent_cost = None
        acceleration_recent_cost = None
        if len(self.velocity_constraint_history) > 0:
            recent_costs = self.velocity_constraint_history[-10:]
            velocity_recent_cost = _finite_float_or_none(np.mean(recent_costs))
            if velocity_recent_cost is not None:
                stats.update({
                    "avg_recent_velocity_cost": velocity_recent_cost,
                    "avg_recent_velocity_violation": velocity_recent_cost - self.velocity_constraint_threshold,
                })

        if len(self.acceleration_constraint_history) > 0:
            recent_costs = self.acceleration_constraint_history[-10:]
            acceleration_recent_cost = _finite_float_or_none(np.mean(recent_costs))
            if acceleration_recent_cost is not None:
                stats.update({
                    "avg_recent_acceleration_cost": acceleration_recent_cost,
                    "avg_recent_acceleration_violation": acceleration_recent_cost - self.acceleration_constraint_threshold,
                })

        if velocity_recent_cost is not None or acceleration_recent_cost is not None:
            total_cost = sum(
                cost for cost in (velocity_recent_cost, acceleration_recent_cost) if cost is not None
            )
            active_threshold = 0.0
            if velocity_recent_cost is not None:
                active_threshold += self.velocity_constraint_threshold
            if acceleration_recent_cost is not None:
                active_threshold += self.acceleration_constraint_threshold
            stats["avg_recent_constraint_cost"] = total_cost
            stats["avg_recent_violation"] = total_cost - active_threshold
        
        return stats 
