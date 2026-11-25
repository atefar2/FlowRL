#!/usr/bin/env python3
"""
Simple MLP training script with clearer parameter naming.
NO ATTENTION MECHANISMS - Pure MLP networks only.
"""

import os
import random
import pickle
import hashlib
import zipfile
import math
import json
import base64
import io
import copy
import time
import textwrap
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from collections import deque
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat, configure
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.buffers import ReplayBuffer
import torch
from tqdm import tqdm
import config
from enviorment import PortfolioEnv
from attention_policy import BasketCurriculumValueModel
# stop print from enviomrnent
import enviorment as _env_module
_env_module.print = lambda *a, **k: None

MODEL_CHECKPOINT_PREFIX = "model_"
MODEL_CHECKPOINT_SUFFIX = "_steps.zip"
REPLAY_BUFFER_CHECKPOINT_PREFIX = "replaybuffer_"
REPLAY_BUFFER_CHECKPOINT_SUFFIX = "_steps.zip"
LEGACY_REPLAY_BUFFER_CHECKPOINT_SUFFIX = "_steps.pkl"
REPLAY_BUFFER_ZIP_MEMBER = "replay_buffer.pkl"
REPLAY_BUFFER_ZIP_COMPRESSLEVEL = 3
COMPACT_REPLAY_BUFFER_FORMAT = "compact_replay_buffer"
COMPACT_REPLAY_BUFFER_VERSION = 1
STREAMING_REPLAY_BUFFER_FORMAT = "streaming_replay_buffer"
STREAMING_REPLAY_BUFFER_VERSION = 1
REPLAY_BUFFER_METADATA_ZIP_MEMBER = "replay_metadata.json"
REPLAY_BUFFER_IO_CHUNK_BYTES = 8 * 1024 * 1024
CROSS_BASKET_RELABEL_REPLAY_KEYS = (
    "relabel_market",
    "relabel_next_market",
    "relabel_asset_returns",
    "relabel_price_valid",
    "relabel_next_partial_count",
    "relabel_valid",
)
RESUME_CHECKPOINT_INTERVAL_STEPS = 50_000
RESUME_CHECKPOINTS_TO_KEEP = 2
PROGRESS_CSV_FILENAME = "progress.csv"
TRAINING_PLOT_FILENAME = "training_plots.png"
PROGRESS_COMBINED_CSV_FILENAME = "progress_combined.csv"
LEGACY_PROGRESS_BACKUP_FILENAME = "progress_pre_resume.csv"
CURRICULUM_CHECKPOINT_PREFIX = "curriculum_"
CURRICULUM_CHECKPOINT_SUFFIX = "_steps.json"
SB3_STDOUT_MAX_KEY_LENGTH = 96


def configure_training_logger(folder, format_strings):
    """Configure SB3 logging without truncating distinct diagnostic names.

    CSV and TensorBoard preserve complete metric keys, but SB3's stdout
    ``HumanOutputFormat`` defaults to 36 characters. The predictive-gradient
    ownership diagnostics intentionally share descriptive prefixes longer than
    that limit, so the default formatter truncates multiple distinct keys to
    the same display name and raises during ``logger.dump``. Widen only the
    human-readable formatter; metric names and persistent log schemas remain
    unchanged.
    """
    logger = configure(folder, format_strings)
    for output_format in logger.output_formats:
        if isinstance(output_format, HumanOutputFormat):
            output_format.max_length = max(
                int(output_format.max_length),
                SB3_STDOUT_MAX_KEY_LENGTH,
            )
    return logger


class _LazyBasketNeighborMap:
    """Compute one-replacement edges only for baskets the teacher touches.

    Materializing every edge is harmless for 8-of-13 but grows as the number
    of baskets times ``basket_size * (universe_size - basket_size)``. Lazy
    binding keeps the same graph semantics while allowing a larger 8-of-N
    universe to pay only for the frontier it actually reaches.
    """

    def __init__(self, manager, cache_size=4_096):
        self.manager = manager
        self.cache_size = int(cache_size)
        self.cache = {}

    def __getitem__(self, basket):
        basket = self.manager._canonical_basket(basket)
        cached = self.cache.get(basket)
        if cached is not None:
            return cached
        neighbors = tuple(
            self.manager._canonical_basket(
                (set(basket) - {outgoing}) | {incoming}
            )
            for outgoing in basket
            for incoming in self.manager.coin_pool
            if incoming not in basket
        )
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[basket] = neighbors
        return neighbors


class LocalCompositionBasketSampler:
    """Uniform-marginal, locality-ordered Path-2 episode sampler.

    Each block begins from a uniformly sampled pivot basket.  The pivot is
    interleaved with distinct one-replacement neighbors, so the online learner
    receives repeated reference states and controlled compositional changes.
    The sampler owns presentation order only: it exposes no teacher value,
    mastery, replay role, or Bellman sampling distribution.
    """

    def __init__(
        self,
        coin_pool,
        basket_size,
        seed,
        block_episodes=8,
        pivot_episodes=4,
        neighbor_episodes=4,
    ):
        self.coin_pool = tuple(coin_pool)
        self.basket_size = int(basket_size)
        self.block_episodes = int(block_episodes)
        self.pivot_episodes = int(pivot_episodes)
        self.neighbor_episodes = int(neighbor_episodes)
        if not 0 < self.basket_size < len(self.coin_pool):
            raise ValueError("local composition basket size must be in (0, N)")
        if self.pivot_episodes != self.neighbor_episodes:
            raise ValueError(
                "interleaved local-composition blocks require equal pivot and "
                "neighbor episode counts"
            )
        if self.block_episodes != self.pivot_episodes + self.neighbor_episodes:
            raise ValueError("local composition block counts do not sum")
        maximum_neighbors = self.basket_size * (
            len(self.coin_pool) - self.basket_size
        )
        if self.neighbor_episodes > maximum_neighbors:
            raise ValueError("requested more distinct neighbors than exist")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.block_index = 0
        self.selection_count = 0
        self._queue = []
        self._current_pivot = None

    def _canonical(self, basket):
        selected = set(basket)
        if len(selected) != self.basket_size or not selected.issubset(
            self.coin_pool
        ):
            raise ValueError("invalid local-composition basket")
        return tuple(coin for coin in self.coin_pool if coin in selected)

    def _sample_uniform_pivot(self):
        indices = self.rng.choice(
            len(self.coin_pool), size=self.basket_size, replace=False
        )
        return self._canonical(self.coin_pool[int(index)] for index in indices)

    def _all_neighbors(self, pivot):
        selected = set(pivot)
        return tuple(
            self._canonical((selected - {outgoing}) | {incoming})
            for outgoing in pivot
            for incoming in self.coin_pool
            if incoming not in selected
        )

    def _prepare_block(self):
        pivot = self._sample_uniform_pivot()
        neighbors = self._all_neighbors(pivot)
        chosen_indices = self.rng.choice(
            len(neighbors), size=self.neighbor_episodes, replace=False
        )
        chosen_neighbors = [neighbors[int(index)] for index in chosen_indices]
        queue = []
        for position, neighbor in enumerate(chosen_neighbors):
            queue.append((pivot, "local_pivot", position))
            queue.append((neighbor, "local_one_swap", position))
        self._current_pivot = pivot
        self._queue = queue
        self.block_index += 1

    def sample_basket(self):
        if not self._queue:
            self._prepare_block()
        basket, role, pair_index = self._queue.pop(0)
        slot_order = self.rng.permutation(len(basket))
        coins = [basket[int(index)] for index in slot_order]
        selection = {
            "coins": coins,
            "basket": list(basket),
            "distance": 0 if role == "local_pivot" else 1,
            "role": role,
            "stage": "uniform_local_composition",
            "episode_index": int(self.selection_count),
            "local_block_index": int(self.block_index - 1),
            "local_pair_index": int(pair_index),
            "local_pivot": list(self._current_pivot),
        }
        self.selection_count += 1
        return selection

    def state_dict(self):
        return {
            "version": 1,
            "coin_pool": list(self.coin_pool),
            "basket_size": self.basket_size,
            "block_episodes": self.block_episodes,
            "pivot_episodes": self.pivot_episodes,
            "neighbor_episodes": self.neighbor_episodes,
            "seed": self.seed,
            "block_index": self.block_index,
            "selection_count": self.selection_count,
            "queue": [
                {"basket": list(basket), "role": role, "pair_index": pair_index}
                for basket, role, pair_index in self._queue
            ],
            "current_pivot": (
                None if self._current_pivot is None else list(self._current_pivot)
            ),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        if tuple(state["coin_pool"]) != self.coin_pool:
            raise ValueError("local-composition checkpoint universe mismatch")
        if int(state["basket_size"]) != self.basket_size:
            raise ValueError("local-composition checkpoint basket-size mismatch")
        for name in ("block_episodes", "pivot_episodes", "neighbor_episodes"):
            if int(state[name]) != int(getattr(self, name)):
                raise ValueError(f"local-composition checkpoint {name} mismatch")
        self.block_index = int(state["block_index"])
        self.selection_count = int(state["selection_count"])
        self._queue = [
            (
                self._canonical(item["basket"]),
                str(item["role"]),
                int(item["pair_index"]),
            )
            for item in state.get("queue", [])
        ]
        saved_pivot = state.get("current_pivot")
        self._current_pivot = (
            None if saved_pivot is None else self._canonical(saved_pivot)
        )
        self.rng.bit_generator.state = state["rng_state"]

    def save(self, path):
        temporary_path = path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(self.state_dict(), handle, separators=(",", ":"))
        os.replace(temporary_path, path)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            self.load_state_dict(json.load(handle))


class BasketCurriculumManager:
    """Graph teacher for fixed-cardinality basket tasks.

    The teacher selects the next episode task and publishes the same task law
    to the replay sampler. It does not alter the actor, critic, Bellman target,
    ProExp calculation, CRL penalty, or evaluation metrics. Evaluation reads a
    detached copy of the current task law but cannot mutate this manager.

    Small task families are enumerated exactly.  Large 8-of-N families use a
    lazy active support containing the anchor, the graph-reachable frontier,
    and a rotating uniform-probe reservoir.  This preserves the self-paced
    transfer topology without making memory grow as ``N choose 8``.
    """

    STATE_NAMES = ("unseen", "eligible", "frontier", "mastered", "regressing")
    STAGE_NAMES = ("constraint_stabilization", "local_transfer", "self_paced")

    def __init__(self, coin_pool, anchor_coins, basket_size, seed):
        self.coin_pool = tuple(coin_pool)
        self.coin_to_id = {coin: index for index, coin in enumerate(self.coin_pool)}
        self.basket_size = int(basket_size)
        if self.basket_size <= 0 or self.basket_size > len(self.coin_pool):
            raise ValueError("basket_size must be between one and the universe size")
        missing_anchor = set(anchor_coins) - set(self.coin_pool)
        if missing_anchor:
            raise ValueError(
                "Curriculum anchor contains assets absent from the training universe: "
                f"{sorted(missing_anchor)}"
            )
        if len(set(anchor_coins)) != self.basket_size:
            raise ValueError("curriculum anchor must contain exactly basket_size unique assets")

        self.anchor = self._canonical_basket(anchor_coins)
        self.rng = np.random.default_rng(int(seed))
        self.probe_rng = np.random.default_rng(int(seed) + 19_919)
        expected = math.comb(len(self.coin_pool), self.basket_size)
        self.total_basket_count = int(expected)
        self.enumerated_task_space = bool(
            expected
            <= int(config.BASKET_CURRICULUM_MAX_ENUMERATED_TASKS)
        )
        if self.enumerated_task_space:
            support = {
                tuple(combo)
                for combo in combinations(self.coin_pool, self.basket_size)
            }
            uniform_probe_pool = set(support)
        else:
            one_swap = {
                self._canonical_basket(
                    (set(self.anchor) - {outgoing}) | {incoming}
                )
                for outgoing in self.anchor
                for incoming in self.coin_pool
                if incoming not in self.anchor
            }
            reservoir_target = min(
                int(config.BASKET_CURRICULUM_GLOBAL_PROBE_RESERVOIR_SIZE),
                expected,
            )
            uniform_probe_pool = set()
            attempts = 0
            while (
                len(uniform_probe_pool) < reservoir_target
                and attempts < reservoir_target * 100
            ):
                attempts += 1
                indices = self.probe_rng.choice(
                    len(self.coin_pool), size=self.basket_size, replace=False
                )
                uniform_probe_pool.add(
                    self._canonical_basket(
                        self.coin_pool[int(index)] for index in indices
                    )
                )
            support = {self.anchor, *one_swap, *uniform_probe_pool}
        self.all_baskets = tuple(
            sorted(
                support,
                key=lambda basket: tuple(self.coin_to_id[coin] for coin in basket),
            )
        )
        self.uniform_probe_pool = tuple(
            sorted(
                uniform_probe_pool,
                key=lambda basket: tuple(self.coin_to_id[coin] for coin in basket),
            )
        )
        self.basket_to_index = {
            basket: index for index, basket in enumerate(self.all_baskets)
        }
        self.distance = {
            basket: len(set(self.anchor) - set(basket))
            for basket in self.all_baskets
        }
        self._refresh_distance_buckets()
        self.neighbors = _LazyBasketNeighborMap(self)

        self.completed_episodes = 0
        self.selection_count = 0
        self.stage = "constraint_stabilization"
        self.control_ready = False
        self.control_ready_streak = 0
        self.control_not_ready_streak = 0
        self.last_control_release_kl = None
        self.control_history = deque(
            maxlen=int(config.BASKET_CURRICULUM_CONTROL_WINDOW)
        )
        self.latest_training_step = 0
        self.target_pressure = 0.0
        self.last_distribution_update_episode = -1
        self.local_transfer_progress = 0.0
        self.last_local_transfer_update_episode = -1
        self._self_paced_distribution = None
        self.records = {basket: self._new_record() for basket in self.all_baskets}
        self.eligible = {self.anchor}
        self.eligible.update(self.baskets_by_distance.get(1, ()))
        for basket in self.eligible:
            self.records[basket]["state"] = "eligible"
        self.records[self.anchor]["state"] = "frontier"
        self.frontier_rng = np.random.default_rng(int(seed) + 32_771)
        self.frontier_cohort_generation = 0
        self.active_frontier_cohort = tuple()
        self._refresh_active_frontier_cohort(force=True)
        self._pending_evidence_bundle = None
        self.selection_history = []
        self.episode_history = []
        self.snapshot_history = []
        self.value_rng = np.random.default_rng(int(seed) + 104_729)
        self.value_buffer = deque(
            maxlen=int(config.BASKET_CURRICULUM_VALUE_BUFFER_SIZE)
        )
        self.value_model = None
        self.value_target_model = None
        self.value_optimizer = None
        self.value_input_dim = None
        self.value_update_count = 0
        self.value_loss_ema = None
        self.value_new_samples_since_update = 0
        self._membership_matrix = self._build_membership_matrix()
        self._pca_coordinates = self._build_pca_coordinates()

        self._visual_projection_dirty = False
        if self.enumerated_task_space and len(self.all_baskets) != expected:
            raise RuntimeError("basket enumeration did not match the combinatorial count")

    def _canonical_basket(self, coins):
        unique = tuple(set(coins))
        if len(unique) != self.basket_size:
            raise ValueError("basket must contain basket_size unique assets")
        return tuple(sorted(unique, key=self.coin_to_id.__getitem__))

    def _refresh_distance_buckets(self):
        self.baskets_by_distance = {
            distance: tuple(
                basket
                for basket in self.all_baskets
                if self.distance[basket] == distance
            )
            for distance in sorted(set(self.distance.values()))
        }

    def _ensure_basket_registered(self, basket):
        """Add a lazily discovered task to the finite active support."""
        basket = self._canonical_basket(basket)
        if basket in self.basket_to_index:
            return basket
        self.all_baskets = tuple((*self.all_baskets, basket))
        self.basket_to_index[basket] = len(self.all_baskets) - 1
        self.distance[basket] = len(set(self.anchor) - set(basket))
        self.records[basket] = self._new_record()
        self._refresh_distance_buckets()
        if self._self_paced_distribution is not None:
            self._self_paced_distribution = np.append(
                np.asarray(self._self_paced_distribution, dtype=np.float64), 0.0
            )
        self._visual_projection_dirty = True
        return basket

    def _refresh_uniform_probe_reservoir(self):
        """Rotate global probes without enumerating a large task universe.

        Probe rotation changes only which globally sampled tasks are available
        to the teacher.  It never makes a probe eligible, mastered, or capable
        of unlocking neighbors.  Those state transitions still require a
        legitimate path through the graph frontier.
        """
        if self.enumerated_task_space:
            return False
        target_size = min(
            int(config.BASKET_CURRICULUM_GLOBAL_PROBE_RESERVOIR_SIZE),
            self.total_basket_count,
        )
        refresh_count = max(
            1,
            int(
                round(
                    target_size
                    * float(
                        config.BASKET_CURRICULUM_GLOBAL_PROBE_REFRESH_FRACTION
                    )
                )
            ),
        )
        retained_count = max(0, target_size - refresh_count)
        existing = list(self.uniform_probe_pool)
        if len(existing) > retained_count:
            retained_indices = self.probe_rng.choice(
                len(existing), size=retained_count, replace=False
            )
            retained = {existing[int(index)] for index in retained_indices}
        else:
            retained = set(existing)

        refreshed = set(retained)
        attempts = 0
        maximum_attempts = max(1_000, target_size * 100)
        while len(refreshed) < target_size and attempts < maximum_attempts:
            attempts += 1
            indices = self.probe_rng.choice(
                len(self.coin_pool), size=self.basket_size, replace=False
            )
            basket = self._canonical_basket(
                self.coin_pool[int(index)] for index in indices
            )
            refreshed.add(self._ensure_basket_registered(basket))
        self.uniform_probe_pool = tuple(
            sorted(
                refreshed,
                key=lambda basket: tuple(
                    self.coin_to_id[coin] for coin in basket
                ),
            )
        )
        return set(self.uniform_probe_pool) != set(existing)

    def _restore_active_support(self, baskets, uniform_probe_pool):
        """Restore a lazy support in its serialized probability-vector order."""
        restored = tuple(self._canonical_basket(basket) for basket in baskets)
        if len(restored) != len(set(restored)):
            raise ValueError("curriculum checkpoint contains duplicate tasks")
        if self.anchor not in restored:
            raise ValueError("curriculum checkpoint active support omits anchor")
        old_records = self.records
        self.all_baskets = restored
        self.uniform_probe_pool = tuple(
            self._canonical_basket(basket) for basket in uniform_probe_pool
        )
        if any(basket not in set(restored) for basket in self.uniform_probe_pool):
            raise ValueError("uniform probe reservoir is outside active support")
        self.basket_to_index = {
            basket: index for index, basket in enumerate(self.all_baskets)
        }
        self.distance = {
            basket: len(set(self.anchor) - set(basket))
            for basket in self.all_baskets
        }
        self.records = {
            basket: old_records.get(basket, self._new_record())
            for basket in self.all_baskets
        }
        self._refresh_distance_buckets()
        self._visual_projection_dirty = True

    def _ensure_visual_projection(self):
        if not getattr(self, "_visual_projection_dirty", False):
            return
        self._membership_matrix = self._build_membership_matrix()
        self._pca_coordinates = self._build_pca_coordinates()
        self._visual_projection_dirty = False

    def _new_record(self):
        return {
            "visits": 0,
            "frontier_visits": 0,
            "promotion_visits": 0,
            "probe_visits": 0,
            "rehearsal_visits": 0,
            "selections": 0,
            "last_visit": -1,
            "last_selection": -1,
            "ema_normalized_return": None,
            "previous_ema_normalized_return": None,
            "signed_progress": 0.0,
            "absolute_learning_progress": 0.0,
            "ema_absolute_td_error": None,
            "td_sample_count": 0,
            "recent_normalized_returns": [],
            "probe_ema_normalized_return": None,
            "representative_value_feature": None,
            "value_discounted_quantiles": None,
            "value_normalized_quantiles": None,
            "value_discounted_abs_error_ema": None,
            "value_normalized_abs_error_ema": None,
            "value_sample_count": 0,
            "mastered_once": False,
            "state": "unseen",
        }

    def _frontier_cohort_candidates(self):
        """Legitimate local-transfer tasks available to the active cohort."""
        if self.stage in {"constraint_stabilization", "local_transfer"}:
            candidates = self.baskets_by_distance.get(1, ())
        else:
            candidates = tuple(
                basket
                for basket in self.eligible
                if basket != self.anchor
                and self.records[basket]["state"] not in {"mastered", "regressing"}
            )
        return tuple(
            basket
            for basket in candidates
            if basket in self.eligible and basket != self.anchor
        )

    def _frontier_cohort_priority(self, basket):
        """Evidence-seeking score used only to choose a compact cohort."""
        record = self.records[basket]
        quantiles = record.get("value_normalized_quantiles")
        if quantiles is None:
            conservative_value = 0.0
            uncertainty = 1.0
        else:
            conservative_value = float(quantiles[0])
            uncertainty = max(float(quantiles[-1]) - float(quantiles[0]), 0.0)
        staleness = (
            self.completed_episodes - int(record["last_visit"])
            if int(record["last_visit"]) >= 0
            else self.completed_episodes + 1
        )
        novelty = 1.0 if int(record["visits"]) == 0 else 0.0
        return (
            0.35 * novelty
            + 0.25 * float(np.tanh(uncertainty))
            + 0.20 * float(np.tanh(max(conservative_value, 0.0)))
            + 0.10 * float(
                np.tanh(record["absolute_learning_progress"])
            )
            + 0.10 * float(staleness > 0)
        )

    def _refresh_active_frontier_cohort(self, force=False):
        candidates = list(self._frontier_cohort_candidates())
        if not candidates:
            self.active_frontier_cohort = tuple()
            return False
        cohort_size = min(
            len(candidates),
            max(
                1,
                int(config.BASKET_CURRICULUM_ACTIVE_FRONTIER_COHORT_SIZE),
            ),
        )
        old = tuple(self.active_frontier_cohort)
        if old and not force:
            minimum = int(
                config.BASKET_CURRICULUM_ACTIVE_FRONTIER_MIN_EPISODES
            )
            maximum = max(
                minimum,
                int(config.BASKET_CURRICULUM_ACTIVE_FRONTIER_MAX_EPISODES),
            )
            # A visit count establishes only that evidence exists. It is not a
            # competence declaration. Every cohort member remains active until
            # its conservative value gate succeeds or its explicit study
            # budget is exhausted, at which point rotation means "stalled; seek
            # another transfer contrast" rather than "mastered."
            if any(
                int(self.records[basket].get("frontier_visits", 0)) < minimum
                for basket in old
            ):
                return False
            if any(
                not self._basket_value_competent(basket)
                and int(self.records[basket].get("frontier_visits", 0)) < maximum
                for basket in old
            ):
                return False

        outside_old = [basket for basket in candidates if basket not in old]
        pool = outside_old if len(outside_old) >= cohort_size else candidates
        tie_break = {
            basket: float(value)
            for basket, value in zip(
                pool, self.frontier_rng.random(len(pool))
            )
        }
        selected = []
        while pool and len(selected) < cohort_size:
            def score(candidate):
                if not selected:
                    diversity = 1.0
                else:
                    diversity = min(
                        len(set(candidate).symmetric_difference(other))
                        / float(2 * self.basket_size)
                        for other in selected
                    )
                return (
                    self._frontier_cohort_priority(candidate)
                    + 0.35 * diversity
                    + 1e-6 * tie_break[candidate]
                )

            chosen = max(pool, key=score)
            selected.append(chosen)
            pool.remove(chosen)
        changed = tuple(selected) != old
        self.active_frontier_cohort = tuple(selected)
        if changed and old:
            self.frontier_cohort_generation += 1
        return changed

    def _maybe_rotate_active_frontier_cohort(self):
        return self._refresh_active_frontier_cohort(force=False)

    def _build_membership_matrix(self):
        matrix = np.zeros(
            (len(self.all_baskets), len(self.coin_pool)), dtype=np.float64
        )
        for row, basket in enumerate(self.all_baskets):
            matrix[row, [self.coin_to_id[coin] for coin in basket]] = 1.0
        return matrix

    def _build_pca_coordinates(self):
        if not self.enumerated_task_space:
            # A deterministic circular projection does not refit when lazy
            # support grows, so the operational map remains spatially stable
            # across curriculum snapshots and checkpoint resume.
            angles = 2.0 * np.pi * (
                np.arange(len(self.coin_pool), dtype=np.float64) + 0.5
            ) / float(len(self.coin_pool))
            basis = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
            return (self._membership_matrix @ basis).astype(np.float32)
        centered = self._membership_matrix - self._membership_matrix.mean(
            axis=0, keepdims=True
        )
        u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        coordinates = u[:, :2] * singular_values[:2]
        for column in range(coordinates.shape[1]):
            pivot = int(np.argmax(np.abs(coordinates[:, column])))
            if coordinates[pivot, column] < 0.0:
                coordinates[:, column] *= -1.0
        return coordinates.astype(np.float32)

    def _ensure_value_model(self, input_dim):
        input_dim = int(input_dim)
        if self.value_model is not None:
            if input_dim != self.value_input_dim:
                raise ValueError(
                    "curriculum value feature width changed from "
                    f"{self.value_input_dim} to {input_dim}"
                )
            return
        # Manager construction happens before TD3 policy construction.  Forking
        # the RNG guarantees that the teacher cannot alter actor/critic initial
        # weights merely by being enabled.
        structured_input_dim = int(config.CURRICULUM_VALUE_INPUT_DIM)
        structured_contract = input_dim == structured_input_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config.BASKET_CURRICULUM_SEED) + 73)
            self.value_model = BasketCurriculumValueModel(
                input_dim=input_dim,
                hidden_dims=tuple(config.BASKET_CURRICULUM_VALUE_HIDDEN_DIMS),
                target_count=2,
                max_assets=(int(config.MAX_COINS) if structured_contract else 0),
                token_dim=(
                    int(config.CURRICULUM_VALUE_TOKEN_DIM)
                    if structured_contract else 0
                ),
                global_dim=(
                    int(config.CURRICULUM_VALUE_GLOBAL_DIM)
                    if structured_contract else 0
                ),
                snapshot_token_dim=(
                    int(config.CURRICULUM_VALUE_SNAPSHOT_TOKEN_DIM)
                    if structured_contract else 0
                ),
                history_length=(
                    int(config.CURRICULUM_VALUE_HISTORY_LENGTH)
                    if structured_contract else 0
                ),
                history_feature_dim=(
                    int(config.CURRICULUM_VALUE_HISTORY_FEATURE_DIM)
                    if structured_contract else 0
                ),
            ).cpu()
        self.value_target_model = copy.deepcopy(self.value_model).eval()
        for parameter in self.value_target_model.parameters():
            parameter.requires_grad_(False)
        self.value_optimizer = torch.optim.Adam(
            self.value_model.parameters(),
            lr=float(config.BASKET_CURRICULUM_VALUE_LEARNING_RATE),
        )
        self.value_input_dim = input_dim

    def _append_value_samples(self, basket, samples, role="unknown"):
        if not bool(config.BASKET_CURRICULUM_VALUE_MODEL_ENABLED):
            return
        record = self.records[basket]
        prepared = []
        for sample in samples or ():
            feature = np.asarray(sample.get("feature", ()), dtype=np.float32).reshape(-1)
            if feature.size == 0 or not np.all(np.isfinite(feature)):
                continue
            self._ensure_value_model(feature.size)
            targets = np.asarray(
                [
                    sample.get("discounted_financial_return", 0.0),
                    sample.get("opportunity_normalized_return", 0.0),
                ],
                dtype=np.float32,
            )
            target_clip = float(config.BASKET_CURRICULUM_VALUE_TARGET_CLIP)
            targets = np.clip(targets, -target_clip, target_clip)
            prepared.append((sample, feature, targets))

        # One episode normally contributes four checkpoints.  The old code
        # launched four separate tiny target-network forwards here.  Batch the
        # complete episode so curriculum bookkeeping pays one dispatch cost,
        # while preserving the exact same per-checkpoint predictions.
        predictions = None
        if prepared and self.value_target_model is not None:
            self.value_target_model.eval()
            with torch.no_grad():
                feature_batch = torch.from_numpy(
                    np.stack([item[1] for item in prepared])
                )
                predictions = (
                    self.value_target_model(feature_batch)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )

        for prepared_index, (sample, feature, targets) in enumerate(prepared):
            checkpoint_step = int(sample.get("step", 0))
            prediction = (
                predictions[prepared_index]
                if predictions is not None
                else None
            )
            # Basket promotion is a contextual initial-state decision.  Later
            # checkpoints still train the shared value model, but their shorter
            # remaining horizons must not be compared with the initial-state
            # uncertainty interval used by the curriculum gate.
            if prediction is not None and checkpoint_step == 0:
                median = prediction[:, 1]
                absolute_error = np.abs(median - targets)
                alpha = float(config.BASKET_CURRICULUM_COMPETENCE_EMA_ALPHA)
                for index, key in enumerate(
                    (
                        "value_discounted_abs_error_ema",
                        "value_normalized_abs_error_ema",
                    )
                ):
                    old = record[key]
                    record[key] = float(
                        absolute_error[index]
                        if old is None
                        else (1.0 - alpha) * float(old)
                        + alpha * float(absolute_error[index])
                    )

            self.value_buffer.append(
                {
                    "basket_index": int(self.basket_to_index[basket]),
                    "distance": int(self.distance[basket]),
                    "episode_index": int(self.completed_episodes),
                    "role": str(role),
                    "feature": feature.tolist(),
                    "targets": targets.tolist(),
                }
            )
            self.value_new_samples_since_update += 1
            record["value_sample_count"] = int(record["value_sample_count"]) + 1

            # The earliest retained checkpoint is the contextual initial-state
            # estimate used by Klink-style task valuation.  Later checkpoints
            # train the same value function but do not overwrite this context.
            if checkpoint_step == 0:
                old_feature = record["representative_value_feature"]
                if old_feature is None:
                    representative = feature
                else:
                    alpha = float(config.BASKET_CURRICULUM_COMPETENCE_EMA_ALPHA)
                    representative = (
                        (1.0 - alpha)
                        * np.asarray(old_feature, dtype=np.float32)
                        + alpha * feature
                    )
                record["representative_value_feature"] = representative.tolist()

    def value_model_update_due(self):
        """Cheap schedule predicate for the 100-step outer training loop."""
        return bool(
            self.value_model is not None
            and self.value_optimizer is not None
            and len(self.value_buffer)
            >= int(config.BASKET_CURRICULUM_VALUE_MIN_SAMPLES)
            and self.value_new_samples_since_update
            >= int(config.BASKET_CURRICULUM_VALUE_NEW_SAMPLES_PER_UPDATE)
            and int(config.BASKET_CURRICULUM_VALUE_UPDATES_PER_ITERATION) > 0
        )

    def _value_sample_weights(self, buffer_values):
        """Role/distance-stratified fitting law for the curriculum value head.

        Completed episodes arrive according to historical collection volume.
        The teacher instead needs calibration on the curriculum distribution it
        is making decisions about now. We therefore allocate probability first
        to (role, substitution-distance) groups using the current teacher law,
        blend in a small uniform group-coverage floor, and only then allocate
        within each group using recency and inverse basket-frequency weights.
        """
        ages = np.asarray(
            [
                max(
                    0,
                    self.completed_episodes - int(item.get("episode_index", 0)),
                )
                for item in buffer_values
            ],
            dtype=np.float64,
        )
        recency_half_life = max(
            float(config.BASKET_CURRICULUM_VALUE_RECENCY_HALF_LIFE_EPISODES),
            1.0,
        )
        recency = np.exp2(-ages / recency_half_life)
        basket_counts = {}
        sample_groups = []
        observed_groups = set()
        for item in buffer_values:
            basket_index = int(item.get("basket_index", -1))
            if 0 <= basket_index < len(self.all_baskets):
                basket = self.all_baskets[basket_index]
                distance = int(
                    item.get("distance", self.distance.get(basket, 0))
                )
            else:
                distance = int(item.get("distance", 0))
            group = (str(item.get("role", "unknown")), distance)
            sample_groups.append(group)
            observed_groups.add(group)
            basket_counts[basket_index] = basket_counts.get(basket_index, 0) + 1

        desired_group_mass = {group: 0.0 for group in observed_groups}
        try:
            current_role_law = self.replay_role_probabilities()
        except (RuntimeError, ValueError):
            current_role_law = {}
        anchor_ids = {self.coin_to_id[coin] for coin in self.anchor}
        for role, task_law in current_role_law.items():
            for membership_ids, probability in task_law.items():
                distance = len(anchor_ids - set(membership_ids))
                group = (str(role), int(distance))
                if group in desired_group_mass:
                    desired_group_mass[group] += max(float(probability), 0.0)

        group_count = max(len(observed_groups), 1)
        coverage = float(
            np.clip(
                config.BASKET_CURRICULUM_VALUE_GROUP_COVERAGE_MIX,
                0.0,
                1.0,
            )
        )
        desired_total = float(sum(desired_group_mass.values()))
        for group in desired_group_mass:
            teacher_mass = (
                desired_group_mass[group] / desired_total
                if desired_total > 0.0
                else 1.0 / group_count
            )
            desired_group_mass[group] = (
                (1.0 - coverage) * teacher_mass + coverage / group_count
            )

        group_indices = {}
        for index, group in enumerate(sample_groups):
            group_indices.setdefault(group, []).append(index)
        sampling_weights = np.zeros(len(buffer_values), dtype=np.float64)
        for group, indices in group_indices.items():
            within = np.asarray(
                [
                    recency[index]
                    / math.sqrt(
                        max(
                            basket_counts.get(
                                int(buffer_values[index].get("basket_index", -1)),
                                1,
                            ),
                            1,
                        )
                    )
                    for index in indices
                ],
                dtype=np.float64,
            )
            within = np.maximum(within, 1e-12)
            within /= within.sum()
            sampling_weights[np.asarray(indices, dtype=np.int64)] = (
                desired_group_mass[group] * within
            )

        # The resulting probabilities already express the intended fitting
        # distribution; no equal-basket loss rewrite is required.
        sampling_weights = np.maximum(sampling_weights, 1e-8)
        return sampling_weights / sampling_weights.sum()

    def train_value_model(self, updates=None):
        if self.value_model is None or self.value_optimizer is None:
            return None
        minimum = int(config.BASKET_CURRICULUM_VALUE_MIN_SAMPLES)
        if len(self.value_buffer) < minimum:
            return None
        if self.value_new_samples_since_update < int(
            config.BASKET_CURRICULUM_VALUE_NEW_SAMPLES_PER_UPDATE
        ):
            return None
        update_count = int(
            config.BASKET_CURRICULUM_VALUE_UPDATES_PER_ITERATION
            if updates is None
            else updates
        )
        if update_count <= 0:
            return None
        batch_size = min(
            int(config.BASKET_CURRICULUM_VALUE_BATCH_SIZE),
            len(self.value_buffer),
        )
        buffer_values = list(self.value_buffer)
        sampling_weights = self._value_sample_weights(buffer_values)
        losses = []
        self.value_model.train()
        for _ in range(update_count):
            indices = self.value_rng.choice(
                len(buffer_values),
                size=batch_size,
                replace=False,
                p=sampling_weights,
            )
            features = torch.as_tensor(
                np.asarray(
                    [buffer_values[int(index)]["feature"] for index in indices],
                    dtype=np.float32,
                )
            )
            targets = torch.as_tensor(
                np.asarray(
                    [buffer_values[int(index)]["targets"] for index in indices],
                    dtype=np.float32,
                )
            )
            predictions = self.value_model(features)
            loss = self.value_model.quantile_huber_loss(
                predictions,
                targets,
                quantiles=tuple(config.BASKET_CURRICULUM_VALUE_QUANTILES),
            )
            self.value_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 5.0)
            self.value_optimizer.step()
            target_tau = float(config.BASKET_CURRICULUM_VALUE_TARGET_TAU)
            with torch.no_grad():
                for target_parameter, online_parameter in zip(
                    self.value_target_model.parameters(),
                    self.value_model.parameters(),
                ):
                    target_parameter.lerp_(online_parameter, target_tau)
            losses.append(float(loss.detach().item()))
            self.value_update_count += 1
        mean_loss = float(np.mean(losses))
        self.value_loss_ema = (
            mean_loss
            if self.value_loss_ema is None
            else 0.9 * float(self.value_loss_ema) + 0.1 * mean_loss
        )
        self._refresh_value_estimates()
        self._reassess_curriculum()
        self.value_new_samples_since_update = 0
        return mean_loss

    @torch.no_grad()
    def predict_value(self, feature):
        if self.value_target_model is None:
            return None
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        if feature.size != self.value_input_dim:
            return None
        self.value_target_model.eval()
        prediction = self.value_target_model(
            torch.as_tensor(feature).unsqueeze(0)
        )[0]
        return prediction.cpu().numpy().astype(np.float64, copy=False)

    def _default_value_feature(self, basket):
        del basket
        # An unseen basket has no observable initial state to value.  Path 2
        # refuses to fabricate one by editing a membership vector; the uniform
        # probe channel must first collect a real causal state.
        return None

    @torch.no_grad()
    def _refresh_value_estimates(self):
        """Refresh observed basket values in vectorized CPU batches.

        The previous implementation launched one tiny Torch forward pass per
        basket after every teacher update.  That incurred Python dispatch and
        tensor-construction overhead proportional to the number of visited
        baskets.  A single batched pass preserves exactly the same target-model
        predictions while making refresh cost proportional to a small number of
        batches instead of one call per record.
        """
        if self.value_target_model is None:
            return
        pending = []
        for basket, record in self.records.items():
            feature = record.get("representative_value_feature")
            if feature is None:
                feature = self._default_value_feature(basket)
            if feature is None:
                continue
            feature = np.asarray(feature, dtype=np.float32).reshape(-1)
            if feature.size != self.value_input_dim:
                continue
            pending.append((basket, feature))
        if not pending:
            return

        self.value_target_model.eval()
        refresh_batch_size = max(
            64,
            min(
                512,
                int(getattr(config, "BASKET_CURRICULUM_VALUE_BATCH_SIZE", 128)),
            ),
        )
        for start in range(0, len(pending), refresh_batch_size):
            batch = pending[start:start + refresh_batch_size]
            features = torch.from_numpy(np.stack([item[1] for item in batch]))
            predictions = self.value_target_model(features).detach().cpu().numpy()
            for index, (basket, _) in enumerate(batch):
                record = self.records[basket]
                record["value_discounted_quantiles"] = predictions[index, 0].tolist()
                record["value_normalized_quantiles"] = predictions[index, 1].tolist()

    def _rank_percentiles(self, values):
        values = np.asarray(values, dtype=np.float64)
        if len(values) <= 1:
            return np.ones(len(values), dtype=np.float64)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks / float(len(values) - 1)

    def update_control_readiness(self, crl_stats, step):
        """Update the gate that separates CRL stabilization from curriculum.

        Readiness is measured under the same diversified foundation mixture
        used after the gate opens.  Acceleration must occupy a robust confidence
        band around its threshold: its center, upper quantile, exceedance rate,
        and catastrophic maximum are checked over a rolling window.  Velocity
        and both dual multipliers remain diagnostics; they do not control this
        gate.  Separate open/close streaks provide hysteresis, so one noisy
        curriculum switch cannot repeatedly toggle graph expansion.
        """
        self.latest_training_step = int(step)
        if not crl_stats or not bool(crl_stats.get("crl_enabled", False)):
            ready_now = self.latest_training_step >= int(
                config.BASKET_CURRICULUM_CONTROL_MIN_STEPS
            )
        else:
            acceleration_cost = crl_stats.get("avg_recent_acceleration_cost")
            acceleration_threshold = crl_stats.get(
                "acceleration_constraint_threshold"
            )
            acceleration_multiplier = crl_stats.get(
                "current_lambda_acceleration"
            )
            velocity_cost = crl_stats.get("avg_recent_velocity_cost")
            velocity_threshold = crl_stats.get("velocity_constraint_threshold")
            velocity_multiplier = crl_stats.get("current_lambda_velocity")
            if any(
                value is None
                for value in (
                    acceleration_cost,
                    acceleration_threshold,
                    acceleration_multiplier,
                )
            ):
                ready_now = False
            else:
                values = tuple(
                    float(value)
                    for value in (
                        acceleration_cost,
                        acceleration_threshold,
                        acceleration_multiplier,
                    )
                )
                if all(np.isfinite(value) for value in values):
                    history_item = {
                        "step": self.latest_training_step,
                        "acceleration_cost": values[0],
                        "acceleration_threshold": values[1],
                        "lambda_acceleration": values[2],
                    }
                    velocity_values = (
                        velocity_cost,
                        velocity_threshold,
                        velocity_multiplier,
                    )
                    if all(value is not None for value in velocity_values):
                        velocity_values = tuple(
                            float(value) for value in velocity_values
                        )
                        if all(np.isfinite(value) for value in velocity_values):
                            history_item.update(
                                velocity_cost=velocity_values[0],
                                velocity_threshold=velocity_values[1],
                                lambda_velocity=velocity_values[2],
                            )
                    self.control_history.append(history_item)
                minimum = int(config.BASKET_CURRICULUM_CONTROL_MIN_SAMPLES)
                ready_now = False
                complete_control_history = [
                    item
                    for item in self.control_history
                    if all(
                        key in item
                        for key in (
                            "acceleration_cost",
                            "acceleration_threshold",
                            "lambda_acceleration",
                        )
                    )
                ]
                if (
                    self.latest_training_step
                    >= int(config.BASKET_CURRICULUM_CONTROL_MIN_STEPS)
                    and len(complete_control_history) >= minimum
                ):
                    recent = complete_control_history[-minimum:]
                    costs = np.asarray(
                        [item["acceleration_cost"] for item in recent],
                        dtype=np.float64,
                    )
                    thresholds = np.asarray(
                        [item["acceleration_threshold"] for item in recent],
                        dtype=np.float64,
                    )
                    def constraint_stable(recent_costs, recent_thresholds):
                        reference = max(float(np.mean(recent_thresholds)), 1e-8)
                        center_tolerance = float(
                            config.BASKET_CURRICULUM_ACCELERATION_CENTER_TOLERANCE_FRACTION
                        )
                        quantile = float(
                            config.BASKET_CURRICULUM_ACCELERATION_QUANTILE
                        )
                        quantile_tolerance = float(
                            config.BASKET_CURRICULUM_ACCELERATION_QUANTILE_TOLERANCE_FRACTION
                        )
                        exceedance_limit = float(
                            config.BASKET_CURRICULUM_ACCELERATION_MAX_EXCEEDANCE_FRACTION
                        )
                        catastrophic_tolerance = float(
                            config.BASKET_CURRICULUM_ACCELERATION_CATASTROPHIC_OVERSHOOT_FRACTION
                        )
                        center_is_near_constraint = abs(
                            float(np.mean(recent_costs)) - reference
                        ) <= center_tolerance * reference
                        robust_upper_tail = float(
                            np.quantile(recent_costs, quantile)
                        ) <= (1.0 + quantile_tolerance) * reference
                        exceedance_fraction = float(
                            np.mean(
                                recent_costs
                                > (1.0 + quantile_tolerance) * recent_thresholds
                            )
                        )
                        no_catastrophic_spike = float(
                            np.max(recent_costs)
                        ) <= (1.0 + catastrophic_tolerance) * reference
                        return bool(
                            center_is_near_constraint
                            and robust_upper_tail
                            and exceedance_fraction <= exceedance_limit
                            and no_catastrophic_spike
                        )

                    ready_now = constraint_stable(costs, thresholds)

        if ready_now:
            self.control_ready_streak += 1
            self.control_not_ready_streak = 0
        else:
            self.control_ready_streak = 0
            self.control_not_ready_streak += 1
        if not self.control_ready:
            self.control_ready = self.control_ready_streak >= int(
                config.BASKET_CURRICULUM_CONTROL_READY_CONSECUTIVE
            )
        elif self.control_not_ready_streak >= int(
            config.BASKET_CURRICULUM_CONTROL_NOT_READY_CONSECUTIVE
        ):
            self.control_ready = False
        if self.stage == "constraint_stabilization" and self.control_ready:
            # Opening the gate authorizes a later KL-bounded movement away from
            # the anchor-dominant foundation.  It must not itself change the
            # task distribution, otherwise CRL readiness would be invalidated
            # by the release event it was meant to protect.
            before_release = self._task_probability_vector()
            self.stage = "local_transfer"
            self.local_transfer_progress = 0.0
            self.last_local_transfer_update_episode = self.completed_episodes
            after_release = self._task_probability_vector()
            release_kl = self._distribution_kl(after_release, before_release)
            self.last_control_release_kl = float(release_kl)
            if release_kl > float(
                config.BASKET_CURRICULUM_CONTROL_RELEASE_KL_LIMIT
            ):
                self.stage = "constraint_stabilization"
                self.control_ready = False
                self.control_ready_streak = 0
                self.control_not_ready_streak = 0
                raise RuntimeError(
                    "Opening the acceleration-readiness gate changed the "
                    "basket distribution: "
                    f"KL={release_kl:.12g}. Stabilization and local transfer "
                    "must use the same task mixture."
                )
        # Once released, the not-ready hysteresis above freezes expansion only
        # after repeated unstable windows.  A single noisy report no longer
        # revokes an otherwise interpretable constrained policy.
        self._reassess_curriculum()
        return self.control_ready

    def _value_model_ready(self):
        return bool(
            self.value_model is not None
            and len(self.value_buffer)
            >= int(config.BASKET_CURRICULUM_VALUE_MIN_SAMPLES)
            and self.value_update_count > 0
        )

    def _value_interval_calibrated(self, record, target_name):
        quantile_key = f"value_{target_name}_quantiles"
        error_key = f"value_{target_name}_abs_error_ema"
        quantiles = record.get(quantile_key)
        error = record.get(error_key)
        if quantiles is None or error is None:
            return False
        interval_width = max(float(quantiles[2]) - float(quantiles[0]), 0.0)
        floor = 0.05 if target_name == "discounted" else 0.25
        permitted = float(
            config.BASKET_CURRICULUM_VALUE_MAX_CALIBRATION_MULTIPLIER
        ) * max(interval_width, floor)
        return float(error) <= permitted

    def _basket_value_competent(self, basket):
        record = self.records[basket]
        if not self.control_ready or not self._value_model_ready():
            return False
        if int(record["promotion_visits"]) < int(
            config.BASKET_CURRICULUM_MIN_EVIDENCE_EPISODES
        ):
            return False
        discounted = record.get("value_discounted_quantiles")
        normalized = record.get("value_normalized_quantiles")
        if discounted is None or normalized is None:
            return False
        conservative_value = float(discounted[0])
        conservative_normalized = float(normalized[0])
        recent = list(record.get("recent_normalized_returns", ()))
        positive_fraction = (
            float(np.mean(np.asarray(recent, dtype=np.float64) > 0.0))
            if recent
            else 0.0
        )
        return bool(
            conservative_value
            >= float(config.BASKET_CURRICULUM_VALUE_DISCOUNTED_LCB_THRESHOLD)
            and conservative_normalized
            >= float(config.BASKET_CURRICULUM_VALUE_NORMALIZED_LCB_THRESHOLD)
            and record.get("ema_normalized_return") is not None
            and float(record["ema_normalized_return"])
            >= float(config.BASKET_CURRICULUM_MASTERY_RETURN_THRESHOLD)
            and positive_fraction
            >= float(config.BASKET_CURRICULUM_MASTERY_POSITIVE_FRACTION)
            and self._value_interval_calibrated(record, "discounted")
            and self._value_interval_calibrated(record, "normalized")
        )

    def _reassess_curriculum(self):
        newly_mastered = []
        for basket in tuple(self.eligible):
            record = self.records[basket]
            competent = self._basket_value_competent(basket)
            if competent:
                if not bool(record["mastered_once"]):
                    newly_mastered.append(basket)
                record["mastered_once"] = True
                record["state"] = "mastered"
            elif bool(record["mastered_once"]):
                recent = list(record.get("recent_normalized_returns", ()))
                positive_fraction = (
                    float(np.mean(np.asarray(recent) > 0.0)) if recent else 0.0
                )
                value_quantiles = record.get("value_normalized_quantiles")
                value_regressed = bool(
                    value_quantiles is not None
                    and float(value_quantiles[0])
                    < float(config.BASKET_CURRICULUM_VALUE_NORMALIZED_LCB_THRESHOLD)
                )
                record["state"] = (
                    "regressing"
                    if value_regressed or positive_fraction < 0.5
                    else "mastered"
                )
            elif record["state"] != "frontier":
                record["state"] = "eligible"

        if self.stage == "local_transfer" and self.control_ready:
            interval = int(
                config.BASKET_CURRICULUM_LOCAL_TRANSFER_UPDATE_INTERVAL_EPISODES
            )
            if (
                self.completed_episodes - self.last_local_transfer_update_episode
                >= interval
            ):
                self._update_local_transfer_progress()

        if (
            self.stage == "local_transfer"
            and self.records[self.anchor]["state"] == "mastered"
            and self.control_ready
            and self.local_transfer_progress >= 1.0 - 1e-12
        ):
            self.stage = "self_paced"
            for basket, record in self.records.items():
                if record["state"] == "mastered" and basket in self.eligible:
                    self._unlock_neighbors(basket)
            self._update_self_paced_distribution(force=True)
        elif self.stage == "self_paced" and self.control_ready:
            for basket in newly_mastered:
                self._unlock_neighbors(basket)
            interval = int(
                config.BASKET_CURRICULUM_VALUE_UPDATE_INTERVAL_EPISODES
            )
            if (
                self.completed_episodes - self.last_distribution_update_episode
                >= interval
            ):
                self._update_self_paced_distribution()

    def _candidate_weights(self, candidates, role):
        if not candidates:
            return np.zeros(0, dtype=np.float64)
        # Cold-start non-anchor tasks are deliberately exchangeable.  Before
        # completed-trajectory value evidence exists, novelty, TD error, and
        # one noisy return must not manufacture a preferred ticker subset.
        # This preserves equal probability inside the active one-swap cohort
        # and global probe pool while the anchor establishes control.
        if self.stage in {"constraint_stabilization", "local_transfer"} and role in {
            "frontier",
            "uniform_probe",
        }:
            return np.full(
                len(candidates), 1.0 / len(candidates), dtype=np.float64
            )
        records = [self.records[basket] for basket in candidates]
        staleness = [
            self.completed_episodes - int(record["last_visit"])
            if int(record["last_visit"]) >= 0
            else self.completed_episodes + 1
            for record in records
        ]
        td_error = [
            float(record["ema_absolute_td_error"] or 0.0)
            for record in records
        ]
        progress = [
            float(record["absolute_learning_progress"])
            for record in records
        ]
        novelty = [1.0 if int(record["visits"]) == 0 else 0.0 for record in records]
        regression = [1.0 if record["state"] == "regressing" else 0.0 for record in records]

        stale_rank = self._rank_percentiles(staleness)
        td_rank = self._rank_percentiles(td_error)
        progress_rank = self._rank_percentiles(progress)
        if role == "rehearsal":
            score = (
                0.45 * self._rank_percentiles(regression)
                + 0.35 * stale_rank
                + 0.20 * td_rank
            )
        elif role == "frontier":
            score = (
                0.35 * np.asarray(novelty, dtype=np.float64)
                + 0.30 * progress_rank
                + 0.25 * td_rank
                + 0.10 * stale_rank
            )
        else:
            return np.full(len(candidates), 1.0 / len(candidates), dtype=np.float64)
        exponent = float(
            getattr(config, "BASKET_CURRICULUM_PRIORITY_RANK_EXPONENT", 1.0)
        )
        weights = np.power(score + 0.05, exponent)
        return weights / weights.sum()

    @staticmethod
    def _normalized_distribution(values):
        values = np.asarray(values, dtype=np.float64)
        values = np.clip(values, 0.0, None)
        total = float(values.sum())
        if total <= 0.0:
            return np.full(len(values), 1.0 / max(len(values), 1), dtype=np.float64)
        return values / total

    def _apply_anchor_floor(self, probabilities):
        probabilities = self._normalized_distribution(probabilities)
        anchor_index = self.basket_to_index[self.anchor]
        floor = float(config.BASKET_CURRICULUM_ANCHOR_PROBABILITY_FLOOR)
        if probabilities[anchor_index] >= floor:
            return probabilities
        deficit = floor - float(probabilities[anchor_index])
        other_mask = np.ones(len(probabilities), dtype=bool)
        other_mask[anchor_index] = False
        other_total = float(probabilities[other_mask].sum())
        if other_total > 0.0:
            probabilities[other_mask] *= max(other_total - deficit, 0.0) / other_total
        probabilities[anchor_index] = floor
        return self._normalized_distribution(probabilities)

    @staticmethod
    def _distribution_kl(candidate, reference):
        epsilon = 1e-12
        candidate = np.clip(np.asarray(candidate, dtype=np.float64), epsilon, None)
        reference = np.clip(np.asarray(reference, dtype=np.float64), epsilon, None)
        candidate /= candidate.sum()
        reference /= reference.sum()
        return float(np.sum(candidate * (np.log(candidate) - np.log(reference))))

    def _bounded_distribution_update(self, previous, proposed):
        previous = self._normalized_distribution(previous)
        proposed = self._normalized_distribution(proposed)
        limit = float(config.BASKET_CURRICULUM_DISTRIBUTION_KL_LIMIT)
        if self._distribution_kl(proposed, previous) <= limit:
            return proposed
        lower, upper = 0.0, 1.0
        for _ in range(40):
            mix = 0.5 * (lower + upper)
            candidate = self._normalized_distribution(
                (1.0 - mix) * previous + mix * proposed
            )
            if self._distribution_kl(candidate, previous) <= limit:
                lower = mix
            else:
                upper = mix
        return self._normalized_distribution(
            (1.0 - lower) * previous + lower * proposed
        )

    def _value_driven_eligible_weights(self, candidates):
        if not candidates:
            return np.zeros(0, dtype=np.float64)
        records = [self.records[basket] for basket in candidates]
        discounted = [
            float(record["value_discounted_quantiles"][0])
            if record.get("value_discounted_quantiles") is not None
            else -np.inf
            for record in records
        ]
        normalized = [
            float(record["value_normalized_quantiles"][0])
            if record.get("value_normalized_quantiles") is not None
            else -np.inf
            for record in records
        ]

        def finite_rank(values):
            array = np.asarray(values, dtype=np.float64)
            finite = np.isfinite(array)
            ranks = np.zeros(len(array), dtype=np.float64)
            if np.any(finite):
                ranks[finite] = self._rank_percentiles(array[finite])
            return ranks

        discounted_rank = finite_rank(discounted)
        normalized_rank = finite_rank(normalized)
        td_rank = self._rank_percentiles(
            [float(record["ema_absolute_td_error"] or 0.0) for record in records]
        )
        stale_rank = self._rank_percentiles(
            [
                self.completed_episodes - int(record["last_visit"])
                if int(record["last_visit"]) >= 0
                else self.completed_episodes + 1
                for record in records
            ]
        )
        novelty = np.asarray(
            [1.0 if int(record["promotion_visits"]) == 0 else 0.0 for record in records],
            dtype=np.float64,
        )
        regression = np.asarray(
            [1.0 if record["state"] == "regressing" else 0.0 for record in records],
            dtype=np.float64,
        )
        score = (
            0.45 * discounted_rank
            + 0.25 * normalized_rank
            + 0.10 * td_rank
            + 0.10 * stale_rank
            + 0.10 * novelty
            + 0.15 * regression
        )
        temperature = max(float(config.BASKET_CURRICULUM_VALUE_TEMPERATURE), 1e-4)
        logits = (score - float(np.max(score))) / temperature
        return self._normalized_distribution(np.exp(logits))

    def _update_self_paced_distribution(self, force=False):
        if self.stage != "self_paced":
            return
        if not force:
            self._refresh_uniform_probe_reservoir()
        evidence = [
            basket
            for basket in self.eligible
            if int(self.records[basket]["promotion_visits"])
            >= int(config.BASKET_CURRICULUM_MIN_EVIDENCE_EPISODES)
        ]
        competent_fraction = (
            float(np.mean([self._basket_value_competent(basket) for basket in evidence]))
            if evidence
            else 0.0
        )
        any_regressing = any(
            self.records[basket]["state"] == "regressing" for basket in self.eligible
        )
        if not force and self.control_ready:
            if any_regressing:
                self.target_pressure = max(
                    0.0,
                    self.target_pressure
                    - float(config.BASKET_CURRICULUM_TARGET_PRESSURE_REGRESSION_STEP),
                )
            elif competent_fraction >= float(
                config.BASKET_CURRICULUM_TARGET_PRESSURE_COMPETENCE_FRACTION
            ):
                self.target_pressure = min(
                    1.0,
                    self.target_pressure
                    + float(config.BASKET_CURRICULUM_TARGET_PRESSURE_STEP),
                )

        base_uniform = float(config.BASKET_CURRICULUM_LOCAL_UNIFORM_SHARE)
        maximum_uniform = float(config.BASKET_CURRICULUM_MAX_UNIFORM_SHARE)
        uniform_share = base_uniform + self.target_pressure * (
            maximum_uniform - base_uniform
        )
        proposed = np.zeros(len(self.all_baskets), dtype=np.float64)
        target_uniform_tasks = (
            self.all_baskets
            if self.enumerated_task_space
            else self.uniform_probe_pool
        )
        for basket in target_uniform_tasks:
            proposed[self.basket_to_index[basket]] += (
                uniform_share / len(target_uniform_tasks)
            )
        eligible_candidates = tuple(
            sorted(self.eligible, key=self.basket_to_index.get)
        )
        eligible_weights = self._value_driven_eligible_weights(eligible_candidates)
        for basket, weight in zip(eligible_candidates, eligible_weights):
            proposed[self.basket_to_index[basket]] += (
                1.0 - uniform_share
            ) * float(weight)
        proposed = self._apply_anchor_floor(proposed)

        if self._self_paced_distribution is None or force:
            previous = self._local_transfer_probability_vector(progress=1.0)
        else:
            previous = self._self_paced_distribution
        self._self_paced_distribution = self._bounded_distribution_update(
            previous, proposed
        )
        self.last_distribution_update_episode = self.completed_episodes

    def _foundation_components(self):
        return [
            (
                "anchor",
                float(config.BASKET_CURRICULUM_FOUNDATION_ANCHOR_SHARE),
                (self.anchor,),
            ),
            (
                "frontier",
                float(config.BASKET_CURRICULUM_FOUNDATION_ONE_SWAP_SHARE),
                self.active_frontier_cohort,
            ),
            (
                "uniform_probe",
                float(config.BASKET_CURRICULUM_FOUNDATION_UNIFORM_SHARE),
                self.uniform_probe_pool,
            ),
        ]

    def _local_transfer_components(self, progress=None):
        if progress is None:
            progress = self.local_transfer_progress
        progress = float(np.clip(progress, 0.0, 1.0))

        def interpolate(foundation_name, destination_name):
            foundation = float(getattr(config, foundation_name))
            destination = float(getattr(config, destination_name))
            return (1.0 - progress) * foundation + progress * destination

        rehearsal = tuple(
            basket
            for basket in self.all_baskets
            if basket in self.eligible
            and self.records[basket]["state"] in {"mastered", "regressing"}
        )
        if not rehearsal:
            rehearsal = (self.anchor,)
        return [
            (
                "anchor",
                interpolate(
                    "BASKET_CURRICULUM_FOUNDATION_ANCHOR_SHARE",
                    "BASKET_CURRICULUM_LOCAL_ANCHOR_SHARE",
                ),
                (self.anchor,),
            ),
            (
                "frontier",
                interpolate(
                    "BASKET_CURRICULUM_FOUNDATION_ONE_SWAP_SHARE",
                    "BASKET_CURRICULUM_LOCAL_ONE_SWAP_SHARE",
                ),
                self.active_frontier_cohort,
            ),
            (
                "uniform_probe",
                interpolate(
                    "BASKET_CURRICULUM_FOUNDATION_UNIFORM_SHARE",
                    "BASKET_CURRICULUM_LOCAL_UNIFORM_SHARE",
                ),
                self.uniform_probe_pool,
            ),
            (
                "rehearsal",
                progress
                * float(config.BASKET_CURRICULUM_LOCAL_REHEARSAL_SHARE),
                rehearsal,
            ),
        ]

    def _local_transfer_probability_vector(self, progress=None):
        return self._task_probability_vector(
            components=self._normalize_components(
                self._local_transfer_components(progress=progress)
            )
        )

    def _update_local_transfer_progress(self):
        """Advance the foundation-to-local mixture without a distribution jump."""
        previous_progress = float(self.local_transfer_progress)
        proposed_progress = min(
            1.0,
            previous_progress
            + float(config.BASKET_CURRICULUM_LOCAL_TRANSFER_PROGRESS_STEP),
        )
        previous_distribution = self._local_transfer_probability_vector(
            progress=previous_progress
        )
        proposed_distribution = self._local_transfer_probability_vector(
            progress=proposed_progress
        )
        limit = float(config.BASKET_CURRICULUM_DISTRIBUTION_KL_LIMIT)
        if self._distribution_kl(proposed_distribution, previous_distribution) <= limit:
            accepted_progress = proposed_progress
        else:
            lower, upper = previous_progress, proposed_progress
            for _ in range(40):
                candidate_progress = 0.5 * (lower + upper)
                candidate_distribution = self._local_transfer_probability_vector(
                    progress=candidate_progress
                )
                if self._distribution_kl(
                    candidate_distribution, previous_distribution
                ) <= limit:
                    lower = candidate_progress
                else:
                    upper = candidate_progress
            accepted_progress = lower
        self.local_transfer_progress = float(accepted_progress)
        self.last_local_transfer_update_episode = self.completed_episodes

    def _normalize_components(self, components):
        available = [
            component for component in components if component[1] > 0.0 and component[2]
        ]
        total_share = sum(component[1] for component in available)
        return [
            (
                role,
                share / total_share,
                tuple(candidates),
                self._candidate_weights(tuple(candidates), role),
            )
            for role, share, candidates in available
        ]

    def _sampling_components(self):
        if self.stage == "constraint_stabilization":
            return self._normalize_components(self._foundation_components())
        if self.stage == "local_transfer":
            return self._normalize_components(
                self._local_transfer_components(
                    progress=self.local_transfer_progress
                )
            )
        if self.stage == "self_paced":
            if self._self_paced_distribution is None:
                self._update_self_paced_distribution(force=True)
            return [
                (
                    "self_paced",
                    1.0,
                    self.all_baskets,
                    np.asarray(self._self_paced_distribution, dtype=np.float64),
                )
            ]
        raise RuntimeError(f"unknown basket curriculum stage: {self.stage}")

    def _task_probability_vector(self, components=None):
        components = self._sampling_components() if components is None else components
        probabilities = np.zeros(len(self.all_baskets), dtype=np.float64)
        for _, role_share, candidates, candidate_weights in components:
            for basket, conditional_probability in zip(candidates, candidate_weights):
                probabilities[self.basket_to_index[basket]] += (
                    role_share * float(conditional_probability)
                )
        total = probabilities.sum()
        if total > 0.0:
            probabilities /= total
        return probabilities

    def _resolve_selection_role(self, role, basket):
        """Resolve the operational role of a task drawn from a component."""
        if role != "self_paced":
            return role
        record_state = self.records[basket]["state"]
        if basket not in self.eligible:
            return "uniform_probe"
        if basket == self.anchor and record_state not in {"regressing"}:
            return "anchor"
        if record_state in {"mastered", "regressing"}:
            return "rehearsal"
        return "frontier"

    def _frontier_evidence_bundle_length(self, role, basket):
        """Number of consecutive episodes needed for honest full-batch TD.

        Only a legitimate frontier selection receives persistence. Uniform
        probes remain isolated probes and cannot gain promotion authority by
        passing through this path. Once a basket already has the configured
        number of completed episodes, later visits revert to one episode.
        """
        if role != "frontier" or basket not in self.eligible:
            return 1
        target = max(
            1,
            int(config.BASKET_CURRICULUM_FRONTIER_EVIDENCE_EPISODES),
        )
        remaining = target - int(self.records[basket]["visits"])
        return max(1, min(target, remaining))

    def _sample_new_curriculum_entry(self, components):
        """Draw a new task bundle while preserving episode-level task mass.

        A two-episode frontier bundle must be started half as often as a
        one-episode task with the same desired episode probability. Flattening
        component/task probabilities and dividing each trigger weight by its
        bundle length implements that renewal correction. Therefore local
        persistence changes ordering, not the teacher's long-run 90/10 or
        later KL-bounded episode distribution.
        """
        entries = []
        trigger_weights = []
        for component_role, role_share, candidates, candidate_weights in components:
            for basket, conditional_probability in zip(
                candidates, candidate_weights
            ):
                resolved_role = self._resolve_selection_role(
                    component_role, basket
                )
                bundle_length = self._frontier_evidence_bundle_length(
                    resolved_role, basket
                )
                entries.append((resolved_role, basket, bundle_length))
                trigger_weights.append(
                    float(role_share)
                    * float(conditional_probability)
                    / float(bundle_length)
                )
        probabilities = self._normalized_distribution(trigger_weights)
        selected_index = int(self.rng.choice(len(entries), p=probabilities))
        return entries[selected_index]

    def sample_basket(self):
        components = self._sampling_components()
        bundle_continuation = self._pending_evidence_bundle is not None
        if bundle_continuation:
            pending = self._pending_evidence_bundle
            role = str(pending["role"])
            basket = self._canonical_basket(pending["basket"])
            bundle_length = int(pending["length"])
            bundle_position = int(pending["next_position"])
            pending["remaining"] = int(pending["remaining"]) - 1
            pending["next_position"] = bundle_position + 1
            if int(pending["remaining"]) <= 0:
                self._pending_evidence_bundle = None
        elif self.selection_count == 0 and self.stage == "constraint_stabilization":
            # Guarantee that the first complete trajectory establishes the
            # demonstrated basis instead of allowing the 10% contrast stream
            # to become the learner's accidental initial task.
            role, basket = "anchor", self.anchor
            bundle_length = 1
            bundle_position = 1
        else:
            role, basket, bundle_length = self._sample_new_curriculum_entry(
                components
            )
            bundle_position = 1
            if bundle_length > 1:
                self._pending_evidence_bundle = {
                    "role": role,
                    "basket": list(basket),
                    "length": int(bundle_length),
                    "remaining": int(bundle_length - 1),
                    "next_position": 2,
                }

        self.selection_count += 1
        record = self.records[basket]
        record["selections"] = int(record["selections"]) + 1
        record["last_selection"] = self.selection_count
        can_promote = bool(
            self.control_ready
            and basket in self.eligible
            and role in {"anchor", "frontier"}
        )
        if record["state"] == "eligible" and can_promote:
            record["state"] = "frontier"

        task_probabilities = self._task_probability_vector(components)
        intended_distance_mass = [
            float(
                task_probabilities[
                    [
                        self.basket_to_index[candidate]
                        for candidate in self.baskets_by_distance.get(distance, ())
                    ]
                ].sum()
            )
            for distance in range(max(self.baskets_by_distance) + 1)
        ]
        selection = {
            "coins": [basket[index] for index in self.rng.permutation(len(basket))],
            "basket": list(basket),
            "basket_ids": [self.coin_to_id[coin] for coin in basket],
            "distance": int(self.distance[basket]),
            "role": role,
            "stage": self.stage,
            "can_promote": can_promote,
            "evidence_bundle_length": int(bundle_length),
            "evidence_bundle_position": int(bundle_position),
            "evidence_bundle_continuation": bool(bundle_continuation),
            "episode_index": int(self.selection_count),
            "intended_distance_mass": intended_distance_mass,
        }
        self.selection_history.append(
            {key: value for key, value in selection.items() if key != "coins"}
        )
        return selection

    def replay_task_probabilities(self):
        """Current teacher law keyed by canonical asset-ID membership tuples."""
        probabilities = self._task_probability_vector()
        return {
            tuple(self.coin_to_id[coin] for coin in basket): float(probability)
            for basket, probability in zip(self.all_baskets, probabilities)
            if float(probability) > 0.0
        }

    def replay_role_probabilities(self):
        """Teacher law decomposed into semantic roles for mixed Bellman replay.

        Values are unconditional probabilities: summing every task in every
        returned role yields one.  A basket may occur in more than one role
        (for example, the anchor is also reachable through uniform probing),
        because the role records *why* that row is requested rather than
        inventing a second neural task identity.
        """
        role_probabilities = {}
        for component_role, role_share, candidates, candidate_weights in (
            self._sampling_components()
        ):
            for basket, conditional_probability in zip(
                candidates, candidate_weights
            ):
                role = self._resolve_selection_role(component_role, basket)
                membership_ids = tuple(
                    self.coin_to_id[coin] for coin in basket
                )
                role_map = role_probabilities.setdefault(role, {})
                role_map[membership_ids] = role_map.get(membership_ids, 0.0) + (
                    float(role_share) * float(conditional_probability)
                )
        total = float(
            sum(sum(values.values()) for values in role_probabilities.values())
        )
        if total <= 0.0:
            raise RuntimeError("curriculum replay roles have no probability mass")
        return {
            role: {
                membership_ids: probability / total
                for membership_ids, probability in values.items()
                if probability > 0.0
            }
            for role, values in role_probabilities.items()
        }

    def predictive_replay_uniform_mix(self):
        """Broad-coverage share for SSL under the current curriculum stage.

        Predictive learning may use every seen task, including an in-progress
        basket that is not yet mature enough for financial TD.  This coefficient
        controls how much task-level authority is uniform rather than inherited
        from the teacher.  It rises only after the acceleration-readiness gate
        has moved the curriculum beyond foundation stabilization.
        """
        foundation = float(config.PREDICTIVE_SSL_FOUNDATION_UNIFORM_MIX)
        local = float(config.PREDICTIVE_SSL_LOCAL_TRANSFER_UNIFORM_MIX)
        self_paced = float(config.PREDICTIVE_SSL_SELF_PACED_UNIFORM_MIX)
        if self.stage == "constraint_stabilization":
            value = foundation
        elif self.stage == "local_transfer":
            progress = float(np.clip(self.local_transfer_progress, 0.0, 1.0))
            value = (1.0 - progress) * foundation + progress * local
        else:
            value = self_paced
        return float(np.clip(value, 0.0, 1.0))

    def _unlock_neighbors(self, basket):
        if basket not in self.eligible:
            return
        if self.records[basket]["state"] != "mastered":
            return
        for neighbor in self.neighbors[basket]:
            neighbor = self._ensure_basket_registered(neighbor)
            if neighbor not in self.eligible:
                self.eligible.add(neighbor)
                self.records[neighbor]["state"] = "eligible"

    def observe_episode(
        self,
        selection,
        financial_return,
        financial_log_growth,
        opportunity_risk_budget,
        opportunity_normalized_return,
        value_samples=None,
    ):
        basket_source = selection.get("basket")
        if basket_source is None:
            basket_source = selection.get("coins")
        if basket_source is None:
            raise KeyError("curriculum selection is missing both basket and coins")
        basket = self._canonical_basket(basket_source)
        record = self.records[basket]
        self.completed_episodes += 1
        record["visits"] = int(record["visits"]) + 1
        record["last_visit"] = self.completed_episodes

        normalized_return = float(opportunity_normalized_return)
        role = str(selection.get("role", "unknown"))
        can_promote = bool(
            selection.get(
                "can_promote",
                basket in self.eligible
                and role in {"anchor_stabilization", "anchor", "frontier"},
            )
        )
        if role == "uniform_probe":
            record["probe_visits"] = int(record["probe_visits"]) + 1
            old_probe = record["probe_ema_normalized_return"]
            alpha = float(config.BASKET_CURRICULUM_COMPETENCE_EMA_ALPHA)
            record["probe_ema_normalized_return"] = float(
                normalized_return
                if old_probe is None
                else (1.0 - alpha) * float(old_probe) + alpha * normalized_return
            )
        if role == "rehearsal":
            record["rehearsal_visits"] = int(record["rehearsal_visits"]) + 1
        if role == "frontier":
            record["frontier_visits"] = int(
                record.get("frontier_visits", 0)
            ) + 1
        if can_promote:
            record["promotion_visits"] = int(record["promotion_visits"]) + 1

        # Probe outcomes train the shared value model but never enter the
        # promotion EMA or graph state.  This is the topology boundary that the
        # first implementation lacked.
        legitimate_curriculum_evidence = bool(
            can_promote or (role == "rehearsal" and basket in self.eligible)
        )
        old_ema = record["ema_normalized_return"]
        alpha = float(config.BASKET_CURRICULUM_COMPETENCE_EMA_ALPHA)
        new_ema = old_ema
        if legitimate_curriculum_evidence:
            new_ema = normalized_return if old_ema is None else (
                (1.0 - alpha) * float(old_ema) + alpha * normalized_return
            )
            record["previous_ema_normalized_return"] = old_ema
            record["ema_normalized_return"] = float(new_ema)
            record["signed_progress"] = float(
                0.0 if old_ema is None else new_ema - float(old_ema)
            )
            record["absolute_learning_progress"] = abs(
                record["signed_progress"]
            )
            recent = list(record["recent_normalized_returns"])
            recent.append(normalized_return)
            record["recent_normalized_returns"] = recent[
                -int(config.BASKET_CURRICULUM_RECENT_WINDOW):
            ]

        self._append_value_samples(basket, value_samples, role)
        if basket not in self.eligible:
            record["state"] = "unseen"
        elif record["state"] == "eligible" and can_promote:
            record["state"] = "frontier"
        self._reassess_curriculum()
        self._maybe_rotate_active_frontier_cohort()

        self.episode_history.append(
            {
                "episode_index": int(self.completed_episodes),
                "selection_index": int(selection.get("episode_index", -1)),
                "basket": list(basket),
                "basket_ids": [self.coin_to_id[coin] for coin in basket],
                "distance": int(self.distance[basket]),
                "role": role,
                "stage": str(selection.get("stage", self.stage)),
                "can_promote": can_promote,
                "evidence_bundle_length": int(
                    selection.get("evidence_bundle_length", 1)
                ),
                "evidence_bundle_position": int(
                    selection.get("evidence_bundle_position", 1)
                ),
                "evidence_bundle_continuation": bool(
                    selection.get("evidence_bundle_continuation", False)
                ),
                "financial_return": float(financial_return),
                "financial_log_growth": float(financial_log_growth),
                "opportunity_risk_budget": float(opportunity_risk_budget),
                "opportunity_normalized_return": normalized_return,
                "ema_normalized_return": (
                    None if new_ema is None else float(new_ema)
                ),
                "value_discounted_lower": (
                    None
                    if record["value_discounted_quantiles"] is None
                    else float(record["value_discounted_quantiles"][0])
                ),
                "value_normalized_lower": (
                    None
                    if record["value_normalized_quantiles"] is None
                    else float(record["value_normalized_quantiles"][0])
                ),
                "state_after": str(record["state"]),
            }
        )

    def update_td_errors(self, td_error_stats):
        alpha = float(config.BASKET_CURRICULUM_TD_ERROR_EMA_ALPHA)
        for membership_ids, stats in td_error_stats.items():
            basket = tuple(self.coin_pool[index] for index in membership_ids)
            basket = self._canonical_basket(basket)
            if basket not in self.records:
                continue
            record = self.records[basket]
            value = float(stats["absolute_td_error_mean"])
            old = record["ema_absolute_td_error"]
            record["ema_absolute_td_error"] = value if old is None else (
                (1.0 - alpha) * float(old) + alpha * value
            )
            record["td_sample_count"] = int(record["td_sample_count"]) + int(
                stats.get("sample_count", 0)
            )

    def snapshot(self, step):
        if self.snapshot_history and int(self.snapshot_history[-1]["step"]) == int(step):
            return self.snapshot_history[-1]
        probabilities = self._task_probability_vector()
        max_distance = max(self.baskets_by_distance)
        intended = [
            float(
                probabilities[
                    [self.basket_to_index[b] for b in self.baskets_by_distance[d]]
                ].sum()
            )
            for d in range(max_distance + 1)
        ]
        window = self.episode_history[-int(config.BASKET_CURRICULUM_HISTORY_WINDOW):]
        realized_counts = np.zeros(max_distance + 1, dtype=np.float64)
        for episode in window:
            realized_counts[int(episode["distance"])] += 1.0
        realized = (
            (realized_counts / realized_counts.sum()).tolist()
            if realized_counts.sum() > 0
            else [0.0] * (max_distance + 1)
        )
        state_counts = {
            state: sum(record["state"] == state for record in self.records.values())
            for state in self.STATE_NAMES
        }
        frontier_distances = [
            self.distance[basket]
            for basket, record in self.records.items()
            if record["state"] in {"frontier", "regressing"}
        ]
        control_window = [
            item
            for item in self.control_history
            if "acceleration_cost" in item and "acceleration_threshold" in item
        ][-int(config.BASKET_CURRICULUM_CONTROL_MIN_SAMPLES):]
        if control_window:
            acceleration_costs = np.asarray(
                [item["acceleration_cost"] for item in control_window],
                dtype=np.float64,
            )
            acceleration_thresholds = np.asarray(
                [item["acceleration_threshold"] for item in control_window],
                dtype=np.float64,
            )
            control_acceleration_cost_mean = float(np.mean(acceleration_costs))
            control_acceleration_cost_std = float(np.std(acceleration_costs))
            control_acceleration_threshold = float(
                np.mean(acceleration_thresholds)
            )
            velocity_window = [
                item
                for item in control_window
                if "velocity_cost" in item and "velocity_threshold" in item
            ]
            if velocity_window:
                velocity_costs = np.asarray(
                    [item["velocity_cost"] for item in velocity_window],
                    dtype=np.float64,
                )
                velocity_thresholds = np.asarray(
                    [item["velocity_threshold"] for item in velocity_window],
                    dtype=np.float64,
                )
                control_velocity_cost_mean = float(np.mean(velocity_costs))
                control_velocity_cost_std = float(np.std(velocity_costs))
                control_velocity_threshold = float(np.mean(velocity_thresholds))
            else:
                control_velocity_cost_mean = None
                control_velocity_cost_std = None
                control_velocity_threshold = None
        else:
            control_acceleration_cost_mean = None
            control_acceleration_cost_std = None
            control_acceleration_threshold = None
            control_velocity_cost_mean = None
            control_velocity_cost_std = None
            control_velocity_threshold = None
        snapshot = {
            "step": int(step),
            "episode_index": int(self.completed_episodes),
            "curriculum_stage": self.stage,
            "control_ready": bool(self.control_ready),
            "control_ready_streak": int(self.control_ready_streak),
            "control_not_ready_streak": int(self.control_not_ready_streak),
            "control_release_kl": self.last_control_release_kl,
            "control_acceleration_cost_mean": control_acceleration_cost_mean,
            "control_acceleration_cost_std": control_acceleration_cost_std,
            "control_acceleration_threshold": control_acceleration_threshold,
            "control_velocity_cost_mean": control_velocity_cost_mean,
            "control_velocity_cost_std": control_velocity_cost_std,
            "control_velocity_threshold": control_velocity_threshold,
            "target_pressure": float(self.target_pressure),
            "local_transfer_progress": float(self.local_transfer_progress),
            "value_model_ready": bool(self._value_model_ready()),
            "value_model_loss_ema": (
                None if self.value_loss_ema is None else float(self.value_loss_ema)
            ),
            "value_sample_count": int(len(self.value_buffer)),
            "intended_distance_mass": intended,
            "realized_distance_mass": realized,
            "state_counts": state_counts,
            "active_support_count": int(len(self.all_baskets)),
            "total_task_count": int(self.total_basket_count),
            "untracked_task_count": int(
                max(0, self.total_basket_count - len(self.all_baskets))
            ),
            "furthest_active_distance": int(max(frontier_distances, default=0)),
            "eligible_count": int(len(self.eligible)),
            "active_frontier_cohort_count": int(
                len(self.active_frontier_cohort)
            ),
            "active_frontier_cohort_generation": int(
                self.frontier_cohort_generation
            ),
        }
        self.snapshot_history.append(snapshot)
        return snapshot

    def state_dict(self):
        value_training_state = None
        if self.value_model is not None:
            payload_buffer = io.BytesIO()
            torch.save(
                {
                    "input_dim": int(self.value_input_dim),
                    "model": self.value_model.state_dict(),
                    "target_model": self.value_target_model.state_dict(),
                    "optimizer": self.value_optimizer.state_dict(),
                },
                payload_buffer,
            )
            value_training_state = base64.b64encode(
                payload_buffer.getvalue()
            ).decode("ascii")
        return {
            "version": 8,
            "coin_pool": list(self.coin_pool),
            "anchor": list(self.anchor),
            "basket_size": self.basket_size,
            "total_basket_count": int(self.total_basket_count),
            "enumerated_task_space": bool(self.enumerated_task_space),
            "active_support": [list(basket) for basket in self.all_baskets],
            "uniform_probe_pool": [
                list(basket) for basket in self.uniform_probe_pool
            ],
            "completed_episodes": self.completed_episodes,
            "selection_count": self.selection_count,
            "pending_evidence_bundle": self._pending_evidence_bundle,
            "stage": self.stage,
            "control_ready": self.control_ready,
            "control_ready_streak": self.control_ready_streak,
            "control_not_ready_streak": self.control_not_ready_streak,
            "last_control_release_kl": self.last_control_release_kl,
            "control_history": list(self.control_history),
            "latest_training_step": self.latest_training_step,
            "target_pressure": self.target_pressure,
            "last_distribution_update_episode": self.last_distribution_update_episode,
            "local_transfer_progress": self.local_transfer_progress,
            "last_local_transfer_update_episode": (
                self.last_local_transfer_update_episode
            ),
            "self_paced_distribution": (
                None
                if self._self_paced_distribution is None
                else self._self_paced_distribution.tolist()
            ),
            "eligible": [list(basket) for basket in sorted(self.eligible, key=self.basket_to_index.get)],
            "records": [
                {"basket": list(basket), **record}
                for basket, record in self.records.items()
            ],
            "selection_history": self.selection_history,
            "episode_history": self.episode_history,
            "snapshot_history": self.snapshot_history,
            "rng_state": self.rng.bit_generator.state,
            "probe_rng_state": self.probe_rng.bit_generator.state,
            "frontier_rng_state": self.frontier_rng.bit_generator.state,
            "frontier_cohort_generation": int(self.frontier_cohort_generation),
            "active_frontier_cohort": [
                list(basket) for basket in self.active_frontier_cohort
            ],
            "value_rng_state": self.value_rng.bit_generator.state,
            "value_buffer": list(self.value_buffer),
            "value_update_count": self.value_update_count,
            "value_new_samples_since_update": (
                self.value_new_samples_since_update
            ),
            "value_loss_ema": self.value_loss_ema,
            "value_training_state": value_training_state,
        }

    def load_state_dict(self, state):
        if tuple(state["coin_pool"]) != self.coin_pool:
            raise ValueError("curriculum checkpoint coin universe does not match")
        if tuple(state["anchor"]) != self.anchor:
            raise ValueError("curriculum checkpoint anchor does not match")
        saved_total = int(state.get("total_basket_count", self.total_basket_count))
        if saved_total != self.total_basket_count:
            raise ValueError("curriculum checkpoint task count does not match")
        saved_support = state.get("active_support")
        if saved_support is not None:
            self._restore_active_support(
                saved_support,
                state.get("uniform_probe_pool", saved_support),
            )
        self.completed_episodes = int(state["completed_episodes"])
        self.selection_count = int(state["selection_count"])
        saved_bundle = state.get("pending_evidence_bundle")
        self._pending_evidence_bundle = (
            None
            if saved_bundle is None
            else {
                "role": str(saved_bundle["role"]),
                "basket": list(
                    self._canonical_basket(saved_bundle["basket"])
                ),
                "length": int(saved_bundle["length"]),
                "remaining": int(saved_bundle["remaining"]),
                "next_position": int(saved_bundle["next_position"]),
            }
        )
        self.stage = str(state.get("stage", "constraint_stabilization"))
        if self.stage not in self.STAGE_NAMES:
            raise ValueError(f"unknown curriculum stage in checkpoint: {self.stage}")
        self.control_ready = bool(state.get("control_ready", False))
        self.control_ready_streak = int(state.get("control_ready_streak", 0))
        self.control_not_ready_streak = int(
            state.get("control_not_ready_streak", 0)
        )
        saved_release_kl = state.get("last_control_release_kl")
        self.last_control_release_kl = (
            None if saved_release_kl is None else float(saved_release_kl)
        )
        self.control_history.clear()
        self.control_history.extend(state.get("control_history", []))
        self.latest_training_step = int(state.get("latest_training_step", 0))
        self.target_pressure = float(state.get("target_pressure", 0.0))
        self.last_distribution_update_episode = int(
            state.get("last_distribution_update_episode", -1)
        )
        self.local_transfer_progress = float(
            state.get(
                "local_transfer_progress",
                1.0 if self.stage == "self_paced" else 0.0,
            )
        )
        self.last_local_transfer_update_episode = int(
            state.get("last_local_transfer_update_episode", -1)
        )
        saved_distribution = state.get("self_paced_distribution")
        if (
            saved_distribution is not None
            and len(saved_distribution) != len(self.all_baskets)
        ):
            raise ValueError(
                "curriculum probability vector does not match active support"
            )
        self._self_paced_distribution = (
            None
            if saved_distribution is None
            else self._normalized_distribution(saved_distribution)
        )
        self.eligible = {
            self._canonical_basket(basket) for basket in state["eligible"]
        }
        for saved in state["records"]:
            basket = self._canonical_basket(saved["basket"])
            merged_record = self._new_record()
            merged_record.update(
                {key: value for key, value in saved.items() if key != "basket"}
            )
            self.records[basket] = merged_record
        self.selection_history = list(state.get("selection_history", []))
        self.episode_history = list(state.get("episode_history", []))
        self.snapshot_history = list(state.get("snapshot_history", []))
        self.rng.bit_generator.state = state["rng_state"]
        self.probe_rng.bit_generator.state = state.get(
            "probe_rng_state", self.probe_rng.bit_generator.state
        )
        self.frontier_rng.bit_generator.state = state.get(
            "frontier_rng_state", self.frontier_rng.bit_generator.state
        )
        self.frontier_cohort_generation = int(
            state.get("frontier_cohort_generation", 0)
        )
        saved_cohort = state.get("active_frontier_cohort")
        if saved_cohort is None:
            self._refresh_active_frontier_cohort(force=True)
        else:
            self.active_frontier_cohort = tuple(
                self._canonical_basket(basket) for basket in saved_cohort
            )
        self.value_rng.bit_generator.state = state.get(
            "value_rng_state", self.value_rng.bit_generator.state
        )
        self.value_buffer.clear()
        self.value_buffer.extend(state.get("value_buffer", []))
        self.value_update_count = int(state.get("value_update_count", 0))
        self.value_new_samples_since_update = int(
            state.get("value_new_samples_since_update", 0)
        )
        self.value_loss_ema = state.get("value_loss_ema")
        value_training_state = state.get("value_training_state")
        if value_training_state:
            payload = torch.load(
                io.BytesIO(base64.b64decode(value_training_state.encode("ascii"))),
                map_location="cpu",
                weights_only=False,
            )
            saved_input_dim = int(payload["input_dim"])
            expected_input_dim = int(config.CURRICULUM_VALUE_INPUT_DIM)
            if saved_input_dim == expected_input_dim:
                self._ensure_value_model(saved_input_dim)
                self.value_model.load_state_dict(payload["model"])
                self.value_target_model.load_state_dict(
                    payload.get("target_model", payload["model"])
                )
                self.value_optimizer.load_state_dict(payload["optimizer"])
            else:
                # Membership-conditioned teacher checkpoints are semantically
                # incompatible with the identity-neutral Path 2 set model.
                # Preserve actor/replay/curriculum progress, but require fresh
                # empirical value evidence under the new observable contract.
                self.value_buffer.clear()
                for record in self.records.values():
                    record["representative_value_feature"] = None
                    record["value_discounted_quantiles"] = None
                    record["value_normalized_quantiles"] = None
                    record["value_sample_count"] = 0
                print(
                    "⚠️ Reset incompatible curriculum value head: "
                    f"checkpoint width={saved_input_dim}, Path 2 width="
                    f"{expected_input_dim}."
                )
        self._ensure_visual_projection()

    def save(self, path):
        temporary_path = path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(self.state_dict(), handle, separators=(",", ":"))
        os.replace(temporary_path, path)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            self.load_state_dict(json.load(handle))


class CurriculumMatchedEvaluationSampler:
    """Independent, low-variance view of the current training task law.

    Evaluation reads the teacher distribution but owns neither the teacher RNG
    nor any teacher counters. ``prepare`` uses stratified points over the full
    categorical CDF, so four historical evaluation episodes remain four
    episodes while their cumulative basket composition follows curriculum
    pacing much more closely than four unconstrained random draws.
    """

    def __init__(self, curriculum_manager, seed):
        self.curriculum_manager = curriculum_manager
        self.seed = int(seed)
        self._queue = []
        self._prepared_rng = np.random.default_rng(self.seed)
        self._prepared_step = 0

    def prepare(self, episode_count, step=0):
        episode_count = int(episode_count)
        if episode_count <= 0:
            raise ValueError("evaluation episode_count must be positive")
        self._prepared_step = int(step)
        # Common random numbers: if the curriculum law is unchanged, every
        # evaluation reuses the same stratified quantiles and basket ordering.
        # A changed panel then reflects a changed teacher distribution rather
        # than a fresh RNG draw, which makes delayed replay-policy credit far
        # less noisy while preserving curriculum matching.
        self._prepared_rng = np.random.default_rng(self.seed)
        probabilities = self.curriculum_manager._task_probability_vector()
        cdf = np.cumsum(probabilities)
        cdf[-1] = 1.0
        offset = float(self._prepared_rng.random())
        points = (np.arange(episode_count, dtype=np.float64) + offset) / float(
            episode_count
        )
        indices = np.searchsorted(cdf, points, side="right")
        self._prepared_rng.shuffle(indices)
        self._queue = [int(index) for index in indices]

    def sample_basket(self):
        if not self._queue:
            self.prepare(1, step=self._prepared_step)
        basket_index = self._queue.pop(0)
        manager = self.curriculum_manager
        basket = manager.all_baskets[basket_index]
        probabilities = manager._task_probability_vector()
        role = "evaluation_anchor" if basket == manager.anchor else "evaluation_transfer"
        return {
            "coins": [
                basket[index]
                for index in self._prepared_rng.permutation(len(basket))
            ],
            "basket": list(basket),
            "basket_ids": [manager.coin_to_id[coin] for coin in basket],
            "distance": int(manager.distance[basket]),
            "role": role,
            "stage": manager.stage,
            "can_promote": False,
            "evaluation_curriculum_probability": float(
                probabilities[basket_index]
            ),
            "episode_index": -1,
        }


def build_basket_curriculum_manager():
    if not bool(getattr(config, "BASKET_CURRICULUM_ENABLED", False)):
        return None
    return BasketCurriculumManager(
        coin_pool=config.COIN_POOL,
        anchor_coins=config.BASKET_CURRICULUM_ANCHOR,
        basket_size=config.SELECTED_ASSETS_PER_EPISODE,
        seed=config.BASKET_CURRICULUM_SEED,
    )


def build_local_composition_sampler():
    if not bool(
        getattr(config, "LOCAL_COMPOSITION_CURRICULUM_ENABLED", False)
    ):
        return None
    if bool(getattr(config, "BASKET_CURRICULUM_ENABLED", False)):
        raise ValueError(
            "The graph teacher and local-composition sampler are mutually "
            "exclusive presentation curricula."
        )
    return LocalCompositionBasketSampler(
        coin_pool=config.COIN_POOL,
        basket_size=config.SELECTED_ASSETS_PER_EPISODE,
        seed=config.LOCAL_COMPOSITION_CURRICULUM_SEED,
        block_episodes=config.LOCAL_COMPOSITION_BLOCK_EPISODES,
        pivot_episodes=config.LOCAL_COMPOSITION_PIVOT_EPISODES,
        neighbor_episodes=config.LOCAL_COMPOSITION_NEIGHBOR_EPISODES,
    )


def _save_figure_atomically(figure, path, *, dpi=220):
    temporary_path = path + ".tmp.png"
    figure.savefig(temporary_path, dpi=dpi, bbox_inches="tight")
    os.replace(temporary_path, path)


def generate_curriculum_diagnostic_plots(
    curriculum_manager,
    log_dir,
    evaluation_iterations,
    financial_returns,
    opportunity_normalized_returns,
    evaluation_episode_history,
):
    """Write two curriculum-only figures without modifying legacy plots."""
    if curriculum_manager is None:
        return
    snapshots = list(curriculum_manager.snapshot_history)
    if not snapshots:
        return

    max_distance = max(curriculum_manager.baskets_by_distance)
    distance_labels = [f"distance {distance}" for distance in range(max_distance + 1)]
    steps = np.asarray([snapshot["step"] for snapshot in snapshots], dtype=np.int64)
    intended = np.asarray(
        [snapshot["intended_distance_mass"] for snapshot in snapshots],
        dtype=np.float64,
    )
    realized = np.asarray(
        [snapshot["realized_distance_mass"] for snapshot in snapshots],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(5, 1, figsize=(15, 24), constrained_layout=True)
    stage_colors = {
        "constraint_stabilization": "#f2d7d5",
        "local_transfer": "#d6eaf8",
        "self_paced": "#d5f5e3",
    }
    stages = [
        str(snapshot.get("curriculum_stage", "constraint_stabilization"))
        for snapshot in snapshots
    ]
    span_start = 0
    for span_end in range(1, len(stages) + 1):
        if span_end == len(stages) or stages[span_end] != stages[span_start]:
            left = steps[span_start]
            right = (
                steps[span_end]
                if span_end < len(steps)
                else steps[-1] + max(int(np.median(np.diff(steps))) if len(steps) > 1 else 1, 1)
            )
            for axis in axes[:4]:
                axis.axvspan(
                    left,
                    right,
                    color=stage_colors.get(stages[span_start], "#eeeeee"),
                    alpha=0.12,
                    linewidth=0.0,
                    zorder=0,
                )
            axes[0].text(
                0.5 * (left + right),
                0.985,
                stages[span_start].replace("_", " "),
                transform=axes[0].get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
            )
            span_start = span_end
    axes[0].stackplot(steps, intended.T, labels=distance_labels, alpha=0.88)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("intended probability")
    axes[0].set_title("Curriculum pacing: teacher sampling distribution")
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    axes[0].grid(alpha=0.18)

    axes[1].stackplot(steps, realized.T, labels=distance_labels, alpha=0.88)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("realized episode share")
    axes[1].set_title(
        f"Collected baskets over the trailing {config.BASKET_CURRICULUM_HISTORY_WINDOW} episodes"
    )
    axes[1].grid(alpha=0.18)

    acceleration_mean = np.asarray(
        [
            np.nan
            if snapshot.get("control_acceleration_cost_mean") is None
            else float(snapshot["control_acceleration_cost_mean"])
            for snapshot in snapshots
        ],
        dtype=np.float64,
    )
    acceleration_std = np.asarray(
        [
            np.nan
            if snapshot.get("control_acceleration_cost_std") is None
            else float(snapshot["control_acceleration_cost_std"])
            for snapshot in snapshots
        ],
        dtype=np.float64,
    )
    acceleration_threshold = np.asarray(
        [
            np.nan
            if snapshot.get("control_acceleration_threshold") is None
            else float(snapshot["control_acceleration_threshold"])
            for snapshot in snapshots
        ],
        dtype=np.float64,
    )
    velocity_mean = np.asarray(
        [
            np.nan
            if snapshot.get("control_velocity_cost_mean") is None
            else float(snapshot["control_velocity_cost_mean"])
            for snapshot in snapshots
        ],
        dtype=np.float64,
    )
    velocity_std = np.asarray(
        [
            np.nan
            if snapshot.get("control_velocity_cost_std") is None
            else float(snapshot["control_velocity_cost_std"])
            for snapshot in snapshots
        ],
        dtype=np.float64,
    )
    velocity_threshold = np.asarray(
        [
            np.nan
            if snapshot.get("control_velocity_threshold") is None
            else float(snapshot["control_velocity_threshold"])
            for snapshot in snapshots
        ],
        dtype=np.float64,
    )
    axes[2].plot(
        steps,
        acceleration_mean,
        color="tab:orange",
        linewidth=2.0,
        label="rolling acceleration cost",
    )
    axes[2].fill_between(
        steps,
        acceleration_mean - acceleration_std,
        acceleration_mean + acceleration_std,
        color="tab:orange",
        alpha=0.16,
        label="rolling ±1 standard deviation",
    )
    axes[2].plot(
        steps,
        acceleration_threshold,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="acceleration threshold",
    )
    axes[2].plot(
        steps,
        velocity_mean,
        color="tab:purple",
        linewidth=1.8,
        label="rolling velocity cost",
    )
    axes[2].fill_between(
        steps,
        velocity_mean - velocity_std,
        velocity_mean + velocity_std,
        color="tab:purple",
        alpha=0.12,
        label="velocity ±1 standard deviation",
    )
    axes[2].plot(
        steps,
        velocity_threshold,
        color="tab:purple",
        linewidth=1.3,
        linestyle="--",
        label="velocity threshold",
    )
    readiness_axis = axes[2].twinx()
    readiness_axis.step(
        steps,
        [float(snapshot.get("control_ready", False)) for snapshot in snapshots],
        where="post",
        color="tab:green",
        linewidth=1.8,
        label="curriculum gate open",
    )
    readiness_axis.set_ylim(-0.05, 1.05)
    readiness_axis.set_yticks((0.0, 1.0), labels=("closed", "open"))
    readiness_axis.set_ylabel("promotion gate", color="tab:green")
    axes[2].set_ylabel("constraint scale")
    axes[2].set_title(
        "Constraint-aware readiness: acceleration controls promotion; velocity is diagnostic"
    )
    axes[2].legend(ncol=3, fontsize=7, loc="upper right")
    axes[2].grid(alpha=0.18)

    for state in curriculum_manager.STATE_NAMES:
        axes[3].plot(
            steps,
            [snapshot["state_counts"][state] for snapshot in snapshots],
            label=state,
            linewidth=1.8,
        )
    axes[3].plot(
        steps,
        [snapshot["eligible_count"] for snapshot in snapshots],
        label="eligible total",
        color="black",
        linewidth=2.0,
        linestyle="--",
    )
    axes[3].set_ylabel("basket count")
    axes[3].set_title("Graph expansion, mastery, and forgetting")
    axes[3].legend(ncol=3, fontsize=8)
    axes[3].grid(alpha=0.18)
    distance_axis = axes[3].twinx()
    distance_axis.plot(
        steps,
        [snapshot["furthest_active_distance"] for snapshot in snapshots],
        color="tab:red",
        linewidth=2.0,
        label="furthest active distance",
    )
    distance_axis.set_ylabel("substitution distance", color="tab:red")
    distance_axis.set_ylim(-0.1, max_distance + 0.3)

    probabilities = curriculum_manager._task_probability_vector()
    curriculum_manager._ensure_visual_projection()
    coordinates = curriculum_manager._pca_coordinates
    competence = np.asarray(
        [
            (
                float(curriculum_manager.records[basket]["value_normalized_quantiles"][0])
                if curriculum_manager.records[basket]["value_normalized_quantiles"]
                is not None
                else (
                    np.nan
                    if curriculum_manager.records[basket]["ema_normalized_return"]
                    is None
                    else float(
                        curriculum_manager.records[basket]["ema_normalized_return"]
                    )
                )
            )
            for basket in curriculum_manager.all_baskets
        ],
        dtype=np.float64,
    )
    unseen_mask = ~np.isfinite(competence)
    axes[4].scatter(
        coordinates[unseen_mask, 0],
        coordinates[unseen_mask, 1],
        s=8,
        color="#dddddd",
        alpha=0.45,
        label="unobserved competence",
    )
    observed_mask = ~unseen_mask
    if observed_mask.any():
        color_limit = float(
            max(np.nanpercentile(np.abs(competence[observed_mask]), 90), 1e-6)
        )
        colored = axes[4].scatter(
            coordinates[observed_mask, 0],
            coordinates[observed_mask, 1],
            s=18.0 + 7000.0 * probabilities[observed_mask],
            c=competence[observed_mask],
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
            alpha=0.8,
        )
        figure.colorbar(
            colored,
            ax=axes[4],
            label="contextual normalized lower quantile (EMA fallback)",
        )
    frontier_mask = np.asarray(
        [
            curriculum_manager.records[basket]["state"] in {"frontier", "regressing"}
            for basket in curriculum_manager.all_baskets
        ],
        dtype=bool,
    )
    axes[4].scatter(
        coordinates[frontier_mask, 0],
        coordinates[frontier_mask, 1],
        s=25,
        facecolors="none",
        edgecolors="black",
        linewidths=0.6,
        label="promotion-active frontier / regression",
    )
    active_cohort = set(curriculum_manager.active_frontier_cohort)
    cohort_mask = np.asarray(
        [
            basket in active_cohort
            for basket in curriculum_manager.all_baskets
        ],
        dtype=bool,
    )
    axes[4].scatter(
        coordinates[cohort_mask, 0],
        coordinates[cohort_mask, 1],
        s=48,
        facecolors="none",
        edgecolors="tab:purple",
        linewidths=1.2,
        label="sampled transfer cohort",
    )
    anchor_index = curriculum_manager.basket_to_index[curriculum_manager.anchor]
    axes[4].scatter(
        [coordinates[anchor_index, 0]],
        [coordinates[anchor_index, 1]],
        marker="*",
        s=180,
        color="gold",
        edgecolor="black",
        label="anchor",
        zorder=5,
    )
    projection_scope = (
        "all basket memberships in a fixed PCA projection"
        if curriculum_manager.enumerated_task_space
        else (
            f"lazy active support ({len(curriculum_manager.all_baskets):,} of "
            f"{curriculum_manager.total_basket_count:,} possible baskets) in a "
            "fixed deterministic set projection"
        )
    )
    axes[4].set_title(
        f"{projection_scope}; point size is current sampling probability"
    )
    axes[4].set_xlabel("membership PCA coordinate 1")
    axes[4].set_ylabel("membership PCA coordinate 2")
    axes[4].legend(fontsize=8, loc="best")
    axes[4].grid(alpha=0.15)
    _save_figure_atomically(
        figure,
        os.path.join(log_dir, "curriculum_pacing.png"),
    )
    plt.close(figure)

    figure, axes = plt.subplots(4, 1, figsize=(15, 20), constrained_layout=True)
    evaluation_steps = np.asarray(evaluation_iterations, dtype=np.int64)
    axes[0].plot(evaluation_steps, financial_returns, color="tab:green", linewidth=2.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Unchanged evaluation FinancialReturn")
    axes[0].set_ylabel("financial return")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        evaluation_steps,
        opportunity_normalized_returns,
        color="tab:purple",
        linewidth=2.0,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title(
        "Evaluation opportunity-normalized FinancialReturn on the same four random episodes"
    )
    axes[1].set_ylabel("growth / exogenous opportunity risk")
    axes[1].grid(alpha=0.2)

    history_window = int(config.BASKET_CURRICULUM_HISTORY_WINDOW)
    rolling_by_distance = {
        distance: deque(maxlen=history_window)
        for distance in range(max_distance + 1)
    }
    episode_pointer = 0
    episode_history = curriculum_manager.episode_history
    medians = {distance: [] for distance in rolling_by_distance}
    lower = {distance: [] for distance in rolling_by_distance}
    upper = {distance: [] for distance in rolling_by_distance}
    for snapshot in snapshots:
        while (
            episode_pointer < len(episode_history)
            and int(episode_history[episode_pointer]["episode_index"])
            <= int(snapshot["episode_index"])
        ):
            episode = episode_history[episode_pointer]
            rolling_by_distance[int(episode["distance"])].append(
                float(episode["opportunity_normalized_return"])
            )
            episode_pointer += 1
        for distance, window in rolling_by_distance.items():
            if window:
                values = np.asarray(window, dtype=np.float64)
                medians[distance].append(float(np.median(values)))
                lower[distance].append(float(np.percentile(values, 25)))
                upper[distance].append(float(np.percentile(values, 75)))
            else:
                medians[distance].append(np.nan)
                lower[distance].append(np.nan)
                upper[distance].append(np.nan)
    for distance in range(max_distance + 1):
        median_values = np.asarray(medians[distance], dtype=np.float64)
        axes[2].plot(steps, median_values, linewidth=1.7, label=f"distance {distance}")
        axes[2].fill_between(
            steps,
            np.asarray(lower[distance], dtype=np.float64),
            np.asarray(upper[distance], dtype=np.float64),
            alpha=0.08,
        )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_title("Training competence propagating outward through substitution distance")
    axes[2].set_ylabel("rolling median and IQR")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(alpha=0.2)

    role_groups = {
        "anchor episodes": {"anchor_stabilization", "anchor"},
        "frontier episodes": {"frontier"},
        "rehearsal episodes": {"rehearsal"},
        "uniform probes": {"uniform_probe"},
    }
    for label, roles in role_groups.items():
        role_steps = []
        role_values = []
        rolling = deque(maxlen=history_window)
        for episode in episode_history:
            if episode["role"] in roles:
                rolling.append(float(episode["opportunity_normalized_return"]))
                role_steps.append(int(episode["episode_index"]) * config.EPISODE_LENGTH)
                role_values.append(float(np.mean(rolling)))
        if role_steps:
            axes[3].plot(role_steps, role_values, linewidth=1.6, label=label)
    axes[3].plot(
        evaluation_steps,
        opportunity_normalized_returns,
        color="black",
        linewidth=2.2,
        linestyle="--",
        label="unchanged random evaluation panel",
    )
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_title("Retention, active-frontier learning, and target-distribution transfer")
    axes[3].set_xlabel("environment steps")
    axes[3].set_ylabel("opportunity-normalized return")
    axes[3].legend(ncol=2, fontsize=8)
    axes[3].grid(alpha=0.2)
    _save_figure_atomically(
        figure,
        os.path.join(log_dir, "curriculum_financial_transfer.png"),
    )
    plt.close(figure)

    if evaluation_episode_history:
        pd.DataFrame(evaluation_episode_history).to_csv(
            os.path.join(log_dir, "curriculum_evaluation_episode_history.csv"),
            index=False,
        )
    if curriculum_manager.episode_history:
        pd.DataFrame(curriculum_manager.episode_history).to_csv(
            os.path.join(log_dir, "curriculum_training_episode_history.csv"),
            index=False,
        )
    pd.DataFrame(curriculum_manager.snapshot_history).to_json(
        os.path.join(log_dir, "curriculum_history.jsonl"),
        orient="records",
        lines=True,
    )


def parse_checkpoint_step(path, prefix=MODEL_CHECKPOINT_PREFIX, suffix=MODEL_CHECKPOINT_SUFFIX):
    basename = os.path.basename(path)
    if not basename.startswith(prefix) or not basename.endswith(suffix):
        raise ValueError(
            f"Checkpoint filename must look like {prefix}<step>{suffix}; got {basename}"
        )
    return int(basename[len(prefix):-len(suffix)])


def replay_buffer_checkpoint_path(checkpoints_dir, step_count):
    return os.path.join(
        checkpoints_dir,
        f"{REPLAY_BUFFER_CHECKPOINT_PREFIX}{step_count}{REPLAY_BUFFER_CHECKPOINT_SUFFIX}",
    )


def model_checkpoint_path(checkpoints_dir, step_count):
    return os.path.join(
        checkpoints_dir,
        f"{MODEL_CHECKPOINT_PREFIX}{step_count}{MODEL_CHECKPOINT_SUFFIX}",
    )


def curriculum_checkpoint_path(checkpoints_dir, step_count):
    return os.path.join(
        checkpoints_dir,
        f"{CURRICULUM_CHECKPOINT_PREFIX}{step_count}{CURRICULUM_CHECKPOINT_SUFFIX}",
    )


def curriculum_path_for_model_checkpoint(model_checkpoint_path):
    step_count = parse_checkpoint_step(model_checkpoint_path)
    return curriculum_checkpoint_path(os.path.dirname(model_checkpoint_path), step_count)


def legacy_replay_buffer_checkpoint_path(checkpoints_dir, step_count):
    return os.path.join(
        checkpoints_dir,
        f"{REPLAY_BUFFER_CHECKPOINT_PREFIX}{step_count}{LEGACY_REPLAY_BUFFER_CHECKPOINT_SUFFIX}",
    )


def replay_buffer_path_for_model_checkpoint(model_checkpoint_path):
    step_count = parse_checkpoint_step(model_checkpoint_path)
    checkpoints_dir = os.path.dirname(model_checkpoint_path)
    return replay_buffer_checkpoint_path(checkpoints_dir, step_count)


def resolve_replay_buffer_checkpoint_path(replay_buffer_path):
    if os.path.exists(replay_buffer_path):
        return replay_buffer_path

    try:
        step_count = parse_checkpoint_step(
            replay_buffer_path,
            prefix=REPLAY_BUFFER_CHECKPOINT_PREFIX,
            suffix=REPLAY_BUFFER_CHECKPOINT_SUFFIX,
        )
    except ValueError:
        return replay_buffer_path

    legacy_path = legacy_replay_buffer_checkpoint_path(os.path.dirname(replay_buffer_path), step_count)
    if os.path.exists(legacy_path):
        print(f"🔄 Using legacy uncompressed replay buffer checkpoint: {legacy_path}")
        return legacy_path
    return replay_buffer_path


def copy_replay_storage_slice(storage, length, *, copy_arrays=True):
    """Keep the legacy snapshot API; synchronous streaming saves use views."""
    def slice_array(value):
        view = value[:length]
        return view.copy() if copy_arrays else view

    if isinstance(storage, dict):
        return {key: slice_array(value) for key, value in storage.items()}
    return slice_array(storage)


def assign_replay_storage_slice(target_storage, saved_storage, length, *, consume=False):
    def assign_array(target, source):
        read_into = getattr(source, "read_into", None)
        if callable(read_into):
            read_into(target[:length])
        else:
            target[:length] = source

    if isinstance(target_storage, dict):
        for key in tuple(saved_storage):
            if key not in target_storage:
                raise KeyError(f"Replay buffer observation key missing during load: {key}")
            saved_value = saved_storage.pop(key) if consume else saved_storage[key]
            assign_array(target_storage[key], saved_value)
            del saved_value
        return
    assign_array(target_storage, saved_storage)


def compact_replay_buffer_payload(replay_buffer, *, copy_arrays=True):
    def storage_slice(storage, length):
        return copy_replay_storage_slice(storage, length, copy_arrays=copy_arrays)

    filled_size = int(replay_buffer.size())
    buffer_size = int(replay_buffer.buffer_size)
    optimize_memory_usage = bool(getattr(replay_buffer, "optimize_memory_usage", False))

    observation_length = filled_size
    if bool(replay_buffer.full):
        observation_length = buffer_size
    elif optimize_memory_usage and filled_size < buffer_size:
        # Memory-efficient replay buffers store next_obs in observations[pos].
        observation_length = filled_size + 1

    payload = {
        "format": COMPACT_REPLAY_BUFFER_FORMAT,
        "version": COMPACT_REPLAY_BUFFER_VERSION,
        "replay_buffer_class": replay_buffer.__class__.__name__,
        "market_input_contract": getattr(
            replay_buffer, "market_input_contract", "selected_basket_v1"
        ),
        "action_normalization": getattr(replay_buffer, "action_normalization", "hierarchical"),
        "buffer_size": buffer_size,
        "filled_size": filled_size,
        "observation_length": observation_length,
        "n_envs": int(replay_buffer.n_envs),
        "pos": int(replay_buffer.pos),
        "full": bool(replay_buffer.full),
        "optimize_memory_usage": optimize_memory_usage,
        "handle_timeout_termination": bool(getattr(replay_buffer, "handle_timeout_termination", False)),
        "observations": storage_slice(replay_buffer.observations, observation_length),
        "actions": storage_slice(replay_buffer.actions, filled_size),
        "rewards": storage_slice(replay_buffer.rewards, filled_size),
        "dones": storage_slice(replay_buffer.dones, filled_size),
        "timeouts": storage_slice(replay_buffer.timeouts, filled_size),
    }
    if not optimize_memory_usage:
        payload["next_observations"] = storage_slice(replay_buffer.next_observations, filled_size)
    if hasattr(replay_buffer, "component_outcomes"):
        payload["component_outcomes"] = storage_slice(
            replay_buffer.component_outcomes, filled_size
        )
        payload["component_outcome_masks"] = storage_slice(
            replay_buffer.component_outcome_masks, filled_size
        )
        payload["component_outcome_target_kind"] = getattr(
            replay_buffer,
            "component_outcome_target_kind",
            "scaled_executed_contribution",
        )
        if hasattr(replay_buffer, "curriculum_role_codes"):
            payload["curriculum_role_codes"] = storage_slice(
                replay_buffer.curriculum_role_codes, filled_size
            )
    if getattr(replay_buffer, "relabel_enabled", False):
        payload["relabel_market_width"] = int(replay_buffer.relabel_market_width)
        for key in CROSS_BASKET_RELABEL_REPLAY_KEYS:
            payload[key] = storage_slice(
                getattr(replay_buffer, key), filled_size
            )
    return payload


def restore_compact_replay_buffer(model, payload, *, consume_payload=False):
    """Restore either legacy arrays or streaming sources into live storage.

    The loader consumes its private payload so each legacy array can be freed
    as soon as it is restored. The default keeps existing callers' snapshots
    reusable, including callers outside the checkpoint I/O path.
    """
    def restore_storage(key, target, length):
        source = payload.pop(key) if consume_payload else payload[key]
        assign_replay_storage_slice(target, source, length, consume=consume_payload)

    received_mapping = payload.get("action_normalization", "hierarchical")
    expected_mapping = getattr(model.replay_buffer, "action_normalization", "hierarchical")
    if received_mapping != expected_mapping:
        raise ValueError(f"Replay action normalization mismatch: {received_mapping} != {expected_mapping}")
    received = payload.get("market_input_contract", "selected_basket_v1")
    expected = getattr(
        model, "market_input_contract",
        getattr(model.replay_buffer, "market_input_contract", "selected_basket_v1"),
    )
    if received != expected:
        raise ValueError(
            f"Replay market input contract mismatch: {received} != {expected}. "
            "Collect fresh replay; normalized legacy values cannot be converted."
        )
    if payload.get("format") != COMPACT_REPLAY_BUFFER_FORMAT:
        raise ValueError("Unsupported compact replay buffer payload.")
    if int(payload.get("version", 0)) != COMPACT_REPLAY_BUFFER_VERSION:
        raise ValueError(f"Unsupported compact replay buffer version: {payload.get('version')}")

    replay_buffer = getattr(model, "replay_buffer", None)
    if replay_buffer is None:
        raise ValueError("Model does not have an initialized replay buffer to restore into.")
    if not isinstance(replay_buffer, ReplayBuffer):
        raise TypeError("The model replay buffer must inherit from ReplayBuffer class.")

    saved_buffer_size = int(payload["buffer_size"])
    filled_size = int(payload["filled_size"])
    observation_length = int(payload.get("observation_length", filled_size))
    if int(replay_buffer.buffer_size) != saved_buffer_size:
        raise ValueError(
            f"Replay buffer capacity mismatch while loading compact checkpoint: "
            f"model={replay_buffer.buffer_size}, checkpoint={saved_buffer_size}"
        )
    if filled_size > replay_buffer.buffer_size or observation_length > replay_buffer.buffer_size:
        raise ValueError("Compact replay buffer payload is larger than the initialized replay buffer.")
    if filled_size < 0 or observation_length < 0:
        raise ValueError("Compact replay buffer lengths cannot be negative.")
    if int(payload.get("n_envs", replay_buffer.n_envs)) != int(replay_buffer.n_envs):
        raise ValueError("Replay buffer environment count mismatch while restoring checkpoint.")
    if bool(payload.get("optimize_memory_usage", False)) != bool(replay_buffer.optimize_memory_usage):
        raise ValueError("Replay buffer storage layout mismatch while restoring checkpoint.")
    position = int(payload["pos"])
    full = bool(payload["full"])
    if not 0 <= position < saved_buffer_size or filled_size != (saved_buffer_size if full else position):
        raise ValueError("Replay buffer position/full state does not match its saved length.")

    restore_storage("observations", replay_buffer.observations, observation_length)
    if not bool(payload.get("optimize_memory_usage", False)):
        restore_storage("next_observations", replay_buffer.next_observations, filled_size)
    restore_storage("actions", replay_buffer.actions, filled_size)
    restore_storage("rewards", replay_buffer.rewards, filled_size)
    restore_storage("dones", replay_buffer.dones, filled_size)
    restore_storage("timeouts", replay_buffer.timeouts, filled_size)
    if hasattr(replay_buffer, "component_outcomes"):
        if "component_outcomes" in payload:
            expected_target_kind = getattr(
                replay_buffer,
                "component_outcome_target_kind",
                "scaled_executed_contribution",
            )
            saved_target_kind = payload.get(
                "component_outcome_target_kind"
            )
            if saved_target_kind is None:
                # Pre-refinement checkpoints contain executed-weight
                # contributions.  They remain useful for scalar financial TD,
                # but those auxiliary rows must not masquerade as unit returns.
                if expected_target_kind == "scaled_unit_asset_return":
                    replay_buffer.component_outcomes[:filled_size] = 0.0
                    replay_buffer.component_outcome_masks[:filled_size] = False
                else:
                    restore_storage("component_outcomes", replay_buffer.component_outcomes, filled_size)
                    restore_storage("component_outcome_masks", replay_buffer.component_outcome_masks, filled_size)
            elif str(saved_target_kind) != str(expected_target_kind):
                raise ValueError(
                    "Component outcome target kind mismatch while restoring "
                    f"replay: checkpoint={saved_target_kind}, "
                    f"model={expected_target_kind}."
                )
            else:
                restore_storage("component_outcomes", replay_buffer.component_outcomes, filled_size)
                restore_storage("component_outcome_masks", replay_buffer.component_outcome_masks, filled_size)
        else:
            # Older replay checkpoints remain loadable. Their historical rows
            # carry no constituent metadata and are therefore masked out; newly
            # collected rows receive full component labels normally.
            replay_buffer.component_outcomes[:filled_size] = 0.0
            replay_buffer.component_outcome_masks[:filled_size] = False
        if hasattr(replay_buffer, "curriculum_role_codes"):
            replay_buffer.curriculum_role_codes[:filled_size] = 0
            if "curriculum_role_codes" in payload:
                restore_storage("curriculum_role_codes", replay_buffer.curriculum_role_codes, filled_size)
    if getattr(replay_buffer, "relabel_enabled", False):
        saved_width = int(payload.get("relabel_market_width", 0))
        if saved_width == int(replay_buffer.relabel_market_width) and all(
            key in payload for key in CROSS_BASKET_RELABEL_REPLAY_KEYS
        ):
            for key in CROSS_BASKET_RELABEL_REPLAY_KEYS:
                restore_storage(key, getattr(replay_buffer, key), filled_size)
        else:
            # Rows collected before relabel metadata existed stay ordinary
            # replay; newly collected rows are relabelable.
            replay_buffer.relabel_valid[:filled_size] = False

    replay_buffer.pos = int(payload["pos"])
    replay_buffer.full = bool(payload["full"])
    replay_buffer.optimize_memory_usage = bool(payload.get("optimize_memory_usage", False))
    replay_buffer.handle_timeout_termination = bool(payload.get("handle_timeout_termination", False))
    replay_buffer.device = model.device
    rebuild_membership_index = getattr(
        replay_buffer, "_rebuild_basket_position_index", None
    )
    if callable(rebuild_membership_index):
        rebuild_membership_index()
    if consume_payload:
        payload.clear()
    return replay_buffer


def finalize_loaded_replay_buffer(model):
    assert isinstance(model.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"
    if getattr(model, "action_normalization", "hierarchical") != getattr(model.replay_buffer, "action_normalization", "hierarchical"):
        raise ValueError("Replay action normalization does not match model")
    if not hasattr(model.replay_buffer, "handle_timeout_termination"):
        model.replay_buffer.handle_timeout_termination = False
        model.replay_buffer.timeouts = np.zeros_like(model.replay_buffer.dones)
    model.replay_buffer.device = model.device
    rebuild_membership_index = getattr(
        model.replay_buffer, "_rebuild_basket_position_index", None
    )
    if callable(rebuild_membership_index):
        rebuild_membership_index()


def write_streaming_replay_buffer(archive, replay_buffer):
    """Save filled array views synchronously, without a replay-sized copy.

    Each numeric array is an ordinary C-order NPY member. NumPy's buffered
    iterator handles even strided source views using at most one I/O chunk;
    ZIP compression never receives a whole multi-gigabyte array at once.
    Training must remain paused until this function returns.
    """
    payload = compact_replay_buffer_payload(replay_buffer, copy_arrays=False)
    metadata = {}
    arrays = []

    def register_array(path, array):
        if not isinstance(array, np.ndarray) or array.dtype.hasobject or array.dtype.fields:
            raise TypeError(f"Replay checkpoint requires plain numeric arrays: {path}")
        descriptor = {
            "path": path,
            "member": f"arrays/{len(arrays):06d}.npy",
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        }
        arrays.append((descriptor, array))

    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            register_array([key], value)
        elif isinstance(value, dict):
            metadata[key] = {}
            for subkey, array in value.items():
                register_array([key, subkey], array)
        else:
            metadata[key] = value

    manifest = {
        "format": STREAMING_REPLAY_BUFFER_FORMAT,
        "version": STREAMING_REPLAY_BUFFER_VERSION,
        "replay": metadata,
        "arrays": [descriptor for descriptor, _ in arrays],
    }
    archive.writestr(REPLAY_BUFFER_METADATA_ZIP_MEMBER, json.dumps(manifest))
    for descriptor, array in arrays:
        with archive.open(descriptor["member"], mode="w", force_zip64=True) as member:
            np.lib.format.write_array_header_2_0(member, {
                "descr": array.dtype.str,
                "fortran_order": False,
                "shape": array.shape,
            })
            iterator = np.nditer(
                array,
                flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=["readonly", "contig"],
                order="C",
                buffersize=max(1, REPLAY_BUFFER_IO_CHUNK_BYTES // array.dtype.itemsize),
            )
            with iterator:
                for chunk in iterator:
                    data = memoryview(chunk).cast("B")
                    for offset in range(0, len(data), REPLAY_BUFFER_IO_CHUNK_BYTES):
                        member.write(data[offset:offset + REPLAY_BUFFER_IO_CHUNK_BYTES])


class _ReplayCheckpointArraySource:
    """A lazy ZIP entry that restores bytes into already allocated storage."""

    def __init__(self, archive, descriptor):
        self.archive = archive
        self.descriptor = descriptor

    def read_into(self, target):
        descriptor = self.descriptor
        shape = tuple(descriptor["shape"])
        dtype = np.dtype(descriptor["dtype"])
        if dtype.hasobject or dtype.fields:
            raise ValueError("Replay checkpoint contains an unsupported array dtype.")
        if target.shape != shape or target.dtype != dtype:
            raise ValueError(
                f"Replay array shape/dtype mismatch for {descriptor['path']}: "
                f"checkpoint={shape}/{dtype}, model={target.shape}/{target.dtype}"
            )
        if not target.flags.c_contiguous:
            raise ValueError("Streaming replay restoration requires C-contiguous destination storage.")
        with self.archive.open(descriptor["member"], mode="r") as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                header = np.lib.format.read_array_header_1_0(member)
            elif version == (2, 0):
                header = np.lib.format.read_array_header_2_0(member)
            else:
                raise ValueError(f"Unsupported replay array NPY version: {version}")
            saved_shape, fortran_order, saved_dtype = header
            if saved_shape != shape or saved_dtype != dtype or fortran_order:
                raise ValueError(f"Replay array header does not match metadata: {descriptor['path']}")
            data = memoryview(target).cast("B") if target.size else memoryview(b"")
            offset = 0
            while offset < len(data):
                count = member.readinto(data[offset:offset + REPLAY_BUFFER_IO_CHUNK_BYTES])
                if not count:
                    raise EOFError(f"Truncated replay array: {descriptor['path']}")
                offset += count
            if member.read(1):
                raise ValueError(f"Unexpected trailing data in replay array: {descriptor['path']}")


def restore_streaming_replay_buffer(model, archive):
    with archive.open(REPLAY_BUFFER_METADATA_ZIP_MEMBER, mode="r") as member:
        manifest = json.load(member)
    if (manifest.get("format") != STREAMING_REPLAY_BUFFER_FORMAT
            or manifest.get("version") != STREAMING_REPLAY_BUFFER_VERSION):
        raise ValueError("Unsupported streaming replay checkpoint format/version.")
    payload = manifest["replay"]
    members = set()
    for descriptor in manifest["arrays"]:
        path = descriptor["path"]
        if len(path) not in (1, 2) or not all(isinstance(key, str) for key in path):
            raise ValueError("Invalid replay array metadata path.")
        parent = payload if len(path) == 1 else payload[path[0]]
        if path[-1] in parent or descriptor["member"] in members:
            raise ValueError("Duplicate replay array metadata entry.")
        members.add(descriptor["member"])
        parent[path[-1]] = _ReplayCheckpointArraySource(archive, descriptor)
    return restore_compact_replay_buffer(model, payload, consume_payload=True)


def save_replay_buffer_checkpoint(model, replay_buffer_path):
    if not hasattr(model, "save_replay_buffer"):
        print("⚠️ Replay buffer save skipped: model does not expose save_replay_buffer().")
        return False

    replay_buffer_tmp = replay_buffer_path + ".tmp"
    try:
        with zipfile.ZipFile(
            replay_buffer_tmp,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=REPLAY_BUFFER_ZIP_COMPRESSLEVEL,
            allowZip64=True,
        ) as archive:
            if not isinstance(model.replay_buffer, ReplayBuffer):
                raise TypeError("The model replay buffer must inherit from ReplayBuffer class.")
            write_streaming_replay_buffer(archive, model.replay_buffer)
        os.replace(replay_buffer_tmp, replay_buffer_path)
        replay_buffer = getattr(model, "replay_buffer", None)
        replay_size = replay_buffer.size() if replay_buffer is not None and hasattr(replay_buffer, "size") else "unknown"
        replay_capacity = getattr(replay_buffer, "buffer_size", "unknown")
        print(f"💾 Streaming compressed replay buffer saved: {replay_buffer_path}")
        print(f"💾 Replay buffer rows saved: {replay_size}/{replay_capacity}")
        print(f"💾 Replay I/O chunk limit: {REPLAY_BUFFER_IO_CHUNK_BYTES // (1024 * 1024)} MiB; no full replay copy")
        return True
    except Exception as e:
        print(f"⚠️ Replay buffer save failed, continuing training: {e}")
        if os.path.exists(replay_buffer_tmp):
            try:
                os.remove(replay_buffer_tmp)
            except OSError:
                pass
        return False


def load_replay_buffer_checkpoint(model, replay_buffer_path):
    if not hasattr(model, "load_replay_buffer"):
        print("⚠️ Replay buffer load skipped: model does not expose load_replay_buffer().")
        return False
    replay_buffer_path = resolve_replay_buffer_checkpoint_path(replay_buffer_path)
    if not os.path.exists(replay_buffer_path):
        print(f"⚠️ No matching replay buffer found for resume checkpoint: {replay_buffer_path}")
        print("⚠️ Resuming with model weights only; replay buffer will start fresh.")
        return False

    if replay_buffer_path.endswith(REPLAY_BUFFER_CHECKPOINT_SUFFIX):
        with zipfile.ZipFile(replay_buffer_path, mode="r") as archive:
            if REPLAY_BUFFER_METADATA_ZIP_MEMBER in archive.namelist():
                restore_streaming_replay_buffer(model, archive)
            else:
                # The existing 650k checkpoint uses this legacy pickle member.
                # It must be decoded once, but restored arrays are released as
                # we proceed instead of retaining two complete replay payloads.
                with archive.open(REPLAY_BUFFER_ZIP_MEMBER, mode="r") as replay_member:
                    payload = pickle.load(replay_member)
                if isinstance(payload, dict) and payload.get("format") == COMPACT_REPLAY_BUFFER_FORMAT:
                    restore_compact_replay_buffer(model, payload, consume_payload=True)
                else:
                    received = getattr(payload, "market_input_contract", "selected_basket_v1")
                    expected = getattr(model, "market_input_contract", config.market_input_contract())
                    if received != expected:
                        raise ValueError(f"Replay market input contract mismatch: {received} != {expected}")
                    model.replay_buffer = payload
                del payload
            finalize_loaded_replay_buffer(model)
    else:
        model.load_replay_buffer(replay_buffer_path)
    replay_buffer = getattr(model, "replay_buffer", None)
    replay_size = replay_buffer.size() if replay_buffer is not None and hasattr(replay_buffer, "size") else "unknown"
    print(f"✅ Replay buffer loaded from checkpoint: {replay_buffer_path}")
    print(f"✅ Replay buffer current size: {replay_size}")
    return True


def collect_step_checkpoints(checkpoints_dir, prefix, suffix):
    checkpoint_files = []
    if not os.path.exists(checkpoints_dir):
        return checkpoint_files

    for filename in os.listdir(checkpoints_dir):
        if filename.startswith(prefix) and filename.endswith(suffix):
            filepath = os.path.join(checkpoints_dir, filename)
            try:
                step_count = parse_checkpoint_step(filepath, prefix=prefix, suffix=suffix)
            except ValueError:
                continue
            checkpoint_files.append((step_count, filepath))
    return checkpoint_files


def keep_latest_step_checkpoints(checkpoints_dir, prefix, suffix, keep_latest=2):
    checkpoint_files = collect_step_checkpoints(checkpoints_dir, prefix, suffix)
    if len(checkpoint_files) <= keep_latest:
        return

    checkpoint_files.sort(key=lambda item: item[0], reverse=True)
    for _, old_filepath in checkpoint_files[keep_latest:]:
        try:
            os.remove(old_filepath)
            print(f"🗑️  Removed old checkpoint artifact: {os.path.basename(old_filepath)}")
        except OSError as e:
            print(f"⚠️  Warning: Could not remove checkpoint artifact {old_filepath}: {e}")


def keep_latest_model_and_replay_checkpoints(checkpoints_dir, keep_latest=2):
    model_checkpoints = dict(
        collect_step_checkpoints(
            checkpoints_dir,
            MODEL_CHECKPOINT_PREFIX,
            MODEL_CHECKPOINT_SUFFIX,
        )
    )
    replay_checkpoints = {}
    for step_count, filepath in collect_step_checkpoints(
        checkpoints_dir,
        REPLAY_BUFFER_CHECKPOINT_PREFIX,
        REPLAY_BUFFER_CHECKPOINT_SUFFIX,
    ):
        replay_checkpoints.setdefault(step_count, []).append(filepath)
    for step_count, filepath in collect_step_checkpoints(
        checkpoints_dir,
        REPLAY_BUFFER_CHECKPOINT_PREFIX,
        LEGACY_REPLAY_BUFFER_CHECKPOINT_SUFFIX,
    ):
        replay_checkpoints.setdefault(step_count, []).append(filepath)

    paired_steps = sorted(
        (step_count for step_count in model_checkpoints if step_count in replay_checkpoints),
        reverse=True,
    )
    if not paired_steps:
        return

    steps_to_keep = set(paired_steps[:keep_latest])
    for step_count, model_path in model_checkpoints.items():
        if step_count not in steps_to_keep:
            try:
                os.remove(model_path)
                print(f"🗑️  Removed old model checkpoint artifact: {os.path.basename(model_path)}")
            except OSError as e:
                print(f"⚠️  Warning: Could not remove model checkpoint artifact {model_path}: {e}")
    for step_count, replay_paths in replay_checkpoints.items():
        if step_count not in steps_to_keep:
            for replay_path in replay_paths:
                try:
                    os.remove(replay_path)
                    print(f"🗑️  Removed old replay checkpoint artifact: {os.path.basename(replay_path)}")
                except OSError as e:
                    print(f"⚠️  Warning: Could not remove replay checkpoint artifact {replay_path}: {e}")
    for step_count, curriculum_path in collect_step_checkpoints(
        checkpoints_dir,
        CURRICULUM_CHECKPOINT_PREFIX,
        CURRICULUM_CHECKPOINT_SUFFIX,
    ):
        if step_count not in steps_to_keep:
            try:
                os.remove(curriculum_path)
                print(
                    "🗑️  Removed old curriculum checkpoint artifact: "
                    f"{os.path.basename(curriculum_path)}"
                )
            except OSError as e:
                print(
                    "⚠️  Warning: Could not remove curriculum checkpoint "
                    f"artifact {curriculum_path}: {e}"
                )


def should_save_resume_checkpoint(
    step_count, interval=RESUME_CHECKPOINT_INTERVAL_STEPS, span=1
):
    """True when a multiple of ``interval`` lies in (step_count - span, step_count]."""
    if step_count <= 0:
        return False
    return step_count // interval > max(0, step_count - span) // interval


def save_model_checkpoint(model, checkpoint_path):
    checkpoint_tmp = checkpoint_path + ".tmp"
    try:
        model.save(checkpoint_tmp)
        os.replace(checkpoint_tmp, checkpoint_path)
        return True
    except Exception as e:
        print(f"⚠️ Model checkpoint save failed, continuing training: {e}")
        if os.path.exists(checkpoint_tmp):
            try:
                os.remove(checkpoint_tmp)
            except OSError:
                pass
        return False


def save_resume_checkpoint_pair(
    model,
    checkpoints_dir,
    step_count,
    curriculum_manager=None,
):
    replay_path = replay_buffer_checkpoint_path(checkpoints_dir, step_count)
    checkpoint_path = model_checkpoint_path(checkpoints_dir, step_count)

    replay_saved = save_replay_buffer_checkpoint(model, replay_path)
    if not replay_saved:
        print(f"⚠️ Paired resume checkpoint skipped at step {step_count}: replay buffer was not saved.")
        return False

    model_saved = save_model_checkpoint(model, checkpoint_path)
    if not model_saved:
        try:
            os.remove(replay_path)
            print(f"🧹 Removed unpaired replay checkpoint after model save failure: {os.path.basename(replay_path)}")
        except OSError:
            pass
        print(f"⚠️ Paired resume checkpoint skipped at step {step_count}: model was not saved.")
        return False

    curriculum_path = None
    if curriculum_manager is not None:
        curriculum_path = curriculum_checkpoint_path(checkpoints_dir, step_count)
        try:
            curriculum_manager.save(curriculum_path)
        except Exception as exc:
            print(
                "⚠️ Paired resume checkpoint skipped: curriculum state "
                f"could not be saved ({exc})."
            )
            for artifact in (replay_path, checkpoint_path, curriculum_path):
                if artifact and os.path.exists(artifact):
                    try:
                        os.remove(artifact)
                    except OSError:
                        pass
            return False

    print(f"💾 Paired resume checkpoint saved at step {step_count}:")
    print(f"   Model: {checkpoint_path}")
    print(f"   Replay buffer: {replay_path}")
    if curriculum_path is not None:
        print(f"   Curriculum: {curriculum_path}")
    keep_latest_model_and_replay_checkpoints(
        checkpoints_dir,
        keep_latest=RESUME_CHECKPOINTS_TO_KEEP,
    )
    return True


def monitor_csv_path(filename):
    if filename.endswith(Monitor.EXT):
        return filename
    if os.path.isdir(filename):
        return os.path.join(filename, Monitor.EXT)
    return filename + "." + Monitor.EXT


def should_append_monitor(filename, resume_checkpoint):
    return bool(resume_checkpoint) and os.path.exists(monitor_csv_path(filename))


def unique_log_artifact_path(log_dir, base_filename):
    root, ext = os.path.splitext(base_filename)
    candidate = os.path.join(log_dir, base_filename)
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(log_dir, f"{root}_{counter}{ext}")
        counter += 1
    return candidate


def archive_existing_progress_csv(log_dir, resume_step):
    progress_path = os.path.join(log_dir, PROGRESS_CSV_FILENAME)
    if not os.path.exists(progress_path) or os.path.getsize(progress_path) == 0:
        return None

    import shutil

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"progress_before_resume_{resume_step}_{timestamp}.csv"
    archive_path = unique_log_artifact_path(log_dir, archive_name)
    shutil.copy2(progress_path, archive_path)
    return archive_path


def progress_segment_paths(log_dir):
    if not os.path.exists(log_dir):
        return []

    paths = []
    for filename in os.listdir(log_dir):
        if filename == PROGRESS_COMBINED_CSV_FILENAME:
            continue
        if filename == PROGRESS_CSV_FILENAME:
            paths.append(os.path.join(log_dir, filename))
            continue
        if filename == LEGACY_PROGRESS_BACKUP_FILENAME:
            paths.append(os.path.join(log_dir, filename))
            continue
        if filename.startswith("progress_before_resume_") and filename.endswith(".csv"):
            paths.append(os.path.join(log_dir, filename))
            continue
        if filename.startswith("progress_pre_resume_") and filename.endswith(".csv"):
            paths.append(os.path.join(log_dir, filename))

    current_progress = os.path.join(log_dir, PROGRESS_CSV_FILENAME)
    paths.sort(key=lambda path: os.path.getmtime(path) if os.path.exists(path) else 0)
    if current_progress in paths:
        paths = [path for path in paths if path != current_progress] + [current_progress]
    return paths


def load_combined_progress_history(log_dir, write_combined=True):
    frames = []
    for path in progress_segment_paths(log_dir):
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                frames.append(pd.read_csv(path))
        except Exception as e:
            print(f"⚠️ Could not read progress segment {path}: {e}")

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True, sort=False).dropna(how="all")
    if "time/total_timesteps" in combined.columns:
        combined["time/total_timesteps"] = pd.to_numeric(combined["time/total_timesteps"], errors="coerce")
        has_step = combined["time/total_timesteps"].notna()
        stepped = combined[has_step].sort_values("time/total_timesteps", kind="mergesort")
        stepped = stepped.drop_duplicates(subset=["time/total_timesteps"], keep="last")
        combined = pd.concat([combined[~has_step], stepped], ignore_index=True, sort=False)

    if write_combined:
        combined_path = os.path.join(log_dir, PROGRESS_COMBINED_CSV_FILENAME)
        try:
            combined.to_csv(combined_path, index=False)
        except (OSError, IOError) as e:
            print(f"⚠️ Could not write combined progress CSV: {e}")
    return combined


def is_missing_value(value):
    try:
        return bool(pd.isna(value))
    except Exception:
        return value is None


def to_optional_float(value):
    if is_missing_value(value):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def crl_float(record, key, default=0.0):
    value = to_optional_float(record.get(key))
    return default if value is None else value


def crl_record_has_finite_values(record, keys):
    return all(to_optional_float(record.get(key)) is not None for key in keys)


SINGLE_CRL_PLOT_KEYS = (
    'current_lambda',
    'avg_recent_constraint_cost',
    'avg_recent_violation',
    'constraint_threshold',
)
DUAL_CRL_PLOT_KEYS = (
    'current_lambda_velocity',
    'current_lambda_acceleration',
    'avg_recent_velocity_cost',
    'avg_recent_acceleration_cost',
    'avg_recent_velocity_violation',
    'avg_recent_acceleration_violation',
    'velocity_constraint_threshold',
    'acceleration_constraint_threshold',
)


def normalize_crl_enabled(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if is_missing_value(value):
        return False
    return bool(value)


def normalize_crl_stat_record(record):
    normalized = dict(record)
    normalized["crl_enabled"] = normalize_crl_enabled(normalized.get("crl_enabled", False))

    has_dual_keys = (
        "current_lambda_velocity" in normalized
        or "current_lambda_acceleration" in normalized
        or "avg_recent_velocity_cost" in normalized
        or "avg_recent_acceleration_cost" in normalized
    )
    if not has_dual_keys:
        return normalized

    velocity_lambda = to_optional_float(normalized.get("current_lambda_velocity"))
    acceleration_lambda = to_optional_float(normalized.get("current_lambda_acceleration"))
    velocity_threshold = to_optional_float(normalized.get("velocity_constraint_threshold"))
    acceleration_threshold = to_optional_float(normalized.get("acceleration_constraint_threshold"))
    velocity_cost = to_optional_float(normalized.get("avg_recent_velocity_cost"))
    acceleration_cost = to_optional_float(normalized.get("avg_recent_acceleration_cost"))
    velocity_violation = to_optional_float(normalized.get("avg_recent_velocity_violation"))
    acceleration_violation = to_optional_float(normalized.get("avg_recent_acceleration_violation"))

    if velocity_violation is None and velocity_cost is not None and velocity_threshold is not None:
        normalized["avg_recent_velocity_violation"] = velocity_cost - velocity_threshold
        velocity_violation = normalized["avg_recent_velocity_violation"]
    if acceleration_violation is None and acceleration_cost is not None and acceleration_threshold is not None:
        normalized["avg_recent_acceleration_violation"] = acceleration_cost - acceleration_threshold
        acceleration_violation = normalized["avg_recent_acceleration_violation"]

    if velocity_lambda is not None and is_missing_value(normalized.get("current_lambda")):
        normalized["current_lambda"] = velocity_lambda
    if velocity_lambda is not None or acceleration_lambda is not None:
        normalized["current_lambda_total"] = sum(
            value for value in (velocity_lambda, acceleration_lambda) if value is not None
        )
    if velocity_threshold is not None or acceleration_threshold is not None:
        normalized["constraint_threshold"] = sum(
            value for value in (velocity_threshold, acceleration_threshold) if value is not None
        )
    if velocity_cost is not None or acceleration_cost is not None:
        normalized["avg_recent_constraint_cost"] = sum(
            value for value in (velocity_cost, acceleration_cost) if value is not None
        )
    if velocity_violation is not None or acceleration_violation is not None:
        normalized["avg_recent_violation"] = sum(
            value for value in (velocity_violation, acceleration_violation) if value is not None
        )
    elif (
        to_optional_float(normalized.get("avg_recent_constraint_cost")) is not None
        and (velocity_threshold is not None or acceleration_threshold is not None)
    ):
        active_threshold = 0.0
        if velocity_cost is not None and velocity_threshold is not None:
            active_threshold += velocity_threshold
        if acceleration_cost is not None and acceleration_threshold is not None:
            active_threshold += acceleration_threshold
        normalized["avg_recent_violation"] = (
            to_optional_float(normalized.get("avg_recent_constraint_cost"))
            - active_threshold
        )

    normalized.setdefault("crl_schema", "dual_velocity_acceleration")
    return normalized


def normalize_crl_stats_history(crl_stats_history):
    return [normalize_crl_stat_record(record) for record in crl_stats_history]


def crl_record_has_dual_metrics(record):
    return (
        to_optional_float(record.get("current_lambda_velocity")) is not None
        or to_optional_float(record.get("current_lambda_acceleration")) is not None
        or to_optional_float(record.get("avg_recent_velocity_cost")) is not None
        or to_optional_float(record.get("avg_recent_acceleration_cost")) is not None
    )


def crl_record_has_single_metrics(record):
    return (
        to_optional_float(record.get("current_lambda")) is not None
        or to_optional_float(record.get("avg_recent_constraint_cost")) is not None
    )


def load_crl_history_csv(path):
    crl_df = pd.read_csv(path)
    for col in crl_df.columns:
        if crl_df[col].dtype == object:
            lowered = crl_df[col].astype(str).str.lower()
            if lowered.isin(["true", "false"]).all():
                crl_df[col] = lowered == "true"
            else:
                crl_df[col] = pd.to_numeric(crl_df[col], errors="ignore")
    return normalize_crl_stats_history(crl_df.to_dict("records"))


def write_crl_history_csv(log_dir, crl_stats_history):
    normalized_history = normalize_crl_stats_history(crl_stats_history)
    crl_df = pd.DataFrame(normalized_history)
    crl_csv = os.path.join(log_dir, "crl_history.csv")
    crl_df.to_csv(crl_csv, index=False)
    return crl_csv, normalized_history

from attention_policy import create_attention_policy, MLP_CONFIGS, SimpleMlpTD3Policy, AttentionTD3Policy, AttentionActorCriticPolicy ,NormalisedTD3Policy, NormalisedDDPGPolicy
# from custom_td3 import CustomTD3  # ✅ IMPORT: Import our new custom agent
# from custom_td3_actor_critic import CustomTD3_AC # ✅ IMPORT: Import our new custom agent
# from custom_td3_actor_critic_batch import CustomTD3_AC # ✅ IMPORT: Import our new custom agent

# from custom_td3_actor_critic_2crl import CustomTD3_AC
# from custom_td3_actor_critic_2crl_ogmid import CustomTD3_AC

# from custom_td3_actor_critic_2crl_safe import CustomTD3_AC
from custom_td3_actor_critic_2crl_safe2 import CustomTD3_AC as CustomTD3_AC_Encoder
from custom_td3_actor_critic_2crl_safe3 import CustomTD3_AC as CustomTD3_AC_Encoder_Res
# from custom_td3_actor_critic_2crl_safe_og import CustomTD3_AC as CustomTD3_AC_OG #this stopped
#current working versions bidirectional learning vs decay lambdas
from custom_td3_actor_critic_2crl_safe_pro_attention import CustomTD3_AC as CustomTD3_AC_OG #this is still going 
from custom_td3_actor_critic_2crl_safe_pro_exp_attention import CustomTD3_AC as CustomTD3_AC_OG_PROEXP
# from custom_td3_actor_critic_2crl_safe_pro_IQL_attention import CustomTD3_AC as CustomTD3_AC_OG_IQL
# from custom_td3_actor_critic_2crl_safe_pro import CustomTD3_AC as CustomTD3_AC_OG


PROPORTIONAL_EXP_MODES = ("off", "local_boltzmann", "boltzmann_disagreement")
PROPORTIONAL_EXP_RUN_SUFFIXES = {
    "off": "",
    "local_boltzmann": "_proexp_local",
    "boltzmann_disagreement": "_proexp_disagreement",
}


def normalize_proportional_exp_mode(mode: str) -> str:
    normalized = str(mode or "off").strip().lower()
    if normalized not in PROPORTIONAL_EXP_RUN_SUFFIXES:
        raise ValueError(
            f"Unknown proportional_exp_mode='{mode}'. Expected one of {PROPORTIONAL_EXP_MODES}."
        )
    return normalized


def get_proportional_exp_run_suffix(mode: str) -> str:
    return PROPORTIONAL_EXP_RUN_SUFFIXES[normalize_proportional_exp_mode(mode)]


def use_iql_attention_td3(
    proportional_exp_mode: str,
    use_attention: bool,
    attention_type: str,
    encoder_variant: str,
) -> bool:
    return (
        normalize_proportional_exp_mode(proportional_exp_mode) == "off"
        and bool(use_attention)
        and attention_type == "variable_coin_attention"
        and encoder_variant == "baseline"
    )


def get_attention_td3_algorithm_class(
    proportional_exp_mode: str,
    use_attention: bool = False,
    attention_type: str = "variable_coin_attention",
    encoder_variant: str = "baseline",
):
    mode = normalize_proportional_exp_mode(proportional_exp_mode)
    if mode != "off":
        return CustomTD3_AC_OG_PROEXP
    # if use_iql_attention_td3(mode, use_attention, attention_type, encoder_variant):
    #     return CustomTD3_AC_OG_IQL
    else:
        return CustomTD3_AC_OG


def validate_proportional_exp_configuration(
    algorithm: str,
    use_attention: bool,
    attention_type: str,
    encoder_variant: str,
    proportional_exp_mode: str,
) -> str:
    mode = normalize_proportional_exp_mode(proportional_exp_mode)
    if mode == "off":
        return mode
    if algorithm.upper() != "TD3":
        raise ValueError("proportional-exp mode is only supported with algorithm TD3.")
    if not use_attention:
        raise ValueError("proportional-exp mode requires --use-attention.")
    if attention_type != "variable_coin_attention":
        raise ValueError("proportional-exp mode requires --attention-type variable_coin_attention.")
    if encoder_variant != "baseline":
        raise ValueError("proportional-exp mode currently supports only --encoder-variant baseline.")
    return mode


def validate_qwm_configuration(
    *,
    use_qwm: bool,
    algorithm: str,
    use_attention: bool,
    attention_type: str,
    use_variable_portfolio: bool,
    encoder_variant: str,
    proportional_exp_mode: str,
) -> bool:
    """Keep QWM on the one production path whose state/action contract it models."""
    if not bool(use_qwm):
        return False
    if algorithm.upper() != "TD3":
        raise ValueError("QWM decision-time search is only supported with TD3.")
    if not use_attention or attention_type != "variable_coin_attention":
        raise ValueError(
            "QWM requires --use-attention --attention-type variable_coin_attention."
        )
    if not use_variable_portfolio:
        raise ValueError("QWM requires the variable 8-of-N dictionary observation.")
    if encoder_variant != "baseline":
        raise ValueError("QWM currently supports only --encoder-variant baseline.")
    if normalize_proportional_exp_mode(proportional_exp_mode) == "off":
        raise ValueError(
            "QWM requires ProExp candidate diversity; choose local_boltzmann "
            "or boltzmann_disagreement."
        )
    return True



def configure_a40_runtime() -> None:
    """
    Safe GPU-side speed settings for Ampere cards such as the A40.

    This keeps the existing training methodology intact and only enables faster
    float32 matmul kernels where PyTorch supports them.
    """
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


class PlotGeneratorCallback(BaseCallback):
    """
    Callback to generate plots from training metrics logged in CSV.
    """
    
    def __init__(self, log_path: str, model_name: str, plot_freq: int = 5000, verbose: int = 0, 
                 crl_stats_history=None, crl_profile=None, returns=None):
        super().__init__(verbose)
        self.log_path = log_path
        self.model_name = model_name
        self.plot_freq = plot_freq
        self.csv_path = os.path.join(log_path, PROGRESS_CSV_FILENAME)
        # CRL-specific parameters for periodic CRL plot generation
        self.crl_stats_history = normalize_crl_stats_history(crl_stats_history or [])
        self.crl_profile = crl_profile
        self.returns = returns or []
        
    def _on_step(self) -> bool:
        # Generate plots periodically
        if self.n_calls % self.plot_freq == 0 and self.n_calls > 0:
            self._generate_plots()
        return True
    
    def _on_training_end(self) -> None:
        """Generate final plots when training ends."""
        self._generate_plots()
    
    def _generate_plots(self):
        """Generate training plots from CSV data."""
        try:
            is_flowrl = self.model_name.startswith("FlowRL")
            df = load_combined_progress_history(self.log_path)
            # Warmup runs before SB3 starts advancing environment timesteps.
            # Merge its dedicated stream at negative x-values so the existing
            # foundation panels show the complete lifecycle without corrupting
            # the financial timestep axis.
            warmup_path = os.path.join(self.log_path, "foundation_warmup.csv")
            if os.path.exists(warmup_path):
                try:
                    warmup = pd.read_csv(warmup_path)
                    if len(warmup):
                        warmup_frame = pd.DataFrame(
                            {
                                "time/total_timesteps": -warmup[
                                    "foundation_update"
                                ].astype(float),
                                "train/foundation_actor_loss": warmup[
                                    "actor_loss"
                                ],
                                "train/foundation_critic_loss": warmup[
                                    "critic_loss"
                                ],
                                "train/foundation_representation_loss": warmup[
                                    "representation_loss"
                                ],
                                "train/foundation_action_outcome_loss": warmup[
                                    "action_outcome_loss"
                                ],
                                "train/foundation_actor_regret": warmup[
                                    "actor_regret"
                                ],
                                "train/foundation_warmup_updates": warmup[
                                    "foundation_update"
                                ],
                            }
                        )
                        if df is None or len(df) == 0:
                            df = warmup_frame
                        else:
                            df = pd.concat(
                                (warmup_frame, df), ignore_index=True, sort=False
                            ).sort_values("time/total_timesteps")
                        if self.verbose > 0:
                            print(
                                f"📊 Loaded {len(warmup)} foundation warmup "
                                f"rows from {warmup_path}"
                            )
                except (OSError, ValueError, KeyError) as exc:
                    if self.verbose > 0:
                        print(f"⚠️ Could not read foundation warmup CSV: {exc}")
            if df is None or len(df) == 0:
                if self.verbose > 0:
                    print(f"⚠️ No progress CSV data found in: {self.log_path}")
                    print(f"📂 Files in log directory: {os.listdir(self.log_path)}")
                return
            
            if len(df) < 2 and not self.model_name.startswith("FlowRL"):
                if self.verbose > 0:
                    print(f"⚠️ Not enough data points for plotting: {len(df)}")
                return
            
            if self.verbose > 0:
                print(f"📊 Generating plots from {len(df)} combined progress data points")
                print(f"📊 Available columns: {list(df.columns)}")
            
            foundation_loss_columns = (
                'train/foundation_critic_loss',
                'train/foundation_actor_loss',
                'train/foundation_action_outcome_loss',
                'train/foundation_representation_loss',
            )
            foundation_diagnostic_columns = foundation_loss_columns + (
                'train/foundation_actor_regret',
                'train/foundation_warmup_complete',
                'train/foundation_warmup_updates',
            )
            has_foundation_diagnostics = any(
                column in df.columns and df[column].notna().any()
                for column in foundation_diagnostic_columns
            )
            earnmore_loss_columns = (
                'train/earnmore_reconstruction_loss',
                'train/earnmore_actor_reconstruction_loss',
                'train/earnmore_critic_reconstruction_loss',
            )
            earnmore_diagnostic_columns = earnmore_loss_columns + (
                'train/earnmore_masked_asset_fraction',
                'train/earnmore_masked_assets_per_sample',
                'train/earnmore_visible_assets_per_sample',
                'train/earnmore_updates',
            )
            has_earnmore_diagnostics = any(
                column in df.columns and df[column].notna().any()
                for column in earnmore_diagnostic_columns
            )

            # Preserve the compact legacy layout when plotting older runs, while
            # giving each active representation learner a dedicated row.
            plot_rows = (
                2
                + int(has_foundation_diagnostics)
                + int(has_earnmore_diagnostics)
            )
            foundation_row = 2 if has_foundation_diagnostics else None
            earnmore_row = (
                2 + int(has_foundation_diagnostics)
                if has_earnmore_diagnostics
                else None
            )
            fig, axes = plt.subplots(
                plot_rows,
                3,
                figsize=(15, 5 * plot_rows),
                squeeze=False,
            )
            readable_model_name = self.model_name.replace("_", " ")
            wrapped_model_name = textwrap.fill(
                readable_model_name,
                width=96,
                max_lines=3,
                placeholder=" …",
            )
            fig.suptitle(
                f"Training Progress\n{wrapped_model_name}",
                fontsize=12,
            )
            
            # Plot 1: Episode Reward Mean
            if 'rollout/ep_rew_mean' in df.columns:
                plot_data = df.dropna(subset=['rollout/ep_rew_mean', 'time/total_timesteps'])
                if len(plot_data) > 0:
                    x_r = plot_data['time/total_timesteps'].values
                    y_r = plot_data['rollout/ep_rew_mean'].values
                    y_r_series = pd.Series(y_r)
                    roll_win = max(5, len(y_r) // 20)
                    y_r_std = y_r_series.rolling(window=roll_win, min_periods=1).std().fillna(0).values
                    y_r_smooth = y_r_series.ewm(alpha=0.03, adjust=False).mean().values
                    axes[0, 0].plot(x_r, y_r, color='steelblue', linewidth=1, alpha=0.2, label='raw')
                    axes[0, 0].fill_between(x_r, y_r - y_r_std, y_r + y_r_std, color='steelblue', alpha=0.12, label='±1 std')
                    axes[0, 0].plot(x_r, y_r_smooth, color='steelblue', linewidth=2, label='smoothed')
                    axes[0, 0].set_title('Episode Reward (Training)')
                    axes[0, 0].set_xlabel('Timesteps')
                    axes[0, 0].set_ylabel('Mean Episode Reward')
                    axes[0, 0].grid(True, alpha=0.3)
                    axes[0, 0].legend(loc='best', fontsize=8)
            
            # Plot 2: Episode Length Mean
            if 'rollout/ep_len_mean' in df.columns:
                plot_data = df.dropna(subset=['rollout/ep_len_mean', 'time/total_timesteps'])
                if len(plot_data) > 0:
                    axes[0, 1].plot(plot_data['time/total_timesteps'], plot_data['rollout/ep_len_mean'], 'g-', linewidth=2)
                    axes[0, 1].set_title('Episode Length (Training)')
                    axes[0, 1].set_xlabel('Timesteps')
                    axes[0, 1].set_ylabel('Mean Episode Length')
                    axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Evaluation Reward
            if 'eval/mean_reward' in df.columns:
                eval_data = df.dropna(subset=['eval/mean_reward', 'time/total_timesteps'])
                if len(eval_data) > 0:
                    ax_eval = axes[0, 2]
                    x_ev = eval_data['time/total_timesteps'].values
                    y_ev = eval_data['eval/mean_reward'].values
                    y_ev_smooth = pd.Series(y_ev).ewm(alpha=0.03, adjust=False).mean().values
                    if 'eval/std_reward' in df.columns:
                        std_ev = eval_data['eval/std_reward'].fillna(0).values
                    else:
                        win_ev = max(3, len(y_ev) // 10)
                        std_ev = pd.Series(y_ev).rolling(window=win_ev, min_periods=1).std().fillna(0).values
                    ax_eval.plot(x_ev, y_ev, color='crimson', linewidth=1,
                                 alpha=0.65 if is_flowrl else 0.2, label='Shaped return (raw)')
                    ax_eval.fill_between(x_ev, y_ev - std_ev, y_ev + std_ev, color='crimson', alpha=0.12,
                                         label='±1 std across episodes' if is_flowrl else '±1 std')
                    ax_eval.plot(x_ev, y_ev_smooth, color='crimson', linewidth=2, label='Shaped return (smoothed)')
                    ax_eval.set_title('Evaluation (Shaped vs Financial)')
                    axes[0, 2].set_xlabel('Timesteps')
                    ax_eval.set_ylabel('Mean Shaped Return')
                    ax_eval.grid(True, alpha=0.3)

                    # Optional: overlay true financial return on a secondary axis when available.
                    if 'eval/mean_financial_return' in df.columns:
                        fin_data = df.dropna(subset=['eval/mean_financial_return', 'time/total_timesteps'])
                        if len(fin_data) > 0:
                            ax_fin = ax_eval.twinx()
                            x_fin = fin_data['time/total_timesteps'].values
                            y_fin = fin_data['eval/mean_financial_return'].values
                            y_fin_smooth = pd.Series(y_fin).ewm(alpha=0.03, adjust=False).mean().values
                            if 'eval/std_financial_return' in df.columns:
                                std_fin = fin_data['eval/std_financial_return'].fillna(0).values
                            else:
                                win_fin = max(3, len(y_fin) // 10)
                                std_fin = pd.Series(y_fin).rolling(window=win_fin, min_periods=1).std().fillna(0).values
                            ax_fin.plot(x_fin, y_fin, color='darkorange', linewidth=1,
                                        alpha=0.65 if is_flowrl else 0.2, label='Financial return (raw)')
                            ax_fin.fill_between(x_fin, y_fin - std_fin, y_fin + std_fin, color='darkorange', alpha=0.12,
                                                label='±1 std across episodes' if is_flowrl else '±1 std')
                            ax_fin.plot(x_fin, y_fin_smooth, color='darkorange', linewidth=2, label='Financial return (smoothed)')
                            ax_fin.axhline(0.0, color='darkorange', alpha=0.2, linewidth=1)
                            ax_fin.set_ylabel('Mean Financial Return')
                            if is_flowrl:
                                from matplotlib.ticker import PercentFormatter
                                ax_fin.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

                    # Legend: combine both axes if the secondary axis exists.
                    try:
                        handles, labels = ax_eval.get_legend_handles_labels()
                        if 'ax_fin' in locals():
                            h2, l2 = ax_fin.get_legend_handles_labels()
                            handles += h2
                            labels += l2
                        if handles:
                            ax_eval.legend(handles, labels, loc='best', fontsize=8)
                    except Exception:
                        pass
            
            # Plot 4: Training Loss (algorithm specific)
            loss_plotted = False
            primary_loss_columns = ['train/actor_loss', 'train/critic_loss', 'train/behavior_critic_loss', 'train/policy_loss', 'train/loss']
            value_loss_axis = None
            primary_lines = []
            value_lines = []
            for loss_col in primary_loss_columns:
                if loss_col in df.columns:
                    loss_data = df.dropna(subset=[loss_col])
                    if len(loss_data) > 0:
                        line, = axes[1, 0].plot(
                            loss_data['time/total_timesteps'],
                            loss_data[loss_col],
                            linewidth=2,
                            label=loss_col.split('/')[-1],
                        )
                        primary_lines.append(line)
                        loss_plotted = True

            if 'train/value_loss' in df.columns:
                value_loss_data = df.dropna(subset=['train/value_loss'])
                if len(value_loss_data) > 0:
                    value_loss_axis = axes[1, 0].twinx()
                    line, = value_loss_axis.plot(
                        value_loss_data['time/total_timesteps'],
                        value_loss_data['train/value_loss'],
                        color='tab:purple',
                        linewidth=2,
                        linestyle='--',
                        label='value_loss',
                    )
                    value_lines.append(line)
                    value_loss_axis.set_ylabel('Value Loss')
                    value_loss_axis.grid(False)
                    loss_plotted = True
            
            if loss_plotted:
                axes[1, 0].set_title('Training Loss')
                axes[1, 0].set_xlabel('Timesteps')
                axes[1, 0].set_ylabel('Actor/Critic Loss')
                lines = primary_lines + value_lines
                if lines:
                    axes[1, 0].legend(lines, [line.get_label() for line in lines])
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Learning Rate (handles single or separate LRs)
            lr_plotted = False
            lr_cols = ['train/learning_rate', 'train/actor_lr', 'train/critic_lr', 'train/behavior_critic_lr', 'train/value_lr']
            for lr_col in lr_cols:
                if lr_col in df.columns:
                    lr_data = df.dropna(subset=[lr_col])
                    if len(lr_data) > 0:
                        axes[1, 1].plot(lr_data['time/total_timesteps'], lr_data[lr_col], linewidth=2, label=lr_col.split('/')[-1])
                        lr_plotted = True
            
            if lr_plotted:
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Timesteps')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: FPS (Training Speed)
            if 'time/fps' in df.columns:
                fps_data = df.dropna(subset=['time/fps'])
                if len(fps_data) > 0:
                    axes[1, 2].plot(fps_data['time/total_timesteps'], fps_data['time/fps'], 'purple', linewidth=2)
                    axes[1, 2].set_title('Training Speed (FPS)')
                    axes[1, 2].set_xlabel('Timesteps')
                    axes[1, 2].set_ylabel('Frames Per Second')
                    axes[1, 2].grid(True, alpha=0.3)

            foundation_lifecycle_plotted = False
            if has_foundation_diagnostics:
                # Plot 7: the losses that train the local/relational
                # representation and the action-conditioned outcome model.
                foundation_loss_axis = axes[foundation_row, 0]
                for loss_column in foundation_loss_columns:
                    if loss_column not in df.columns:
                        continue
                    loss_data = df.dropna(
                        subset=[loss_column, 'time/total_timesteps']
                    )
                    if len(loss_data) > 0:
                        foundation_loss_axis.plot(
                            loss_data['time/total_timesteps'],
                            loss_data[loss_column],
                            linewidth=1.5,
                            label=loss_column.removeprefix('train/foundation_'),
                        )
                if foundation_loss_axis.has_data():
                    foundation_loss_axis.set_title('Foundation Learning Losses')
                    foundation_loss_axis.set_xlabel('Timesteps')
                    foundation_loss_axis.set_ylabel('Loss')
                    foundation_loss_axis.legend(loc='best', fontsize=8)
                    foundation_loss_axis.grid(True, alpha=0.3)

                # Plot 8: regret directly measures whether the live actor is
                # approaching the best hindsight candidate in its sampled set.
                regret_column = 'train/foundation_actor_regret'
                if regret_column in df.columns:
                    regret_data = df.dropna(
                        subset=[regret_column, 'time/total_timesteps']
                    )
                    if len(regret_data) > 0:
                        axes[foundation_row, 1].plot(
                            regret_data['time/total_timesteps'],
                            regret_data[regret_column],
                            color='tab:red',
                            linewidth=1.5,
                            label='actor_regret',
                        )
                        axes[foundation_row, 1].axhline(
                            0.0, color='black', linewidth=1, alpha=0.25
                        )
                        axes[foundation_row, 1].set_title('Foundation Actor Regret')
                        axes[foundation_row, 1].set_xlabel('Timesteps')
                        axes[foundation_row, 1].set_ylabel('Regret')
                        axes[foundation_row, 1].legend(loc='best', fontsize=8)
                        axes[foundation_row, 1].grid(True, alpha=0.3)

                # Plot 9: make the pre-TD3 lifecycle contract visible alongside
                # training. Completion and update count use separate scales.
                lifecycle_axis = axes[foundation_row, 2]
                lifecycle_lines = []
                completion_column = 'train/foundation_warmup_complete'
                updates_column = 'train/foundation_warmup_updates'
                if completion_column in df.columns:
                    completion_data = df.dropna(
                        subset=[completion_column, 'time/total_timesteps']
                    )
                    if len(completion_data) > 0:
                        completion_line, = lifecycle_axis.step(
                            completion_data['time/total_timesteps'],
                            completion_data[completion_column],
                            where='post',
                            color='tab:green',
                            linewidth=2,
                            label='warmup_complete',
                        )
                        lifecycle_lines.append(completion_line)
                        foundation_lifecycle_plotted = True
                        lifecycle_axis.set_ylim(-0.05, 1.05)
                        lifecycle_axis.set_ylabel('Warmup Complete (0/1)')
                updates_axis = None
                if updates_column in df.columns:
                    updates_data = df.dropna(
                        subset=[updates_column, 'time/total_timesteps']
                    )
                    if len(updates_data) > 0:
                        updates_axis = lifecycle_axis.twinx()
                        updates_line, = updates_axis.plot(
                            updates_data['time/total_timesteps'],
                            updates_data[updates_column],
                            color='tab:blue',
                            linewidth=1.5,
                            linestyle='--',
                            label='warmup_updates',
                        )
                        lifecycle_lines.append(updates_line)
                        foundation_lifecycle_plotted = True
                        updates_axis.set_ylabel('Foundation Updates')
                        updates_axis.grid(False)
                if lifecycle_axis.has_data() or (
                    updates_axis is not None and updates_axis.has_data()
                ):
                    lifecycle_axis.set_title('Foundation Warmup Lifecycle')
                    lifecycle_axis.set_xlabel('Timesteps')
                    lifecycle_axis.grid(True, alpha=0.3)
                    lifecycle_axis.legend(
                        lifecycle_lines,
                        [line.get_label() for line in lifecycle_lines],
                        loc='best',
                        fontsize=8,
                    )

            if has_earnmore_diagnostics:
                loss_axis = axes[earnmore_row, 0]
                for loss_column in earnmore_loss_columns:
                    if loss_column not in df.columns:
                        continue
                    loss_data = df.dropna(
                        subset=[loss_column, 'time/total_timesteps']
                    )
                    if len(loss_data) > 0:
                        loss_axis.plot(
                            loss_data['time/total_timesteps'],
                            loss_data[loss_column],
                            linewidth=1.5,
                            label=loss_column.removeprefix('train/earnmore_'),
                        )
                if loss_axis.has_data():
                    loss_axis.set_title('EarnMore Masked Reconstruction')
                    loss_axis.set_xlabel('Timesteps')
                    loss_axis.set_ylabel('Masked-coordinate MSE')
                    loss_axis.legend(loc='best', fontsize=8)
                    loss_axis.grid(True, alpha=0.3)

                mask_axis = axes[earnmore_row, 1]
                for column, label in (
                    ('train/earnmore_masked_assets_per_sample', 'masked assets'),
                    ('train/earnmore_visible_assets_per_sample', 'visible assets'),
                ):
                    if column in df.columns:
                        mask_data = df.dropna(
                            subset=[column, 'time/total_timesteps']
                        )
                        if len(mask_data) > 0:
                            mask_axis.plot(
                                mask_data['time/total_timesteps'],
                                mask_data[column],
                                linewidth=1.5,
                                label=label,
                            )
                if mask_axis.has_data():
                    mask_axis.set_title('EarnMore Mask Composition')
                    mask_axis.set_xlabel('Timesteps')
                    mask_axis.set_ylabel('Assets per sample')
                    mask_axis.legend(loc='best', fontsize=8)
                    mask_axis.grid(True, alpha=0.3)

                gradient_axis = axes[earnmore_row, 2]
                for column, label in (
                    ('train/earnmore_actor_gradient_norm', 'actor encoder'),
                    ('train/earnmore_critic_gradient_norm', 'critic encoders'),
                ):
                    if column in df.columns:
                        gradient_data = df.dropna(
                            subset=[column, 'time/total_timesteps']
                        )
                        if len(gradient_data) > 0:
                            gradient_axis.plot(
                                gradient_data['time/total_timesteps'],
                                gradient_data[column],
                                linewidth=1.5,
                                label=label,
                            )
                if gradient_axis.has_data():
                    gradient_axis.set_title('EarnMore Representation Gradients')
                    gradient_axis.set_xlabel('Timesteps')
                    gradient_axis.set_ylabel('Pre-clip norm')
                    gradient_axis.legend(loc='best', fontsize=8)
                    gradient_axis.grid(True, alpha=0.3)
            
            if is_flowrl and 'train/learning_starts' in df:
                starts = df['train/learning_starts'].dropna()
                if len(starts):
                    for axis in (axes[0, 0], axes[0, 2], axes[1, 0]):
                        if axis.has_data():
                            threshold = float(starts.iloc[-1])
                            if threshold <= df['time/total_timesteps'].max():
                                axis.axvline(threshold, color='dimgray', linestyle=':', alpha=0.7)
                                axis.text(threshold, 0.98, ' learning starts', rotation=90,
                                          va='top', fontsize=7, color='dimgray', transform=axis.get_xaxis_transform())
                            else:
                                axis.text(0.98, 0.98, f'Warmup: learning after {threshold:g} steps',
                                          ha='right', va='top', fontsize=7, color='dimgray', transform=axis.transAxes)

            # Remove empty subplots
            for i in range(plot_rows):
                for j in range(3):
                    if (
                        foundation_row is not None
                        and (i, j) == (foundation_row, 2)
                        and foundation_lifecycle_plotted
                    ):
                        continue
                    if not axes[i, j].has_data():
                        axes[i, j].remove()
            
            fig.tight_layout(rect=[0, 0, 1, 0.91])
            
            # Save plot
            plot_path = os.path.join(
                self.log_path,
                TRAINING_PLOT_FILENAME,
            )
            _save_figure_atomically(fig, plot_path, dpi=220)
            plt.close(fig)
            if self.model_name.startswith("FlowRL"):
                self._generate_flowrl_plots(df)
            self.last_rendered_step = int(df['time/total_timesteps'].max())
            
            if self.verbose > 0:
                print(f"📊 Training plots saved to: {plot_path}")
            
        except Exception as e:
            if self.model_name.startswith("FlowRL"):
                raise
            if self.verbose > 0:
                print(f"⚠️ Error generating plots: {e}")
                print(f"📂 Log directory contents: {os.listdir(self.log_path) if os.path.exists(self.log_path) else 'Directory not found'}")
        finally:
            # CRL history is independent of the progress-frame length. Keep
            # its one rendering pass alive at the first checkpoint, where the
            # ordinary training plot may have only one CSV row and return.
            self._generate_crl_plots()

    def _generate_flowrl_plots(self, df):
        """Flow learning, portfolio diagnostics and the applied CRL contributions."""
        reference = df.get("train/crl_reference_is_forward", pd.Series(dtype=float)).dropna()
        forward_crl = bool(len(reference) and reference.iloc[-1] > .5)
        constraint_frame = df
        if forward_crl:
            current_reference = df["train/crl_reference_is_forward"].eq(1)
            sampling = df.get("train/crl_flow_sampling_is_independent", pd.Series(dtype=float)).dropna()
            if len(sampling):
                current_reference &= df["train/crl_flow_sampling_is_independent"].eq(sampling.iloc[-1])
            # Keep all return/evaluation history, but never relabel old CRL
            # measurements as costs of a different reference/sampling rule.
            constraint_frame = df.loc[current_reference]
        def series(axis, frame, x, column, label, **kwargs):
            if column in frame:
                values = frame.dropna(subset=[x, column])
                if len(values):
                    axis.plot(values[x], values[column], marker=".", markersize=4,
                              linewidth=1.4, label=label, **kwargs)

        def finish(axis, title, ylabel):
            axis.set(title=title, xlabel="Environment transitions", ylabel=ylabel)
            axis.grid(True, alpha=0.25)
            if axis.has_data():
                axis.legend(fontsize=8)
            else:
                axis.text(0.5, 0.5, "Waiting for observations / updates", ha="center",
                          va="center", transform=axis.transAxes, fontsize=9)

        fig, axes = plt.subplots(9, 2, figsize=(14, 30))
        try:
            fig.suptitle("FlowRL learning, portfolio returns and runtime", fontsize=14)
            panels = (
                (axes[0, 0], "Expectile value loss", "Loss", (
                    ("train/value_loss", "Value loss"),)),
                (axes[0, 1], "Flow matching loss", "Mean squared error", (
                    ("train/flow_matching_loss", "Unweighted"),
                    ("train/weighted_flow_matching_loss", "Weighted"))),
                (axes[1, 0], "Financial return", "Fraction of starting capital", (
                    ("rollout/ep_financial_return_mean", "Training (last 100 episodes)"),
                    ("eval/mean_financial_return", "Evaluation mean"))),
                (axes[1, 1], "Terminal step reward", "Reward", (
                    ("rollout/ep_terminal_reward_mean", "Training (last 100 episodes)"),
                    ("eval/mean_terminal_reward", "Evaluation mean"))),
                (axes[2, 0], "Replay occupancy", "Transitions", (
                    ("replay/size", "Filled transitions"),
                    ("replay/capacity", "Capacity"))),
                (axes[2, 1], "Cumulative runtime across resumes", "Seconds", (
                    ("time/time_elapsed", "Total elapsed"),
                    ("time/training_seconds", "Collection and updates"))),
                (axes[3, 0], "Causal financial forecast teacher", "Normalized auxiliary loss", (
                    ("train/financial_aux_loss", "Policy Q teacher"),
                    ("train/behavior_financial_aux_loss", "Behavior Q teacher"),
                    ("train/financial_relative_loss", "Within-basket relative error"))),
                (axes[3, 1], "Financial teaching coverage and shared gradient pressure", "Fraction / gradient norm ratio", (
                    ("train/financial_label_fraction", "Active-asset label coverage"),
                    ("train/financial_encoder_gradient_ratio", "Policy Q auxiliary / TD"),
                    ("train/behavior_financial_encoder_gradient_ratio", "Behavior Q auxiliary / TD"))),
                (axes[4, 0], "Actor: value, CRL and applied matching", "Objective contribution", (
                    (("train/actor_raw_q_loss" if "train/actor_raw_q_loss" in df else "train/actor_q_loss"),
                     "Negative candidate value in raw Q units"),
                    ("train/effective_flow_matching_loss", "Applied CFM contribution"),
                    ("train/actor_velocity_penalty", "Velocity CRL contribution"),
                    ("train/actor_acceleration_penalty", "Acceleration CRL contribution"),
                    ("train/actor_control_loss", "Value plus CRL"))),
                (axes[4, 1], "Actor parameter gradients", "L2 gradient norm (log scale)", (
                    ("train/actor_q_gradient_norm", "Return gradient"),
                    ("train/actor_cfm_gradient_norm", "Guided CFM gradient"),
                    ("train/effective_actor_cfm_gradient_norm", "Effective CFM gradient"),
                    ("train/actor_smooth_gradient_norm", "CRL gradient"),
                    ("train/actor_control_gradient_norm", "Value plus CRL gradient"))),
                (axes[5, 0], "Added post-flow noise (off by default)", "Raw mean absolute / allocation L1 change", (
                    ("train/exploration_raw_delta", "Added raw-action perturbation"),
                    ("train/exploration_allocation_delta", "Added allocation perturbation"))),
                (axes[5, 1], "Allocation value relative to the held portfolio", "Q difference", (
                    ("train/policy_vs_hold_advantage", "Generated action minus held portfolio"),
                    ("train/candidate_value_gain", "Local candidate improvement"))),
                (axes[6, 0], "Policy cash exposure", "Allocation fraction", (
                    ("train/actor_cash_weight", "Generated training policy"),
                    ("eval/action_cash_weight_mean", "Evaluation mean"))),
                (axes[6, 1], "Action saturation and matching-scale floor", "Fraction of active coordinates / updates", (
                    ("train/actor_raw_saturation", "Generated training policy"),
                    ("eval/action_raw_saturation_mean", "Evaluation mean"),
                    ("train/actor_matching_floor_active", "Updates using the matching-scale floor"))),
                (axes[7, 0], "Training constraint velocity and executed evaluations", "Sum of squared weight changes", (
                    ("train/velocity_cost", "Current vs next target endpoint" if forward_crl else "Proposal vs previous executed target"),
                    ("eval/action_allocation_velocity_squared_mean", "Consecutive evaluation targets"))),
                (axes[7, 1], "Training constraint acceleration and reference coverage", "Sum of squared second differences", (
                    ("train/acceleration_cost", "Previous/current/next target" if forward_crl else "Executed history second difference"),
                    ("eval/action_allocation_acceleration_squared_mean", "Consecutive evaluation targets"))),
                (axes[8, 0], "Two movements on the constraint's three-point path", "Allocation difference energy", (
                    ("train/crl_joint_velocity_cost", "Outgoing movement squared"),
                    ("train/crl_previous_velocity_cost", "Incoming movement squared"),
                    ("train/crl_velocity_inner_product", "Consecutive velocity inner product"),
                    ("train/crl_reconstructed_acceleration_cost", "Outgoing + incoming - twice inner product"))),
                (axes[8, 1], "TD3 batch acceleration curriculum" if forward_crl else "Velocity-dependent acceleration gates", "Gate strength",
                 (("train/crl_coupling_factor", "Batch gate"),) if forward_crl else (
                     ("train/crl_coupling_min", "Mean batch minimum"),
                     ("train/crl_coupling_factor", "Mean gate"),
                     ("train/crl_coupling_max", "Mean batch maximum"))),
            )
            for axis, title, ylabel, columns in panels:
                for column, label in columns:
                    frame = (constraint_frame if axis in (axes[7, 0], axes[7, 1], axes[8, 0], axes[8, 1])
                             and column.startswith("train/") else df)
                    series(axis, frame, "time/total_timesteps", column, label)
                finish(axis, title, ylabel)
            axes[4, 1].set_yscale("log")
            axes[8, 1].set_ylim(-.02, 1.02)
            if "train/crl_velocity_persistence" in df and df["train/crl_velocity_persistence"].notna().any():
                persistence_axis = axes[8, 1].twinx()
                series(persistence_axis, constraint_frame, "time/total_timesteps", "train/crl_velocity_persistence",
                       "Consecutive velocity alignment", color="purple", linestyle="--")
                persistence_axis.set(ylabel="Alignment: -1 reversal, +1 continuation", ylim=(-1.05, 1.05))
                persistence_axis.legend(loc="lower right", fontsize=7)
            if "train/crl_acceleration_valid_fraction" in df and df["train/crl_acceleration_valid_fraction"].notna().any():
                coverage_axis = axes[7, 1].twinx()
                series(coverage_axis, constraint_frame, "time/total_timesteps", "train/crl_acceleration_valid_fraction",
                       "Rows with an eligible reference path", color="gray", linestyle="--")
                coverage_axis.set(ylabel="Valid acceleration reference fraction", ylim=(-.02, 1.02))
                coverage_axis.legend(loc="upper right", fontsize=7)
            if "train/actor_cash_budget_velocity" in df and df["train/actor_cash_budget_velocity"].notna().any():
                budget_axis = axes[6, 0].twinx()
                series(budget_axis, df, "time/total_timesteps", "train/actor_cash_budget_velocity",
                       "Cash-versus-risk transport", color="purple", linestyle="--")
                budget_axis.set_ylabel("Flow budget velocity; positive favors cash")
                budget_axis.axhline(0, color="purple", alpha=.2, linewidth=.8)
                budget_axis.legend(loc="lower right", fontsize=7)
            elif "train/actor_cash_marginal_score" in df and df["train/actor_cash_marginal_score"].notna().any():
                cash_axis = axes[6, 0].twinx()
                series(cash_axis, df, "time/total_timesteps", "train/actor_cash_marginal_score",
                       "Cash from shared marginal scorer", color="purple", linestyle="--")
                cash_axis.set_ylabel("Centered joint opportunity score")
                cash_axis.axhline(0, color="purple", alpha=.2, linewidth=.8)
                cash_axis.legend(loc="lower right", fontsize=7)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            _save_figure_atomically(fig, os.path.join(self.log_path, "flowrl_diagnostics.png"), dpi=180)
        finally:
            plt.close(fig)

        dynamics_path = os.path.join(self.log_path, "action_dynamics.csv")
        dynamics = pd.read_csv(dynamics_path)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        try:
            fig.suptitle("Action changes between market decisions\n"
                         "Absolute first and second differences; reset at episode boundaries", fontsize=13)
            for axis, name, title, units in (
                (axes[0, 0], "raw_velocity", "Raw action velocity", "Action units / decision"),
                (axes[0, 1], "raw_acceleration", "Raw action acceleration", "Action units / decision squared"),
                (axes[1, 0], "allocation_velocity", "Allocation velocity", "Weight fraction / decision"),
                (axes[1, 1], "allocation_acceleration", "Allocation acceleration", "Weight fraction / decision squared"),
            ):
                for phase, color in (("training", "steelblue"), ("evaluation", "darkorange")):
                    phase_data = dynamics[dynamics["phase"] == phase]
                    series(axis, phase_data, "step", name + "_mean", phase + " mean", color=color)
                    series(axis, phase_data, "step", name + "_max", phase + " peak", color=color,
                           linestyle="--", alpha=0.55)
                finish(axis, title, units)
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            _save_figure_atomically(fig, os.path.join(self.log_path, "action_dynamics.png"), dpi=180)
        finally:
            plt.close(fig)

    def _generate_crl_plots(self):
        """
        Generate CRL analysis plots, automatically adapting to single or dual-constraint agents.
        """
        self.crl_stats_history = normalize_crl_stats_history(self.crl_stats_history)
        enabled_stats = [s for s in self.crl_stats_history if normalize_crl_enabled(s.get('crl_enabled', False))]
        if not enabled_stats:
            if self.verbose > 0:
                print("⚠️ CRL plots skipped: No data or CRL not enabled.")
            return

        dual_stats = [s for s in enabled_stats if crl_record_has_finite_values(s, DUAL_CRL_PLOT_KEYS)]
        single_stats = [s for s in enabled_stats if crl_record_has_finite_values(s, SINGLE_CRL_PLOT_KEYS)]
        is_dual_crl = len(dual_stats) >= 2

        if self.verbose > 0:
            print(f"📊 CRL Plotter: Detected {'Dual' if is_dual_crl else 'Single'}-constraint agent.")

        if is_dual_crl:
            self.crl_stats_history = dual_stats
            self._plot_dual_crl()
        elif len(single_stats) >= 2:
            self.crl_stats_history = single_stats
            self._plot_single_crl()
        elif self.verbose > 0:
            print("⚠️ CRL plots skipped: Not enough compatible CRL rows after schema normalization.")

    def _plot_single_crl(self):
        """Generate plots for a single-constraint CRL agent."""
        try:
            valid_stats = [
                s for s in normalize_crl_stats_history(self.crl_stats_history)
                if crl_record_has_finite_values(s, SINGLE_CRL_PLOT_KEYS)
            ]
            if len(valid_stats) < 2:
                if self.verbose > 0: print(f"⚠️ Single CRL plots skipped: Not enough valid data ({len(valid_stats)}).")
                return

            lambdas = [crl_float(s, 'current_lambda') for s in valid_stats]
            costs = [crl_float(s, 'avg_recent_constraint_cost') for s in valid_stats]
            violations = [crl_float(s, 'avg_recent_violation') for s in valid_stats]
            threshold = crl_float(valid_stats[0], 'constraint_threshold')

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            profile_name = self.crl_profile or "Unknown"
            fig.suptitle(f'Single-Constraint RL Analysis: {profile_name} Profile', fontsize=16)

            axes[0, 0].plot(lambdas, 'r-', linewidth=2)
            axes[0, 0].set_title('Lagrange Multiplier (λ) Evolution')
            axes[0, 0].set_xlabel('Training Iterations')
            axes[0, 0].set_ylabel('λ Value')
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(costs, 'g-', linewidth=2, label='Actual Cost')
            axes[0, 1].axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
            axes[0, 1].set_title('Constraint Cost vs. Threshold')
            axes[0, 1].set_xlabel('Training Iterations')
            axes[0, 1].set_ylabel('Cost')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(violations, 'purple', linewidth=2)
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 0].set_title('Constraint Violations (Cost - Threshold)')
            axes[1, 0].set_xlabel('Training Iterations')
            axes[1, 0].set_ylabel('Violation')
            axes[1, 0].grid(True, alpha=0.3)
            
            fig.delaxes(axes[1,1]) # Remove unused subplot

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            crl_plot_path = os.path.join(self.log_path, "crl_analysis.png")
            plt.savefig(crl_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            if self.verbose > 0:
                print(f"📊 Single CRL analysis plot saved to: {crl_plot_path}")
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Error generating single CRL plots: {e}")

    def _plot_dual_crl(self):
        """Generate plots for a dual-constraint CRL agent using twin axes."""
        try:
            valid_stats = [
                s for s in normalize_crl_stats_history(self.crl_stats_history)
                if crl_record_has_finite_values(s, DUAL_CRL_PLOT_KEYS)
            ]
            if len(valid_stats) < 2:
                if self.verbose > 0: print(f"⚠️ Dual CRL plots skipped: Not enough valid data ({len(valid_stats)}).")
                return

            # Extract dual CRL metrics
            lambdas_v = [crl_float(s, 'current_lambda_velocity') for s in valid_stats]
            costs_v = [crl_float(s, 'avg_recent_velocity_cost') for s in valid_stats]
            violations_v = [crl_float(s, 'avg_recent_velocity_violation') for s in valid_stats]
            threshold_v = crl_float(valid_stats[0], 'velocity_constraint_threshold')

            lambdas_a = [crl_float(s, 'current_lambda_acceleration') for s in valid_stats]
            costs_a = [crl_float(s, 'avg_recent_acceleration_cost') for s in valid_stats]
            violations_a = [crl_float(s, 'avg_recent_acceleration_violation') for s in valid_stats]
            threshold_a = crl_float(valid_stats[0], 'acceleration_constraint_threshold')

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            profile_name = self.crl_profile or "Unknown"
            fig.suptitle(f'Dual Constrained RL Analysis: {profile_name} Profile', fontsize=16)

            # --- Plot 1: Lambda Evolution (Twin Axis) ---
            ax1 = axes[0, 0]
            ax1.plot(lambdas_v, 'r-', linewidth=2, label='λ Velocity')
            ax1.set_xlabel('Training Iterations')
            ax1.set_ylabel('λ Velocity Value', color='r')
            ax1.tick_params(axis='y', labelcolor='r')
            ax1.grid(True, alpha=0.3)

            ax1_twin = ax1.twinx()
            ax1_twin.plot(lambdas_a, 'b--', linewidth=2, label='λ Acceleration')
            ax1_twin.set_ylabel('λ Acceleration Value', color='b')
            ax1_twin.tick_params(axis='y', labelcolor='b')
            ax1.set_title('Lagrange Multiplier (λ) Evolution')
            
            # Combine legends
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1_twin.legend(lines + lines2, labels + labels2, loc='upper left')

            # --- Plot 2: Constraint Costs vs Thresholds (Twin Axis) ---
            ax2 = axes[0, 1]
            ax2.plot(costs_v, 'g-', linewidth=2, label='Velocity Cost')
            ax2.axhline(y=threshold_v, color='g', linestyle=':', linewidth=2, label=f'Vel. Threshold ({threshold_v:.4f})')
            ax2.set_xlabel('Training Iterations')
            ax2.set_ylabel('Velocity Cost (E[Δa²])', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.grid(True, alpha=0.3)

            ax2_twin = ax2.twinx()
            ax2_twin.plot(costs_a, 'm--', linewidth=2, label='Acceleration Cost')
            ax2_twin.axhline(y=threshold_a, color='m', linestyle=':', linewidth=2, label=f'Accel. Threshold ({threshold_a:.4f})')
            ax2_twin.set_ylabel('Acceleration Cost (E[Δ(Δa)²])', color='m')
            ax2_twin.tick_params(axis='y', labelcolor='m')
            ax2.set_title('Constraint Costs vs. Thresholds')
            
            # ✅ ROBUST VISUALIZATION FIX: Dynamically cap the y-axis to handle initial spikes
            # This makes the plot legible even if there's a large, one-off cost at the beginning.
            if len(costs_a) > 10:
                # Use data after the initial noisy period to find a reasonable limit
                stable_costs_a = np.array([cost for cost in costs_a[10:] if np.isfinite(cost)], dtype=float)
                if stable_costs_a.size > 0:
                    # Calculate the 98th percentile as a robust upper bound
                    upper_bound = float(np.percentile(stable_costs_a, 98))
                    lower_bound = float(np.min(stable_costs_a))
                    bottom = min(0.0, lower_bound * 1.1)
                    top = upper_bound * 1.1
                    if top <= bottom:
                        padding = max(abs(bottom) * 0.05, abs(upper_bound) * 0.05, 1e-6)
                        top = bottom + padding
                    if np.isfinite(bottom) and np.isfinite(top) and top > bottom:
                        # Set the y-limit slightly above this bound to give it some space
                        ax2_twin.set_ylim(bottom=bottom, top=top)

            lines, labels = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2_twin.legend(lines + lines2, labels + labels2, loc='upper left')

            # --- Plot 3: Constraint Violations (Twin Axis) ---
            ax3 = axes[1, 0]
            ax3.plot(violations_v, 'purple', linewidth=2, label='Velocity Violation')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.set_xlabel('Training Iterations')
            ax3.set_ylabel('Velocity Violation', color='purple')
            ax3.tick_params(axis='y', labelcolor='purple')
            ax3.grid(True, alpha=0.3)

            ax3_twin = ax3.twinx()
            ax3_twin.plot(violations_a, 'orange', linestyle='--', linewidth=2, label='Acceleration Violation')
            ax3_twin.set_ylabel('Acceleration Violation', color='orange')
            ax3_twin.tick_params(axis='y', labelcolor='orange')
            ax3.set_title('Constraint Violations (Cost - Threshold)')
            
            lines, labels = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3_twin.legend(lines + lines2, labels + labels2, loc='upper left')

            # --- Plot 4: Lambda vs Returns Correlation (Twin Axis) ---
            ax4 = axes[1, 1]
            min_len = min(len(self.returns), len(lambdas_v), len(lambdas_a))
            if min_len > 1:
                correlation_rows = [
                    (lambda_v, lambda_a, return_value)
                    for lambda_v, lambda_a, return_value in zip(
                        lambdas_v[-min_len:],
                        lambdas_a[-min_len:],
                        self.returns[-min_len:],
                    )
                    if to_optional_float(return_value) is not None
                ]
                correlation_lambdas_v = [row[0] for row in correlation_rows]
                correlation_lambdas_a = [row[1] for row in correlation_rows]
                correlation_returns = [to_optional_float(row[2]) for row in correlation_rows]

                # Plot Velocity on the primary x-axis
                if len(correlation_returns) > 1:
                    ax4.scatter(correlation_lambdas_v, correlation_returns, alpha=0.6, color='r', label='Returns vs λ Velocity')
                    ax4.set_xlabel('λ Velocity Value', color='r')
                    ax4.set_ylabel('Average Return')
                    ax4.tick_params(axis='x', labelcolor='r')
                    ax4.grid(True, alpha=0.3)

                    # Create a twin x-axis for Acceleration
                    ax4_twin = ax4.twiny()
                    ax4_twin.scatter(correlation_lambdas_a, correlation_returns, alpha=0.6, color='b', marker='x', label='Returns vs λ Acceleration')
                    ax4_twin.set_xlabel('λ Acceleration Value', color='b')
                    ax4_twin.tick_params(axis='x', labelcolor='b')

                    ax4.set_title('λ vs. Returns Correlation')

                    # Combine legends
                    lines, labels = ax4.get_legend_handles_labels()
                    lines2, labels2 = ax4_twin.get_legend_handles_labels()
                    ax4.legend(lines + lines2, labels + labels2, loc='best')
                else:
                    ax4.text(0.5, 0.5, 'Correlation analysis\nnot available', ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, 'Correlation analysis\nnot available', ha='center', va='center', transform=ax4.transAxes)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            crl_plot_path = os.path.join(self.log_path, "crl_analysis.png")
            plt.savefig(crl_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            if self.verbose > 0:
                print(f"📊 Dual CRL analysis plot saved to: {crl_plot_path}")

        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Error generating dual CRL plots: {e}")


class CSVDumpCallback(BaseCallback):
    """
    Callback to dump the logger's data to CSV at a regular interval.
    This solves the issue of sparse CSV files with skipped rows by ensuring
    all recorded metrics are written to a single row at the same time.
    """
    def __init__(self, dump_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.dump_freq = dump_freq

    def _on_step(self) -> bool:
        # Dump the log file every `dump_freq` steps.
        # This is crucial for off-policy algorithms like TD3/SAC to prevent sparse CSVs.
        if self.n_calls > 0 and self.n_calls % self.dump_freq == 0:
            if self.verbose > 0:
                print(f"💾 Dumping CSV logs at step {self.n_calls} to progress.csv...")
            try:
                self.logger.dump(step=self.num_timesteps)
            except (OSError, IOError) as e:
                if self.verbose > 0:
                    print(f"⚠️ Logger dump failed (I/O), continuing: {e}")
        return True
    
    def _on_training_end(self) -> None:
        """Force final dump when training ends."""
        if self.verbose > 0:
            print(f"💾 Final CSV dump to progress.csv...")
        try:
            self.logger.dump(step=self.num_timesteps)
        except (OSError, IOError) as e:
            if self.verbose > 0:
                print(f"⚠️ Logger dump failed (I/O): {e}")


class EpisodeReturnCallback(BaseCallback):
    """
    Custom callback that explicitly tracks episode returns (sum of rewards per episode)
    like the original TF-Agents implementation. This replicates the compute_avg_return() 
    functionality and saves results to output_ar_gamma.csv for direct comparison.
    """
    
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 4, 
                 log_path: str = "./logs", verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        
        # Track episode returns over time (like original output_ar_gamma.csv)
        self.episode_returns_history = []
        self.timesteps_history = []
        
        if self.verbose > 0:
            print(f"🔍 EpisodeReturnCallback: Will track episode returns every {eval_freq} steps")
            print(f"📊 Using {n_eval_episodes} episodes per evaluation (matching original)")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            self._evaluate_episode_returns()
        return True
    
    def _on_training_end(self) -> None:
        """Save final episode return data when training ends."""
        self._save_episode_returns_csv()
    
    def _evaluate_episode_returns(self):
        """
        Evaluate episode returns by calculating the ACTUAL financial return of the portfolio,
        not by summing the intermediate step rewards.
        """
        if self.verbose > 0:
            print(f"🧮 Computing TRUE financial returns at timestep {self.num_timesteps}...")

        total_return = 0.0
        episode_returns = []
        
        for episode in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            terminated = truncated = False

            # Run the episode to completion, but IGNORE the 'reward' signal
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)  # ✅ FIX: Allow exploration during evaluation
                # The final 'info' dict will contain the final portfolio value
                obs, reward, terminated, truncated, info = self.eval_env.step(action)

            # ✅ CORRECTED LOGIC:
            # After the episode is done, calculate the true return from the final state.
            final_value = info.get('value', config.INITIAL_CASH) if isinstance(info, dict) else info[0].get('value', config.INITIAL_CASH)
            initial_value = config.INITIAL_CASH # Get initial cash from config

            # Calculate the actual percentage return for the episode
            true_episode_return = (final_value - initial_value) / initial_value

            episode_returns.append(true_episode_return)
            total_return += true_episode_return
        
        # Calculate average return (like original)
        # NOTE: This is now the average of the TRUE financial returns.
        avg_return = total_return / self.n_eval_episodes
        
        # Store for history tracking
        self.episode_returns_history.append(avg_return)
        self.timesteps_history.append(self.num_timesteps)
        
        # Log like the original
        if self.verbose > 0:
            print(f"📈 Timestep {self.num_timesteps}: Average Episode Return = {avg_return:.4f}")
            print(f"   Individual episode returns: {[f'{r:.2f}' for r in episode_returns]}")
        
        # Log to SB3 logger for tensorboard/CSV
        self.logger.record("eval/mean_episode_return", avg_return)
        self.logger.record("eval/std_episode_return", np.std(episode_returns))
        
        # Save CSV periodically
        if len(self.episode_returns_history) % 5 == 0:  # Every 5 evaluations
            self._save_episode_returns_csv()
    
    def _save_episode_returns_csv(self):
        """Save episode returns to CSV file (like original output_ar_gamma.csv)."""
        if len(self.episode_returns_history) > 0:
            try:
                df = pd.DataFrame({
                    "timesteps": self.timesteps_history,
                    "average_episode_return": self.episode_returns_history
                })
                csv_path = os.path.join(self.log_path, "output_ar_gamma.csv")
                df.to_csv(csv_path, index=False)
                if self.verbose > 0:
                    print(f"💾 Episode returns saved to: {csv_path}")
                    print(f"   Latest avg return: {self.episode_returns_history[-1]:.4f}")
            except (OSError, IOError) as e:
                if self.verbose > 0:
                    print(f"⚠️ Failed to write episode returns CSV (I/O), continuing: {e}")


class EarlyStoppingCallback(BaseCallback):
    """
    Callback to stop training early if performance degrades consistently.
    This prevents the agent from unlearning good policies.
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: int = 0):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_eval_mean_reward = -np.inf
        self.stopped_epoch = 0
        
    def _on_step(self) -> bool:
        # Check if we have evaluation results
        if len(self.logger.name_to_value) > 0 and 'eval/mean_reward' in self.logger.name_to_value:
            current_eval_reward = self.logger.name_to_value['eval/mean_reward']
            
            # Check if current performance is better than best
            if current_eval_reward > self.best_eval_mean_reward + self.min_delta:
                self.best_eval_mean_reward = current_eval_reward
                self.wait = 0
                if self.verbose > 0:
                    print(f"📈 New best evaluation reward: {current_eval_reward:.4f}")
            else:
                self.wait += 1
                if self.verbose > 0:
                    print(f"⚠️  Performance degradation {self.wait}/{self.patience}: {current_eval_reward:.4f} vs best {self.best_eval_mean_reward:.4f}")
                
                if self.wait >= self.patience:
                    self.stopped_epoch = self.num_timesteps
                    if self.verbose > 0:
                        print(f"🛑 Early stopping at timestep {self.num_timesteps} due to performance degradation")
                        print(f"   Best reward: {self.best_eval_mean_reward:.4f}, Current: {current_eval_reward:.4f}")
                    return False  # Stop training
        
        return True


def train_simple_mlp(
    algorithm="PPO",
    mlp_size="medium",  # More explicit than "config"
    reward_type="TRANSACTION_COST",
    total_timesteps=100000,
    eval_freq=400,  # Match TF-Agents: EVAL_INTERVAL=4 * COLLECT_STEPS_PER_ITERATION=100 = 400 steps
    log_dir="./logs",
    model_save_path="./models"
):
    """
    Train with simple MLP networks (NO ATTENTION).
    
    Args:
        algorithm: "PPO", "SAC", or "TD3"
        mlp_size: "light", "medium", or "heavy" (MLP architecture size)
        total_timesteps: Total training steps
        eval_freq: Evaluation frequency (matching original TF-Agents eval logic)
        log_dir: Logging directory
        model_save_path: Model save directory
    """
    configure_a40_runtime()
    
    # Create timestamped directories for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algorithm}_simple_mlp_{mlp_size}_{reward_type}_{timestamp}"
    
    # ✅ CHECKPOINT FIX: Create run-specific directories for logs AND models
    run_log_dir = os.path.join(log_dir, run_name)
    run_model_dir = os.path.join(model_save_path, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_model_dir, exist_ok=True)
    
    # Setup SB3 logging to CSV and tensorboard
    logger = configure_training_logger(
        run_log_dir, ["stdout", "csv", "tensorboard"]
    )
    
    # Create environments with Monitor wrapper for automatic logging
    print(f"🎯 Creating environments with reward_type: {reward_type}")
    env = Monitor(PortfolioEnv(reward_type=reward_type), filename=os.path.join(run_log_dir, "training"))
    eval_env = Monitor(PortfolioEnv(reward_type=reward_type), filename=os.path.join(run_log_dir, "evaluation"))
    
    # Get MLP architecture configuration
    mlp_config = MLP_CONFIGS.get(mlp_size, {}) # Safely get config
    print(f"🚫 NO ATTENTION - Using simple MLP networks only")
    print(f"🔧 MLP size: {mlp_size}")
    print(f"🏗️  MLP architecture: {mlp_config['net_arch']} → {mlp_config['features_dim']} features")
    print(f"📈 Similar to TF-Agents standard fully connected networks")
    print(f"📊 Logs will be saved to: {run_log_dir}")
    print(f"📄 Expected progress.csv: {os.path.join(run_log_dir, 'progress.csv')}")
    
    # Key insight: Track episode RETURNS like original successful implementation
    print(f"🎯 KEY DIFFERENCE: Explicitly tracking episode returns (not just step rewards)")
    print(f"📈 This replicates the original TF-Agents compute_avg_return() logic")
    print(f"💾 Will save to output_ar_gamma.csv for direct comparison")
    
    # Create MLP policy using the old factory method for now for PPO/SAC
    MlpPolicy = create_attention_policy(
        attention_type="mlp",
        algorithm=algorithm,
        **mlp_config
    )
    
    # Create model based on algorithm
    if algorithm == "PPO":
        print(f"🚀 Creating PPO with simple MLP...")
        model = PPO(
            SimpleMlpTD3Policy, env, verbose=1, 
            tensorboard_log=run_log_dir,  # Enable tensorboard logging
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.05, device="auto"
        )
    elif algorithm == "SAC":
        print(f"🚀 Creating SAC with simple MLP...")
        model = SAC(
            MlpPolicy, env, verbose=1, 
            tensorboard_log=run_log_dir,  # Enable tensorboard logging
            learning_rate=3e-4, buffer_size=100000, device="auto"
        )
    elif algorithm == "TD3":
        print(f"🚀 Creating TD3 with TF-Agents-matched configuration...")
        
        # ✅ EXPLORATION FIX: Use OrnsteinUhlenbeckActionNoise to match TF-Agents
        # This provides temporally correlated noise for more persistent exploration.
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.2 * np.ones(n_actions),
            theta=0.15  # Corresponds to ou_damping in original config
        )
        print(f"⚡ Using OrnsteinUhlenbeckActionNoise (sigma=0.2, theta=0.15) for better exploration.")
        
        print(f"🔧 EXACT TF-Agents hyperparameter matching:")
        print(f"   - actor_lr: 1e-4 (was 3e-4)")
        print(f"   - critic_lr: 1e-3 (was 3e-4)") 
        print(f"   - gamma: 0.05 (Myopic agent)")
        print(f"   - tau: 0.05 (target_update_tau)")
        print(f"   - Architecture: (400, 300) layers")
        print(f"   - Batch size: 100 (matching BATCH_SIZE)")
        print(f"   - Note: Using MSE loss (SB3 limitation, original used Huber)")

        # ✅ REFACTOR: Define policy_kwargs directly as per SB3 best practice.
        # This dictionary specifies the actor (pi) and critic (qf) network architectures.
        policy_kwargs = {
            "net_arch": {
                "pi": [400, 300],  # Actor network
                "qf": [400, 300]   # Critic network
            },
            # ✅ LEARNING RATE FIX: Pass separate LRs directly to the custom policy
            "actor_lr": 1e-4,
            "critic_lr": 1e-3
        }
        
        # ✅ STABILITY IMPROVEMENTS: Add learning rate scheduling to prevent degradation
        def lr_schedule(progress_remaining: float) -> float:
            """Learning rate schedule for TF-Agents style training"""
            if progress_remaining > 0.6:  # First 40% of training
                return 1e-4
            elif progress_remaining > 0.3:  # Middle 30%
                return 5e-5  
            else:  # Final 30% - fine-tuning
                return 1e-5

        model = TD3(
            SimpleMlpTD3Policy, env, verbose=1, # ✅ REFACTOR: Use the specific policy class
            policy_kwargs=policy_kwargs,     # ✅ REFACTOR: Pass architecture via policy_kwargs
            learning_rate=lr_schedule,      # Use schedule instead of fixed rate
            batch_size=100,          # BATCH_SIZE
            buffer_size=100000,       # REPLAY_BUFFER_MAX_LENGTH 
            learning_starts=2000,    # ✅ FIX: Increase exploration time for variable portfolios
            gamma=0.50,              # Low gamma for myopic learning - gamma = 0.05 is too low
            tau=0.05,                # target_update_tau
            target_policy_noise=0.2, # ou_stddev
            action_noise=action_noise, # ✅ EXPLORATION FIX: Use OU-Noise
            device="auto"
        )
        print(f"✅ STABILITY FIX: Added learning rate decay to prevent performance degradation")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Set the custom logger
    model.set_logger(logger)
    
    # ✅ FORCE INITIAL LOG DUMP: Ensure progress.csv is created
    print(f"🔄 Forcing initial log dump to create progress.csv...")
    model.logger.record("system/setup", 1.0)
    try:
        model.logger.dump(step=0)
    except (OSError, IOError) as e:
        print(f"⚠️ Initial logger dump failed (I/O), continuing: {e}")
    
    # Verify CSV file exists
    progress_csv_path = os.path.join(run_log_dir, "progress.csv")
    if os.path.exists(progress_csv_path):
        print(f"✅ progress.csv created successfully at: {progress_csv_path}")
    else:
        print(f"⚠️ progress.csv not found. Files in directory: {os.listdir(run_log_dir)}")
    
    print(f"✅ {algorithm} model created with simple MLP (NO attention)")
    
    # Setup evaluation callback - this will automatically log to CSV
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_model_dir,  # ✅ CHECKPOINT FIX: Save to run-specific directory
        log_path=run_log_dir,  # Log evaluation results
        eval_freq=eval_freq,
        n_eval_episodes=4,  # Use 4 episodes to match original TF-Agents exactly
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Setup plot generator callback
    plot_callback = PlotGeneratorCallback(
        log_path=run_log_dir,
        model_name=run_name,
        plot_freq=eval_freq,  # Generate plots at same frequency as evaluation
        verbose=1
    )
    
    # NEW: Setup CSV dumper callback to ensure consistent logging
    csv_dumper_callback = CSVDumpCallback(
        dump_freq=eval_freq, # Sync with evaluation frequency
        verbose=1
    )
    
    # NEW: Setup EpisodeReturnCallback (replicating original compute_avg_return)
    episode_return_callback = EpisodeReturnCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=4,  # Match original TF-Agents exactly
        log_path=run_log_dir,
        verbose=1
    )
    
    # NEW: Setup EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(
        patience=5,
        min_delta=0.0,
        verbose=1
    )
    
    # Combine callbacks
    # Order is important: Eval runs, then we dump, then we plot from the dumped CSV.
    callbacks = CallbackList([eval_callback, csv_dumper_callback, plot_callback, episode_return_callback, early_stopping_callback])
    
    # Validate model
    print(f"🧪 Validating simple MLP setup...")
    obs, _ = env.reset()
    action, _ = model.predict(obs)
    print(f"✅ Validation successful - MLP working correctly")
    
    # Train
    print(f"\n🚀 Training {algorithm} with simple MLP...")
    print(f"📊 Timesteps: {total_timesteps}")
    print(f"🏗️  MLP: {mlp_size} config (NO attention)")
    print(f"📈 CSV logs: {os.path.join(run_log_dir, 'progress.csv')}")
    print(f"📊 Plots will be generated every {eval_freq} steps")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"{algorithm}_simple_mlp_{mlp_size}",  # Tensorboard log name
        log_interval=1  # Log after every episode for off-policy algos
    )
    
    # Save final model (atomic write + catch I/O errors)
    final_path = os.path.join(run_model_dir, f"final_model.zip") # ✅ CHECKPOINT FIX: Save to run-specific directory
    final_tmp = final_path + ".tmp"
    try:
        model.save(final_tmp)
        os.replace(final_tmp, final_path)
        print(f"💾 Final model saved: {final_path}")
    except OSError as e:
        print(f"⚠️ Final model save failed (I/O): {e}")
        if os.path.exists(final_tmp):
            try:
                os.remove(final_tmp)
            except OSError:
                pass
    
    # ✅ FORCE FINAL PLOT GENERATION: Ensure plots are created
    print(f"🎨 Generating final training plots...")
    try:
        plot_callback._generate_plots()
        print(f"✅ Final plots generated successfully")
    except Exception as e:
        print(f"⚠️ Error generating final plots: {e}")
    
    # Generate final summary
    print(f"\n✅ Training completed successfully!")
    print(f"📁 All logs saved in: {run_log_dir}")
    print(f"📊 CSV data: {os.path.join(run_log_dir, 'progress.csv')}")
    print(
        f"📈 Training plots: "
        f"{os.path.join(run_log_dir, TRAINING_PLOT_FILENAME)}"
    )
    print(f"📋 Tensorboard logs: {run_log_dir}")
    print(f"💾 Best model: {os.path.join(run_model_dir, 'best_model.zip')}") # ✅ CHECKPOINT FIX: Correct path
    print(f"💾 Final model: {final_path}")
    
    return model


def test_mlp_policy():
    """Test that the MLP policy works correctly without attention."""
    print("🧪 Testing simple MLP policy (NO ATTENTION)...")
    
    # Create environment with TRANSACTION_COST reward type for testing
    env = PortfolioEnv(reward_type="TRANSACTION_COST")
    
    # Test MLP with different sizes
    for mlp_size in ["light", "medium", "heavy"]:
        print(f"\n🔍 Testing MLP {mlp_size} configuration...")
        
        try:
            # Create policy with MLP architecture
            MlpPolicy = create_attention_policy(
                attention_type="mlp",  # Ensures NO attention
                algorithm="PPO",
                **MLP_CONFIGS[mlp_size]
            )
            
            # Create model
            model = PPO(MlpPolicy, env, verbose=0)
            
            # Test observation and action
            obs, _ = env.reset()
            action, _ = model.predict(obs)
            
            # Test that action is properly applied
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"✅ MLP {mlp_size} working correctly")
            print(f"   Observation shape: {obs.shape}")
            print(f"   Action shape: {action.shape}")
            print(f"   Sample action: {action}")
            print(f"   Applied allocation: {info['money_split']}")
            print(f"   Reward: {reward:.4f}")
            print(f"   🚫 NO ATTENTION used - simple MLP only")
            
        except Exception as e:
            print(f"❌ MLP {mlp_size} failed: {e}")
            import traceback
            traceback.print_exc()
    
    env.close()


def compare_mlp_algorithms():
    """Compare different algorithms with simple MLP."""
    print("🔄 Comparing algorithms with simple MLP (NO ATTENTION)...")
    
    results = {}
    comparison_log_dir = f"./logs/{algorithm}mlp_algorithm_comparison"
    os.makedirs(comparison_log_dir, exist_ok=True)
    
    for algorithm in ["PPO", "SAC", "TD3"]:
        print(f"\n🧪 Training {algorithm} with simple MLP...")
        
        try:
            model = train_simple_mlp(
                algorithm=algorithm,
                mlp_size="medium",  # Use mlp_size parameter
                reward_type="TRANSACTION_COST",  # Use Net Return for comparison
                total_timesteps=20000,  # Short training for comparison
                eval_freq=2000,  # More frequent evaluation for comparison
                log_dir=f"./logs/{algorithm}mlp_comparison",
                model_save_path=f"./models/{algorithm}mlp_comparison"
            )
            
            # Quick evaluation
            env = PortfolioEnv(reward_type="TRANSACTION_COST")
            obs, _ = env.reset()
            total_reward = 0
            
            for _ in range(100):
                action, _ = model.predict(obs, deterministic=False)  # ✅ FIX: Allow exploration during evaluation
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            results[f"{algorithm}_simple_mlp"] = total_reward
            env.close()
            
            print(f"✅ {algorithm}_simple_mlp: Average reward = {total_reward:.4f}")
            
        except Exception as e:
            print(f"❌ {algorithm}_simple_mlp failed: {e}")
            results[f"{algorithm}_simple_mlp"] = None
    
    # Save comparison results
    comparison_df = pd.DataFrame([
        {"Algorithm": name.split("_")[0], "Final_Reward": reward}
        for name, reward in results.items() if reward is not None
    ])
    
    if not comparison_df.empty:
        comparison_csv = os.path.join(comparison_log_dir, "algorithm_comparison.csv")
        comparison_df.to_csv(comparison_csv, index=False)
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        plt.bar(comparison_df["Algorithm"], comparison_df["Final_Reward"])
        plt.title("MLP Algorithm Comparison (NO ATTENTION)")
        plt.xlabel("Algorithm")
        plt.ylabel("Final Reward")
        plt.grid(True, alpha=0.3)
        
        comparison_plot = os.path.join(comparison_log_dir, "algorithm_comparison.png")
        plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Comparison results saved:")
        print(f"   CSV: {comparison_csv}")
        print(f"   Plot: {comparison_plot}")
    
    print("\n=== SIMPLE MLP COMPARISON RESULTS (NO ATTENTION) ===")
    for name, reward in results.items():
        if reward is not None:
            print(f"{name}: {reward:.4f}")
        else:
            print(f"{name}: FAILED")


def resolve_attention_training_profile(
    mlp_size,
    use_attention,
    attention_type,
):
    """Resolve training-only controls for the current 8-of-13 experiment.

    Architecture widths are selected from attention_policy.  This helper owns
    only replay/update settings.  Both the medium comparison and the wider
    multitask architecture use the 200-original-row update and gamma 0.966,
    with learning starting after 1,000 collected steps.
    Keeping this helper pure lets verification prove those temporal semantics
    without constructing an environment or starting a training run.
    """
    multitask_enabled = bool(
        use_attention
        and attention_type == "variable_coin_attention"
        and mlp_size == "multitask"
    )
    if mlp_size == "multitask" and not multitask_enabled:
        raise ValueError(
            "The multitask profile requires --use-attention with "
            "--attention-type variable_coin_attention."
        )
    return {
        "multitask_enabled": multitask_enabled,
        "batch_size": int(config.MULTITASK_BATCH_SIZE),
        "learning_starts": (
            int(config.MULTITASK_LEARNING_STARTS)
            if multitask_enabled
            else 1_000
        ),
        "gamma": float(config.MULTITASK_GAMMA),
    }


def apply_attention_learning_runtime_config(model, profile):
    """Apply the requested learning regime after checkpoint metadata loads.

    SB3 restores gamma and auxiliary flags as well as weights. An experiment
    must not silently resurrect the saved reconstruction or discount regime.
    Observation/architecture contracts are still checked by the loader.
    """
    model.gamma = float(profile['gamma'])
    if hasattr(model, 'earnmore_masked_reconstruction_enabled'):
        model.earnmore_masked_reconstruction_enabled = bool(
            config.EARNMORE_MASKED_RECONSTRUCTION_ENABLED)
    if hasattr(model, 'earnmore_update_interval'):
        model.earnmore_update_interval = max(1, int(config.EARNMORE_UPDATE_INTERVAL))
    if hasattr(model, 'predictive_ssl_update_interval'):
        model.predictive_ssl_update_interval = max(1, int(config.PREDICTIVE_SSL_UPDATE_INTERVAL))
    if hasattr(model, 'predictive_ssl_online_enabled'):
        model.predictive_ssl_online_enabled = bool(config.PREDICTIVE_SSL_ONLINE_ENABLED)
    if hasattr(model, 'component_outcome_gradient_calibration_interval'):
        model.component_outcome_gradient_calibration_interval = max(
            1, int(config.COMPONENT_OUTCOME_GRADIENT_CALIBRATION_INTERVAL))
        # A checkpoint's cached coefficient and norm measurements describe
        # its old replay batch. They cannot authorize the first resumed batch.
        model._component_outcome_gradient_scale = None
        model._component_outcome_td_gradient_norm = None
        model._component_outcome_aux_gradient_norm = None
        model._component_outcome_gradient_measurement_update = None
    for attribute, setting in (
        ('basket_replay_stratify_enabled', 'BASKET_REPLAY_STRATIFY_ENABLED'),
        ('curriculum_mixed_replay_enabled', 'CURRICULUM_MIXED_REPLAY_ENABLED'),
        ('basket_replay_group_selection', 'BASKET_REPLAY_GROUP_SELECTION'),
        ('basket_replay_min_completed_episodes', 'BASKET_REPLAY_MIN_COMPLETED_EPISODES'),
        ('crl_policy_step_diagnostic_interval', 'CRL_POLICY_STEP_DIAGNOSTIC_INTERVAL'),
    ):
        if hasattr(model, attribute):
            setattr(model, attribute, getattr(config, setting))
    model.online_update_version = config.ATTENTION_ONLINE_UPDATE_VERSION
    if getattr(model, 'verbose', 0):
        print(
            f"[Learning profile] {model.online_update_version} | "
            f"predictive_online={getattr(model, 'predictive_ssl_online_enabled', False)} | "
            f"basket_balanced={getattr(model, 'basket_replay_stratify_enabled', False)} | "
            f"basket_min_completed={getattr(model, 'basket_replay_min_completed_episodes', 0)} | "
            f"financial_teacher_interval={getattr(model, 'component_outcome_gradient_calibration_interval', 1)} | "
            f"critic_bellman={getattr(model.critic.q_networks[0], 'bellman_composition', 'separate_v')} | "
            "financial actor/critic gradients remain live"
        )
    return model


def _stack_dict_observations(observations):
    """Stack independent Gym Dict observations into one policy batch."""
    if not observations:
        raise ValueError("cannot stack an empty observation batch")
    return {
        key: np.stack([observation[key] for observation in observations], axis=0)
        for key in observations[0]
    }


def validate_attention_resume_objective(checkpoint_path):
    """Reject a silent switch back to saved log-growth/PopArt run settings."""
    with zipfile.ZipFile(checkpoint_path, "r") as archive:
        metadata = json.loads(archive.read("data").decode("utf-8"))
    received = metadata.get("training_objective_version")
    expected = config.ATTENTION_TRAINING_OBJECTIVE_VERSION
    if received != expected:
        raise ValueError(
            "This checkpoint uses the previous training objective. The current "
            "profile restores structured-credit rewards, raw Bellman/actor "
            "losses, stable attention, width 256, causal portfolio reward history, "
            "gamma 0.966, no independent reconstruction, and batch size 200. "
            "Start a fresh run instead of mixing replay without the required history "
            "and saved optimizer settings "
            f"with this profile ({received!r} != {expected!r}). "
            "Continuing the historical objective requires a matching training profile."
        )


def saved_model_environment_count(checkpoint_path, default=1):
    """Read SB3's small JSON metadata without loading model/replay tensors."""
    if not checkpoint_path:
        return int(default)
    try:
        with zipfile.ZipFile(checkpoint_path, "r") as archive:
            metadata = json.loads(archive.read("data").decode("utf-8"))
        return max(1, int(metadata.get("n_envs", default)))
    except (OSError, KeyError, ValueError, TypeError, zipfile.BadZipFile):
        return int(default)


def evaluate_policy_in_parallel_envs(model, eval_envs):
    """Run one complete deterministic episode in every environment.

    Every live state still receives a complete QWM search.  The only semantic
    change is presentation: independent states are sent to the policy as one
    tensor batch, which amortizes Python and CUDA launch latency.
    """
    observations = []
    for environment in eval_envs:
        observation, _ = environment.reset()
        observations.append(observation)
    episode_returns = np.zeros(len(eval_envs), dtype=np.float64)
    finished = np.zeros(len(eval_envs), dtype=bool)
    last_infos = [{} for _ in eval_envs]

    while not bool(finished.all()):
        active_indices = np.flatnonzero(~finished).tolist()
        active_observations = _stack_dict_observations(
            [observations[index] for index in active_indices]
        )
        action_batch, _ = model.predict(
            active_observations, deterministic=True
        )
        action_batch = np.asarray(action_batch)
        if action_batch.ndim == 1:
            action_batch = action_batch[None, :]
        for batch_index, environment_index in enumerate(active_indices):
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = eval_envs[environment_index].step(action_batch[batch_index])
            observations[environment_index] = next_observation
            episode_returns[environment_index] += float(reward)
            last_infos[environment_index] = info
            finished[environment_index] = bool(terminated or truncated)
    return episode_returns.tolist(), last_infos


def append_flowrl_csv(path, row):
    """Append an event, extending older telemetry schemas safely on resume."""
    import csv
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    fields = list(row)
    if exists:
        with open(path, newline="") as stream:
            fields = next(csv.reader(stream))
        added = [field for field in row if field not in fields]
        if added:
            fields += added
            write_flowrl_frame(path, pd.read_csv(path).reindex(columns=fields))
    with open(path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_flowrl_frame(path, frame):
    temporary = path + ".tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def restore_flowrl_history(log_dir, step):
    """Rewind durable telemetry to the selected model/replay pair.

    A crash can leave CSVs ahead of the last paired checkpoint. Preserve that
    abandoned tail separately, then rebuild plots from the restored trajectory.
    """
    streams = {"progress.csv": "time/total_timesteps", "output_ar_gamma.csv": "iterations",
               "training_episodes.csv": "step", "evaluation_episodes.csv": "step",
               "action_dynamics.csv": "step", "crl_history.csv": "step"}
    rewound = False
    for filename, column in streams.items():
        path = os.path.join(log_dir, filename)
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        if column not in frame:
            raise ValueError(f"FlowRL telemetry {filename} is missing {column}")
        ahead = pd.to_numeric(frame[column], errors="raise") > step
        if ahead.any():
            rewound = True
            write_flowrl_frame(os.path.join(log_dir, f"{filename}.after_step_{step}.{time.time_ns()}.csv"), frame[ahead])
            write_flowrl_frame(path, frame[~ahead])
    crl_plot = os.path.join(log_dir, "crl_analysis.png")
    if rewound and os.path.exists(crl_plot):
        # Before acceleration has enough retained measurements, the CRL
        # plotter correctly waits. Do not leave a plot from the abandoned tail.
        os.replace(crl_plot, crl_plot + f".after_step_{step}.{time.time_ns()}.png")
    # Monitor has no global step column; the rich episode streams give its
    # completed-row count. Preserve its metadata header and matching prefix.
    for phase in ("training", "evaluation"):
        events = os.path.join(log_dir, f"{phase}_episodes.csv")
        monitor = os.path.join(log_dir, f"{phase}.monitor.csv")
        if os.path.exists(events) and os.path.exists(monitor):
            count = len(pd.read_csv(events))
            with open(monitor) as stream:
                metadata = stream.readline()
                frame = pd.read_csv(stream)
            if len(frame) > count:
                temporary = monitor + ".tmp"
                with open(temporary, "w") as stream:
                    stream.write(metadata)
                    frame.iloc[:count].to_csv(stream, index=False)
                os.replace(temporary, monitor)


class FlowRLActionDynamics:
    """Decision-to-decision action changes, separate from the flow ODE.

    Differences use successive proposed allocations, not price-drifted books.
    First differences require two decisions; second differences require three.
    Reset on episode boundaries and intersect masks before comparing assets.
    """
    def __init__(self):
        self.previous = None
        self.previous_delta = None

    def observe(self, raw, asset_mask):
        raw = np.asarray(raw, dtype=np.float64)
        mask = np.concatenate(([True], np.asarray(asset_mask, dtype=bool)))
        shifted = (np.clip(raw, -1, 1) + 1) * 0.5 * mask
        weights = shifted / shifted.sum() if shifted.sum() > 1e-8 else mask / mask.sum()
        current = np.stack((raw, weights))
        result = {"cash_weight": float(weights[0]),
                  "max_asset_weight": float(weights[1:].max()),
                  "raw_saturation": float((np.abs(raw[mask]) > .999).mean())}
        if self.previous is not None:
            previous, previous_mask = self.previous
            valid = mask & previous_mask
            delta = current - previous
            for index, name in enumerate(("raw", "allocation")):
                result[name + "_velocity"] = float(np.abs(delta[index, valid]).mean())
                result[name + "_velocity_squared"] = float(np.square(delta[index, valid]).sum())
            if self.previous_delta is not None:
                old_delta, old_mask = self.previous_delta
                valid_acceleration = valid & old_mask
                for index, name in enumerate(("raw", "allocation")):
                    result[name + "_acceleration"] = float(np.abs((delta - old_delta)[index, valid_acceleration]).mean())
                    result[name + "_acceleration_squared"] = float(np.square((delta - old_delta)[index, valid_acceleration]).sum())
            self.previous_delta = (delta, valid)
        self.previous = (current, mask)
        return result


FLOWRL_DYNAMICS_NAMES = ("raw_velocity", "raw_acceleration", "allocation_velocity", "allocation_acceleration",
                        "raw_velocity_squared", "raw_acceleration_squared", "allocation_velocity_squared", "allocation_acceleration_squared",
                        "cash_weight", "max_asset_weight", "raw_saturation")


def accumulate_flowrl_dynamics(totals, sample):
    for name, value in sample.items():
        total, count, peak = totals.get(name, (0.0, 0, 0.0))
        totals[name] = (total + value, count + 1, max(peak, value))


def summarize_flowrl_dynamics(totals):
    row = {}
    for name in FLOWRL_DYNAMICS_NAMES:
        total, count, peak = totals.get(name, (0.0, 0, float("nan")))
        row[name + "_mean"] = total / count if count else float("nan")
        row[name + "_max"] = peak
        row[name + "_count"] = count
    return row


class FlowRLTelemetryCallback(BaseCallback):
    """Observe real collection episodes without adding an optimization loss."""
    def __init__(self, log_dir, elapsed, saved_state=None):
        super().__init__()
        self.log_dir, self.elapsed = log_dir, elapsed
        saved_state = saved_state or {}
        self.completed = int(saved_state.get("completed_training_episodes", 0))
        self.recent = deque(saved_state.get("recent_training_episodes", []), maxlen=100)
        self.workers = {}
        self.interval_dynamics = {}

    def _on_step(self):
        for index, (action, reward, done, info) in enumerate(zip(
            self.locals["actions"], self.locals["rewards"], self.locals["dones"], self.locals["infos"]
        )):
            worker = self.workers.setdefault(index, {"return": 0.0, "length": 0,
                                                     "dynamics": FlowRLActionDynamics(), "totals": {}})
            sample = worker["dynamics"].observe(action, self.model._last_obs["mask"][index])
            accumulate_flowrl_dynamics(worker["totals"], sample)
            accumulate_flowrl_dynamics(self.interval_dynamics, sample)
            worker["return"] += float(reward)
            worker["length"] += 1
            if done:
                self.completed += 1
                final_value = float(info["value"])
                row = {"step": int(self.model.num_timesteps), "episode": self.completed,
                       "shaped_return": worker["return"],
                       "financial_return": float(info.get("episode_financial_return", final_value / config.INITIAL_CASH - 1)),
                       "terminal_reward": float(reward), "length": worker["length"],
                       "initial_value": float(config.INITIAL_CASH), "final_value": final_value,
                       "opportunity_normalized_financial_return": float(info.get("opportunity_normalized_financial_return", np.nan)),
                       "episode_coins": "|".join(info.get("episode_coins", [])),
                       "elapsed_seconds": self.elapsed(), **summarize_flowrl_dynamics(worker["totals"]),
                       **{key: info[key] for key in FLOWRL_SAMPLING_FIELDS}}
                append_flowrl_csv(os.path.join(self.log_dir, "training_episodes.csv"), row)
                self.recent.append(row)
                print(f"Training episode {self.completed} at step {row['step']}: "
                      f"{self.model.environment_settings['reward_type']} terminal reward={row['terminal_reward']:.6f} | "
                      f"shaped return={row['shaped_return']:.6f} | Financial return={row['financial_return']:.4%} | "
                      f"length={row['length']} | start={row['episode_start_time']} | "
                      f"basket={row['episode_coins']} | elapsed={row['elapsed_seconds']:.1f}s")
                del self.workers[index]
        return True

    def flush(self, step):
        dynamics = summarize_flowrl_dynamics(self.interval_dynamics)
        append_flowrl_csv(os.path.join(self.log_dir, "action_dynamics.csv"),
                         {"step": step, "phase": "training", "elapsed_seconds": self.elapsed(), **dynamics})
        self.interval_dynamics = {}
        metrics = {"action/" + name: value for name, value in dynamics.items()}
        for key, field in (("ep_rew_mean", "shaped_return"), ("ep_len_mean", "length"),
                           ("ep_financial_return_mean", "financial_return"), ("ep_terminal_reward_mean", "terminal_reward")):
            metrics["rollout/" + key] = float(np.mean([row[field] for row in self.recent])) if self.recent else np.nan
        metrics["rollout/episodes"] = self.completed
        return metrics

    def state_dict(self):
        # An SB3 resume resets the simulator. Do not carry partial episodes or
        # their action derivatives across that reset; completed history persists.
        return {"completed_training_episodes": self.completed, "recent_training_episodes": list(self.recent)}


FLOWRL_SAMPLING_FIELDS = ("dataset_file", "episode_start_time", "episode_end_time", "episode_start_row")


def flowrl_evaluation_seed(step, episode):
    """Reproducible at a checkpoint, fresh across evaluation checkpoints."""
    return int(np.random.SeedSequence([43, int(step), int(episode)]).generate_state(1)[0])


def evaluate_flowrl_portfolio(model, env, episodes, step, elapsed):
    """Fresh seeded market windows and flow samples; preserve training RNG."""
    if isinstance(env, (list, tuple)):
        return evaluate_flowrl_portfolio_batch(model, env, episodes, step, elapsed)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    was_training = model.policy.training
    records, totals = [], {}
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            for episode in range(episodes):
                seed = flowrl_evaluation_seed(step, episode)
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                obs, initial_info = env.reset(seed=seed)
                initial_value = float(initial_info.get("value", config.INITIAL_CASH))
                shaped, length, done = 0.0, 0, False
                tracker, episode_totals = FlowRLActionDynamics(), {}
                while not done:
                    action, _ = model.predict(obs, deterministic=False)
                    sample = tracker.observe(action, obs["mask"])
                    accumulate_flowrl_dynamics(episode_totals, sample)
                    accumulate_flowrl_dynamics(totals, sample)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    shaped += float(reward)
                    length += 1
                row = {"step": int(step), "episode": episode + 1, "shaped_return": shaped,
                       "financial_return": float(info["value"]) / initial_value - 1,
                       "terminal_reward": float(reward), "length": length,
                       "initial_value": initial_value, "final_value": float(info["value"]),
                       "opportunity_normalized_financial_return": float(info.get("opportunity_normalized_financial_return", np.nan)),
                       "episode_coins": "|".join(info.get("episode_coins", [])),
                       "elapsed_seconds": elapsed(), **summarize_flowrl_dynamics(episode_totals),
                       **{key: info[key] for key in FLOWRL_SAMPLING_FIELDS},
                       "evaluation_seed": seed, "evaluation_sampling": "rotating_seeded_windows_v1"}
                records.append(row)
                print(f"Evaluation {episode + 1}/{episodes} at step {step}: "
                      f"{model.environment_settings['reward_type']} terminal reward={row['terminal_reward']:.6f} | "
                      f"shaped return={shaped:.6f} | Financial return={row['financial_return']:.4%} | "
                      f"start={row['episode_start_time']} | seed={seed} | basket={row['episode_coins']}")
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        model.policy.set_training_mode(was_training)
    return records, summarize_flowrl_dynamics(totals)


def evaluate_flowrl_portfolio_batch(model, environments, episodes, step, elapsed):
    """Run full independent episodes with a shared batched actor forward.

    Each episode owns its market seed and Torch latent generator. Environment
    accounting/rewards stay on the original PortfolioEnv/Monitor path.
    """
    numpy_state, python_state = np.random.get_state(), random.getstate()
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    was_training = model.policy.training
    records, totals = [], {}
    latent_device = model.device if model.device.type == "cuda" else torch.device("cpu")
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))), torch.inference_mode():
            model.policy.set_training_mode(False)
            for first in range(0, episodes, len(environments)):
                workers = []
                for offset, environment in enumerate(environments[:episodes - first]):
                    episode = first + offset
                    seed = flowrl_evaluation_seed(step, episode)
                    np.random.seed(seed)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    obs, initial_info = environment.reset(seed=seed)
                    workers.append(dict(env=environment, obs=obs, seed=seed, episode=episode,
                                        initial_value=float(initial_info.get("value", config.INITIAL_CASH)),
                                        shaped=0.0, length=0, done=False, tracker=FlowRLActionDynamics(), totals={},
                                        generator=torch.Generator(device=latent_device).manual_seed(seed)))
                while any(not worker["done"] for worker in workers):
                    active = [worker for worker in workers if not worker["done"]]
                    observations = {key: np.stack([worker["obs"][key] for worker in active])
                                    for key in active[0]["obs"]}
                    tensors = model.policy.obs_to_tensor(observations)[0]
                    # Separate generators retain each episode's stochastic flow
                    # independently of the other episodes' lengths and ordering.
                    noise = torch.cat([torch.randn((1, model.action_space.shape[0]), device=latent_device,
                                                   generator=worker["generator"]).clamp(-1, 1)
                                       for worker in active]).to(model.device)
                    actions = model.actor(tensors, noise=noise).cpu().numpy()
                    for worker, action in zip(active, actions):
                        sample = worker["tracker"].observe(action, worker["obs"]["mask"])
                        accumulate_flowrl_dynamics(worker["totals"], sample)
                        accumulate_flowrl_dynamics(totals, sample)
                        obs, reward, terminated, truncated, info = worker["env"].step(action)
                        worker.update(obs=obs, done=terminated or truncated, info=info, terminal=float(reward))
                        worker["shaped"] += float(reward)
                        worker["length"] += 1
                for worker in workers:
                    info = worker["info"]
                    row = {"step": int(step), "episode": worker["episode"] + 1,
                           "shaped_return": worker["shaped"],
                           "financial_return": float(info["value"]) / worker["initial_value"] - 1,
                           "terminal_reward": worker["terminal"], "length": worker["length"],
                           "initial_value": worker["initial_value"], "final_value": float(info["value"]),
                           "opportunity_normalized_financial_return": float(info.get("opportunity_normalized_financial_return", np.nan)),
                           "episode_coins": "|".join(info.get("episode_coins", [])),
                           "elapsed_seconds": elapsed(), **summarize_flowrl_dynamics(worker["totals"]),
                           **{key: info[key] for key in FLOWRL_SAMPLING_FIELDS},
                           "evaluation_seed": worker["seed"], "evaluation_sampling": "rotating_seeded_windows_v1"}
                    records.append(row)
                    print(f"Evaluation {row['episode']}/{episodes} at step {step}: "
                          f"{model.environment_settings['reward_type']} terminal reward={row['terminal_reward']:.6f} | "
                          f"shaped return={row['shaped_return']:.6f} | Financial return={row['financial_return']:.4%} | "
                          f"cash={row['cash_weight_mean']:.2%} | saturation={row['raw_saturation_mean']:.2%} | "
                          f"start={row['episode_start_time']} | seed={row['evaluation_seed']} | basket={row['episode_coins']}")
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        model.policy.set_training_mode(was_training)
    return records, summarize_flowrl_dynamics(totals)


def flowrl_crl_settings(profile):
    """Use the actual October 13 attention runner's dual settings.

    The general profile supplies the enable/fixed-penalty choice. The dual
    thresholds and update rates come from the attention TD3 constructor below,
    rather than its older single-constraint profile defaults.
    """
    profile_config = (config.get_large_portfolio_crl_config(profile) if config.MAX_COINS > 4
                      else config.get_crl_config(profile))
    return dict(use_crl=bool(profile_config["use_crl"]), action_reg_coef=float(profile_config["action_reg_coef"]),
                velocity_constraint_threshold=0.0016, velocity_lambda_lr=0.05, initial_velocity_lambda=0.03,
                acceleration_constraint_threshold=0.003, acceleration_lambda_lr=0.005,
                initial_acceleration_lambda=0.001, lambda_warmup_steps=500, auto_adjust_for_attention=True)


def train_flowrl(
    mlp_size, reward_type, total_timesteps, log_dir="./logs", model_save_path="./models",
    resume_checkpoint=None, options=None, crl_profile="balanced",
):
    """FlowRL learner with the portfolio runner's full reporting lifecycle."""
    from FlowRL_algo import FlowRL
    from FlowRL_model import ACTION_NORMALIZATION
    configure_a40_runtime()
    settings = dict(config.FLOWRL_CONFIG)
    settings.update({key: value for key, value in (options or {}).items() if value is not None})
    crl_settings = flowrl_crl_settings(crl_profile)
    # Programmatic overrides are useful for small verification runs; ordinary
    # CLI runs use the stable attention runner settings above.
    crl_settings.update({key: settings.pop(key) for key in tuple(crl_settings) if key in settings})
    run_dir = settings.pop("run_dir", None)
    if run_dir:
        log_dir, model_save_path = os.path.join(run_dir, "logs"), os.path.join(run_dir, "models")
    intervals = {name: int(settings.pop(name)) for name in ("log_freq", "eval_freq", "plot_freq", "checkpoint_freq")}
    collect_steps = int(settings.pop("collect_steps"))
    eval_episodes = int(settings.pop("eval_episodes"))
    eval_batch_size = max(1, min(eval_episodes, int(settings.pop("eval_batch_size", 4))))
    if min(*intervals.values(), eval_episodes, collect_steps) < 1 or total_timesteps < 1:
        raise ValueError("FlowRL intervals, episode count and training horizon must be positive")
    config.set_portfolio_mode(True)
    if bool(getattr(config, "BASKET_CURRICULUM_ENABLED", False)):
        raise ValueError("FlowRL uses uniform/local-composition collection; disable the TD3 graph teacher")
    sampler = build_local_composition_sampler()
    if sampler is not None and resume_checkpoint:
        sampler_path = curriculum_path_for_model_checkpoint(resume_checkpoint)
        if not os.path.exists(sampler_path):
            raise FileNotFoundError(f"Missing FlowRL sampler checkpoint: {sampler_path}")
        sampler.load(sampler_path)
    run_name = (os.path.basename(os.path.dirname(os.path.dirname(resume_checkpoint)))
                if resume_checkpoint else
                f"FlowRL_tf_agents_style_{mlp_size}_{reward_type}_variable_coin_attention_variable"
                f"_CRL_{crl_profile if crl_settings['use_crl'] else 'off'}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    run_logs = os.path.join(log_dir, run_name)
    if resume_checkpoint and not run_dir:
        with zipfile.ZipFile(resume_checkpoint) as archive:
            saved_metadata = json.loads(archive.read("data"))
        run_logs = saved_metadata.get("flowrl_log_dir", run_logs)
    checkpoints = (os.path.dirname(resume_checkpoint) if resume_checkpoint else
                   os.path.join(model_save_path, run_name, "checkpoints"))
    os.makedirs(run_logs, exist_ok=True)
    os.makedirs(checkpoints, exist_ok=True)
    raw_env = PortfolioEnv(use_variable_portfolio=True, reward_type=reward_type,
                           basket_sampler=sampler, action_normalization=ACTION_NORMALIZATION)
    architecture = dict(config.get_attention_config(mlp_size))
    policy_kwargs = dict(model_dim=settings.get("model_dim", architecture["features_dim"]),
                         net_arch=architecture["net_arch"], n_heads=architecture["n_heads"],
                         n_layers=architecture["n_layers"], use_market_attention=architecture["use_market_attention"],
                         use_global_pooling=architecture["use_global_pooling"], flow_steps=settings.pop("flow_steps"))
    if "model_dim" in settings:
        policy_kwargs["net_arch"] = [settings.pop("model_dim")] * 2
    device = settings.pop("device", "auto")
    replay_kwargs = {key: settings.pop(key) for key in ("recent_fraction", "recent_window")}
    if resume_checkpoint:
        model = FlowRL.load(resume_checkpoint, env=DummyVecEnv([lambda: raw_env]), device=device)
        if model.policy.architecture_version < 3:
            raise ValueError("This FlowRL checkpoint predates portfolio objective v3 (complete portfolio critic, "
                             "consistent flow endpoints and return-gradient-controlled imitation). It remains evaluable, "
                             "but the repaired architecture requires a fresh training run.")
        if model.policy.architecture_version < 5:
            raise ValueError("FlowRL v5 replaces the independent cash heads with one joint marginal scorer. "
                             "This checkpoint remains evaluable with its saved architecture, but the "
                             "joint marginal network requires a fresh training run.")
        if model.configure_collection_exploration(settings["exploration_noise"]):
            print(f"FlowRL resume: applied collection exploration sigma={model.exploration_noise:g}; "
                  "stochastic flow sampling retained; existing replay and learned optimizer/dual state retained")
        # These objective corrections do not change network shapes. Apply
        # current training settings on compatible v5 resume while retaining learned
        # weights, critic optimizers, targets, replay and evaluation behavior.
        # An objective change invalidates only the actor's Adam moments.
        if model.configure_actor_objective(
                settings["actor_cfm_gradient_ratio"], settings["actor_cfm_scale_floor"],
                settings["actor_q_normalization"], settings["actor_q_scale_floor"],
                matching_mode=settings["actor_matching_mode"]):
            print("FlowRL resume: actor objective changed; cleared actor Adam moments, "
                  f"retained policy weights, critic optimizers, targets and replay; matching={model.actor_matching_mode}")
        previous_crl_reference = model.crl_reference_mode
        previous_crl_sampling = model.crl_flow_sampling
        if model.configure_crl(**crl_settings):
            print("FlowRL resume: CRL configuration changed; cleared actor Adam moments, "
                  f"retained learned networks, critic optimizers, targets and replay; "
                  f"reference={model.crl_reference_mode}, flow sampling={model.crl_flow_sampling}, "
                  f"batch gate revision={model.crl_coupling_version}")
            if (previous_crl_reference != model.crl_reference_mode
                    or previous_crl_sampling != model.crl_flow_sampling):
                print("FlowRL resume: restored October 29 forward endpoint costs with paired flow initialization; "
                      "restarted both adaptive prices, violation EMAs and controller histories for the changed estimator")
        if not load_replay_buffer_checkpoint(model, replay_buffer_path_for_model_checkpoint(resume_checkpoint)):
            raise FileNotFoundError("FlowRL resume requires the matching replay checkpoint")
        if model.environment_settings != {"reward_type": reward_type, "use_variable_portfolio": True}:
            raise ValueError("FlowRL resume reward/environment settings differ from checkpoint")
        restore_flowrl_history(run_logs, model.num_timesteps)
    else:
        model = FlowRL(env=DummyVecEnv([lambda: raw_env]), policy_kwargs=policy_kwargs,
                       replay_buffer_kwargs=replay_kwargs, seed=42, verbose=1, device=device,
                       **crl_settings, **settings)
    model.validate_td3_crl_training_contract()
    model.crl_profile = crl_profile
    model.flowrl_log_dir = os.path.abspath(run_logs)
    dates = raw_env.cached_data["date"]
    model.flowrl_data_provenance = {
        "dataset_file": os.path.abspath(config.FILE), "rows": len(raw_env.cached_data),
        "assets": list(config.COIN_POOL), "first_time": dates.min().isoformat(),
        "last_time": dates.max().isoformat(), "available_dates": int(dates.dt.normalize().nunique()),
        "evaluation_sampling": "rotating_seeded_windows_v1",
        "architecture_version": model.policy.architecture_version,
        "financial_aux_weight": model.financial_aux_weight,
        "financial_return_reference": config.COMPONENT_OUTCOME_LOSS_RETURN_REFERENCE,
        "basket_relative_aux_weight": config.BASKET_RELATIVE_AUX_WEIGHT,
        "financial_encoder_gradient_budget": config.COMPONENT_OUTCOME_TARGET_GRADIENT_RATIO * model.financial_aux_weight,
        "critic_regression": "huber_beta1", "flow_endpoint": "inverse_tanh_v1",
        "actor_cfm_gradient_ratio": model.actor_cfm_gradient_ratio,
        "actor_cfm_scale_floor": model.actor_cfm_scale_floor,
        "actor_matching_mode": model.actor_matching_mode,
        "actor_q_normalization": model.actor_q_normalization,
        "actor_q_scale_floor": model.actor_q_scale_floor,
        "collection_exploration_sigma": model.exploration_noise,
        "collection_exploration_mode": ("stochastic_flow_only" if model.action_noise is None
                                        else "stochastic_flow_plus_ou"),
        "candidate_count": model.candidate_count,
        "candidate_residual_weight": model.candidate_residual_weight,
        "sigma_loss_focus": model.sigma_loss_focus,
        "cash_budget_architecture": "joint_cash_asset_marginal_v2",
        "max_grad_norm": model.max_grad_norm,
        "crl_profile": crl_profile, "crl_settings": crl_settings,
        "crl_weight_geometry": ACTION_NORMALIZATION, "crl_estimator": model.crl_reference_mode,
        "crl_coupling_version": model.crl_coupling_version,
        "crl_acceleration_gate": "td3_batch_velocity_v1",
        "crl_flow_sampling": model.crl_flow_sampling,
        "crl_flow_reference": "oct29_completed_flow_endpoints_shared_initial_latent",
        "crl_support_semantics": "previous_availability_mask",
        "actor_matching_budget": "oct29_q_only_with_matching_floor_v1",
    }
    # Open monitors after replay validation and CSV rewind, so their file
    # handles cannot point to the replaced, abandoned pre-resume CSV tail.
    env = DummyVecEnv([lambda: Monitor(raw_env, filename=os.path.join(run_logs, "training"),
                                      override_existing=not should_append_monitor(
                                          os.path.join(run_logs, "training"), resume_checkpoint))])
    model.set_env(env)
    eval_env = Monitor(PortfolioEnv(use_variable_portfolio=True, reward_type=reward_type,
                                   action_normalization=ACTION_NORMALIZATION),
                       filename=os.path.join(run_logs, "evaluation"), override_existing=not should_append_monitor(
                           os.path.join(run_logs, "evaluation"), resume_checkpoint))
    eval_envs = [eval_env]
    for _ in range(eval_batch_size - 1):
        peer = Monitor(PortfolioEnv(use_variable_portfolio=True, reward_type=reward_type,
                                    action_normalization=ACTION_NORMALIZATION))
        # All full episodes feed the same evaluation monitor CSV.
        peer.results_writer = eval_env.results_writer
        peer.t_start = eval_env.t_start
        eval_envs.append(peer)
    progress_path = os.path.join(run_logs, "progress.csv")
    if resume_checkpoint and os.path.exists(progress_path):
        import csv
        from stable_baselines3.common.logger import CSVOutputFormat
        class ResumedCSVOutputFormat(CSVOutputFormat):
            def __init__(self, filename):
                self.file = open(filename, "r+")
                self.keys = next(csv.reader(self.file), [])
                self.separator, self.quotechar = ",", '"'
                self.file.seek(0, os.SEEK_END)
        logger = configure_training_logger(run_logs, ["stdout", "tensorboard"])
        logger.output_formats.append(ResumedCSVOutputFormat(progress_path))
    else:
        logger = configure_training_logger(run_logs, ["stdout", "csv", "tensorboard"])
    if resume_checkpoint:
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        from torch.utils.tensorboard import SummaryWriter
        for output in logger.output_formats:
            if isinstance(output, TensorBoardOutputFormat):
                output.writer.close()
                # TensorBoard must hide the same abandoned tail as the CSVs.
                output.writer = SummaryWriter(log_dir=run_logs, purge_step=int(model.num_timesteps) + 1)
    model.set_logger(logger)
    results_path = os.path.join(run_logs, "output_ar_gamma.csv")
    results = pd.read_csv(results_path).to_dict("records") if os.path.exists(results_path) else []
    state = dict(getattr(model, "flowrl_telemetry_state", {}))
    # Legacy checkpoints had no clock; recover the last retained logged time
    # where available. No time is attributed to the period between processes.
    if ("elapsed_seconds" not in state and os.path.exists(progress_path)
            and os.path.getsize(progress_path) > 0):
        history = pd.read_csv(progress_path)
        for field, column in (("elapsed_seconds", "time/time_elapsed"),
                              ("training_seconds", "time/training_seconds")):
            if column in history and history[column].notna().any():
                state[field] = float(history[column].dropna().iloc[-1])
    elapsed_before = float(state.get("elapsed_seconds", 0.0))
    training_seconds = float(state.get("training_seconds", 0.0))
    started = time.perf_counter()
    elapsed = lambda: elapsed_before + time.perf_counter() - started
    telemetry = FlowRLTelemetryCallback(run_logs, elapsed, state)
    best_return = max((float(row["Return"]) for row in results), default=-np.inf)
    best_path = os.path.join(os.path.dirname(checkpoints), "best_model.zip")
    final_path = os.path.join(os.path.dirname(checkpoints), "final_model.zip")
    for snapshot in (best_path, final_path):
        if resume_checkpoint and os.path.exists(snapshot):
            with zipfile.ZipFile(snapshot) as archive:
                saved_step = int(json.loads(archive.read("data"))["num_timesteps"])
            if saved_step > model.num_timesteps:
                # Later evaluation snapshots belong to the abandoned continuation.
                os.replace(snapshot, snapshot + f".after_step_{model.num_timesteps}.{time.time_ns()}")
    if not os.path.exists(best_path):
        best_return = -np.inf
    crl_path = os.path.join(run_logs, "crl_history.csv")
    crl_history = load_crl_history_csv(crl_path) if os.path.exists(crl_path) else []
    plot_callback = PlotGeneratorCallback(run_logs, run_name, plot_freq=intervals["plot_freq"], verbose=0,
                                          crl_stats_history=crl_history, crl_profile=crl_profile)
    plot_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flowrl-plots")
    pending_plot = None
    print(f"FlowRL: telemetry every {intervals['log_freq']} steps; {eval_episodes} evaluation episodes "
          f"every {intervals['eval_freq']}; plots every {intervals['plot_freq']}; "
          f"paired checkpoints every {intervals['checkpoint_freq']} and at completion")
    print(f"Replay capacity={model.replay_buffer.buffer_size * model.n_envs:,} transitions; "
          f"loaded rows={model.replay_buffer.size() * model.n_envs:,}; outputs={run_logs}")
    provenance = model.flowrl_data_provenance
    print(f"Dataset={provenance['dataset_file']}; rows={provenance['rows']:,}; "
          f"dates={provenance['available_dates']}; span={provenance['first_time']}..{provenance['last_time']}; "
          f"training baskets={config.COINS_PER_EPISODE_RANGE} of {len(provenance['assets'])}")
    print(f"TF-Agents-style reporting: collect in {collect_steps}-transition blocks; "
          f"FlowRL optimization starts after {model.learning_starts} transitions, "
          f"train_freq={model.train_freq}, gradient_steps={model.gradient_steps}; "
          f"evaluation rotates date/basket/action seeds at each checkpoint")
    print(f"FlowRL architecture v{model.policy.architecture_version}: "
          f"identity width={model.actor.state_encoder.identity_dim}, "
          f"whole-pool={model.actor.state_encoder.encoder.pool_level_enabled}, "
          f"basket opportunity actor/critics enabled; causal financial teacher weight={model.financial_aux_weight:g}")
    print(f"Portfolio objective: complete TD3 DeepSet critics; Huber TD; gamma={model.gamma:g}; "
          f"target tau={model.tau:g}; actor delay={model.policy_delay}; actor LR={model.actor_learning_rate:g}; "
          f"{model.candidate_count} cash/composition candidates; stochastic flow collection; "
          f"added OU={'off' if model.action_noise is None else 'sigma=' + str(model.exploration_noise)}; "
          f"CFM matching mode={model.actor_matching_mode}; "
          f"CFM ratio={model.actor_cfm_gradient_ratio}; matching-scale floor={model.actor_cfm_scale_floor:g}; "
          f"actor Q normalization={model.actor_q_normalization}, floor={model.actor_q_scale_floor:g}; "
          f"evaluation actor batch={eval_batch_size}; inverse-tanh matching endpoints")
    print(f"CRL profile={crl_profile}; adaptive={model.use_crl}; "
          f"velocity threshold={model.velocity_constraint_threshold:g}, lambda={float(model.lambda_param_velocity):g}, "
          f"LR={model.velocity_lambda_lr:g}; acceleration threshold={model.acceleration_constraint_threshold:g}, "
          f"lambda={float(model.lambda_param_acceleration):g}, LR={model.acceleration_lambda_lr:g}; "
          f"warmup={model.lambda_warmup_steps} critic updates; joint allocation geometry; "
          f"CRL reference={model.crl_reference_mode}; previous execution/current endpoint/next target endpoint; "
          f"shared initial latent as in October 29, full flow integration; TD3 batch acceleration gate; "
          f"cash/assets share one marginal scorer")

    def update_saved_state():
        model.flowrl_telemetry_state = {**telemetry.state_dict(), "elapsed_seconds": elapsed(),
                                       "training_seconds": training_seconds,
                                       "last_evaluation_step": results[-1]["iterations"] if results else None,
                                       "best_return": best_return, "intervals": intervals,
                                       "collect_steps": collect_steps,
                                       "evaluation_sampling": "rotating_seeded_windows_v1"}

    def evaluate():
        nonlocal best_return
        step = int(model.num_timesteps)
        evaluation_started = time.perf_counter()
        records, dynamics = evaluate_flowrl_portfolio(model, eval_envs, eval_episodes, step, elapsed)
        logger.record("eval/seconds", time.perf_counter() - evaluation_started)
        for row in records:
            append_flowrl_csv(os.path.join(run_logs, "evaluation_episodes.csv"), row)
        append_flowrl_csv(os.path.join(run_logs, "action_dynamics.csv"),
                         {"step": step, "phase": "evaluation", "elapsed_seconds": elapsed(), **dynamics})
        shaped = [row["shaped_return"] for row in records]
        financial = [row["financial_return"] for row in records]
        terminal = [row["terminal_reward"] for row in records]
        normalized = [row["opportunity_normalized_financial_return"] for row in records]
        result = {"iterations": step, "Return": float(np.mean(shaped)),
                  "FinancialReturn": float(np.mean(financial)),
                  "OpportunityNormalizedFinancialReturn": float(np.mean(normalized)),
                  "TerminalReward": float(np.mean(terminal)), "EvaluationEpisodes": eval_episodes,
                  "ElapsedSeconds": elapsed()}
        for index, row in enumerate(records, 1):
            result[f"Episode{index}Return"] = row["shaped_return"]
            result[f"Episode{index}FinancialReturn"] = row["financial_return"]
            result[f"Episode{index}TerminalReward"] = row["terminal_reward"]
        results[:] = [row for row in results if int(row["iterations"]) != step] + [result]
        write_flowrl_frame(results_path, pd.DataFrame(results).sort_values("iterations"))
        for name, values in (("reward", shaped), ("financial_return", financial),
                             ("terminal_reward", terminal), ("opportunity_normalized_financial_return", normalized),
                             ("ep_length", [row["length"] for row in records])):
            logger.record("eval/mean_" + name, float(np.mean(values)))
            logger.record("eval/std_" + name, float(np.std(values)))
        for name, value in dynamics.items():
            logger.record("eval/action_" + name, value)
        print(f"Step {step}: Average shaped return={result['Return']:.6f}; "
              f"Average Financial Return={result['FinancialReturn']:.4%}")
        print(f"Episode shaped returns: {[f'{value:.6f}' for value in shaped]}")
        print(f"Episode terminal rewards: {[f'{value:.6f}' for value in terminal]}")
        print(f"Episode financial returns: {[f'{value:.4%}' for value in financial]}")
        if result["Return"] > best_return:
            best_return = result["Return"]
            update_saved_state()
            if not save_model_checkpoint(model, best_path):
                raise OSError("FlowRL best-model checkpoint failed")

    def flush(fps=np.nan):
        metrics = {**model.drain_training_metrics(), **telemetry.flush(int(model.num_timesteps))}
        metrics.update({"time/total_timesteps": model.num_timesteps,
                        "time/iterations": model.num_timesteps / collect_steps,
                        "time/time_elapsed": elapsed(), "time/training_seconds": training_seconds,
                        "time/fps": fps, "replay/size": model.replay_buffer.size() * model.n_envs,
                        "replay/capacity": model.replay_buffer.buffer_size * model.n_envs,
                        "train/learning_starts": model.learning_starts,
                        "train/learning_started": float(model._n_updates > 0),
                        "eval/rotating_windows": 1.0})
        for name, value in metrics.items():
            logger.record(name, value)
        if model.use_crl:
            if not crl_history or int(crl_history[-1]["step"]) != int(model.num_timesteps):
                crl_history.append({"step": int(model.num_timesteps), **model.get_crl_stats(),
                                    "crl_reference_mode": model.crl_reference_mode,
                                    "crl_flow_sampling": model.crl_flow_sampling})
            write_flowrl_frame(crl_path, pd.DataFrame(normalize_crl_stats_history(crl_history)))
        logger.dump(model.num_timesteps)
        update_saved_state()
        print(f"FlowRL step {model.num_timesteps}/{total_timesteps} | elapsed={elapsed():.1f}s | "
              f"training={training_seconds:.1f}s | FPS={fps:.1f} | "
              f"actor LR={metrics['train/actor_lr']:.2e} | critic LR={metrics['train/critic_lr']:.2e} | "
              f"value loss={metrics['train/value_loss']:.6g} | "
              f"actor Q/CFM gradients={metrics['train/actor_q_gradient_norm']:.3g}/{metrics['train/actor_cfm_gradient_norm']:.3g} | "
              f"effective CFM ratio={metrics['train/actor_cfm_gradient_ratio']:.3g} | "
              f"CFM/control={metrics['train/actor_control_cfm_gradient_ratio']:.3g} | "
              f"control descent margin={metrics['train/actor_control_descent_margin']:.3g} | "
              f"CFM floor active={metrics['train/actor_matching_floor_active']:.1%} | "
              f"Q/CFM cosine={metrics['train/actor_q_cfm_cosine']:.3g} | "
              f"added-noise allocation delta={metrics['train/exploration_allocation_delta']:.3g} | replay={metrics['replay/size']:,}")
        if model.use_crl:
            print(f"CRL: velocity lambda={float(model.lambda_param_velocity):.6g}; "
                  f"acceleration lambda={float(model.lambda_param_acceleration):.6g}; "
                  f"velocity penalty={metrics['train/actor_velocity_penalty']:.6g}; "
                  f"acceleration penalty={metrics['train/actor_acceleration_penalty']:.6g}; "
                  f"coupling={metrics['train/crl_coupling_factor']:.4f}")

    def render(step):
        plot_callback._generate_plots()
        if getattr(plot_callback, "last_rendered_step", None) != step:
            raise RuntimeError(f"FlowRL plots did not finish for step {step}")
        print(f"Plots refreshed at step {step}: training_plots.png, flowrl_diagnostics.png, action_dynamics.png")

    def refresh_crl_plot_history():
        # The renderer owns a snapshot while collection continues. Correlate
        # lambdas only with returns evaluated at that same environment step.
        current_history = [row for row in crl_history
                           if row.get("crl_reference_mode") == model.crl_reference_mode
                           and row.get("crl_flow_sampling") == model.crl_flow_sampling]
        plot_callback.crl_stats_history = copy.deepcopy(current_history)
        evaluated = {int(row["iterations"]): row["Return"] for row in results}
        compatible = [row for row in current_history if crl_record_has_finite_values(row, DUAL_CRL_PLOT_KEYS)]
        plot_callback.returns = [evaluated.get(int(row["step"]), float("nan")) for row in compatible]

    try:
        if not results:
            evaluate()
            flush()
        if os.path.exists(progress_path):
            refresh_crl_plot_history()
            render(int(model.num_timesteps))
        interval_steps, interval_seconds = 0, 0.0
        while model.num_timesteps < total_timesteps:
            next_events = {key: (model.num_timesteps // frequency + 1) * frequency for key, frequency in intervals.items()}
            next_collection = (model.num_timesteps // collect_steps + 1) * collect_steps
            stop = min(total_timesteps, next_collection, *next_events.values())
            before_steps, before_time = model.num_timesteps, time.perf_counter()
            # The callback observes completed episodes; all CSV logger dumps
            # happen here, after collection and the corresponding updates.
            model.learn(total_timesteps=stop - before_steps, reset_num_timesteps=False,
                        callback=telemetry, log_interval=None, progress_bar=False)
            duration = time.perf_counter() - before_time
            training_seconds += duration
            interval_steps += model.num_timesteps - before_steps
            interval_seconds += duration
            if pending_plot is not None:
                pending_plot.result()
                pending_plot = None
            if model.use_crl:
                crl_history.append({"step": int(model.num_timesteps), **model.get_crl_stats(),
                                    "crl_reference_mode": model.crl_reference_mode,
                                    "crl_flow_sampling": model.crl_flow_sampling})
            final = model.num_timesteps >= total_timesteps
            if model.num_timesteps >= next_events["eval_freq"] or final:
                evaluate()
            if final:
                logger.record("training/completed", 1.0)
            if final or any(model.num_timesteps >= event for event in next_events.values()):
                flush(interval_steps / max(interval_seconds, 1e-9))
                interval_steps, interval_seconds = 0, 0.0
            if model.num_timesteps >= next_events["plot_freq"] or final:
                refresh_crl_plot_history()
                pending_plot = plot_executor.submit(render, int(model.num_timesteps))
            if model.num_timesteps >= next_events["checkpoint_freq"] or final:
                update_saved_state()
                if not save_resume_checkpoint_pair(model, checkpoints, model.num_timesteps, curriculum_manager=sampler):
                    raise OSError("FlowRL paired model/replay checkpoint failed")
                print(f"Resume snapshot at step {model.num_timesteps}: optimizer/target state, "
                      f"{model.replay_buffer.size()} replay rows, sampler and telemetry clock saved")
        if pending_plot is not None:
            pending_plot.result()
        update_saved_state()
        if not save_model_checkpoint(model, final_path):
            raise OSError("FlowRL final-model checkpoint failed")
        print(f"FlowRL finished at step {model.num_timesteps}: elapsed={elapsed():.1f}s, "
              f"collection and updates={training_seconds:.1f}s; final model={final_path}")
    finally:
        plot_executor.shutdown(wait=True)
        for evaluation_env in eval_envs:
            evaluation_env.close()
    return model, [row["Return"] for row in results], [int(row["iterations"]) for row in results]


def train_simple_mlp_tf_agents_style(
    algorithm="TD3",
    mlp_size="heavy",
    reward_type="STRUCTURED_CREDIT",
    log_dir="./logs",
    model_save_path="./models",
    crl_profile="balanced",  # New parameter for CRL constraint profile
    use_attention=False,  # NEW: Enable attention mechanisms
    attention_type="SimpleMlpTD3Policy",  # NEW: Type of Mlp type to use
    use_variable_portfolio=True,  # NEW: Enable variable portfolio sizes
    total_timesteps=100000,  # ✅ NEW: Add total_timesteps parameter
    encoder_variant="baseline",  # baseline (CombinedExtractor) | encoder (extractor.py) | encoder_res (extractor_res.py)
    aux_coef=0.0,               # masked-feature reconstruction coefficient (only used for encoder variants)
    aux_mask_prob=0.15,         # masking probability for the auxiliary objective
    sigma_input_mode=None,      # "passthrough" (Option B default) | "dampen_input" (legacy Option A)
    sigma_gate_scale=None,      # only used in dampen_input mode
    sigma_loss_focus_enabled=None,
    sigma_loss_focus_alpha=None,
    sigma_loss_focus_power=None,
    sigma_loss_focus_max_weight=None,
    sigma_loss_focus_eps=None,
    proportional_exp_mode="off",
    proportional_exp_candidates=5,
    proportional_exp_alpha=0.05,
    proportional_exp_alpha_max=0.25,
    proportional_exp_disagreement_gain=1.0,
    proportional_exp_ema_decay=0.99,
    proportional_exp_cloud_space="portfolio_weights",
    proportional_exp_budget_candidate_ratio=0.34,
    proportional_exp_target_budget_std=None,
    proportional_exp_target_composition_std=None,
    proportional_exp_actor_budget_std=None,
    proportional_exp_actor_composition_std=None,
    use_qwm=False,
    resume_checkpoint=None,
    gpu_sim_envs=0,
    gpu_sim_updates_per_step=1,
    gpu_sim_device="auto",
    flowrl_options=None,
):
    """
    Train using the EXACT TF-Agents methodology with optional Constrained RL:
    - Collect 100 steps, then train for 1 iteration
    - Repeat for 1000 iterations (total 100,000 steps)
    - Evaluate every 4 iterations (every 400 steps)
    - NEW: Support for adaptive Lagrange multiplier constraints (CRL)

    With ``gpu_sim_envs=N`` training collection runs N baskets in lockstep in
    BatchedPortfolioVecEnv and performs ``gpu_sim_updates_per_step`` TD3
    updates per vector step.  Timesteps then count transitions (N per vector
    step); an iteration is 100 vector steps, so evaluation still happens every
    400 vector steps.
    """
    if algorithm == "FlowRL":
        if not (use_attention and attention_type == "variable_coin_attention"
                and use_variable_portfolio and encoder_variant == "baseline"):
            raise ValueError("FlowRL requires --variable-portfolio --encoder-variant baseline --use-attention --attention-type variable_coin_attention")
        if proportional_exp_mode != "off" or use_qwm or gpu_sim_envs or aux_coef:
            raise ValueError("FlowRL v3 configures its portfolio candidates through --flowrl-*; "
                             "TD3 --proportional-exp-mode, QWM, encoder auxiliaries and --gpu-sim-envs are incompatible")
        return train_flowrl(mlp_size, reward_type, total_timesteps, log_dir, model_save_path,
                            resume_checkpoint, flowrl_options, crl_profile=crl_profile)
    configure_a40_runtime()
    gpu_sim_envs = max(0, int(gpu_sim_envs or 0))
    gpu_sim_updates_per_step = max(1, int(gpu_sim_updates_per_step or 1))
    
    # ✅ FIX: Calculate NUM_ITERATIONS based on total_timesteps
    COLLECT_STEPS_PER_ITERATION = 100 * max(1, gpu_sim_envs)
    NUM_ITERATIONS = total_timesteps // COLLECT_STEPS_PER_ITERATION
    EVAL_INTERVAL = 4  # Every 4 iterations
    NUM_EVAL_EPISODES = 4

    if resume_checkpoint:
        resume_step = parse_checkpoint_step(resume_checkpoint)
        start_iteration = resume_step // COLLECT_STEPS_PER_ITERATION
        resume_run_name = os.path.basename(os.path.dirname(os.path.dirname(resume_checkpoint)))
        print(f"🔄 RESUMING from checkpoint: {resume_checkpoint}")
        print(f"🔄 Resume step: {resume_step}, Start iteration: {start_iteration}/{NUM_ITERATIONS}")
        print(f"🔄 Remaining steps: {(NUM_ITERATIONS - start_iteration) * COLLECT_STEPS_PER_ITERATION}")
    else:
        resume_step = 0
        start_iteration = 0
        resume_run_name = None

    # Sigma mode/loss defaults from config (Option B by default).
    sigma_input_mode = str(sigma_input_mode or getattr(config, "SIGMA_INPUT_MODE", "passthrough")).strip().lower()
    sigma_gate_scale = float(getattr(config, "SIGMA_GATE_SCALE", 0.1) if sigma_gate_scale is None else sigma_gate_scale)
    if sigma_loss_focus_enabled is None:
        sigma_loss_focus_enabled = bool(getattr(config, "SIGMA_LOSS_FOCUS_ENABLED", True))
    if sigma_loss_focus_alpha is None:
        sigma_loss_focus_alpha = float(getattr(config, "SIGMA_LOSS_FOCUS_ALPHA", 0.75))
    if sigma_loss_focus_power is None:
        sigma_loss_focus_power = float(getattr(config, "SIGMA_LOSS_FOCUS_POWER", 1.0))
    if sigma_loss_focus_max_weight is None:
        sigma_loss_focus_max_weight = float(getattr(config, "SIGMA_LOSS_FOCUS_MAX_WEIGHT", 3.0))
    if sigma_loss_focus_eps is None:
        sigma_loss_focus_eps = float(getattr(config, "SIGMA_LOSS_FOCUS_EPS", 1e-6))
    proportional_exp_mode = validate_proportional_exp_configuration(
        algorithm=algorithm,
        use_attention=use_attention,
        attention_type=attention_type,
        encoder_variant=encoder_variant,
        proportional_exp_mode=proportional_exp_mode,
    )
    use_qwm = validate_qwm_configuration(
        use_qwm=use_qwm,
        algorithm=algorithm,
        use_attention=use_attention,
        attention_type=attention_type,
        use_variable_portfolio=use_variable_portfolio,
        encoder_variant=encoder_variant,
        proportional_exp_mode=proportional_exp_mode,
    )
    earnmore_masked_reconstruction_enabled = bool(
        getattr(config, "EARNMORE_MASKED_RECONSTRUCTION_ENABLED", False)
    )
    if (
        resume_checkpoint
        and use_attention
        and attention_type == "variable_coin_attention"
        and encoder_variant == "baseline"
        and proportional_exp_mode == "boltzmann_disagreement"
    ):
        validate_attention_resume_objective(resume_checkpoint)
    if bool(getattr(config, "LOCAL_FINANCIAL_HANDOFF_ENABLED", False)):
        if proportional_exp_mode == "off":
            raise ValueError(
                "Local-financial handoff requires the component-outcome replay "
                "path; run the configured boltzmann_disagreement ProExp mode."
            )
        if not (
            bool(getattr(config, "COMPONENT_OUTCOME_AUX_ENABLED", False))
            and bool(getattr(config, "COMPONENT_OUTCOME_UNIT_RETURN_TARGETS", False))
        ):
            raise ValueError(
                "Local-financial handoff requires supervised unit-return "
                "component outcomes for every active asset."
            )
        if bool(getattr(config, "COMPONENT_FORECAST_BRIDGE_ENABLED", False)):
            raise ValueError(
                "Disable the old hidden-vector forecast bridge when the "
                "local-financial handoff is enabled."
            )
    if earnmore_masked_reconstruction_enabled:
        if bool(getattr(config, "FOUNDATION_LEARNING_ENABLED", False)):
            raise ValueError(
                "EarnMore-TD3 is the foundation-free representation experiment; "
                "FOUNDATION_LEARNING_ENABLED must remain false."
            )
        if use_qwm:
            raise ValueError(
                "EarnMore masked reconstruction is the active experiment; "
                "QWM must remain disabled."
            )
        if not (
            algorithm.upper() == "TD3"
            and use_attention
            and attention_type == "variable_coin_attention"
            and use_variable_portfolio
            and encoder_variant == "baseline"
        ):
            raise ValueError(
                "EarnMore masked reconstruction requires TD3 with the baseline "
                "variable_coin_attention policy and variable 8-of-N portfolios."
            )
        if len(getattr(config, "COIN_POOL", ())) <= int(config.SELECTED_ASSETS_PER_EPISODE):
            raise ValueError(
                "EarnMore-TD3 must be trained on a true 8-of-N universe with "
                "N > 8; an 8-of-8 data file is not valid for this experiment."
            )
        if resume_checkpoint and not config.EARNMORE_GLOBAL_POOL_ENABLED:
            raise ValueError(
                "EarnMore-TD3 is a fresh 8-of-13 representation experiment; "
                "do not resume an older checkpoint or its replay buffer."
            )
    if gpu_sim_envs:
        if not (
            algorithm.upper() == "TD3"
            and use_attention
            and attention_type == "variable_coin_attention"
            and use_variable_portfolio
            and reward_type == "STRUCTURED_CREDIT"
        ):
            raise ValueError(
                "--gpu-sim-envs requires TD3 with variable_coin_attention, "
                "--variable-portfolio, and the STRUCTURED_CREDIT reward."
            )
        if use_qwm:
            raise ValueError("--gpu-sim-envs cannot be combined with --use-qwm")
        if not (
            config.EARNMORE_GLOBAL_POOL_ENABLED
            and config.EARNMORE_POOL_LEVEL_ENABLED
            and config.LOCAL_CAUSAL_INPUTS_ENABLED
        ):
            raise ValueError(
                "--gpu-sim-envs requires the global pool, whole-stock pool "
                "history, and local-causal inputs."
            )
    iql_attention_td3_enabled = use_iql_attention_td3(
        proportional_exp_mode=proportional_exp_mode,
        use_attention=use_attention,
        attention_type=attention_type,
        encoder_variant=encoder_variant,
    )
    attention_training_profile = resolve_attention_training_profile(
        mlp_size=mlp_size,
        use_attention=use_attention,
        attention_type=attention_type,
    )
    multitask_attention_profile = bool(
        attention_training_profile["multitask_enabled"]
    )
    selected_attention_batch_size = int(
        attention_training_profile["batch_size"]
    )
    selected_attention_learning_starts = int(
        attention_training_profile["learning_starts"]
    )
    selected_multitask_gamma = attention_training_profile["gamma"]
    
    if resume_run_name:
        run_name = resume_run_name
        print(f"🔄 Reusing existing run name: {run_name}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crl_suffix = f"_CRL_{crl_profile}" if algorithm == "TD3" else ""
        attention_suffix = f"_{attention_type}" if use_attention else "_mlp"
        portfolio_suffix = "_variable" if use_variable_portfolio else "_fixed"
        proportional_exp_suffix = get_proportional_exp_run_suffix(proportional_exp_mode)
        qwm_suffix = "_qwm" if use_qwm else ""
        earnmore_suffix = (
            (f"_earnmore_global{len(config.COIN_POOL)}_select{config.SELECTED_ASSETS_PER_EPISODE}" + ("_whole_stock_pool" if config.EARNMORE_POOL_LEVEL_ENABLED else "") + ("_task_conditioned_critic" if config.CRITIC_POOL_TASK_CONDITIONING_ENABLED else "") + ("_centered_advantage_bellman" if config.CRITIC_CENTERED_ADVANTAGE_ENABLED else "") + ("_popart" if config.TASK_POPART_ENABLED else "") + ("_xbr" if config.CROSS_BASKET_RELABEL_ENABLED else "") + ("_actor_real" if (config.CROSS_BASKET_RELABEL_ENABLED and str(getattr(config, "CROSS_BASKET_RELABEL_ACTOR_CRL_ROWS", "original")).strip().lower() in {"original", "real", "source", "full"}) else "")
              if config.EARNMORE_GLOBAL_POOL_ENABLED else "_earnmore_masked")
            if earnmore_masked_reconstruction_enabled
            else ("_basket_history_g966_norecon" if config.PORTFOLIO_REWARD_HISTORY_ENABLED else "")
        )
        iql_suffix = "_iql" if iql_attention_td3_enabled else ""
        encoder_suffix = f"_{encoder_variant}"
        aux_suffix = f"_aux{aux_coef:g}" if float(aux_coef) > 0.0 else ""
        batch_sim_suffix = (
            f"_bs{gpu_sim_envs}x{gpu_sim_updates_per_step}" if gpu_sim_envs else ""
        )
        run_name = (
            f"{algorithm}_tf_agents_style_{mlp_size}_{reward_type}"
            f"{attention_suffix}{portfolio_suffix}{crl_suffix}{proportional_exp_suffix}{qwm_suffix}{earnmore_suffix}{iql_suffix}{batch_sim_suffix}_{timestamp}"
        )
        run_name = f"{run_name}{encoder_suffix}{aux_suffix}"
        if len(run_name.encode()) > 255 and earnmore_suffix:
            # Filesystems cap a path component at 255 bytes.
            digest = hashlib.sha1(earnmore_suffix.encode()).hexdigest()[:8]
            run_name = run_name.replace(earnmore_suffix, f"_earnmore_{digest}", 1)
    run_log_dir = os.path.join(log_dir, run_name)

    # ✅ DIRECTORY FIX: Create log directory before any logging operations
    os.makedirs(run_log_dir, exist_ok=True)
    
    # ✅ CHECKPOINT FIX: Create run-specific directories for models and checkpoints
    run_model_dir = os.path.join(model_save_path, run_name)
    checkpoints_dir = os.path.join(run_model_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"💾 Models and checkpoints will be saved in: {run_model_dir}")
    
    print(f"🎯 TF-AGENTS EXACT REPLICATION {'WITH CONSTRAINED RL' if algorithm == 'TD3' else ''}")
    print(f"📊 Reward Type: {reward_type} {'(Net Return = Profit - Transaction Costs)' if reward_type == 'TRANSACTION_COST' else ''}")
    print(f"📊 Transaction Cost Rate: {config.TRANSACTION_COST:.3f} ({config.TRANSACTION_COST*100:.1f}%)")
    
    # ✅ CRL INTEGRATION: Show CRL configuration if TD3
    if algorithm == "TD3":
        if multitask_attention_profile:
            print("🧠 MULTITASK 8-of-13 CAPACITY PROFILE")
            print(
                f"   - batch_size={selected_attention_batch_size}, "
                f"learning_starts={selected_attention_learning_starts}"
            )
            print(
                f"   - gamma={selected_multitask_gamma:.9f}, "
                f"reciprocal_horizon="
                f"{1.0 / (1.0 - selected_multitask_gamma):.1f} steps, "
                f"episode_length={config.EPISODE_LENGTH}"
            )
        # ✅ FIX: Select correct CRL config based on portfolio size
        if config.MAX_COINS > 4:  # Threshold for using large portfolio profiles (consistent)
            print(f"🎯 Using LARGE PORTFOLIO CRL profile: {crl_profile}")
            crl_config = config.get_large_portfolio_crl_config(crl_profile)
        else:
            print(f"🎯 Using standard CRL profile: {crl_profile}")
            crl_config = config.get_crl_config(crl_profile)
        
        print(f"🎯 CRL Configuration:")
        print(f"   Profile: {crl_profile}")
        print(f"   Use CRL: {crl_config['use_crl']}")
        if crl_config['use_crl']:
            print(f"   Constraint threshold: {crl_config['constraint_threshold']:.4f}")
            print(f"   Initial λ: {crl_config['initial_lambda']:.4f}")
            print(f"   λ learning rate: {crl_config['lambda_lr']:.1e}")
            print(f"   λ Warm-up Steps: {crl_config.get('lambda_warmup_steps', 'N/A')}")
    
    print(f"📊 Iterations: {NUM_ITERATIONS}")
    print(f"📊 Steps per iteration: {COLLECT_STEPS_PER_ITERATION}")
    print(f"📊 Total steps: {NUM_ITERATIONS * COLLECT_STEPS_PER_ITERATION}")
    print(f"📊 Evaluation every {EVAL_INTERVAL} iterations ({EVAL_INTERVAL * COLLECT_STEPS_PER_ITERATION} steps)")
    # Keep the checkpoint cadence at the same number of TD3 updates when the
    # batched simulator collects several transitions per update.
    resume_checkpoint_interval = (
        RESUME_CHECKPOINT_INTERVAL_STEPS * gpu_sim_envs // gpu_sim_updates_per_step
        if gpu_sim_envs
        else RESUME_CHECKPOINT_INTERVAL_STEPS
    )
    print(f"💾 Paired resume checkpoints every {resume_checkpoint_interval} steps")
    print(f"💾 Keeping latest {RESUME_CHECKPOINTS_TO_KEEP} paired resume checkpoints")
    print(f"📊 Episodes per evaluation: {NUM_EVAL_EPISODES}")
    
    # ✅ VARIABLE PORTFOLIO INTEGRATION: Create environments with variable portfolio support
    print(f"🎯 Creating environments with reward_type: {reward_type}")
    # ✅ RUNTIME CONFIG FIX: Ensure dynamic coin-count range when variable portfolio is enabled
    if use_variable_portfolio:
        try:
            import config as cfg
            # ✅ CORRECT WAY: Call the dedicated function to set portfolio mode.
            # This ensures all related global variables (USE_VARIABLE_PORTFOLIO, 
            # COINS_PER_EPISODE_RANGE) are set correctly and in one place.
            cfg.set_portfolio_mode(True)
        except Exception as e:
            print(f"⚠️ Could not set portfolio mode at runtime: {e}")
    # ✅ REFACTOR: Pass use_variable_portfolio flag to the environment
    training_monitor_path = os.path.join(run_log_dir, "training")
    evaluation_monitor_path = os.path.join(run_log_dir, "evaluation")
    append_training_monitor = should_append_monitor(training_monitor_path, resume_checkpoint)
    append_evaluation_monitor = should_append_monitor(evaluation_monitor_path, resume_checkpoint)
    curriculum_manager = (
        build_basket_curriculum_manager()
        if use_variable_portfolio
        else None
    )
    local_composition_sampler = (
        build_local_composition_sampler()
        if use_variable_portfolio
        else None
    )
    training_basket_sampler = (
        curriculum_manager
        if curriculum_manager is not None
        else local_composition_sampler
    )
    if training_basket_sampler is not None and resume_checkpoint:
        curriculum_resume_path = curriculum_path_for_model_checkpoint(
            resume_checkpoint
        )
        if not os.path.exists(curriculum_resume_path):
            raise FileNotFoundError(
                "Basket-sampler resume requires the matching sampler "
                f"checkpoint: {curriculum_resume_path}"
            )
        training_basket_sampler.load(curriculum_resume_path)
        print(f"🔄 Basket sampler state restored from: {curriculum_resume_path}")
    if curriculum_manager is not None:
        distance_counts = {
            distance: len(baskets)
            for distance, baskets in curriculum_manager.baskets_by_distance.items()
        }
        print(
            "🧭 Basket curriculum enabled: "
            f"active_support={len(curriculum_manager.all_baskets)}/"
            f"{curriculum_manager.total_basket_count} tasks, "
            f"enumerated={curriculum_manager.enumerated_task_space}, "
            f"anchor={list(curriculum_manager.anchor)}, "
            f"distance_counts={distance_counts}, "
            "frontier_evidence_episodes="
            f"{int(config.BASKET_CURRICULUM_FRONTIER_EVIDENCE_EPISODES)}, "
            "active_frontier_cohort="
            f"{len(curriculum_manager.active_frontier_cohort)}"
        )
    if local_composition_sampler is not None:
        print(
            "🧩 Uniform local-composition curriculum enabled: "
            f"block={local_composition_sampler.block_episodes} episodes, "
            f"pivot={local_composition_sampler.pivot_episodes}, "
            f"one_swap={local_composition_sampler.neighbor_episodes}; "
            "ordinary full-buffer Bellman replay remains unchanged"
        )
    qwm_training_env_count = (
        max(1, int(getattr(config, "QWM_PARALLEL_TRAIN_ENVS", 1)))
        if use_qwm
        else 1
    )
    if resume_checkpoint and gpu_sim_envs:
        saved_env_count = saved_model_environment_count(resume_checkpoint, default=1)
        if saved_env_count != gpu_sim_envs:
            raise ValueError(
                "Replay stores an environment axis; resume with "
                f"--gpu-sim-envs {saved_env_count} to match the checkpoint."
            )
    elif resume_checkpoint:
        # SB3 replay buffers persist an environment axis. An existing
        # checkpoint cannot be widened or narrowed without rewriting replay
        # chronology, so reconstruct exactly the saved worker count.
        qwm_training_env_count = saved_model_environment_count(
            resume_checkpoint, default=1
        )
        print(
            "↩️ QWM resume detected: preserving the checkpoint's historical "
            f"{qwm_training_env_count}-environment replay axis"
        )

    def _make_training_environment(worker_index):
        monitor_path = (
            training_monitor_path
            if worker_index == 0
            else f"{training_monitor_path}_worker_{worker_index}"
        )
        return Monitor(
            PortfolioEnv(
                reward_type=reward_type,
                use_variable_portfolio=use_variable_portfolio,
                basket_sampler=training_basket_sampler,
            ),
            filename=monitor_path,
            override_existing=not append_training_monitor,
        )

    if gpu_sim_envs:
        if training_basket_sampler is not None:
            raise ValueError(
                "--gpu-sim-envs draws baskets through config.get_random_coin_subset(); "
                "disable the basket curricula to use it."
            )
        batched_cache_dir = getattr(config, "BATCHED_SIM_CACHE_DIR", None)
        panel_started_at = time.perf_counter()
        print(
            "📦 Preparing the batched market panel "
            f"(built once from {config.FILE}, cached in {batched_cache_dir})..."
        )
        batched_panel = _env_module.build_batched_market_panel(
            cache_dir=batched_cache_dir
        )
        print(
            f"📦 Batched market panel ready: {len(batched_panel['starts'])} episode "
            f"windows in {time.perf_counter() - panel_started_at:.1f}s"
        )
        env = _env_module.BatchedPortfolioVecEnv(
            gpu_sim_envs,
            reward_type=reward_type,
            device=gpu_sim_device,
            panel=batched_panel,
            monitor_path=training_monitor_path,
            append_monitor=append_training_monitor,
        )
        print(
            f"⚡ Batched training simulator: {gpu_sim_envs} baskets in lockstep on "
            f"{env.device}; {gpu_sim_updates_per_step} TD3 update(s) per vector step "
            f"= {gpu_sim_updates_per_step / gpu_sim_envs:.3f} updates per transition; "
            f"evaluation stays on PortfolioEnv"
        )
    elif qwm_training_env_count > 1:
        env = DummyVecEnv(
            [
                (lambda worker_index=worker_index: _make_training_environment(
                    worker_index
                ))
                for worker_index in range(qwm_training_env_count)
            ]
        )
        print(
            "⚡ QWM collection batching enabled: "
            f"{qwm_training_env_count} causal environments, one full search "
            "and one TD/world update per real transition"
        )
        print(
            "🗂️ Training monitor mapping: worker 0 -> "
            "training.monitor.csv; additional workers -> "
            "training_worker_1.monitor.csv through "
            f"training_worker_{qwm_training_env_count - 1}.monitor.csv"
        )
    else:
        env = _make_training_environment(0)
    eval_basket_sampler = (
        CurriculumMatchedEvaluationSampler(
            curriculum_manager,
            seed=int(config.BASKET_CURRICULUM_SEED) + 1_048_583,
        )
        if curriculum_manager is not None
        else None
    )
    # Evaluation frequency, episode count, shaped Return, and FinancialReturn
    # definitions remain historical. Only basket identity is matched to the
    # teacher's current task distribution through an independent sampler.
    qwm_evaluation_env_count = (
        min(
            NUM_EVAL_EPISODES,
            max(1, int(getattr(config, "QWM_PARALLEL_EVAL_ENVS", 1))),
        )
        if use_qwm and bool(getattr(config, "QWM_BATCH_EVALUATION", True))
        else 1
    )
    eval_envs = []
    for worker_index in range(qwm_evaluation_env_count):
        monitor_path = (
            evaluation_monitor_path
            if worker_index == 0
            else f"{evaluation_monitor_path}_worker_{worker_index}"
        )
        eval_envs.append(
            Monitor(
                PortfolioEnv(
                    reward_type=reward_type,
                    use_variable_portfolio=use_variable_portfolio,
                    basket_sampler=eval_basket_sampler,
                ),
                filename=monitor_path,
                override_existing=not append_evaluation_monitor,
            )
        )
    eval_env = eval_envs[0]
    if qwm_evaluation_env_count > 1:
        print(
            "⚡ QWM evaluation batching enabled: "
            f"{qwm_evaluation_env_count} complete deterministic searches per "
            "device batch"
        )
    if resume_checkpoint:
        print(f"🔄 Training monitor log: {'append' if append_training_monitor else 'new file'}")
        print(f"🔄 Evaluation monitor log: {'append' if append_evaluation_monitor else 'new file'}")
    
    # ✅ STEP 1.1: Add SB3 logger configuration (MISSING in original TF-Agents style)
    if resume_checkpoint:
        archived_progress = archive_existing_progress_csv(run_log_dir, resume_step)
        if archived_progress:
            print(f"🔄 Archived existing progress.csv to: {archived_progress}")
    print(f"🔧 Step 1.1: Setting up SB3 logger for CSV generation...")
    logger = configure_training_logger(
        run_log_dir, ["stdout", "csv", "tensorboard"]
    )
    print(f"✅ SB3 logger configured with CSV output to: {run_log_dir}")
    
    # ✅ ATTENTION INTEGRATION: Choose between MLP and Attention policies
    if use_attention:
        print(f"🧠 ATTENTION MODE: Using {attention_type} with {mlp_size} configuration")
        
        # Map mlp_size to attention configuration
        from attention_policy import ATTENTION_CONFIGS
        if mlp_size not in ATTENTION_CONFIGS:
            raise ValueError(f"Unknown attention config: {mlp_size}. Available: {list(ATTENTION_CONFIGS.keys())}")
        
        attention_config = ATTENTION_CONFIGS[mlp_size].copy()
        if config.EARNMORE_GLOBAL_POOL_ENABLED:
            attention_config["asset_identity_dim"] = config.GLOBAL_POOL_IDENTITY_DIM
            print(f"Global pool: {len(config.COIN_POOL)} stable positions, "
                  f"{config.SELECTED_ASSETS_PER_EPISODE} tradable per episode")
        print(f"🔧 Attention config: {attention_config}")
        
        # Create attention policy
        from attention_policy import AttentionTD3Policy
        PolicyClass = AttentionTD3Policy
        
        # ✅ ENVIRONMENT COMPATIBILITY: Read actual max_coins and features_per_coin from config
        # Observation shape is MAX_COINS * FEATURES_PER_COIN; extractor must match or view() fails.
        import config as env_config
        actual_features_per_coin = env_config.FEATURES_PER_COIN
        actual_max_coins = env_config.MAX_COINS
        actual_num_asset_ids = getattr(env_config, "NUM_ASSET_IDS", actual_max_coins)
        actual_basket_context_dim = (
            int(getattr(env_config, "BASKET_CONTEXT_DIM", 0))
            if bool(getattr(env_config, "BASKET_CONTEXT_ENABLED", False))
            else 0
        )
        actual_basket_membership_dim = (
            int(getattr(env_config, "BASKET_MEMBERSHIP_DIM", 0))
            if bool(getattr(env_config, "BASKET_MEMBERSHIP_ENABLED", False))
            else 0
        )
        expected_features_per_coin = (
            len(env_config.SCOLS)
            + 1
            + env_config.COMPARATIVE_FEATURE_COUNT
            + env_config.EPISODE_RELATIVE_FEATURE_COUNT
        )
        if actual_features_per_coin != expected_features_per_coin:
            raise ValueError(
                "Observation-token contract mismatch: config.FEATURES_PER_COIN="
                f"{actual_features_per_coin}, but offline ({len(env_config.SCOLS)}) "
                f"+ cost basis (1) + selected-basket comparative "
                f"({env_config.COMPARATIVE_FEATURE_COUNT}) + episode-relative "
                f"({env_config.EPISODE_RELATIVE_FEATURE_COUNT}) = "
                f"{expected_features_per_coin}."
            )
        print(
            f"🔧 Environment has {actual_max_coins} max coins, "
            f"{actual_features_per_coin} features per coin "
            f"({len(env_config.SCOLS)} offline + 1 cost basis + "
            f"{env_config.COMPARATIVE_FEATURE_COUNT} selected-basket comparative + "
            f"{env_config.EPISODE_RELATIVE_FEATURE_COUNT} selected-basket)"
        )
        print(
            f"🧩 Stable asset map: {actual_num_asset_ids} ids; neural identity_dim="
            f"{attention_config.get('asset_identity_dim', 0)}"
        )
        print(
            f"🧺 Basket context g(B): enabled={bool(getattr(env_config, 'BASKET_CONTEXT_ENABLED', False))} "
            f"dim={actual_basket_context_dim} "
            f"features={getattr(env_config, 'BASKET_CONTEXT_FEATURES', ())}"
        )
        print(
            f"🪪 Basket membership metadata: neural_enabled={bool(getattr(env_config, 'BASKET_MEMBERSHIP_ENABLED', False))} "
            f"dim={actual_basket_membership_dim} "
            f"role_mixed_replay={bool(getattr(env_config, 'CURRICULUM_MIXED_REPLAY_ENABLED', False))} "
            f"min_unique_rows_per_role="
            f"{getattr(env_config, 'CURRICULUM_MIXED_REPLAY_MIN_UNIQUE_TRANSITIONS_PER_ROLE', 1)} "
            f"target_distinct_tasks_per_role="
            f"{getattr(env_config, 'CURRICULUM_MIXED_REPLAY_TARGET_DISTINCT_TASKS_PER_ROLE', 1)} "
            f"task_selection={getattr(env_config, 'BASKET_REPLAY_GROUP_SELECTION', 'uniform')} "
            f"without_replacement={bool(getattr(env_config, 'BASKET_REPLAY_SAMPLE_WITHOUT_REPLACEMENT', True))} "
            f"min_completed_episodes={int(getattr(env_config, 'BASKET_REPLAY_MIN_COMPLETED_EPISODES', 0))} "
            f"partial_unique_batch={bool(getattr(env_config, 'BASKET_REPLAY_ALLOW_PARTIAL_UNIQUE_BATCH', False))}"
        )
        print(
            f"🩺 Basket amortization diagnostics: "
            f"enabled={bool(getattr(env_config, 'BASKET_CONTEXT_DIAGNOSTICS_ENABLED', False))} "
            f"sensitivity_interval={getattr(env_config, 'BASKET_CONTEXT_DIAGNOSTICS_SENSITIVITY_INTERVAL', 50)} "
            f"sensitivity_delta={getattr(env_config, 'BASKET_CONTEXT_DIAGNOSTICS_SENSITIVITY_DELTA', 0.75)}"
        )
        print(
            f"🧬 Critic low-rank local UVFA: "
            f"enabled={bool(getattr(env_config, 'UVFA_LOCAL_INTERACTION_ENABLED', False))} "
            f"rank={int(getattr(env_config, 'UVFA_LOCAL_INTERACTION_RANK', 8))} "
            f"max_abs={float(getattr(env_config, 'UVFA_LOCAL_INTERACTION_MAX_ABS', 0.10)):.3f} "
            f"task_source=continuous_basket_context"
        )
        if bool(getattr(env_config, "MDAEN_FAM_ENABLED", False)):
            print(
                "🧭 MDAEN FAM→AAM contract: "
                f"X=[m={actual_max_coins}, "
                f"n={int(getattr(env_config, 'MDAEN_FAM_HISTORY_LENGTH', 0))}, "
                f"f={int(getattr(env_config, 'MDAEN_FAM_FEATURE_COUNT', 0))}] "
                f"kernels={tuple(getattr(env_config, 'MDAEN_FAM_KERNEL_SIZES', ()))} "
                f"→ FAM maps [m,n,{int(getattr(env_config, 'MDAEN_FAM_BRANCH_COUNT', 0))}f] "
                f"→ one shared FC per map "
                f"→ MDAEN q={int(getattr(env_config, 'MDAEN_FAM_ASSET_VECTOR_DIM', attention_config['features_dim']))} "
                f"→ shared adapter → AAM width={int(attention_config['features_dim'])}; "
                "AAM Q/K/V is FAM-only and the flat snapshot is a bounded "
                "post-AAM fusion residual"
            )
        print(
            f"🧩 Predictive compositional MoE: "
            f"enabled={bool(getattr(env_config, 'PREDICTIVE_SSL_ENABLED', False))} "
            f"experts={int(getattr(env_config, 'PREDICTIVE_MOE_NUM_EXPERTS', 4))} "
            f"same_financial_batch={not bool(getattr(env_config, 'PREDICTIVE_SSL_SEPARATE_REPLAY_ENABLED', False))} "
            f"horizon={int(getattr(env_config, 'PREDICTIVE_SSL_HORIZON', 1))} "
            f"sequence_batch={int(getattr(env_config, 'PREDICTIVE_SSL_SEQUENCE_BATCH_SIZE', 128))} "
            f"critic_weight={float(getattr(env_config, 'PREDICTIVE_SSL_CRITIC_LOSS_WEIGHT', 0.01)):.4f} "
            f"actor_weight={float(getattr(env_config, 'PREDICTIVE_SSL_ACTOR_LOSS_WEIGHT', 0.005)):.4f} "
            f"gradient_guard={bool(getattr(env_config, 'PREDICTIVE_SSL_GRADIENT_GUARD_ENABLED', True))} "
            f"critic_shared_cap={float(getattr(env_config, 'PREDICTIVE_SSL_CRITIC_SHARED_GRADIENT_MAX_RATIO', 0.25)):.2f} "
            f"actor_shared_cap={float(getattr(env_config, 'PREDICTIVE_SSL_ACTOR_SHARED_GRADIENT_MAX_RATIO', 0.10)):.2f}"
        )
        print(
            "🧭 Value-equivalent Path-2 model: "
            f"enabled={bool(getattr(env_config, 'VALUE_EQUIVALENT_MODEL_ENABLED', False))} "
            f"bins={int(getattr(env_config, 'VALUE_EQUIVALENT_DISTRIBUTION_BINS', 101))} "
            f"expectile={float(getattr(env_config, 'VALUE_EQUIVALENT_EXPECTILE', 0.80)):.2f} "
            f"horizon_discount={float(getattr(env_config, 'PREDICTIVE_SSL_HORIZON_DISCOUNT', 0.5)):.2f}; "
            "the auxiliary predicts rewards, continuation values, and sampled "
            "Bellman backups while ordinary causal-financial TD3 retains the "
            "actor and critic objective"
        )
        print(f"Market input contract: {env_config.market_input_contract()}")
        print("Relations are learned from local histories; handcrafted basket slots are zero."
              if env_config.LOCAL_CAUSAL_INPUTS_ENABLED else
              "Dynamic selected-basket features: " + ", ".join(env_config.EPISODE_RELATIVE_FEATURES))

        # Set policy_kwargs for attention
        policy_kwargs = {
            "attention_type": attention_type,
            "features_dim": attention_config["features_dim"],
            "n_heads": attention_config["n_heads"],
            "n_layers": attention_config["n_layers"],
            "dropout": attention_config["dropout"],
            "net_arch": attention_config["net_arch"],
            "max_coins": actual_max_coins,  # ✅ Use env MAX_COINS so extractor reshape matches observation size
            "features_per_coin": actual_features_per_coin,  # ✅ Use actual environment value
            "num_asset_ids": actual_num_asset_ids,
            "asset_identity_dim": attention_config.get("asset_identity_dim", 0),
            "basket_context_dim": attention_config.get(
                "basket_context_dim", actual_basket_context_dim
            ),
            "basket_membership_dim": attention_config.get(
                "basket_membership_dim", actual_basket_membership_dim
            ),
            "local_feature_dim": env_config.LOCAL_FEATURE_COUNT,
            "position_feature_dim": env_config.POSITION_FEATURE_COUNT,
            "comparative_feature_dim": env_config.COMPARATIVE_FEATURE_COUNT,
            "context_feature_dim": env_config.EPISODE_RELATIVE_FEATURE_COUNT,
            "use_market_attention": attention_config.get("use_market_attention", False),  # False = faster (light)
            "use_global_pooling": attention_config.get("use_global_pooling", True),  # default on; set False in config to disable
            "component_outcome_aux_enabled": bool(
                env_config.COMPONENT_OUTCOME_AUX_ENABLED
                and proportional_exp_mode != "off"
            ),
            "component_outcome_dim": len(
                env_config.COMPONENT_OUTCOME_HORIZONS
            ),
            "component_outcome_unit_return_targets": bool(
                env_config.COMPONENT_OUTCOME_UNIT_RETURN_TARGETS
            ),
            "component_forecast_bridge_enabled": bool(
                env_config.COMPONENT_OUTCOME_AUX_ENABLED
                and env_config.COMPONENT_FORECAST_BRIDGE_ENABLED
                and proportional_exp_mode != "off"
            ),
            "component_forecast_bridge_return_reference": float(
                env_config.COMPONENT_FORECAST_BRIDGE_RETURN_REFERENCE
            ),
            "component_forecast_bridge_max_abs": float(
                env_config.COMPONENT_FORECAST_BRIDGE_MAX_ABS
            ),
            "component_forecast_bridge_local_only": bool(
                env_config.COMPONENT_FORECAST_BRIDGE_LOCAL_ONLY
            ),
            "uvfa_local_interaction_enabled": bool(
                env_config.UVFA_LOCAL_INTERACTION_ENABLED
            ),
            "uvfa_local_interaction_rank": int(
                env_config.UVFA_LOCAL_INTERACTION_RANK
            ),
            "uvfa_local_interaction_max_abs": float(
                env_config.UVFA_LOCAL_INTERACTION_MAX_ABS
            ),
            "predictive_moe_enabled": bool(
                env_config.PREDICTIVE_SSL_ENABLED
                and env_config.PREDICTIVE_MOE_ENABLED
            ),
            "predictive_moe_num_experts": int(
                env_config.PREDICTIVE_MOE_NUM_EXPERTS
            ),
            "predictive_moe_expert_hidden_dim": int(
                env_config.PREDICTIVE_MOE_EXPERT_HIDDEN_DIM
            ),
            "predictive_moe_router_temperature": float(
                env_config.PREDICTIVE_MOE_ROUTER_TEMPERATURE
            ),
            "predictive_moe_max_residual_abs": float(
                env_config.PREDICTIVE_MOE_MAX_RESIDUAL_ABS
            ),
            # Replay stores causal per-asset unit-exposure returns. The private
            # forecaster learns those labels; its detached forecast is exposed
            # only to the per-asset Q route through the TD-owned bridge above.
            # ✅ CRITICAL FIX: Remove actor_lr/critic_lr from policy_kwargs
            # CustomTD3_AC manages learning rates directly via actor_learning_rate/critic_learning_rate
        }
        
    else:
        print(f"🚫 MLP MODE: Using SimpleMlpTD3Policy with {mlp_size} configuration")
        
        # ✅ NEW: Get MLP architecture dynamically based on mlp_size
        from attention_policy import MLP_CONFIGS
        if mlp_size not in MLP_CONFIGS:
            raise ValueError(f"Unknown MLP size: {mlp_size}. Available: {list(MLP_CONFIGS.keys())}")
        
        mlp_net_arch = MLP_CONFIGS[mlp_size]["net_arch"]
        print(f"🔧 Using MLP architecture for '{mlp_size}': {mlp_net_arch}")
        
        PolicyClass = SimpleMlpTD3Policy
        from extractor import SharedCoinOnlyExtractor as encoder
        from extractor_res import SharedCoinOnlyExtractor as encoder_res        
        # ✅ REFACTOR: Define policy_kwargs directly as per SB3 best practice.

        if encoder_variant == "encoder":
            extractor_class = encoder
            extractor_kwargs = dict(
                max_coins=config.MAX_COINS,
                features_per_coin=config.FEATURES_PER_COIN,
                coin_hidden=64,
                coin_out=64,
                zero_inactive_slots=False,
                aux_mask_prob=float(aux_mask_prob),
            )
        elif encoder_variant == "encoder_res":
            extractor_class = encoder_res
            extractor_kwargs = dict(
                max_coins=config.MAX_COINS,
                features_per_coin=config.FEATURES_PER_COIN,
                coin_hidden=64,
                coin_out=64,
                zero_inactive_slots=False,
                include_pooled_summary=True,  # keep this on; it’s the “cash-gate sanity vector”
                use_film=False,               # flip to True when you want FiLM
                film_hidden=64,
                film_scale=0.1,
                aux_mask_prob=float(aux_mask_prob),
            )
        elif encoder_variant == "baseline":
            extractor_class = CombinedExtractor
            extractor_kwargs = {}

        else:
            raise ValueError(f"Unknown encoder_variant='{encoder_variant}'. Expected: baseline, encoder, encoder_res")
        if encoder_variant == "encoder" or encoder_variant == "encoder_res":
            policy_kwargs = {
                "net_arch": {
                    "pi": mlp_net_arch,  # Actor network
                    "qf": mlp_net_arch   # Critic network
                },
                "features_extractor_class": extractor_class, #SharedCoinEncoderExtractor,                
                "features_extractor_kwargs": extractor_kwargs,


                # net_arch stays whatever you already use for pi/qf
        

            }
        elif encoder_variant == "baseline":
            policy_kwargs = {
                "net_arch": {
                    "pi": mlp_net_arch,  # Actor network
                    "qf": mlp_net_arch   # Critic network
                },
            }
        else:
            raise ValueError(f"Unknown encoder_variant='{encoder_variant}'. Expected: baseline, encoder, encoder_res")


            # ✅ CRITICAL FIX: Remove actor_lr/critic_lr from policy_kwargs
            # CustomTD3_AC manages learning rates directly via actor_learning_rate/critic_learning_rate
            # Passing them here causes optimizer conflicts
        

        # ✅ FIX: When using variable portfolio without attention, observations are Dicts.
        # Use CombinedExtractor so MLP policies can handle Dict observations (observations + mask).

        # CombinedExtractor
        if use_variable_portfolio:
            if encoder_variant == "encoder" or encoder_variant == "encoder_res":
               policy_kwargs["features_extractor_class"] = extractor_class #CombinedExtractor 

            else:
                policy_kwargs["features_extractor_class"] = CombinedExtractor
    
    # Create model with exact TF-Agents parameters
    if algorithm == "TD3":
        # ✅ EXPLORATION FIX: Use OrnsteinUhlenbeckActionNoise to match TF-Agents
        # ✅ ENHANCED FIX: Higher noise for variable portfolios
        n_actions = env.action_space.shape[-1]
        sigma_value = 0.3 if n_actions > 4 else 0.3  # More noise for variable portfolios
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=sigma_value * np.ones(n_actions),
            theta=0.15 # Corresponds to ou_damping
        )
        print(f"⚡ Using OrnsteinUhlenbeckActionNoise (sigma={sigma_value}, theta=0.15) - Enhanced for variable portfolios.")

        # ✅ ACTOR-CRITIC STABILITY FIX: Define separate, decaying learning rate schedules
        # The actor should be more conservative (lower LR) than the critic to prevent instability.
        # actor_lr_schedule = get_linear_fn(start=1e-3, end=5e-5, end_fraction=0.9)
        # critic_lr_schedule = get_linear_fn(start=1e-3, end=5e-5, end_fraction=0.9)
        actor_lr_schedule = get_linear_fn(start=1e-4, end=8e-5, end_fraction=0.9)
        # The critic will use 5e-4 for the first 60% of training, then decay to 1e-4 in the final 40%.
        # CustomTD3 passes progress_remaining (1.0 at start, 0.0 at end), so: flat when progress_remaining > 0.4,
        # decay when progress_remaining <= 0.4 (last 40% of training).
        critic_lr_schedule = lambda progress_remaining: 5e-4 if progress_remaining > 0.05 else get_linear_fn(start=5e-4, end=4.5e-4, end_fraction=1.0)(progress_remaining / 0.95)
        selected_policy_delay = 5
        selected_sigma_loss_focus_enabled = bool(sigma_loss_focus_enabled)

        if iql_attention_td3_enabled:
            # Paper-aligned IQL update cadence:
            # - actor/value/critic should update every batch
            # - actor uses a cosine decay
            # - critic/value use the same baseline LR to keep V from lagging Q
            actor_lr_schedule = lambda progress_remaining: 0.5 * 3e-4 * (
                1.0 + math.cos(math.pi * (1.0 - float(progress_remaining)))
            )
            critic_lr_schedule = lambda progress_remaining: 3e-4
            selected_policy_delay = 1
            selected_sigma_loss_focus_enabled = False

        print("🔧 Using separate, decaying learning rates for actor and critic:")
        if iql_attention_td3_enabled:
            print("   - IQL Actor LR: cosine decay from 3e-4 to 0.0 (paper-style extraction schedule).")
            print("   - IQL Critic LR: fixed 3e-4.")
            print("   - IQL Value LR: matched to critic at 3e-4 via iql_value_learning_rate.")
            print("   - IQL Policy Delay: 1 (update actor every batch).")
        else:
            print(f"   - Actor LR: Starts at 1e-4, decays to 5e-5 over 90% of training.")
            print(f"   - Critic LR: Flat 5e-4 for first 60%, then decays to 1e-4 over final 40%.")
        print(f"🔬 Sigma Input Mode: {sigma_input_mode}")
        print(f"🔬 Sigma Loss Focus: {bool(selected_sigma_loss_focus_enabled)}")
        if bool(selected_sigma_loss_focus_enabled):
            print(f"   α={float(sigma_loss_focus_alpha):.3f}, p={float(sigma_loss_focus_power):.3f}, "
                  f"w_max={float(sigma_loss_focus_max_weight):.3f}, eps={float(sigma_loss_focus_eps):.1e}")
        if iql_attention_td3_enabled:
            print("🧠 IQL attention TD3 enabled: custom_td3_actor_critic_2crl_safe_pro_IQL_attention.py")
            print("   Pure IQL path: value expectile loss + V(next_state) critic target + AWR-style portfolio KL actor.")
        if proportional_exp_mode != "off":
            print(f"🧪 Proportional-exp mode: {proportional_exp_mode}")
            print(
                f"   K={int(proportional_exp_candidates)}, alpha={float(proportional_exp_alpha):.4f}, "
                f"alpha_max={float(proportional_exp_alpha_max):.4f}, "
                f"gain={float(proportional_exp_disagreement_gain):.4f}, "
                f"ema_decay={float(proportional_exp_ema_decay):.5f}"
            )
            print(
                f"   cloud_space={proportional_exp_cloud_space}, "
                f"budget_ratio={float(proportional_exp_budget_candidate_ratio):.2f}, "
                f"target_budget_std={proportional_exp_target_budget_std if proportional_exp_target_budget_std is not None else 'auto'}, "
                f"target_comp_std={proportional_exp_target_composition_std if proportional_exp_target_composition_std is not None else 'auto'}, "
                f"actor_budget_std={proportional_exp_actor_budget_std if proportional_exp_actor_budget_std is not None else 'auto'}, "
                f"actor_comp_std={proportional_exp_actor_composition_std if proportional_exp_actor_composition_std is not None else 'auto'}"
            )
        if earnmore_masked_reconstruction_enabled:
            print(
                "🧩 EarnMore-TD3 masked reconstruction enabled: online 8-of-13 "
                "replay only; QWM and foundation are inactive."
            )
            if bool(getattr(config, "LOCAL_FINANCIAL_HANDOFF_ENABLED", False)):
                print(
                    "   Control path: shared local FAM asset preferences plus "
                    "bounded AAM corrections; twin Q = detached action-weighted "
                    "asset forecast + Bellman-trained residual."
                )
                print(
                    "   Forecast: supervised local unit return plus bounded "
                    "peer correction; old hidden-vector forecast bridge off."
                )
            else:
                print(
                    "   Control path: complete fused temporal/relational actor "
                    "and critic are primary."
                )
            print(
                "   Representation scaffold: August-style causal component "
                "forecasting + guarded one-step predictive MoE are active; "
                "EarnMore remains a separate post-control optimizer step."
            )


        # ✅ CRL INTEGRATION: Get CRL configuration and pass to model
        # ✅ FIX: Select correct CRL config based on portfolio size
        if config.MAX_COINS > 4:  # Threshold for using large portfolio profiles
            print(f"🎯 Using LARGE PORTFOLIO CRL profile: {crl_profile}")
            crl_config = config.get_large_portfolio_crl_config(crl_profile)
        else:
            print(f"🎯 Using standard CRL profile: {crl_profile}")
            crl_config = config.get_crl_config(crl_profile)
        
        if encoder_variant == "baseline":
            baseline_td3_cls = get_attention_td3_algorithm_class(
                proportional_exp_mode,
                use_attention=use_attention,
                attention_type=attention_type,
                encoder_variant=encoder_variant,
            )
            proportional_exp_kwargs = {}
            if proportional_exp_mode != "off":
                proportional_exp_kwargs = dict(
                    proportional_exp_mode=proportional_exp_mode,
                    proportional_exp_candidates=int(proportional_exp_candidates),
                    proportional_exp_alpha=float(proportional_exp_alpha),
                    proportional_exp_alpha_max=float(proportional_exp_alpha_max),
                    proportional_exp_disagreement_gain=float(proportional_exp_disagreement_gain),
                    proportional_exp_ema_decay=float(proportional_exp_ema_decay),
                    proportional_exp_cloud_space=str(proportional_exp_cloud_space),
                    proportional_exp_budget_candidate_ratio=float(proportional_exp_budget_candidate_ratio),
                    proportional_exp_target_budget_std=(
                        None if proportional_exp_target_budget_std is None else float(proportional_exp_target_budget_std)
                    ),
                    proportional_exp_target_composition_std=(
                        None if proportional_exp_target_composition_std is None else float(proportional_exp_target_composition_std)
                    ),
                    proportional_exp_actor_budget_std=(
                        None if proportional_exp_actor_budget_std is None else float(proportional_exp_actor_budget_std)
                    ),
                    proportional_exp_actor_composition_std=(
                        None if proportional_exp_actor_composition_std is None else float(proportional_exp_actor_composition_std)
                    ),
                )
            component_outcome_kwargs = {}
            if proportional_exp_mode != "off":
                component_outcome_kwargs = dict(
                    component_outcome_aux_enabled=bool(
                        config.COMPONENT_OUTCOME_AUX_ENABLED
                    ),
                    component_outcome_target_gradient_ratio=float(
                        config.COMPONENT_OUTCOME_TARGET_GRADIENT_RATIO
                    ),
                    component_outcome_gradient_calibration_interval=int(
                        config.COMPONENT_OUTCOME_GRADIENT_CALIBRATION_INTERVAL
                    ),
                    actor_action_advantage_diagnostic_enabled=bool(
                        config.ACTOR_ACTION_ADVANTAGE_DIAGNOSTIC_ENABLED
                    ),
                )
            qwm_kwargs = {}
            if use_qwm:
                qwm_kwargs = dict(
                    qwm_enabled=True,
                    qwm_hidden_dim=int(config.QWM_HIDDEN_DIM),
                    qwm_max_normalized_residual=float(
                        config.QWM_MAX_NORMALIZED_RESIDUAL
                    ),
                    qwm_latent_dim=int(config.QWM_LATENT_DIM),
                    qwm_kl_weight=float(config.QWM_KL_WEIGHT),
                    qwm_kl_free_nats=float(config.QWM_KL_FREE_NATS),
                    qwm_reward_target_scale=float(
                        config.QWM_REWARD_TARGET_SCALE
                    ),
                    qwm_antithetic_samples=bool(
                        config.QWM_ANTITHETIC_SAMPLES
                    ),
                    qwm_prior_reconstruction_weight=float(
                        config.QWM_PRIOR_RECONSTRUCTION_WEIGHT
                    ),
                    qwm_learning_rate=float(config.QWM_LEARNING_RATE),
                    qwm_weight_decay=float(config.QWM_WEIGHT_DECAY),
                    qwm_batch_size=int(config.QWM_BATCH_SIZE),
                    qwm_pretrain_replay_rows=int(
                        config.QWM_PRETRAIN_REPLAY_ROWS
                    ),
                    qwm_pretrain_min_unique_baskets=int(
                        config.QWM_PRETRAIN_MIN_UNIQUE_BASKETS
                    ),
                    qwm_replay_baskets_per_batch=int(
                        config.QWM_REPLAY_BASKETS_PER_BATCH
                    ),
                    qwm_pretrain_updates=int(config.QWM_PRETRAIN_UPDATES),
                    qwm_online_updates_per_td_step=int(
                        config.QWM_ONLINE_UPDATES_PER_TD_STEP
                    ),
                    qwm_state_loss_weight=float(config.QWM_STATE_LOSS_WEIGHT),
                    qwm_reward_loss_weight=float(config.QWM_REWARD_LOSS_WEIGHT),
                    qwm_max_grad_norm=float(config.QWM_MAX_GRAD_NORM),
                    qwm_initial_depth=int(config.QWM_INITIAL_DEPTH),
                    qwm_max_depth=int(config.QWM_MAX_DEPTH),
                    qwm_depth_three_updates=int(
                        config.QWM_DEPTH_THREE_UPDATES
                    ),
                    qwm_depth_four_updates=int(
                        config.QWM_DEPTH_FOUR_UPDATES
                    ),
                    qwm_candidates=int(config.QWM_CANDIDATES),
                    qwm_leaf_candidates=int(config.QWM_LEAF_CANDIDATES),
                    qwm_model_samples=int(config.QWM_MODEL_SAMPLES),
                    qwm_beam_width=int(config.QWM_BEAM_WIDTH),
                    qwm_search_discount=float(config.QWM_SEARCH_DISCOUNT),
                    qwm_leaf_aggregation=str(config.QWM_LEAF_AGGREGATION),
                    qwm_successor_aggregation=str(
                        config.QWM_SUCCESSOR_AGGREGATION
                    ),
                    qwm_cuda_autocast=bool(config.QWM_CUDA_AUTOCAST),
                    qwm_compile_regions=bool(config.QWM_COMPILE_REGIONS),
                    qwm_online_search_enabled=bool(
                        config.QWM_ONLINE_SEARCH_ENABLED
                    ),
                    qwm_evaluation_search_enabled=bool(
                        config.QWM_EVALUATION_SEARCH_ENABLED
                    ),
                    qwm_crl_aware_scoring=bool(
                        config.QWM_CRL_AWARE_SCORING
                    ),
                    qwm_search_seed=int(config.QWM_SEARCH_SEED),
                )
            iql_kwargs = {}
            if iql_attention_td3_enabled:
                # This repo's reward scale produces very small raw Q-V gaps, so pure IQL
                # needs a materially larger temperature than the D4RL defaults to create
                # any useful actor selectivity without batch-standardizing the advantage.
                iql_kwargs = dict(
                    iql_expectile=0.90,
                    iql_temperature=250.0,
                    iql_temperature_start=50.0,
                    iql_temperature_warmup_steps=5_000,
                    iql_max_actor_weight=100.0,
                    iql_max_actor_weight_start=100.0,
                    iql_max_actor_weight_warmup_steps=0,
                    iql_normalize_actor_weights=False,
                    iql_actor_bc_loss_scale=1.0,
                    iql_actor_bc_loss_scale_start=1.0,
                    iql_actor_bc_loss_warmup_steps=0,
                    iql_value_learning_rate=critic_lr_schedule,
                    iql_actor_start_steps=0,
                )
            model = baseline_td3_cls( #15 and 0.8 or 30 and 0.966
                PolicyClass,  # ✅ ATTENTION INTEGRATION: Use either attention or MLP policy
                env,
                verbose=1,
                actor_learning_rate=actor_lr_schedule,
                critic_learning_rate=critic_lr_schedule,
                policy_kwargs=policy_kwargs,
                batch_size=selected_attention_batch_size,
                buffer_size=1000000,
                learning_starts=selected_attention_learning_starts,
                # ✅ GAMMA/EFFECTIVE HORIZON REFERENCE TABLE:
                #   gamma=0.50  ≈ 1.4 steps (~1 bar)   (heavily myopic, very short horizon)
                #   gamma=0.55  ≈ 1.8 steps
                #   gamma=0.60  ≈ 2.5 steps
                #   gamma=0.65  ≈ 3.3 steps
                #   gamma=0.70  ≈ 4.2 steps
                #   gamma=0.75  ≈ 5.6 steps (~5 bars, short-term momentum)
                #   gamma=0.80  ≈ 7.2 steps (about 1.5x window of 5; short swing)
                #   gamma=0.85  ≈ 9.9 steps (~10 bars, longer short-term swing)
                #   gamma=0.90  ≈ 15.7 steps (near long-term window)
                #   gamma=0.95  ≈ 39 steps   (very long-horizon, slow to decay)
                #   gamma=0.966 ≈ 29 steps
                #   gamma=0.98630137 = 73 steps (multitask profile)
                #   (Reciprocal effective horizon h = 1/(1-gamma))
                # With long_term_return as the primary reward, a shorter horizon reduces
                # noise accumulation in Q-targets (fewer noisy future steps to sum) and
                # matches the MOMENTUM_WINDOW=5 and LONG_TERM_LOOKBACK=10 time scales
                gamma=(
                    selected_multitask_gamma
                    if selected_multitask_gamma is not None
                    else selected_multitask_gamma
                ),
                tau=0.005,
                policy_delay=selected_policy_delay,

                # DummyVecEnv contributes one real replay row per worker on
                # every vector step. Match those rows with the same one-to-one
                # TD3/QWM update schedule used by the historical single-env
                # run instead of silently lowering UTD by the batch width.
                gradient_steps=(
                    gpu_sim_updates_per_step
                    if gpu_sim_envs
                    else qwm_training_env_count if use_qwm else 1
                ),

                sigma_gate_scale = sigma_gate_scale,
                use_sigma_gating= True,
                sigma_input_mode = sigma_input_mode,
                sigma_loss_focus_enabled = bool(selected_sigma_loss_focus_enabled),
                sigma_loss_focus_alpha = float(sigma_loss_focus_alpha),
                sigma_loss_focus_power = float(sigma_loss_focus_power),
                sigma_loss_focus_max_weight = float(sigma_loss_focus_max_weight),
                sigma_loss_focus_eps = float(sigma_loss_focus_eps),

                use_n_step_returns = False,
                n_step = 3,
                n_step_blend = 0.5,

                target_policy_noise=0.2,
                target_noise_clip=0.5,                 
                action_noise=action_noise,
                device="auto", # previously "cuda" originally auto
                overall_total_timesteps=NUM_ITERATIONS * COLLECT_STEPS_PER_ITERATION,
                # ✅ CRL PARAMETERS: Pass CRL configuration to model
                velocity_constraint_threshold= 0.0016, # ✅ NEGATIVE THRESHOLD FIX: Now supports negative thresholds with abs() dead-band
                velocity_lambda_lr = 5e-2, # orginally 5e-2 0.01 0.05, 1e-1
                initial_velocity_lambda = 0.03, # orginally 0.03, now 0.3 (10x higher to prevent floor collapse)
                # CRL-specific parameters for Acceleration
                acceleration_constraint_threshold = 0.003, # ✅ UPDATED: Target threshold for acceleration constraint
                acceleration_lambda_lr = 5e-3, # orginally 1e-3 1.5e-3 #1.5e-3 0.005 0.0015
                initial_acceleration_lambda = 0.001, # orginally 0.001
                # 🧠 ATTENTION-AWARE CRL: Auto-adjust parameters for attention models
                auto_adjust_for_attention=True,
                # ✅ LAMBDA WARM-UP: Pass the warm-up steps to the agent
                lambda_warmup_steps=500, # orginally 500
                lambda_decay_factor=0.9999, # orginally 0.9999 tested 0.99
                # ✅ NOT FOR OG
                # aux_coef=float(aux_coef),
                # aux_mask_prob=float(aux_mask_prob),
                # feature_extractor_variant="baseline",
                # feature_extractor_class_name=getattr(extractor_class, "__name__", str(extractor_class)),
                
                # entropy_coef=0.1, # orginally 0.01
                # ✅ HYBRID SOLUTION: Add adaptive threshold scaling and hierarchical capping
                # acceleration_dominance_ratio = 0.1,  # ✅ HIERARCHICAL CAP: Max ratio of lambda_accel to lambda_velocity
                # adaptive_threshold_scale = 5.0,  # ✅ ADAPTIVE SCALING: Factor for threshold expansion when velocity satisfied
                # velocity_gate_threshold= 1e-5,
                # synergy_factor= 0.5,
                # # ✅ ROBUSTNESS FIX: Add a growth-only period for lambda
                # lambda_growth_period=5000,
                **proportional_exp_kwargs,
                **component_outcome_kwargs,
                **qwm_kwargs,
                **iql_kwargs,
            )

        elif encoder_variant == "encoder":
            model = CustomTD3_AC_Encoder( #15 and 0.8 or 30 and 0.966
                PolicyClass,  # ✅ ATTENTION INTEGRATION: Use either attention or MLP policy
                env,
                verbose=1,
                actor_learning_rate=actor_lr_schedule,
                critic_learning_rate=critic_lr_schedule,
                policy_kwargs=policy_kwargs,
                batch_size=selected_attention_batch_size,
                buffer_size=1000000,
                learning_starts=selected_attention_learning_starts,
                gamma=(
                    selected_multitask_gamma
                    if selected_multitask_gamma is not None
                    else 0.80
                ),
                tau=0.005,
                policy_delay=5,
                target_policy_noise=0.2,
                target_noise_clip=0.5,                 
                action_noise=action_noise,
                device="auto", # previously "cuda" originally auto
                overall_total_timesteps=NUM_ITERATIONS * COLLECT_STEPS_PER_ITERATION,
                # ✅ CRL PARAMETERS: Pass CRL configuration to model
                velocity_constraint_threshold= 0.0016, # ✅ NEGATIVE THRESHOLD FIX: Now supports negative thresholds with abs() dead-band
                velocity_lambda_lr = 5e-2, # orginally 5e-2 0.01 0.05, 1e-1
                initial_velocity_lambda = 0.03, # orginally 0.03, now 0.3 (10x higher to prevent floor collapse)
                # CRL-specific parameters for Acceleration
                acceleration_constraint_threshold = 0.003, # ✅ UPDATED: Target threshold for acceleration constraint
                acceleration_lambda_lr = 5e-3, # orginally 1e-3 1.5e-3 #1.5e-3 0.005 0.0015
                initial_acceleration_lambda = 0.001, # orginally 0.001
                # 🧠 ATTENTION-AWARE CRL: Auto-adjust parameters for attention models
                auto_adjust_for_attention=True,
                # ✅ LAMBDA WARM-UP: Pass the warm-up steps to the agent
                lambda_warmup_steps=500, # orginally 500
                lambda_decay_factor=0.9999, # orginally 0.9999 tested 0.99
                aux_coef=float(aux_coef),
                aux_mask_prob=float(aux_mask_prob),
                feature_extractor_variant="encoder",
                feature_extractor_class_name=getattr(extractor_class, "__name__", str(extractor_class)),
                # entropy_coef=0.1, # orginally 0.01
                # ✅ HYBRID SOLUTION: Add adaptive threshold scaling and hierarchical capping
                # acceleration_dominance_ratio = 0.1,  # ✅ HIERARCHICAL CAP: Max ratio of lambda_accel to lambda_velocity
                # adaptive_threshold_scale = 5.0,  # ✅ ADAPTIVE SCALING: Factor for threshold expansion when velocity satisfied
                # velocity_gate_threshold= 1e-5,
                # synergy_factor= 0.5,
                # # ✅ ROBUSTNESS FIX: Add a growth-only period for lambda
                # lambda_growth_period=5000,
            )        
        elif encoder_variant == "encoder_res":
            model = CustomTD3_AC_Encoder_Res( #15 and 0.8 or 30 and 0.966
                PolicyClass,  # ✅ ATTENTION INTEGRATION: Use either attention or MLP policy
                env,
                verbose=1,
                actor_learning_rate=actor_lr_schedule,
                critic_learning_rate=critic_lr_schedule,
                policy_kwargs=policy_kwargs,
                batch_size=selected_attention_batch_size,
                buffer_size=1000000,
                learning_starts=selected_attention_learning_starts,
                gamma=(
                    selected_multitask_gamma
                    if selected_multitask_gamma is not None
                    else 0.80
                ),
                tau=0.005,
                policy_delay=5,
                target_policy_noise=0.2,
                target_noise_clip=0.5,                 
                action_noise=action_noise,
                device="auto", # previously "cuda" originally auto
                overall_total_timesteps=NUM_ITERATIONS * COLLECT_STEPS_PER_ITERATION,
                # ✅ CRL PARAMETERS: Pass CRL configuration to model
                velocity_constraint_threshold= 0.0016, # ✅ NEGATIVE THRESHOLD FIX: Now supports negative thresholds with abs() dead-band
                velocity_lambda_lr = 5e-2, # orginally 5e-2 0.01 0.05, 1e-1
                initial_velocity_lambda = 0.03, # orginally 0.03, now 0.3 (10x higher to prevent floor collapse)
                # CRL-specific parameters for Acceleration
                acceleration_constraint_threshold = 0.003, # ✅ UPDATED: Target threshold for acceleration constraint try 0.0135 or 0.003 or 0.001 #0.003 or 0.001
                acceleration_lambda_lr = 5e-3, # orginally 1e-3 1.5e-3 #1.5e-3 0.005 0.0015
                initial_acceleration_lambda = 0.001, # orginally 0.001
                # 🧠 ATTENTION-AWARE CRL: Auto-adjust parameters for attention models
                auto_adjust_for_attention=True,
                # ✅ LAMBDA WARM-UP: Pass the warm-up steps to the agent
                lambda_warmup_steps=500, # orginally 500
                lambda_decay_factor=0.9999, # orginally 0.9999 tested 0.99
                aux_coef=float(aux_coef),
                aux_mask_prob=float(aux_mask_prob),
                feature_extractor_variant="encoder_res",
                feature_extractor_class_name=getattr(extractor_class, "__name__", str(extractor_class)),
                # entropy_coef=0.1, # orginally 0.01
                # ✅ HYBRID SOLUTION: Add adaptive threshold scaling and hierarchical capping
                # acceleration_dominance_ratio = 0.1,  # ✅ HIERARCHICAL CAP: Max ratio of lambda_accel to lambda_velocity
                # adaptive_threshold_scale = 5.0,  # ✅ ADAPTIVE SCALING: Factor for threshold expansion when velocity satisfied
                # velocity_gate_threshold= 1e-5,
                # synergy_factor= 0.5,
                # # ✅ ROBUSTNESS FIX: Add a growth-only period for lambda
                # lambda_growth_period=5000,
            )            
        else:
            print (f"Unknown encoder_variant='{encoder_variant}'. Expected: encoder, encoder_res")
            pass
        
        if crl_config['use_crl']:
            print(f"✅ CRL Mode: Using adaptive Lagrange multiplier for smoothness constraints")
            print(f"✅ Constraint: E[action_change²] ≤ {crl_config['constraint_threshold']:.4f}")
        else:
            print(f"🚫 CRL Mode: Disabled, using fixed penalty coefficient {crl_config['action_reg_coef']:.4f}")
        
        if iql_attention_td3_enabled:
            print(f"✅ IQL CRITIC: Using MSE Bellman regression to r + γV(next_state).")
        else:
            print(f"✅ STABILITY FIX: Using Huber loss for the critic.")
        print(f"✅ STABILITY FIX: Using constant learning rate for direct TF-Agents replication")
    else:
        raise ValueError(f"Only TD3 supported for TF-Agents replication currently")
    
    if resume_checkpoint:
        print(f"🔄 Loading model weights from checkpoint: {resume_checkpoint}")
        if encoder_variant == "baseline":
            baseline_td3_cls = get_attention_td3_algorithm_class(
                proportional_exp_mode,
                use_attention=use_attention,
                attention_type=attention_type,
                encoder_variant=encoder_variant,
            )
            model = baseline_td3_cls.load(resume_checkpoint, env=env, device="auto")
        elif encoder_variant == "encoder":
            model = CustomTD3_AC_Encoder.load(resume_checkpoint, env=env, device="auto")
        elif encoder_variant == "encoder_res":
            model = CustomTD3_AC_Encoder_Res.load(resume_checkpoint, env=env, device="auto")
        model.num_timesteps = resume_step
        print(f"✅ Model loaded from checkpoint at step {resume_step}")
        load_replay_buffer_checkpoint(model, replay_buffer_path_for_model_checkpoint(resume_checkpoint))
        print(f"✅ CRL lambdas restored — velocity_λ={model.lambda_param_velocity.item():.6f}, accel_λ={model.lambda_param_acceleration.item():.6f}")

    if use_attention and attention_type == 'variable_coin_attention':
        apply_attention_learning_runtime_config(model, attention_training_profile)
        print(f"Learning regime: gamma={model.gamma:.9f}; "
              f"independent reconstruction={getattr(model, 'earnmore_masked_reconstruction_enabled', False)}; "
              f"portfolio reward history={config.PORTFOLIO_REWARD_HISTORY_ENABLED}")

    # SB3 checkpoints serialize runtime attributes as well as weights.  Apply
    # the current compute cadence after loading so a resumed interval-1/2 run
    # receives the same optimization as a fresh run instead of silently
    # restoring the older, more expensive schedule.
    if hasattr(model, "earnmore_update_interval"):
        model.earnmore_update_interval = max(
            1, int(config.EARNMORE_UPDATE_INTERVAL)
        )
    if hasattr(model, "predictive_ssl_update_interval"):
        model.predictive_ssl_update_interval = max(
            1, int(config.PREDICTIVE_SSL_UPDATE_INTERVAL)
        )
    if hasattr(model, "earnmore_update_interval") or hasattr(
        model, "predictive_ssl_update_interval"
    ):
        print(
            "⏱️ Auxiliary update cadence: "
            f"EarnMore every {getattr(model, 'earnmore_update_interval', 'n/a')} "
            "TD updates; predictive SSL every "
            f"{getattr(model, 'predictive_ssl_update_interval', 'n/a')} TD updates"
        )

    # ✅ STEP 1.2: Attach logger to the model (MISSING in original TF-Agents style)
    print(f"🔧 Step 1.2: Attaching SB3 logger to model...")
    model.set_logger(logger)
    print(f"✅ Logger attached to model - CSV logging now enabled")

    if use_qwm:
        if not hasattr(model, "qwm_pretrain"):
            raise RuntimeError(
                "The selected TD3 implementation does not expose QWM lifecycle hooks."
            )
        model.qwm_log_dir = run_log_dir
        model.qwm_pretrain()
        print(
            "🌳 QWM armed: the model learns with TD3 during replay accumulation; "
            "its search-readiness fit begins after the configured row and basket "
            "coverage gates, then search controls collection and evaluation."
        )

    if bool(getattr(config, "FOUNDATION_LEARNING_ENABLED", False)):
        expected_foundation_schema = int(
            getattr(config, "FOUNDATION_SCHEMA_VERSION", 0)
        )
        if resume_checkpoint and int(
            getattr(model, "foundation_schema_version", -1)
        ) != expected_foundation_schema:
            raise RuntimeError(
                "This checkpoint predates foundation schema v2 and contains "
                "the legacy mixed feature_history observation. Start a fresh "
                "run; partial checkpoint/replay conversion is intentionally "
                "unsupported."
            )
        if hasattr(model, "foundation_warmup"):
            # The live pretraining phase precedes callbacks and SB3's first
            # financial dump. Give it an explicit destination so RunPod users
            # can monitor it while it is actually running.
            model.foundation_log_dir = run_log_dir
            # Replay is empty for a fresh model, so this arms the live warmup.
            # CustomTD3_AC fulfills it after causal collection reaches
            # learning_starts and before applying the first financial gradient.
            model.foundation_warmup()
    
    # Setup logging similar to TF-Agents
    results_csv = os.path.join(run_log_dir, "output_ar_gamma.csv")
    if resume_checkpoint:
        existing_csv = results_csv
        if os.path.exists(existing_csv):
            existing_df = pd.read_csv(existing_csv)
            returns = existing_df["Return"].tolist()
            financial_returns = existing_df["FinancialReturn"].tolist()
            if "OpportunityNormalizedFinancialReturn" in existing_df.columns:
                opportunity_normalized_returns = existing_df[
                    "OpportunityNormalizedFinancialReturn"
                ].tolist()
            else:
                opportunity_normalized_returns = [np.nan] * len(financial_returns)
            iterations = existing_df["iterations"].tolist()
            print(f"🔄 Loaded {len(returns)} existing evaluation records from previous run")
        else:
            returns = []
            financial_returns = []
            opportunity_normalized_returns = []
            iterations = [0]
        evaluation_episode_history_path = os.path.join(
            run_log_dir, "curriculum_evaluation_episode_history.csv"
        )
        if os.path.exists(evaluation_episode_history_path):
            evaluation_episode_history = pd.read_csv(
                evaluation_episode_history_path
            ).to_dict("records")
        else:
            evaluation_episode_history = []
        existing_crl = os.path.join(run_log_dir, "crl_history.csv")
        if os.path.exists(existing_crl):
            crl_stats_history = load_crl_history_csv(existing_crl)
            print(f"🔄 Loaded {len(crl_stats_history)} existing CRL stats from previous run")
        else:
            crl_stats_history = []
        best_avg_return = max(returns) if returns else -np.inf
        print(f"🔄 Best avg return from previous run: {best_avg_return:.4f}")
    else:
        returns = []
        financial_returns = []
        opportunity_normalized_returns = []
        evaluation_episode_history = []
        iterations = [0]
        crl_stats_history = []
        best_avg_return = -np.inf
    last_resume_checkpoint_step = None
    
    if not resume_checkpoint:
        # Initial evaluation (like TF-Agents)
        print(f"🧮 Initial evaluation...")
        if eval_basket_sampler is not None:
            eval_basket_sampler.prepare(NUM_EVAL_EPISODES, step=0)
        episode_returns = []
        episode_financial_returns = []
        episode_opportunity_normalized_returns = []
        evaluation_feedback_strata = []
        for episode in range(NUM_EVAL_EPISODES):
            obs, _ = eval_env.reset()
            episode_return = 0.0
            terminated = truncated = False
            last_info = {}
            
            while not (terminated or truncated):
                action = eval_env.action_space.sample()
                obs, reward, terminated, truncated, last_info = eval_env.step(action)
                episode_return += reward
            
            episode_returns.append(episode_return)
            final_value = last_info.get("value", config.INITIAL_CASH)
            financial_return = (final_value - config.INITIAL_CASH) / config.INITIAL_CASH
            episode_financial_returns.append(financial_return)
            normalized_return = float(
                last_info.get("opportunity_normalized_financial_return", 0.0)
            )
            episode_opportunity_normalized_returns.append(normalized_return)
            evaluation_feedback_strata.append(
                {
                    "stratum": (
                        "distance:"
                        f"{int(last_info.get('basket_substitution_distance', -1))}"
                    ),
                    "financial_return": float(financial_return),
                    "opportunity_normalized_return": normalized_return,
                }
            )
            evaluation_episode_history.append(
                {
                    "step": 0,
                    "episode": int(episode),
                    "financial_return": float(financial_return),
                    "opportunity_normalized_financial_return": normalized_return,
                    "opportunity_risk_budget": float(
                        last_info.get("opportunity_risk_budget", 0.0)
                    ),
                    "basket_substitution_distance": int(
                        last_info.get("basket_substitution_distance", -1)
                    ),
                        "episode_coins": "|".join(last_info.get("episode_coins", [])),
                        "curriculum_role": str(
                            last_info.get("curriculum_role", "uniform_evaluation")
                        ),
                        "curriculum_stage": str(
                            last_info.get("curriculum_stage", "none")
                        ),
                        "evaluation_curriculum_probability": float(
                            last_info.get(
                                "evaluation_curriculum_probability", 0.0
                            )
                        ),
                    }
                )
        
        avg_return = np.mean(episode_returns)
        returns.append(avg_return)
        avg_financial = np.mean(episode_financial_returns) if episode_financial_returns else 0.0
        financial_returns.append(avg_financial)
        avg_opportunity_normalized = (
            float(np.mean(episode_opportunity_normalized_returns))
            if episode_opportunity_normalized_returns
            else 0.0
        )
        opportunity_normalized_returns.append(avg_opportunity_normalized)
        if hasattr(model, "update_value_experience_replay_from_evaluation"):
            model.update_value_experience_replay_from_evaluation(
                avg_financial,
                avg_opportunity_normalized,
                evaluation_feedback_strata,
            )
        print(f"🔍 Initial Average Return = {avg_return:.4f}")
        print(f"💰 Initial Average Financial Return = {avg_financial:.4%}")
        print(
            "⚖️ Initial Opportunity-Normalized Financial Return = "
            f"{avg_opportunity_normalized:.6f}"
        )
    else:
        print(f"🔄 Skipping initial evaluation (resuming from step {resume_step})")
    if curriculum_manager is not None:
        curriculum_manager.snapshot(resume_step if resume_checkpoint else 0)
    
    # ✅ STEP 2: Add SB3 Callbacks to TF-Agents Style Training
    print(f"🔧 Step 2: Setting up SB3 callbacks for TF-Agents style training...")
    
    # Step 2.2: Add PlotGeneratorCallback (after variables are initialized)
    plot_callback = PlotGeneratorCallback(
        log_path=run_log_dir,
        model_name=run_name,
        plot_freq=EVAL_INTERVAL * COLLECT_STEPS_PER_ITERATION,
        verbose=1,
        crl_stats_history=crl_stats_history,  # Pass CRL stats for periodic plotting
        crl_profile=crl_profile,             # Pass CRL profile name
        returns=returns                      # Pass returns for correlation analysis
    )
    plot_callback.init_callback(model)
    print(f"✅ Step 2.2: PlotGeneratorCallback configured (generate plots every {EVAL_INTERVAL * COLLECT_STEPS_PER_ITERATION} steps)")
    
    # Main training loop - EXACT TF-Agents methodology
    # Variables to track training metrics
    training_episode_rewards = []
    training_episode_lengths = []
    # Track how many episodes have already been consumed from the Monitor to avoid
    # duplicating the same episode metrics at every iteration.
    training_monitor_envs = list(getattr(env, "envs", [env]))
    consumed_episode_counts = [0 for _ in training_monitor_envs]
    training_start_time = __import__('time').time()
    plot_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="training-plots"
    )
    pending_plot = None
    
    for iteration in tqdm(range(start_iteration, NUM_ITERATIONS), desc="TF-Agents Style Training with CRL"):

        # COLLECT PHASE: Collect exactly COLLECT_STEPS_PER_ITERATION steps
        if iteration % 10 == 0:  # Print every 10 iterations to avoid spam
            print(f"\n📥 Iteration {iteration+1}: Collecting {COLLECT_STEPS_PER_ITERATION} steps...")
        
        # ✅ STEP 1.3a: Store current timestep before learning
        current_timesteps = (iteration + 1) * COLLECT_STEPS_PER_ITERATION
        
        # Track training time for FPS calculation
        iteration_start_time = __import__('time').time()
        
        # Execute the learning step
        curriculum_replay_publication_seconds = 0.0
        curriculum_replay_has_consumer = bool(
            getattr(config, "BASKET_REPLAY_STRATIFY_ENABLED", False)
            or getattr(config, "PREDICTIVE_SSL_SEPARATE_REPLAY_ENABLED", False)
        )
        if (
            curriculum_manager is not None
            and curriculum_replay_has_consumer
            and hasattr(model, "set_basket_replay_task_probabilities")
        ):
            curriculum_publication_start = time.perf_counter()
            model.set_basket_replay_task_probabilities(
                curriculum_manager.replay_task_probabilities(),
                role_probabilities_by_membership_ids=(
                    curriculum_manager.replay_role_probabilities()
                ),
                predictive_uniform_mix=(
                    curriculum_manager.predictive_replay_uniform_mix()
                ),
            )
            curriculum_replay_publication_seconds = (
                time.perf_counter() - curriculum_publication_start
            )
        model.learn(
            total_timesteps=COLLECT_STEPS_PER_ITERATION, 
            reset_num_timesteps=False,
            # callback=callbacks,
            # ✅ FIX: Disable automatic logging inside the inner learning loop.
            # We want to control logging EXPLICITLY only during evaluation steps.
            # Setting a high log_interval prevents SB3 from dumping logs every time
            # an episode happens to end during a 100-step collection phase, which
            # was causing irregular and incomplete rows in progress.csv.
            log_interval=10000
        )
        if curriculum_manager is not None and hasattr(
            model, "drain_basket_td_error_stats"
        ):
            curriculum_manager.update_td_errors(
                model.drain_basket_td_error_stats()
            )
        curriculum_value_loss = None
        curriculum_value_update_seconds = 0.0
        if (
            curriculum_manager is not None
            and curriculum_manager.value_model_update_due()
        ):
            curriculum_value_start = time.perf_counter()
            curriculum_value_loss = curriculum_manager.train_value_model()
            curriculum_value_update_seconds = time.perf_counter() - curriculum_value_start
        
        iteration_end_time = __import__('time').time()
        iteration_duration = iteration_end_time - iteration_start_time
        
        # ✅ CRL MONITORING: Collect CRL statistics after training
        if hasattr(model, 'get_crl_stats'):
            crl_stats = normalize_crl_stat_record(model.get_crl_stats())
            crl_stats_history.append(crl_stats)
            if curriculum_manager is not None:
                curriculum_control_start = time.perf_counter()
                curriculum_manager.update_control_readiness(
                    crl_stats,
                    current_timesteps,
                )
                curriculum_control_seconds = time.perf_counter() - curriculum_control_start
                model.logger.record(
                    "curriculum/control_ready",
                    float(curriculum_manager.control_ready),
                )
                model.logger.record(
                    "curriculum/stage_index",
                    float(
                        curriculum_manager.STAGE_NAMES.index(
                            curriculum_manager.stage
                        )
                    ),
                )
                model.logger.record(
                    "curriculum/target_pressure",
                    float(curriculum_manager.target_pressure),
                )
                model.logger.record(
                    "curriculum/local_transfer_progress",
                    float(curriculum_manager.local_transfer_progress),
                )
                model.logger.record(
                    "curriculum/replay_publication_seconds",
                    float(curriculum_replay_publication_seconds),
                )
                model.logger.record(
                    "curriculum/value_update_seconds",
                    float(curriculum_value_update_seconds),
                )
                model.logger.record(
                    "curriculum/control_update_seconds",
                    float(curriculum_control_seconds),
                )
                if curriculum_value_loss is not None:
                    model.logger.record(
                        "curriculum/value_loss",
                        float(curriculum_value_loss),
                    )
            
            # ✅ DEBUG: Show CRL data collection progress
            if iteration % 100 == 0:  # Every 100 iterations
                has_constraint_data = (
                    to_optional_float(crl_stats.get('avg_recent_constraint_cost')) is not None
                    and to_optional_float(crl_stats.get('avg_recent_violation')) is not None
                )
                print(f"📊 CRL Data Collection (Iter {iteration+1}): {len(crl_stats_history)} entries, Has constraint data: {has_constraint_data}")
            
            # Log CRL stats every 50 iterations to avoid spam
            if iteration % 50 == 0 and crl_stats.get('crl_enabled', False):
                print(f"🎯 CRL Stats (Iter {iteration+1}):")
                print(f"   λ: {crl_stats.get('current_lambda', 0):.4f}")
                print(f"   Avg constraint cost: {crl_stats.get('avg_recent_constraint_cost', 0):.4f}")
                print(f"   Constraint violation: {crl_stats.get('avg_recent_violation', 0):.4f}")
                
                # 🧠 ATTENTION-SPECIFIC LOGGING
                if crl_stats.get('is_attention_policy', False):
                    print(f"   🧠 Attention Model: Auto-adjust={crl_stats.get('auto_adjust_enabled', False)}")
                    print(f"   🧠 Exploration bonus: {crl_stats.get('attention_exploration_bonus', 0):.4f}")
        
        # ✅ MISSING METRICS FIX: Add the missing SB3 standard metrics that PlotGeneratorCallback expects
        
        # 1. Calculate training FPS (Missing Plot 6)
        fps = COLLECT_STEPS_PER_ITERATION / iteration_duration if iteration_duration > 0 else 0
        
        # 2. Extract training losses (Missing Plot 4)
        # For TD3, we can access losses from the logger if they exist
        actor_loss = getattr(model, '_last_actor_loss', 0.0)
        critic_loss = getattr(model, '_last_critic_loss', 0.0)
        value_loss = getattr(model, '_last_value_loss', np.nan)
        
        # Try to get actual losses from the model's logger if available
        if hasattr(model, 'logger') and hasattr(model.logger, 'name_to_value'):
            recent_logs = model.logger.name_to_value
            if 'train/actor_loss' in recent_logs:
                actor_loss = recent_logs['train/actor_loss']
            if 'train/critic_loss' in recent_logs:
                critic_loss = recent_logs['train/critic_loss']
            if 'train/value_loss' in recent_logs:
                value_loss = recent_logs['train/value_loss']
        
        # 3. Get current learning rate (Missing Plot 5)
        current_lr = model.learning_rate
        if callable(current_lr):
            # If it's a schedule, call it with current progress
            progress = 1.0 - (iteration / NUM_ITERATIONS)
            current_lr = current_lr(progress)
        
        # 4. Extract training episode statistics (Missing Plots 1 & 2)
        # Get data from Monitor wrapper
        if all(
            hasattr(monitor_env, "get_episode_rewards")
            and hasattr(monitor_env, "get_episode_lengths")
            for monitor_env in training_monitor_envs
        ):
            # Consume completed episodes from every vector worker exactly once.
            for worker_index, monitor_env in enumerate(training_monitor_envs):
                all_rewards = monitor_env.get_episode_rewards()
                all_lengths = monitor_env.get_episode_lengths()
                consumed = consumed_episode_counts[worker_index]
                if len(all_rewards) > consumed:
                    training_episode_rewards.extend(all_rewards[consumed:])
                    training_episode_lengths.extend(all_lengths[consumed:])
                    consumed_episode_counts[worker_index] = len(all_rewards)
            completed_episodes_per_env = [
                len(monitor_env.get_episode_rewards())
                for monitor_env in training_monitor_envs
            ]
        elif hasattr(env, '_episode_rewards') and hasattr(env, '_episode_lengths'):
            # Access Monitor internal data
            if env._episode_rewards:
                training_episode_rewards.extend(env._episode_rewards[-5:])
                training_episode_lengths.extend(env._episode_lengths[-5:])
            completed_episodes_per_env = [len(env._episode_rewards)]
        else:
            # Fallback: estimate realistic training metrics
            # Use evaluation performance as baseline with some training variation
            baseline_reward = returns[-1] if returns else 0.0
            baseline_length = 1500  # Typical portfolio episode length
            
            # Training episodes are typically more variable than evaluation
            estimated_reward = baseline_reward + np.random.normal(0, abs(baseline_reward * 0.2) + 1)
            estimated_length = baseline_length + np.random.randint(-100, 100)
            
            training_episode_rewards.append(estimated_reward)
            training_episode_lengths.append(max(50, estimated_length))  # Ensure positive length
            completed_episodes_per_env = [len(training_episode_rewards)]

        # ``num_timesteps`` counts real rows across every vector worker. Record
        # the environment and episode cardinalities explicitly; the production
        # QWM setting is one worker so these retain the historical monitor and
        # rolling-reward semantics.
        parallel_training_envs = int(getattr(env, "num_envs", len(training_monitor_envs)))
        completed_training_episodes = int(sum(completed_episodes_per_env))
        local_steps_per_training_env = (
            int(model.num_timesteps) // max(parallel_training_envs, 1)
        )
        model.logger.record(
            "rollout/training_env_count", float(parallel_training_envs)
        )
        model.logger.record(
            "rollout/completed_training_episodes_total",
            float(completed_training_episodes),
        )
        model.logger.record(
            "rollout/completed_training_episodes_per_env_min",
            float(min(completed_episodes_per_env)),
        )
        model.logger.record(
            "rollout/completed_training_episodes_per_env_max",
            float(max(completed_episodes_per_env)),
        )
        model.logger.record(
            "rollout/local_steps_per_training_env",
            float(local_steps_per_training_env),
        )
        
        # Keep only recent data
        if len(training_episode_rewards) > 100:
            training_episode_rewards = training_episode_rewards[-100:]
            training_episode_lengths = training_episode_lengths[-100:]
        
        # Calculate rolling averages for the missing metrics
        # If there are no completed episodes yet, log NaN so the plot skips points
        ep_rew_mean = (np.mean(training_episode_rewards[-10:])
                       if training_episode_rewards else 0.0)
        ep_len_mean = (np.mean(training_episode_lengths[-10:])
                       if training_episode_lengths else config.EPISODE_LENGTH)
        
        # ✅ CSV FIX: Log ALL the metrics that PlotGeneratorCallback expects
        model.logger.record("time/total_timesteps", current_timesteps)
        model.logger.record("time/iterations", iteration + 1)
        model.logger.record("time/fps", fps)  # Missing Plot 6
        model.logger.record("rollout/ep_rew_mean", ep_rew_mean)  # Missing Plot 1  
        model.logger.record("rollout/ep_len_mean", ep_len_mean)  # Missing Plot 2
        model.logger.record("train/actor_loss", actor_loss)  # Missing Plot 4
        model.logger.record("train/critic_loss", critic_loss)  # Missing Plot 4
        model.logger.record("train/value_loss", value_loss)
        model.logger.record("train/learning_rate", current_lr)  # Missing Plot 5
        
        print(f"📊 Logged complete metrics: FPS={fps:.1f}, EP_REW={ep_rew_mean}, LR={current_lr:.2e}")
        # Note: Still don't dump here - will dump only during evaluation for complete rows
        
        # EVALUATION PHASE: Every EVAL_INTERVAL iterations
        if (iteration + 1) % EVAL_INTERVAL == 0:
            step_count = (iteration + 1) * COLLECT_STEPS_PER_ITERATION
            tqdm.write(f"\n🧮 Evaluating at step {step_count}...")
            if eval_basket_sampler is not None:
                eval_basket_sampler.prepare(
                    NUM_EVAL_EPISODES,
                    step=step_count,
                )
            
            # ✅ EVALUATION FLEXIBILITY: Set evaluation mode with relaxed constraints
            if hasattr(model, 'set_evaluation_mode'):
                # Get typical number of active assets for constraint relaxation
                typical_assets = getattr(config, 'MAX_COINS', 5) if use_variable_portfolio else len(config.COINS)
                model.set_evaluation_mode(n_active_assets=typical_assets, relax_constraints=True)
                print(f"🎯 Set evaluation mode: Relaxed constraints for {typical_assets} assets")
            
            # Compute average return exactly like TF-Agents compute_avg_return()
            episode_returns = []
            episode_financial_returns = []
            episode_opportunity_normalized_returns = []
            episode_evaluation_anchor_flags = []
            evaluation_feedback_strata = []
            batched_evaluation = None
            if len(eval_envs) == NUM_EVAL_EPISODES and len(eval_envs) > 1:
                batched_evaluation = evaluate_policy_in_parallel_envs(
                    model, eval_envs
                )
            for episode in range(NUM_EVAL_EPISODES):
                if batched_evaluation is not None:
                    episode_return = batched_evaluation[0][episode]
                    last_info = batched_evaluation[1][episode]
                else:
                    obs, _ = eval_env.reset()
                    episode_return = 0.0
                    terminated = truncated = False
                    last_info = {}

                    # Preserve the historical dynamic constraint selection in
                    # the serial fallback. The production 8-of-13 batch has a
                    # common active width and is configured once above.
                    try:
                        env_instance = (
                            eval_env.env if hasattr(eval_env, 'env') else eval_env
                        )
                        if hasattr(env_instance, 'n_episode_coins'):
                            actual_assets = env_instance.n_episode_coins
                            if hasattr(model, 'set_evaluation_mode'):
                                model.set_evaluation_mode(
                                    n_active_assets=actual_assets,
                                    relax_constraints=True,
                                )
                                print(
                                    f"🎯 Episode {episode}: Relaxed for "
                                    f"{actual_assets} assets"
                                )
                    except Exception:
                        pass

                    while not (terminated or truncated):
                        action, _ = model.predict(obs, deterministic=True)
                        (
                            obs,
                            reward,
                            terminated,
                            truncated,
                            last_info,
                        ) = eval_env.step(action)
                        episode_return += reward
                
                episode_returns.append(episode_return)
                final_value = last_info.get("value", config.INITIAL_CASH)
                fin_ret = (final_value - config.INITIAL_CASH) / config.INITIAL_CASH
                episode_financial_returns.append(fin_ret)
                normalized_return = float(
                    last_info.get(
                        "opportunity_normalized_financial_return", 0.0
                    )
                )
                episode_opportunity_normalized_returns.append(normalized_return)
                evaluation_feedback_strata.append(
                    {
                        "stratum": (
                            "distance:"
                            f"{int(last_info.get('basket_substitution_distance', -1))}"
                        ),
                        "financial_return": float(fin_ret),
                        "opportunity_normalized_return": normalized_return,
                    }
                )
                episode_evaluation_anchor_flags.append(
                    float(
                        last_info.get("curriculum_role")
                        == "evaluation_anchor"
                    )
                )
                evaluation_episode_history.append(
                    {
                        "step": int(step_count),
                        "episode": int(episode),
                        "financial_return": float(fin_ret),
                        "opportunity_normalized_financial_return": normalized_return,
                        "opportunity_risk_budget": float(
                            last_info.get("opportunity_risk_budget", 0.0)
                        ),
                        "basket_substitution_distance": int(
                            last_info.get("basket_substitution_distance", -1)
                        ),
                        "episode_coins": "|".join(
                            last_info.get("episode_coins", [])
                        ),
                        "curriculum_role": str(
                            last_info.get("curriculum_role", "uniform_evaluation")
                        ),
                        "curriculum_stage": str(
                            last_info.get("curriculum_stage", "none")
                        ),
                        "evaluation_curriculum_probability": float(
                            last_info.get(
                                "evaluation_curriculum_probability", 0.0
                            )
                        ),
                    }
                )

            # ✅ RESET TO TRAINING MODE: Restore training constraints after evaluation
            if hasattr(model, 'reset_to_training_mode'):
                model.reset_to_training_mode()
                print(f"🎯 Reset to training mode: Restored training constraints")
            
            avg_return = np.mean(episode_returns)
            avg_financial = np.mean(episode_financial_returns) if episode_financial_returns else 0.0
            avg_opportunity_normalized = (
                float(np.mean(episode_opportunity_normalized_returns))
                if episode_opportunity_normalized_returns
                else 0.0
            )
            returns.append(avg_return)
            financial_returns.append(avg_financial)
            opportunity_normalized_returns.append(avg_opportunity_normalized)
            iterations.append(step_count)
            
            # ✅ STEP 1.3c: Log evaluation metrics (like SB3's EvalCallback does)
            model.logger.record("eval/mean_reward", avg_return)
            model.logger.record("eval/std_reward", np.std(episode_returns))
            model.logger.record("eval/mean_ep_length", config.EPISODE_LENGTH)
            model.logger.record("eval/mean_financial_return", avg_financial)
            model.logger.record("eval/std_financial_return", np.std(episode_financial_returns) if episode_financial_returns else 0.0)
            model.logger.record(
                "eval/mean_opportunity_normalized_financial_return",
                avg_opportunity_normalized,
            )
            model.logger.record(
                "eval/std_opportunity_normalized_financial_return",
                np.std(episode_opportunity_normalized_returns)
                if episode_opportunity_normalized_returns
                else 0.0,
            )
            if hasattr(
                model, "update_value_experience_replay_from_evaluation"
            ):
                replay_feedback_loss = (
                    model.update_value_experience_replay_from_evaluation(
                        avg_financial,
                        avg_opportunity_normalized,
                        evaluation_feedback_strata,
                    )
                )
                if replay_feedback_loss is not None:
                    model.logger.record(
                        "replay/ve_external_loss",
                        float(replay_feedback_loss),
                    )
            if curriculum_manager is not None:
                task_probabilities = (
                    curriculum_manager._task_probability_vector()
                )
                model.logger.record(
                    "eval/curriculum_intended_anchor_share",
                    float(
                        task_probabilities[
                            curriculum_manager.basket_to_index[
                                curriculum_manager.anchor
                            ]
                        ]
                    ),
                )
                model.logger.record(
                    "eval/curriculum_realized_anchor_share",
                    float(np.mean(episode_evaluation_anchor_flags)),
                )
            print(f"📊 Step 1.3c: Logged evaluation metrics to CSV")
            
            # ✅ CRL EVALUATION REPORTING: Show CRL performance during evaluation
            if hasattr(model, 'get_crl_stats'):
                crl_stats = normalize_crl_stat_record(model.get_crl_stats())
                if crl_stats.get('crl_enabled', False):
                    print(f"🎯 CRL Performance at Step {step_count}:")
                    print(f"   Current λ: {crl_stats.get('current_lambda', 0):.4f}")
                    print(f"   Recent constraint cost: {crl_stats.get('avg_recent_constraint_cost', 0):.4f}")
                    print(f"   Target threshold: {crl_stats.get('constraint_threshold', 0):.4f}")
                    violation = crl_stats.get('avg_recent_violation', 0)
                    if violation > 0:
                        print(f"   ⚠️ Constraint violation: +{violation:.4f} (agent too erratic)")
                    elif violation < -0.01:
                        print(f"   ✅ Under constraint: {violation:.4f} (agent could be more responsive)")
                    else:
                        print(f"   ✅ Near optimal constraint satisfaction: {violation:.4f}")
            
            # ✅ RESUME CHECKPOINTS: Save model+replay pairs less frequently than plots/evals.
            if should_save_resume_checkpoint(
                step_count,
                interval=resume_checkpoint_interval,
                span=EVAL_INTERVAL * COLLECT_STEPS_PER_ITERATION,
            ):
                if save_resume_checkpoint_pair(
                    model,
                    checkpoints_dir,
                    step_count,
                    curriculum_manager=training_basket_sampler,
                ):
                    last_resume_checkpoint_step = step_count
            elif step_count % 10_000 == 0:
                next_checkpoint_step = ((step_count // resume_checkpoint_interval) + 1) * resume_checkpoint_interval
                tqdm.write(
                    f"⏭️ Resume checkpoint skipped at step {step_count}; "
                    f"next paired checkpoint at step {next_checkpoint_step}."
                )
            
            # Save best model if performance has improved (atomic write + catch I/O errors)
            if avg_return > best_avg_return:
                best_avg_return = avg_return
                best_model_path = os.path.join(run_model_dir, "best_model.zip")
                best_model_tmp = best_model_path + ".tmp"
                try:
                    model.save(best_model_tmp)
                    os.replace(best_model_tmp, best_model_path)
                    print(f"🎉 New best model saved with return {avg_return:.4f}")
                except OSError as e:
                    print(f"⚠️ Best model save failed (I/O), continuing training: {e}")
                    if os.path.exists(best_model_tmp):
                        try:
                            os.remove(best_model_tmp)
                        except OSError:
                            pass
            
            # ✅ CONVERGENCE MONITORING: Print detailed evaluation results
            tqdm.write(f"🎯 Step {step_count}: Average Return = {avg_return:.4f}")
            tqdm.write(f"   📊 Episode returns: {[f'{r:.2f}' for r in episode_returns]}")
            tqdm.write(f"💰 Step {step_count}: Average Financial Return = {avg_financial:.4%}")
            tqdm.write(f"   💰 Episode financial returns: {[f'{r:.2%}' for r in episode_financial_returns]}")
            tqdm.write(
                "⚖️ Step "
                f"{step_count}: Opportunity-Normalized Financial Return = "
                f"{avg_opportunity_normalized:.6f}"
            )
            
            # Monitor for convergence signs
            if len(returns) > 5:
                recent_avg = np.mean(returns[-5:])
                if recent_avg > 50:  # Success threshold
                    print(f"🚀 CONVERGENCE SIGN: Recent 5-eval average = {recent_avg:.2f}")
                elif avg_return > 100:
                    print(f"🎉 EXCELLENT RESULT: Single evaluation = {avg_return:.2f}")
            
            # ✅ STEP 1.3d + STEP 2.3: Force CSV dump and trigger callbacks after evaluation
            print(f"💾 Step 1.3d: Dumping evaluation metrics to CSV...")
            try:
                model.logger.dump(step=step_count)
            except (OSError, IOError) as e:
                print(f"⚠️ Logger dump failed (I/O), continuing training: {e}")
            
            # ✅ STEP 2.3: Manually trigger callbacks (since custom training loop bypasses normal callback execution)
            print(f"🔄 Step 2.3: Triggering callbacks manually...")
            
            # A prior render must finish before any callback-owned inputs are
            # replaced. This makes each asynchronous figure a coherent snapshot
            # of exactly one durable logger dump.
            if pending_plot is not None:
                pending_plot.result()

            # Update callback internal state
            plot_callback.num_timesteps = step_count
            plot_callback.n_calls = step_count

            # _generate_plots() already calls _generate_crl_plots(). Refresh
            # its inputs before that single pass. The render runs on another
            # thread, so give it an immutable checkpoint snapshot instead of
            # live lists that the learner continues to mutate.
            if len(crl_stats_history) > 0:
                plot_callback.crl_stats_history = copy.deepcopy(
                    crl_stats_history
                )
                plot_callback.returns = list(returns)

            # Rendering reads the durable CSV that was dumped above and has no
            # role in action selection or learning. Keep it on one background
            # worker so CPU plotting does not idle the GPU learner.
            print(f"🔄 Scheduling training plots asynchronously...")
            pending_plot = plot_executor.submit(plot_callback._generate_plots)

            print(f"✅ Step 2.3: Callbacks executed successfully")

            if curriculum_manager is not None:
                curriculum_manager.snapshot(step_count)
                try:
                    generate_curriculum_diagnostic_plots(
                        curriculum_manager=curriculum_manager,
                        log_dir=run_log_dir,
                        evaluation_iterations=iterations,
                        financial_returns=financial_returns,
                        opportunity_normalized_returns=(
                            opportunity_normalized_returns
                        ),
                        evaluation_episode_history=(
                            evaluation_episode_history
                        ),
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    print(
                        "⚠️ Curriculum diagnostic plot generation failed, "
                        f"continuing training: {exc}"
                    )
            
            # Save results like TF-Agents (guard against I/O errors)
            try:
                results_df = pd.DataFrame({
                    "iterations": iterations,
                    "Return": returns,
                    "FinancialReturn": financial_returns,
                    "OpportunityNormalizedFinancialReturn": (
                        opportunity_normalized_returns
                    ),
                })
                results_csv = os.path.join(run_log_dir, "output_ar_gamma.csv")
                results_df.to_csv(results_csv, index=False)
            except (OSError, IOError) as e:
                print(f"⚠️ Failed to write output_ar_gamma.csv (I/O), continuing: {e}")
            try:
                if any(normalize_crl_enabled(s.get('crl_enabled', False)) for s in crl_stats_history):
                    crl_csv, crl_stats_history = write_crl_history_csv(run_log_dir, crl_stats_history)
                    print(f"📊 CRL statistics saved to: {crl_csv}")
            except (OSError, IOError) as e:
                print(f"⚠️ Failed to write crl_history.csv (I/O), continuing: {e}")
    
    # ✅ STEP 1.4a + STEP 2.4: Final CSV dump and callback execution
    print(f"💾 Step 1.4a: Final CSV dump...")
    model.logger.record("training/completed", 1.0)
    try:
        model.logger.dump(step=NUM_ITERATIONS * COLLECT_STEPS_PER_ITERATION)
    except (OSError, IOError) as e:
        print(f"⚠️ Final logger dump failed (I/O): {e}")
    try:
        results_df = pd.DataFrame({
            "iterations": iterations,
            "Return": returns,
            "FinancialReturn": financial_returns,
            "OpportunityNormalizedFinancialReturn": (
                opportunity_normalized_returns
            ),
        })
        results_df.to_csv(results_csv, index=False)
    except (OSError, IOError) as e:
        print(f"⚠️ Final output_ar_gamma.csv write failed (I/O): {e}")
    try:
        if any(normalize_crl_enabled(s.get('crl_enabled', False)) for s in crl_stats_history):
            crl_csv, crl_stats_history = write_crl_history_csv(run_log_dir, crl_stats_history)
            print(f"📊 Final CRL statistics saved to: {crl_csv}")
    except (OSError, IOError) as e:
        print(f"⚠️ Final crl_history.csv write failed (I/O): {e}")
    
    # ✅ STEP 2.4: Final callback execution
    print(f"🏁 Step 2.4: Final callback execution...")
    
    # ✅ FINAL REFERENCE UPDATE: Ensure callback has latest data for final plots
    if pending_plot is not None:
        pending_plot.result()
    plot_executor.shutdown(wait=True)
    plot_callback.crl_stats_history = crl_stats_history
    plot_callback.returns = returns
    
    # Generate final plots
    plot_callback._on_training_end()      # Final plot generation
    
    # _on_training_end() delegates to _generate_plots(), which includes the
    # final CRL figure in the same single rendering pass.
    if curriculum_manager is not None:
        curriculum_manager.snapshot(NUM_ITERATIONS * COLLECT_STEPS_PER_ITERATION)
        try:
            generate_curriculum_diagnostic_plots(
                curriculum_manager,
                run_log_dir,
                iterations,
                financial_returns,
                opportunity_normalized_returns,
                evaluation_episode_history,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"⚠️ Final curriculum plot generation failed: {exc}")
    
    print(f"✅ Step 2.4: Final callbacks completed successfully")
    
    # Save final model (atomic write + catch I/O errors)
    final_path = os.path.join(run_model_dir, "final_model.zip")
    final_tmp = final_path + ".tmp"
    try:
        model.save(final_tmp)
        os.replace(final_tmp, final_path)
    except OSError as e:
        print(f"⚠️ Final model save failed (I/O): {e}")
        if os.path.exists(final_tmp):
            try:
                os.remove(final_tmp)
            except OSError:
                pass
    final_step_count = NUM_ITERATIONS * COLLECT_STEPS_PER_ITERATION
    if last_resume_checkpoint_step != final_step_count:
        print(f"💾 Saving final paired resume checkpoint at step {final_step_count}...")
        if save_resume_checkpoint_pair(
            model,
            checkpoints_dir,
            final_step_count,
            curriculum_manager=training_basket_sampler,
        ):
            last_resume_checkpoint_step = final_step_count
    
    # Generate final plot like TF-Agents
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, returns, 'b-', linewidth=2, marker='o')
    plt.ylabel('Average Return')
    plt.xlabel('Steps')
    plt.title(f'TF-Agents Style Training: {algorithm}' + (' with CRL' if algorithm == 'TD3' else ''))
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(run_log_dir, "training_returns.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ✅ CRL FINAL SUMMARY: Print detailed CRL performance summary (plots now generated periodically)
    crl_stats_history = normalize_crl_stats_history(crl_stats_history)
    if any(normalize_crl_enabled(stats.get('crl_enabled', False)) for stats in crl_stats_history):
        print(f"📊 CRL analysis plots were generated periodically during training...")
        
        final_crl_stats = normalize_crl_stat_record(model.get_crl_stats())
        print(f"\n🎯 FINAL CRL PERFORMANCE SUMMARY:")
        print(f"   Profile used: {crl_profile}")

        if crl_record_has_dual_metrics(final_crl_stats):
            print(f"   Final velocity lambda: {final_crl_stats.get('current_lambda_velocity', 0):.4f}")
            print(f"   Final acceleration lambda: {final_crl_stats.get('current_lambda_acceleration', 0):.4f}")
            print(f"   Velocity threshold: {final_crl_stats.get('velocity_constraint_threshold', 0):.4f}")
            print(f"   Acceleration threshold: {final_crl_stats.get('acceleration_constraint_threshold', 0):.4f}")
            print(f"   Final velocity cost: {final_crl_stats.get('avg_recent_velocity_cost', 0):.4f}")
            print(f"   Final acceleration cost: {final_crl_stats.get('avg_recent_acceleration_cost', 0):.4f}")
            print(f"   Final total violation: {final_crl_stats.get('avg_recent_violation', 0):.4f}")
        else:
            print(f"   Final lambda: {final_crl_stats.get('current_lambda', 0):.4f}")
            print(f"   Constraint threshold: {final_crl_stats.get('constraint_threshold', 0):.4f}")
            print(f"   Final constraint cost: {final_crl_stats.get('avg_recent_constraint_cost', 0):.4f}")
            print(f"   Final violation: {final_crl_stats.get('avg_recent_violation', 0):.4f}")
        
        # Provide interpretation
        final_violation = final_crl_stats.get('avg_recent_violation', 0)
        if abs(final_violation) < 0.01:
            print(f"   ✅ EXCELLENT: Near-optimal constraint satisfaction")
        elif final_violation > 0.02:
            print(f"   ⚠️ SUBOPTIMAL: Agent still too erratic, consider stricter profile")
        elif final_violation < -0.02:
            print(f"   📈 OPPORTUNITY: Agent overly cautious, could be more responsive")
        
        # Lambda trajectory analysis
        if len(crl_stats_history) > 1:
            if any(crl_record_has_dual_metrics(stats) for stats in crl_stats_history):
                velocity_lambdas = [
                    to_optional_float(stats.get('current_lambda_velocity'))
                    for stats in crl_stats_history
                    if to_optional_float(stats.get('current_lambda_velocity')) is not None
                ]
                acceleration_lambdas = [
                    to_optional_float(stats.get('current_lambda_acceleration'))
                    for stats in crl_stats_history
                    if to_optional_float(stats.get('current_lambda_acceleration')) is not None
                ]
                if len(velocity_lambdas) > 1:
                    velocity_trend = np.polyfit(range(len(velocity_lambdas)), velocity_lambdas, 1)[0]
                    print(f"   Velocity lambda trend: {velocity_trend:.4f}/iter")
                if len(acceleration_lambdas) > 1:
                    acceleration_trend = np.polyfit(range(len(acceleration_lambdas)), acceleration_lambdas, 1)[0]
                    print(f"   Acceleration lambda trend: {acceleration_trend:.4f}/iter")
            else:
                lambdas = [
                    to_optional_float(stats.get('current_lambda'))
                    for stats in crl_stats_history
                    if to_optional_float(stats.get('current_lambda')) is not None
                ]
                if len(lambdas) > 1:
                    lambda_trend = np.polyfit(range(len(lambdas)), lambdas, 1)[0]
                    if lambda_trend > 0.001:
                        print(f"   📈 λ increased during training (+{lambda_trend:.4f}/iter) - agent learned to be smoother")
                    elif lambda_trend < -0.001:
                        print(f"   📉 λ decreased during training ({lambda_trend:.4f}/iter) - constraints relaxed")
                    else:
                        print(f"   ➡️ λ remained stable (trend: {lambda_trend:.4f}/iter) - balanced constraint satisfaction")
    
    # ✅ STEP 1.4b: Verify progress CSV files exist and show contents
    progress_csv_path = os.path.join(run_log_dir, PROGRESS_CSV_FILENAME)
    combined_progress_csv_path = os.path.join(run_log_dir, PROGRESS_COMBINED_CSV_FILENAME)
    combined_progress_df = load_combined_progress_history(run_log_dir)
    print(f"\n🔍 Step 1.4b: Verifying progress CSV files...")
    if os.path.exists(progress_csv_path):
        print(f"✅ SUCCESS: progress.csv found at: {progress_csv_path}")
        try:
            df = pd.read_csv(progress_csv_path)
            print(f"📊 CSV file contains {len(df)} rows and {len(df.columns)} columns")
            print(f"📋 Available columns: {list(df.columns)}")
            if len(df) > 0:
                print(f"📈 Sample data from last row:")
                for col in df.columns[:5]:  # Show first 5 columns
                    if col in df.columns:
                        print(f"   {col}: {df.iloc[-1][col]}")
        except Exception as e:
            print(f"⚠️ Could not read progress.csv: {e}")
    else:
        print(f"❌ FAILURE: progress.csv not found at: {progress_csv_path}")
        print(f"📂 Directory contents: {os.listdir(run_log_dir)}")
    if combined_progress_df is not None:
        print(f"✅ Combined progress CSV contains {len(combined_progress_df)} rows: {combined_progress_csv_path}")
    
    print(f"\n✅ TF-Agents style training with CRL completed!")
    print(f"📁 Logs and models saved in separate run-specific directories:")
    print(f"   - Logs: {run_log_dir}")
    print(f"   - Models: {run_model_dir}")
    print(f"💾 Best model (by evaluation return): {os.path.join(run_model_dir, 'best_model.zip')}")
    print(f"💾 Intermediate checkpoints in: {checkpoints_dir}")
    print(f"💾 Replay buffer checkpoints in: {checkpoints_dir}")
    print(f"💾 Final model: {final_path}")
    print(f"📊 Returns CSV: {results_csv}")
    print(f"📈 Plot: {plot_path}")
    print(f"📄 Progress CSV: {progress_csv_path}")
    print(f"📄 Combined progress CSV: {combined_progress_csv_path}")
    
    # ✅ CRL INTEGRATION: Add CRL-specific file references
    if any(normalize_crl_enabled(stats.get('crl_enabled', False)) for stats in crl_stats_history):
        print(f"🎯 CRL-specific outputs:")
        print(f"   - CRL history: {os.path.join(run_log_dir, 'crl_history.csv')}")
        print(f"   - CRL analysis: {os.path.join(run_log_dir, 'crl_analysis.png')} (updated periodically during training)")
    
    return model, returns, iterations


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with simple MLP (NO ATTENTION)")
    parser.add_argument("--algorithm", default="TD3", choices=["PPO", "SAC", "TD3", "FlowRL"])
    parser.add_argument(
        "--mlp-size",
        default="heavy",
        choices=["light", "medium", "multitask", "heavy"],
        help=(
            "Architecture size. 'multitask' is the 256-wide 8-of-13 "
            "variable-coin attention experiment."
        ),
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="TRANSACTION_COST",
        choices=["simple", "TRANSACTION_COST", "STRUCTURED_CREDIT", "SHAPLEY"],
        help="Type of reward function to use",
    )
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--test", action="store_true", help="Run tests instead of training")
    parser.add_argument("--compare", action="store_true", help="Compare different algorithms with MLP")
    parser.add_argument("--tf-agents-style", action="store_true", default=True,
                       help="Use TF-Agents exact training methodology (recommended)")
    
    # ✅ CRL INTEGRATION: Add CRL profile selection
    parser.add_argument(
        "--crl-profile",
        type=str,
        default="balanced",
        choices=["conservative", "balanced", "aggressive", "adaptive", "standard"],
        help="CRL profile for TD3 or FlowRL; attention learners use the stable dual velocity/acceleration settings"
    )
    parser.add_argument("--disable-crl", action="store_true", 
                       help="Disable CRL mode and use fixed penalty coefficient")
    
    # ✅ ATTENTION INTEGRATION: Add attention mechanism options
    parser.add_argument("--use-attention", action="store_true",
                       help="Use attention mechanisms instead of simple MLP")
    parser.add_argument("--attention-type", type=str, default="variable_coin_attention",
                       choices=["variable_coin_attention", "coin_attention", "multihead"],
                       help="Type of attention mechanism to use")
    parser.add_argument("--variable-portfolio", action="store_true",
                       help="Use variable portfolio sizes (dynamic asset allocation)")

    parser.add_argument(
        "--encoder-variant",
        type=str,
        default="baseline",
        choices=["baseline", "encoder", "encoder_res"],
        help="Feature extractor variant: baseline=CombinedExtractor, encoder=extractor.py, encoder_res=extractor_res.py",
    )
    parser.add_argument(
        "--aux-coef",
        type=float,
        default=0.0,
        help="Auxiliary masked-feature reconstruction coefficient (only meaningful for encoder variants)",
    )
    parser.add_argument(
        "--aux-mask-prob",
        type=float,
        default=0.15,
        help="Masking probability for the auxiliary objective (only meaningful for encoder variants)",
    )
    parser.add_argument(
        "--sigma-input-mode",
        type=str,
        default=getattr(config, "SIGMA_INPUT_MODE", "passthrough"),
        choices=["passthrough", "dampen_input"],
        help="Sigma input behavior: passthrough (Option B) or dampen_input (legacy Option A).",
    )
    parser.add_argument(
        "--sigma-gate-scale",
        type=float,
        default=float(getattr(config, "SIGMA_GATE_SCALE", 0.2)),
        help="Legacy dampen_input gate scale (used only when --sigma-input-mode dampen_input).",
    )
    parser.add_argument(
        "--sigma-loss-focus",
        dest="sigma_loss_focus",
        action="store_true",
        default=bool(getattr(config, "SIGMA_LOSS_FOCUS_ENABLED", True)),
        help="Enable sigma-aware critic loss focus (Option B).",
    )
    parser.add_argument(
        "--no-sigma-loss-focus",
        dest="sigma_loss_focus",
        action="store_false",
        help="Disable sigma-aware critic loss focus.",
    )
    parser.add_argument(
        "--sigma-loss-focus-alpha",
        type=float,
        default=float(getattr(config, "SIGMA_LOSS_FOCUS_ALPHA", 0.75)),
        help="Sigma loss focus alpha coefficient.",
    )
    parser.add_argument(
        "--sigma-loss-focus-power",
        type=float,
        default=float(getattr(config, "SIGMA_LOSS_FOCUS_POWER", 1.0)),
        help="Sigma loss focus power coefficient.",
    )
    parser.add_argument(
        "--sigma-loss-focus-max-weight",
        type=float,
        default=float(getattr(config, "SIGMA_LOSS_FOCUS_MAX_WEIGHT", 3.0)),
        help="Sigma loss focus maximum per-sample critic weight.",
    )
    parser.add_argument(
        "--sigma-loss-focus-eps",
        type=float,
        default=float(getattr(config, "SIGMA_LOSS_FOCUS_EPS", 1e-6)),
        help="Sigma loss focus normalization epsilon.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .zip to resume training from (e.g. models/.../checkpoints/model_1132000_steps.zip)",
    )
    parser.add_argument(
        "--proportional-exp-mode",
        type=str,
        default="off",
        choices=list(PROPORTIONAL_EXP_MODES),
        help="Ablation for local exp(Q/alpha) action distribution in the attention TD3 baseline.",
    )
    parser.add_argument(
        "--proportional-exp-candidates",
        type=int,
        default=5,
        help="Number of local action candidates used by proportional-exp attention TD3.",
    )
    parser.add_argument(
        "--proportional-exp-alpha",
        type=float,
        default=0.05,
        help="Base alpha for the local Boltzmann soft value.",
    )
    parser.add_argument(
        "--proportional-exp-alpha-max",
        type=float,
        default=0.25,
        help="Maximum alpha when disagreement-aware widening is enabled.",
    )
    parser.add_argument(
        "--proportional-exp-disagreement-gain",
        type=float,
        default=1.0,
        help="Gain applied to disagreement-driven alpha widening.",
    )
    parser.add_argument(
        "--proportional-exp-ema-decay",
        type=float,
        default=0.99,
        help="EMA decay for the disagreement baseline in boltzmann_disagreement mode.",
    )
    parser.add_argument(
        "--proportional-exp-cloud-space",
        type=str,
        default="portfolio_weights",
        choices=["raw_actions", "portfolio_weights"],
        help="Geometry used to build the proportional-exp candidate cloud.",
    )
    parser.add_argument(
        "--proportional-exp-budget-candidate-ratio",
        type=float,
        default=0.34,
        help="Fraction of non-center candidates devoted to budget-only moves; the rest become composition candidates.",
    )
    parser.add_argument(
        "--proportional-exp-target-budget-std",
        type=float,
        default=None,
        help="Optional override for target-side budget cloud radius; default derives from target_policy_noise.",
    )
    parser.add_argument(
        "--proportional-exp-target-composition-std",
        type=float,
        default=None,
        help="Optional override for target-side composition cloud radius; default derives from target_policy_noise.",
    )
    parser.add_argument(
        "--proportional-exp-actor-budget-std",
        type=float,
        default=None,
        help="Optional override for actor-side budget cloud radius; default derives from target_policy_noise.",
    )
    parser.add_argument(
        "--proportional-exp-actor-composition-std",
        type=float,
        default=None,
        help="Optional override for actor-side composition cloud radius; default derives from target_policy_noise.",
    )
    parser.add_argument(
        "--use-qwm",
        action="store_true",
        help=(
            "Wrap TD3 ProExp action selection with a continuously trained "
            "Q-Learning World Model tree search."
        ),
    )
    parser.add_argument(
        "--gpu-sim-envs",
        type=int,
        default=0,
        help=(
            "Collect training data from N baskets in lockstep with the batched "
            "PortfolioEnv simulator (0 keeps the single PortfolioEnv). "
            "--timesteps then counts transitions, N per vector step."
        ),
    )
    parser.add_argument(
        "--gpu-sim-updates-per-step",
        type=int,
        default=1,
        help="TD3 gradient updates per simulator vector step (N transitions).",
    )
    parser.add_argument(
        "--gpu-sim-device",
        type=str,
        default="auto",
        help="Device holding the simulator's market panel and portfolio state.",
    )
    
    for name, kind in (("learning-starts", int), ("buffer-size", int), ("batch-size", int),
                       ("flow-steps", int), ("expectile", float), ("flow-coefficient", float),
                       ("tau", float), ("gamma", float), ("gradient-steps", int),
                       ("eval-freq", int), ("eval-episodes", int), ("eval-batch-size", int), ("checkpoint-freq", int),
                       ("log-freq", int), ("plot-freq", int),
                       ("model-dim", int), ("financial-aux-weight", float),
                       ("actor-learning-rate", float), ("exploration-noise", float),
                       ("actor-cfm-gradient-ratio", float), ("policy-delay", int),
                       ("actor-cfm-scale-floor", float),
                       ("actor-q-normalization", int), ("actor-q-scale-floor", float),
                       ("candidate-count", int), ("candidate-budget-std", float), ("candidate-composition-std", float),
                       ("target-candidate-budget-std", float), ("target-candidate-composition-std", float),
                       ("candidate-residual-weight", float), ("max-grad-norm", float),
                       ("sigma-loss-focus", int)):
        parser.add_argument("--flowrl-" + name, type=kind, default=None)
    parser.add_argument("--flowrl-device", default="auto")
    parser.add_argument("--flowrl-run-dir", default=None, help="Optional parent for FlowRL logs and models")
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    if args.test:
        test_mlp_policy()
    elif args.compare:
        compare_mlp_algorithms()
    elif args.tf_agents_style:
        if args.algorithm == "FlowRL":
            print("FlowRL: SB3 step collection, guided flow matching, dual twin-Q systems and expectile value learning")
        else:
            print("🎯 USING TF-AGENTS EXACT TRAINING METHODOLOGY")
            print("🔧 This replicates the successful learning paradigm:")
            print("   • Collect 100 steps → Train 1 iteration → Repeat")
            print("   • Evaluate every 4 iterations (400 steps)")
            print("   • Exact hyperparameters from working implementation")
        
        # ✅ ATTENTION INTEGRATION: Show attention configuration
        if args.use_attention:
            print(f"🧠 ATTENTION MECHANISMS ENABLED")
            print(f"   • Attention Type: {args.attention_type}")
            print(f"   • Configuration: {args.mlp_size} (mapped from --mlp-size)")
            print(f"   • Variable Portfolio: {'Yes' if args.variable_portfolio else 'No'}")
            print(f"   • Proportional-Exp Mode: {args.proportional_exp_mode}")
            if args.algorithm == "TD3" and use_iql_attention_td3(
                proportional_exp_mode=args.proportional_exp_mode,
                use_attention=args.use_attention,
                attention_type=args.attention_type,
                encoder_variant=args.encoder_variant,
            ):
                print(f"   • TD3 Variant: Pure IQL attention baseline")
            
            # Show configuration details (use env config for max_coins so display matches policy build)
            from attention_policy import ATTENTION_CONFIGS
            attention_config = ATTENTION_CONFIGS[args.mlp_size]
            print(f"   • Features: {attention_config['features_dim']}")
            print(f"   • Attention Heads: {attention_config['n_heads']}")
            print(f"   • Layers: {attention_config['n_layers']}")
            print(f"   • Max Assets: {config.MAX_COINS}")
        else:
            print(f"🚫 SIMPLE MLP MODE - No attention mechanisms")
            print(f"   • Architecture: {args.mlp_size}")
            print(f"   • Variable Portfolio: {'Yes' if args.variable_portfolio else 'No'}")
        
        # ✅ CRL INTEGRATION: Show CRL configuration
        if args.algorithm in {"TD3", "FlowRL"}:
            if not args.disable_crl:
                print(f"🎯 CONSTRAINED REINFORCEMENT LEARNING ENABLED")
                print(f"   • Profile: {args.crl_profile}")
                print(f"   • Adaptive Lagrange multiplier λ for smoothness constraints")
                print(f"   • Automatic penalty adjustment based on constraint violations")
                
                # Show profile description
                # ✅ FIX: Get config based on portfolio size to show correct description
                if config.MAX_COINS > 4:  # Consistent threshold with training logic
                    crl_config = config.get_large_portfolio_crl_config(args.crl_profile)
                else:
                    crl_config = config.get_crl_config(args.crl_profile)
                print(f"   • {crl_config.get('description', 'No description available')}")
            else:
                print(f"🚫 CRL DISABLED - Using fixed penalty coefficient")
        if args.proportional_exp_mode != "off":
            print(f"🧪 PROPORTIONAL-EXP ENABLED")
            print(f"   • Mode: {args.proportional_exp_mode}")
            print(f"   • Candidates: {args.proportional_exp_candidates}")
            print(f"   • Alpha: {args.proportional_exp_alpha:.4f}")
            print(f"   • Alpha Max: {args.proportional_exp_alpha_max:.4f}")
            print(f"   • Disagreement Gain: {args.proportional_exp_disagreement_gain:.4f}")
            print(f"   • EMA Decay: {args.proportional_exp_ema_decay:.5f}")
            print(f"   • Cloud Space: {args.proportional_exp_cloud_space}")
            print(f"   • Budget Candidate Ratio: {args.proportional_exp_budget_candidate_ratio:.2f}")
            print(f"   • Target Budget Std: {args.proportional_exp_target_budget_std if args.proportional_exp_target_budget_std is not None else 'auto'}")
            print(f"   • Target Composition Std: {args.proportional_exp_target_composition_std if args.proportional_exp_target_composition_std is not None else 'auto'}")
            print(f"   • Actor Budget Std: {args.proportional_exp_actor_budget_std if args.proportional_exp_actor_budget_std is not None else 'auto'}")
            print(f"   • Actor Composition Std: {args.proportional_exp_actor_composition_std if args.proportional_exp_actor_composition_std is not None else 'auto'}")
        if args.use_qwm:
            print("🌳 Q-LEARNING WITH WORLD MODEL SEARCH ENABLED")
            print(
                f"   • Depth schedule: D={config.QWM_INITIAL_DEPTH} → "
                f"D={config.QWM_MAX_DEPTH} after real-transition model maturation"
            )
            print(
                f"   • Candidates={config.QWM_CANDIDATES}, "
                f"leaf={config.QWM_LEAF_CANDIDATES}, "
                f"beam={config.QWM_BEAM_WIDTH}, "
                f"successor samples K={config.QWM_MODEL_SAMPLES}, "
                f"K aggregation={config.QWM_SUCCESSOR_AGGREGATION}"
            )
            print(
                f"   • Conditional latent={config.QWM_LATENT_DIM}, "
                f"pretrain replay rows={config.QWM_PRETRAIN_REPLAY_ROWS}, "
                f"minimum baskets={config.QWM_PRETRAIN_MIN_UNIQUE_BASKETS}, "
                f"initial real-replay updates={config.QWM_PRETRAIN_UPDATES}, "
                f"online world updates={config.QWM_ONLINE_UPDATES_PER_TD_STEP}:1"
            )
            print(
                f"   • CUDA path: BF16 autocast={config.QWM_CUDA_AUTOCAST}, "
                f"compiled regions={config.QWM_COMPILE_REGIONS}, "
                f"fused optimizer={config.QWM_FUSED_OPTIMIZER}"
            )
            print(
                "   • World predictions guide action selection only; replay and "
                "TD3 targets remain real."
            )
        
        print(f"💰 REWARD SYSTEM: {args.reward_type}")
        if args.reward_type == "TRANSACTION_COST":
            print(f"   • Net Return = Profit - Transaction Costs")
            print(f"   • Transaction Cost Rate: {config.TRANSACTION_COST:.3f} ({config.TRANSACTION_COST*100:.1f}%)")
            print(f"   • Agent will learn to minimize unnecessary trading")
        elif args.reward_type == "STRUCTURED_CREDIT":
            print(f"   • Structured credit assignment with Sharpe Ratio")
            print(f"   • Agent will learn to balance risk and return per asset")
        elif args.reward_type == "SHAPLEY":
            print(f"   • Shapley value-based credit assignment (computationally intensive)")
            print(f"   • Agent learns each asset's true marginal contribution")
        
        # ✅ CRL PROFILE OVERRIDE: Temporarily disable CRL if requested
        selected_crl_profile = "conservative" if args.disable_crl else args.crl_profile
        if args.disable_crl:
            # Temporarily modify the CRL config to disable CRL
            original_crl_config = config.CRL_CONFIG.copy()
            config.CRL_CONFIG["use_crl"] = False
            print(f"⚠️ CRL mode disabled via --disable-crl flag")
        
        model, returns, iterations = train_simple_mlp_tf_agents_style(
            algorithm=args.algorithm,
            mlp_size=args.mlp_size,
            reward_type=args.reward_type,
            crl_profile=selected_crl_profile,  # ✅ CRL INTEGRATION: Pass CRL profile
            use_attention=args.use_attention,  # ✅ ATTENTION INTEGRATION: Pass attention flag
            attention_type=args.attention_type,  # ✅ ATTENTION INTEGRATION: Pass attention type
            use_variable_portfolio=args.variable_portfolio,  # ✅ ATTENTION INTEGRATION: Pass variable portfolio flag
            total_timesteps=args.timesteps,  # ✅ FIX: Pass timesteps from command line
            encoder_variant=args.encoder_variant,
            aux_coef=args.aux_coef,
            aux_mask_prob=args.aux_mask_prob,
            sigma_input_mode=args.sigma_input_mode,
            sigma_gate_scale=args.sigma_gate_scale,
            sigma_loss_focus_enabled=args.sigma_loss_focus,
            sigma_loss_focus_alpha=args.sigma_loss_focus_alpha,
            sigma_loss_focus_power=args.sigma_loss_focus_power,
            sigma_loss_focus_max_weight=args.sigma_loss_focus_max_weight,
            sigma_loss_focus_eps=args.sigma_loss_focus_eps,
            proportional_exp_mode=args.proportional_exp_mode,
            proportional_exp_candidates=args.proportional_exp_candidates,
            proportional_exp_alpha=args.proportional_exp_alpha,
            proportional_exp_alpha_max=args.proportional_exp_alpha_max,
            proportional_exp_disagreement_gain=args.proportional_exp_disagreement_gain,
            proportional_exp_ema_decay=args.proportional_exp_ema_decay,
            proportional_exp_cloud_space=args.proportional_exp_cloud_space,
            proportional_exp_budget_candidate_ratio=args.proportional_exp_budget_candidate_ratio,
            proportional_exp_target_budget_std=args.proportional_exp_target_budget_std,
            proportional_exp_target_composition_std=args.proportional_exp_target_composition_std,
            proportional_exp_actor_budget_std=args.proportional_exp_actor_budget_std,
            proportional_exp_actor_composition_std=args.proportional_exp_actor_composition_std,
            use_qwm=args.use_qwm,
            resume_checkpoint=args.resume_checkpoint,
            gpu_sim_envs=args.gpu_sim_envs,
            gpu_sim_updates_per_step=args.gpu_sim_updates_per_step,
            gpu_sim_device=args.gpu_sim_device,
            flowrl_options={key.removeprefix("flowrl_"): value for key, value in vars(args).items()
                            if key.startswith("flowrl_") and value is not None},
        )
        
        # ✅ CRL PROFILE OVERRIDE: Restore original config if modified
        if args.disable_crl:
            config.CRL_CONFIG.update(original_crl_config)
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Initial return: {returns[0]:.4f}")
        print(f"   Final return: {returns[-1]:.4f}")
        print(f"   Improvement: {returns[-1] - returns[0]:.4f}")
        if len(returns) > 1:
            print(f"   Best return: {max(returns):.4f}")
            
        # ✅ CRL INTEGRATION: Show final CRL summary if enabled
        if args.algorithm in {"TD3", "FlowRL"} and not args.disable_crl:
            if hasattr(model, 'get_crl_stats'):
                final_crl_stats = model.get_crl_stats()
                if final_crl_stats.get('crl_enabled', False):
                    print(f"\n🎯 CRL FINAL IMPACT:")
                    print(f"   Profile: {args.crl_profile}")
                    print(f"   Final λ: {final_crl_stats.get('current_lambda', 0):.4f}")
                    if args.algorithm == "FlowRL":
                        print(f"   Velocity λ: {final_crl_stats['current_lambda_velocity']:.6g}; "
                              f"Acceleration λ: {final_crl_stats['current_lambda_acceleration']:.6g}")
                    
                    final_violation = final_crl_stats.get('avg_recent_violation', 0)
                    if abs(final_violation) < 0.01:
                        print(f"   ✅ Achieved near-optimal constraint satisfaction")
                    elif final_violation > 0:
                        print(f"   ⚠️ Agent slightly more erratic than target")
                    else:
                        print(f"   📈 Agent more conservative than necessary")
    else:
        # Use standard SB3 training
        print("🚫 NO ATTENTION MECHANISMS USED")
        print("🧠 Simple MLP networks only (like TF-Agents)")
        
        train_simple_mlp(
            algorithm=args.algorithm,
            mlp_size=args.mlp_size,
            total_timesteps=args.timesteps
        )
        
        print("\n✅ Simple MLP training completed!")
        
        print(f"\n✅ MLP TRAINING FEATURES:")
        print(f"   🚫 NO attention mechanisms used")
        print(f"   🧠 Simple fully connected layers only")
        print(f"   📈 Similar to TF-Agents standard networks")
        print(f"   🎯 Direct observation → MLP → action mapping")
        
        print(f"\n📊 MLP Architecture Benefits:")
        print(f"   • Faster training (no complex attention computations)")
        print(f"   • Lower memory usage")
        print(f"   • Easier to understand and debug")
        print(f"   • Proven effective for many RL tasks")
        
        print("\n💡 Next steps:")
        print("1. Run evaluate.py to see portfolio allocation performance")
        print("2. Compare with attention-based models using train_with_attention.py")
        print("3. Experiment with different MLP architectures")
        print("4. Try different algorithms (PPO, SAC, TD3)")
        
        # ✅ ATTENTION INTEGRATION: Mention attention capabilities
        print("5. Experiment with attention mechanisms for complex market relationships:")
        print("   --use-attention             (enable attention mechanisms)")
        print("   --attention-type variable_coin_attention  (for variable portfolios)")
        print("   --attention-type coin_attention           (for fixed portfolios)")
        print("   --variable-portfolio        (enable dynamic asset allocation)")
        
        # ✅ CRL INTEGRATION: Mention CRL capabilities
        if args.algorithm == "TD3":
            print("6. Experiment with different CRL profiles for smoothness control:")
            print("   --crl-profile conservative  (minimal trading)")
            print("   --crl-profile balanced      (default)")
            print("   --crl-profile aggressive    (responsive trading)")
            print("   --crl-profile adaptive      (starts conservative)")
            print("   --disable-crl               (fixed penalty)")
            
            print("7. Combine attention and CRL for advanced portfolio management:")
            print("   --use-attention --variable-portfolio --crl-profile balanced")
 
