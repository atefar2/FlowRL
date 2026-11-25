# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# Adapted for the portfolio project; licensed under the Apache License, Version
# 2.0. See FlowRL_LICENSE.txt and http://www.apache.org/licenses/LICENSE-2.0.
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""SB3 port of FlowRL-main/model/algo.py (ByteDance, Apache-2.0).

The source's two Q systems, expectile V and guided midpoint flow are retained.
Version 3 adapts the optimizer to the portfolio task: the complete baseline
DeepSet critics, Huber TD, local cash/composition candidates, slow target actor,
and measured actor objective gradients. Collection uses stochastic flow alone
by default; post-flow OU remains an explicit historical-comparison option.
Optional adaptive dual CRL ports the stable attention TD3 velocity/acceleration
curriculum to joint executed weights; its dual state is saved with the learner.
Version 5 shares one cash/asset marginal rule. New training otherwise restores
the October 29 learning rules: paired final flow endpoints, the forward CRL
stencil, the batch curriculum, and a Q-only matching budget with its scale floor.
Network shape does not select the actor objective. Older combined-control and
independent-sampling checkpoints retain their saved behavior until the runner
explicitly configures the rollback.
Saved v5 executed-history objectives remain loadable and migrate explicitly.
Matching uses inverse-tanh endpoints to reconstruct executed bounded actions.
Dictionary replay preserves private causal unit-return labels and the original
80/20 historical/recent mixture; losses balance represented basket tasks.
Versions 1 through 4 remain loadable for inference with their original behavior.
"""
import math
from copy import deepcopy
from typing import NamedTuple
import numpy as np
import torch
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import polyak_update

import config
from FlowRL_model import ACTION_NORMALIZATION, action_mask, portfolio_weights, portfolio_action_candidates, weights_to_flow_actions
from FlowRL_policy import FlowRLPolicy


class FlowRLReplaySamples(NamedTuple):
    observations: dict
    actions: torch.Tensor
    next_observations: dict
    dones: torch.Tensor
    rewards: torch.Tensor
    component_outcomes: torch.Tensor
    component_outcome_mask: torch.Tensor


class FlowRLReplayBuffer(DictReplayBuffer):
    """80% uniform / 20% recent replay, sampling complete dictionary rows.

    Recent chronology is calculated relative to the circular write pointer.
    Sampling with replacement keeps small-buffer minibatches full sized.
    SB3 supplies terminal observations and timeout-aware bootstrap masks.
    """
    def __init__(self, *args, recent_fraction=0.2, recent_window=2048,
                 store_component_outcomes=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= recent_fraction <= 1 or recent_window < 1:
            raise ValueError("Invalid FlowRL replay mixture")
        self.recent_fraction = recent_fraction
        self.recent_window = int(recent_window)
        self.action_normalization = ACTION_NORMALIZATION
        self.market_input_contract = config.market_input_contract()
        self.store_component_outcomes = bool(store_component_outcomes)
        if self.store_component_outcomes:
            assets = self.observation_space["mask"].shape[0]
            shape = (self.buffer_size, self.n_envs, assets, 1)
            self.component_outcomes = np.zeros(shape, dtype=np.float32)
            self.component_outcome_masks = np.zeros(shape, dtype=bool)
            self.component_outcome_target_kind = "scaled_unit_asset_return"

    def add(self, obs, next_obs, action, reward, done, infos):
        position = self.pos
        if self.store_component_outcomes:
            self.component_outcomes[position] = 0
            self.component_outcome_masks[position] = False
            for worker, info in enumerate(infos):
                if "component_outcomes" not in info:
                    continue
                if info.get("component_outcome_target_kind") != self.component_outcome_target_kind:
                    raise ValueError("FlowRL financial teacher requires unit asset returns, not executed contributions")
                targets = np.asarray(info["component_outcomes"], dtype=np.float32)
                valid = np.asarray(info["component_outcome_mask"], dtype=bool)
                expected = self.component_outcomes.shape[2:]
                if targets.shape != expected or valid.shape != expected:
                    raise ValueError("FlowRL per-asset labels do not match global replay coordinates")
                valid = valid & np.asarray(obs["mask"][worker], dtype=bool)[:, None] & np.isfinite(targets)
                self.component_outcomes[position, worker] = np.where(valid, targets, 0)
                self.component_outcome_masks[position, worker] = valid
        super().add(obs, next_obs, action, reward, done, infos)

    def sample_indices(self, batch_size):
        size = self.size()
        if size == 0:
            raise ValueError("Cannot sample an empty replay buffer")
        uniform_count = int(batch_size * (1 - self.recent_fraction))
        uniform = np.random.randint(size, size=uniform_count)
        # Window is in transitions, including all vector-environment workers.
        window = min(size, max(1, math.ceil(self.recent_window / self.n_envs)))
        recent = (self.pos - 1 - np.random.randint(window, size=batch_size - uniform_count)) % self.buffer_size
        return np.concatenate((uniform, recent))

    def sample(self, batch_size, env=None):
        return self._get_samples(self.sample_indices(batch_size), env=env)

    def _get_samples(self, batch_inds, env=None):
        if not self.store_component_outcomes:
            return super()._get_samples(batch_inds, env=env)
        workers = np.random.randint(self.n_envs, size=len(batch_inds))
        obs = self._normalize_obs({key: value[batch_inds, workers] for key, value in self.observations.items()}, env)
        next_obs = self._normalize_obs({key: value[batch_inds, workers] for key, value in self.next_observations.items()}, env)
        dones = self.dones[batch_inds, workers] * (1 - self.timeouts[batch_inds, workers])
        return FlowRLReplaySamples(
            {key: self.to_torch(value) for key, value in obs.items()},
            self.to_torch(self.actions[batch_inds, workers]),
            {key: self.to_torch(value) for key, value in next_obs.items()},
            self.to_torch(dones.reshape(-1, 1)),
            self.to_torch(self._normalize_reward(self.rewards[batch_inds, workers].reshape(-1, 1), env)),
            self.to_torch(self.component_outcomes[batch_inds, workers]),
            self.to_torch(self.component_outcome_masks[batch_inds, workers]))


def guided_flow_weights(behavior_q, policy_q, coefficient):
    gap = (behavior_q - policy_q).detach().relu()
    # Algebraically identical to upstream lambda*exp(gap-mean), but stable.
    return (math.log(coefficient) + gap - gap.mean()).clamp(math.log(1e-3), 0).exp()


def basket_balanced_mean(values, observations, valid=None, task_groups=None):
    """Give each represented availability task equal optimizer authority."""
    rows = values.reshape(values.shape[0], -1).mean(-1)
    valid = torch.ones_like(rows) if valid is None else valid.to(rows.dtype)
    inverse = (torch.unique(observations["mask"], dim=0, return_inverse=True)[1]
               if task_groups is None else task_groups)
    totals = torch.zeros_like(rows).scatter_add(0, inverse, rows * valid)
    counts = torch.zeros_like(rows).scatter_add(0, inverse, valid)
    return (totals / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)


def financial_forecast_losses(predictions, data, balance_baskets=False, task_groups=None):
    """Absolute cash-vs-assets evidence plus within-basket return differences.

    Only the auxiliary error is normalized. Predictions and their direct
    contribution to Q keep the environment's scaled unit-return units.
    """
    targets = getattr(data, "component_outcomes", None)
    valid = getattr(data, "component_outcome_mask", None)
    zero = data.rewards.new_zeros(())
    if targets is None or valid is None or predictions[0] is None:
        return zero, zero, False
    valid = (valid.bool() & data.observations["mask"].bool().unsqueeze(-1)
             & torch.isfinite(targets))
    targets = torch.where(valid, targets.float(), torch.zeros_like(targets.float()))
    counts = valid.sum(dim=(1, 2))
    reference = float(config.COMPONENT_OUTCOME_LOSS_RETURN_REFERENCE)
    beta = float(config.COMPONENT_OUTCOME_HUBER_BETA)
    if reference <= 0 or beta <= 0:
        raise ValueError("FlowRL financial forecast scales must be positive")
    def masked_row_mean(errors, eligible):
        per_row = (errors * valid).sum(dim=(1, 2)) / counts.clamp_min(1)
        if balance_baskets:
            return basket_balanced_mean(per_row, data.observations, eligible, task_groups)
        return (per_row * eligible).sum() / eligible.sum().clamp_min(1)
    absolute, relative = zero, zero
    for prediction in predictions:
        residual = (prediction - targets) / reference
        absolute = absolute + masked_row_mean(
            F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=beta, reduction="none"),
            counts > 0)
        # Remove the common basket move, preserving only allocation-relevant
        # differences. Absolute supervision separately retains the cash signal.
        centered = residual - (residual * valid).sum(dim=1, keepdim=True) / counts[:, None, None].clamp_min(1)
        relative = relative + masked_row_mean(
            F.smooth_l1_loss(centered, torch.zeros_like(centered), beta=beta, reduction="none"),
            counts > 1)
    return absolute, relative, bool(valid.any())


class FlowRL(OffPolicyAlgorithm):
    policy_aliases = {"FlowRLPolicy": FlowRLPolicy, "MultiInputPolicy": FlowRLPolicy}

    def __init__(self, policy="FlowRLPolicy", env=None, learning_rate=5e-4,
                 buffer_size=1000000, learning_starts=1000, batch_size=200,
                 tau=0.005, gamma=0.966, train_freq=1, gradient_steps=1,
                 expectile=0.9, flow_coefficient=0.1, policy_delay=5,
                 replay_buffer_class=FlowRLReplayBuffer, replay_buffer_kwargs=None,
                 policy_kwargs=None, tensorboard_log=None, verbose=0, device="auto",
                 seed=None, financial_aux_weight=1.0, actor_learning_rate=1e-4,
                 exploration_noise=0.0, actor_cfm_gradient_ratio=0.5,
                 actor_cfm_scale_floor=0.01,
                 actor_q_normalization=False, actor_q_scale_floor=0.01,
                 candidate_count=5, candidate_budget_std=0.05, candidate_composition_std=0.04,
                 target_candidate_budget_std=0.08, target_candidate_composition_std=0.05,
                 candidate_residual_weight=0.1, max_grad_norm=1.0, sigma_loss_focus=True,
                 use_crl=False, action_reg_coef=0.0,
                 velocity_constraint_threshold=0.0016, velocity_lambda_lr=0.05,
                 initial_velocity_lambda=0.03, acceleration_constraint_threshold=0.003,
                 acceleration_lambda_lr=0.005, initial_acceleration_lambda=0.001,
                 lambda_warmup_steps=500, auto_adjust_for_attention=True,
                 actor_matching_mode="q_only",
                 _init_setup_model=True):
        if not 0 < expectile < 1 or flow_coefficient <= 0 or policy_delay < 1:
            raise ValueError("Invalid FlowRL objective hyperparameters")
        if not 0 < tau <= 1 or not 0 <= gamma <= 1:
            raise ValueError("Invalid target interpolation or discount")
        self.expectile = expectile
        self.flow_coefficient = flow_coefficient
        self.policy_delay = int(policy_delay)
        self.action_normalization = ACTION_NORMALIZATION
        self.algorithm_name = "FlowRL"
        self.flowrl_format_version = 5 if _init_setup_model else 1
        # Network architecture and controller semantics are independent.
        # A field-less v5 checkpoint used executed-history CRL. Infer that
        # after loading metadata; never silently change a saved objective.
        self.crl_reference_mode = "td3_forward_endpoints" if _init_setup_model else "legacy"
        self.crl_flow_sampling = "paired" if _init_setup_model else "legacy"
        self.crl_coupling_version = 1
        self.actor_matching_mode = str(actor_matching_mode) if _init_setup_model else "legacy"
        if _init_setup_model and self.actor_matching_mode not in {"q_only", "combined_control"}:
            raise ValueError("Unknown FlowRL actor matching mode")
        self.financial_aux_weight = float(financial_aux_weight)
        if self.financial_aux_weight < 0:
            raise ValueError("FlowRL financial auxiliary weight must be nonnegative")
        self.actor_learning_rate = float(actor_learning_rate)
        self.exploration_noise = float(exploration_noise)
        self.actor_cfm_gradient_ratio = (None if actor_cfm_gradient_ratio is None
                                         else float(actor_cfm_gradient_ratio))
        self.actor_cfm_scale_floor = float(actor_cfm_scale_floor) if _init_setup_model else 0.0
        self.actor_q_normalization = bool(actor_q_normalization)
        self.actor_q_scale_floor = float(actor_q_scale_floor)
        self.candidate_count = int(candidate_count)
        self.candidate_budget_std = float(candidate_budget_std)
        self.candidate_composition_std = float(candidate_composition_std)
        self.target_candidate_budget_std = float(target_candidate_budget_std)
        self.target_candidate_composition_std = float(target_candidate_composition_std)
        self.candidate_residual_weight = float(candidate_residual_weight)
        self.max_grad_norm = float(max_grad_norm)
        self.sigma_loss_focus = bool(sigma_loss_focus)
        self._set_crl_configuration(
            use_crl, action_reg_coef, velocity_constraint_threshold, velocity_lambda_lr,
            initial_velocity_lambda, acceleration_constraint_threshold, acceleration_lambda_lr,
            initial_acceleration_lambda, lambda_warmup_steps, auto_adjust_for_attention)
        if (self.actor_learning_rate <= 0 or not math.isfinite(self.exploration_noise) or self.exploration_noise < 0
                or (self.actor_cfm_gradient_ratio is not None and self.actor_cfm_gradient_ratio < 0)
                or not 0 <= self.actor_cfm_scale_floor <= 1
                or self.actor_q_scale_floor <= 0
                or self.candidate_count < 1 or min(self.candidate_budget_std, self.candidate_composition_std,
                    self.target_candidate_budget_std, self.target_candidate_composition_std) < 0
                or not 0 <= self.candidate_residual_weight <= .5 or self.max_grad_norm <= 0):
            raise ValueError("Invalid FlowRL portfolio optimization/exploration settings")
        if _init_setup_model:
            policy_kwargs = dict(policy_kwargs or {})
            policy_kwargs.setdefault("architecture_version", 5)
        self.market_input_contract = config.market_input_contract()
        super().__init__(policy, env, learning_rate, buffer_size, learning_starts,
                         batch_size, tau, gamma, train_freq, gradient_steps,
                         replay_buffer_class=replay_buffer_class,
                         replay_buffer_kwargs=replay_buffer_kwargs,
                         policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log,
                         verbose=verbose, device=device, seed=seed,
                         support_multi_env=True, sde_support=False,
                         supported_action_spaces=(spaces.Box,))
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        if self.market_input_contract != config.market_input_contract():
            raise ValueError("FlowRL checkpoint market input contract differs from the current configuration")
        if int(self.policy_kwargs.get("architecture_version", 1)) >= 2:
            self.replay_buffer_kwargs.setdefault("store_component_outcomes", True)
        super()._setup_model()
        self.flowrl_format_version = self.policy.architecture_version
        if self.actor_matching_mode == "legacy":
            self.actor_matching_mode = ("combined_control" if self.policy.architecture_version >= 5 else "q_only")
        if self.actor_matching_mode not in {"q_only", "combined_control"}:
            raise ValueError("Unknown saved FlowRL actor matching mode")
        if self.crl_reference_mode == "legacy":
            self.crl_reference_mode = ("executed_history" if self.policy.architecture_version >= 5
                                       else "td3_forward_endpoints")
        if self.crl_reference_mode not in {"td3_forward_endpoints", "executed_history"}:
            raise ValueError("Unknown saved FlowRL CRL reference mode")
        if self.crl_flow_sampling == "legacy":
            self.crl_flow_sampling = ("paired" if self.crl_reference_mode == "td3_forward_endpoints"
                                      else "independent")
        if self.crl_flow_sampling not in {"paired", "independent"}:
            raise ValueError("Unknown saved FlowRL CRL flow sampling mode")
        if self.policy.architecture_version < 5:
            self.crl_coupling_version = 1
            self.crl_flow_sampling = "paired"
        self.actor, self.critic = self.policy.actor, self.policy.critic
        self.critic_target = self.policy.critic_target
        if self.policy.architecture_version >= 3:
            self.configure_collection_exploration(self.exploration_noise)
        self._check_environment(self.env)
        if self.env is not None and not hasattr(self, "environment_settings"):
            self.environment_settings = {
                "reward_type": self.env.get_attr("reward_type")[0],
                "use_variable_portfolio": self.env.get_attr("use_variable_portfolio")[0],
            }
        self.evaluation_action_mode = "stochastic_flow"
        self._setup_crl_state()

    def _set_crl_configuration(self, use_crl, action_reg_coef, velocity_constraint_threshold,
                               velocity_lambda_lr, initial_velocity_lambda,
                               acceleration_constraint_threshold, acceleration_lambda_lr,
                               initial_acceleration_lambda, lambda_warmup_steps,
                               auto_adjust_for_attention):
        values = (action_reg_coef, velocity_constraint_threshold, velocity_lambda_lr,
                  initial_velocity_lambda, acceleration_constraint_threshold,
                  acceleration_lambda_lr, initial_acceleration_lambda)
        if (not all(math.isfinite(float(value)) for value in values)
                or action_reg_coef < 0 or velocity_constraint_threshold < 0
                or acceleration_constraint_threshold < 0
                or min(velocity_lambda_lr, acceleration_lambda_lr,
                       initial_velocity_lambda, initial_acceleration_lambda) <= 0
                or lambda_warmup_steps < 0):
            raise ValueError("Invalid FlowRL dual CRL settings")
        self.use_crl = bool(use_crl)
        self.action_reg_coef = float(action_reg_coef)
        self.velocity_constraint_threshold = float(velocity_constraint_threshold)
        self.velocity_lambda_lr = float(velocity_lambda_lr)
        self.initial_velocity_lambda = float(initial_velocity_lambda)
        self.acceleration_constraint_threshold = float(acceleration_constraint_threshold)
        self.acceleration_lambda_lr = float(acceleration_lambda_lr)
        self.initial_acceleration_lambda = float(initial_acceleration_lambda)
        self.lambda_warmup_steps = int(lambda_warmup_steps)
        self.auto_adjust_for_attention = bool(auto_adjust_for_attention)

    def _setup_crl_state(self, reset=False):
        """Keep dual state outside actor parameters and portable across devices."""
        if self.use_crl and self.policy.architecture_version < 3:
            raise ValueError("Adaptive portfolio CRL requires FlowRL architecture v3 or newer")
        if reset:
            self._last_crl_diagnostics = {}
        for name, initial in (("lambda_param_velocity", self.initial_velocity_lambda),
                              ("lambda_param_acceleration", self.initial_acceleration_lambda),
                              ("_viol_ema_v", 0.0), ("_viol_ema_a", 0.0)):
            previous = None if reset else getattr(self, name, None)
            value = torch.tensor(initial, device=self.device) if previous is None else torch.as_tensor(
                previous, device=self.device, dtype=torch.float32).detach()
            setattr(self, name, value)
        for name in ("velocity_constraint_history", "velocity_lambda_history",
                     "acceleration_constraint_history", "acceleration_lambda_history"):
            if reset or not hasattr(self, name):
                setattr(self, name, [])
        if reset or not hasattr(self, "max_velocity_lambda_reference"):
            self.max_velocity_lambda_reference = self.initial_velocity_lambda
        self.max_accel_to_velocity_lambda_ratio = 10.0
        self.is_attention_policy = True
        self.attention_exploration_bonus = 0.0

    def configure_crl(self, **settings):
        """Explicitly migrate old history CRL; unchanged resumes keep dual state.

        Historical/forward costs and paired/independent samples have different
        gradients and distributions.
        Their prices/EMAs and actor Adam moments are not interchangeable. Keep
        the learned networks, critic optimizers and replay, but restart that
        controller state when switching the estimator used for training.
        """
        names = ("use_crl", "action_reg_coef", "velocity_constraint_threshold", "velocity_lambda_lr",
                 "initial_velocity_lambda", "acceleration_constraint_threshold", "acceleration_lambda_lr",
                 "initial_acceleration_lambda", "lambda_warmup_steps", "auto_adjust_for_attention")
        previous = tuple(getattr(self, name) for name in names)
        enabled_before = self.use_crl
        self._set_crl_configuration(**settings)
        changed = previous != tuple(getattr(self, name) for name in names)
        reference_changed = self.crl_reference_mode != "td3_forward_endpoints"
        sampling = "paired"
        sampling_changed = self.crl_flow_sampling != sampling
        if reference_changed or sampling_changed:
            self.crl_reference_mode = "td3_forward_endpoints"
            self.crl_flow_sampling = sampling
            # Do not average pending old-estimator metrics into the first
            # reporting interval of the new objective.
            self._telemetry_loss_totals = {}
            changed = True
        if self.crl_coupling_version != 1:
            self.crl_coupling_version = 1
            changed = True
        self._setup_crl_state(reset=reference_changed or sampling_changed or (self.use_crl and not enabled_before))
        if changed:
            self.policy.actor_optimizer.state.clear()
        return changed

    def validate_td3_crl_training_contract(self):
        """Keep portfolio training on the explicitly restored TD3 references.

        Saved experimental controllers remain loadable for evaluation. The
        runner calls this after any explicit resume migration, before learning.
        Pairing coordinates current/next flow draws; it cannot reconstruct the
        old latent or OU realization in the previous executed allocation.
        """
        if not self.use_crl and self.action_reg_coef <= 0:
            return
        expected = dict(crl_reference_mode="td3_forward_endpoints",
                        crl_flow_sampling="paired", crl_coupling_version=1)
        differences = [f"{name}={getattr(self, name, None)!r} (expected {value!r})"
                       for name, value in expected.items() if getattr(self, name, None) != value]
        if differences:
            raise ValueError("FlowRL portfolio training requires completed, paired TD3 forward references; "
                             + "; ".join(differences) + ". Apply configure_crl() before training.")
        target = getattr(self.policy, "actor_target", None)
        if target is None or any(parameter.requires_grad for parameter in target.parameters()):
            raise ValueError("FlowRL CRL requires a frozen target flow actor for the next completed proposal")

    def _crl_penalty(self, data, actions, next_actions, task_groups=None):
        """TD3 path: previous execution -> current proposal -> next target.

        Both proposals are FINAL tanh-bounded flow endpoints. New training uses
        the October 29 common initializer, comparing one flow realization across
        replay-adjacent states. This is a policy-trajectory regularizer; separate
        executed-action telemetry still measures independent collection draws.
        Only the current actor owns a gradient, as in TD3. Cash/assets use
        joint physical weights.
        Pairing does not match the past execution's latent or collection OU
        noise. Acceleration can therefore include historical sampling/policy
        disagreement even when current-to-next velocity is small.
        The old history controller is retained solely for checkpoint semantics.
        """
        if self.crl_reference_mode == "executed_history":
            return self._causal_crl_penalty(data, actions, task_groups)
        mask = action_mask(data.observations)
        current = portfolio_weights(actions, mask)
        with torch.no_grad():
            following = portfolio_weights(next_actions.detach(), mask)
            previous = data.observations.get("prev_action")
            previous = None if previous is None else previous.detach().to(actions)
            previous_mask = data.observations.get("prev_mask")
            if self.policy.architecture_version >= 4 and previous_mask is not None:
                # Zero exposure is an observed allocation, not missing history.
                # The availability mask keeps cash-only/concentrated books
                # eligible for acceleration control on the known asset set.
                historical = torch.cat((torch.ones_like(previous_mask[:, :1]), previous_mask), -1).to(mask)
                shared = historical * mask
            else:
                shared = mask if previous is None else ((previous * mask) > 1e-9).to(mask) * mask
            valid = shared.sum(-1) > 1e-8
            shared_dims = shared.sum(-1).clamp_min(1)
            shared_space = (previous is not None and bool(valid.any())
                            and bool(shared_dims.mean() / mask.sum(-1).clamp_min(1).mean() >= .3))
        compared = shared if shared_space else mask
        eligible = valid if shared_space else None
        def average(rows):
            return basket_balanced_mean(rows, data.observations, eligible, task_groups)
        dimensions = compared.sum(-1).clamp_min(1)
        velocity_rows = ((following - current).square() * compared).sum(-1)
        velocity_cost = average(velocity_rows)
        velocity_violation = average(velocity_rows - dimensions * self.velocity_constraint_threshold)
        acceleration_cost = actions.new_zeros(())
        acceleration_violation = None
        if shared_space:
            acceleration_rows = ((following - 2 * current + previous).square() * compared).sum(-1)
            acceleration_cost = average(acceleration_rows)
            acceleration_violation = average(
                acceleration_rows - dimensions * self.acceleration_constraint_threshold)
        with torch.no_grad():
            coupling = torch.sigmoid(-25.0 * velocity_violation)
        acceleration_penalty = actions.new_zeros(())
        if self.use_crl:
            velocity_penalty = self.lambda_param_velocity.clamp_min(0) * velocity_cost
            if previous is not None and self._n_updates >= self.lambda_warmup_steps:
                acceleration_penalty = coupling * self.lambda_param_acceleration.clamp_min(0) * acceleration_cost
            self._update_lambdas(velocity_violation.detach(),
                                 acceleration_violation.detach() if acceleration_violation is not None
                                 and self._n_updates >= self.lambda_warmup_steps else None)
        else:
            velocity_penalty = self.action_reg_coef * velocity_cost
        smooth = velocity_penalty + acceleration_penalty
        diagnostics = {
            "actor_velocity_penalty": velocity_penalty, "actor_acceleration_penalty": acceleration_penalty,
            "actor_smooth_loss": smooth, "velocity_cost": velocity_cost,
            "acceleration_cost": acceleration_cost, "velocity_violation": velocity_violation,
            "crl_coupling_factor": coupling, "crl_shared_subspace": actions.new_tensor(float(shared_space)),
            "crl_acceleration_active": actions.new_tensor(float(
                self.use_crl and acceleration_violation is not None and self._n_updates >= self.lambda_warmup_steps)),
        }
        if acceleration_violation is not None:
            diagnostics["raw_acceleration_violation"] = acceleration_violation
            diagnostics["gated_acceleration_violation"] = acceleration_violation * (.05 + .95 * coupling)
            diagnostics["dynamic_gate"] = .05 + .95 * coupling
        if self.policy.architecture_version >= 5:
            # Explain the SAME three-point stencil used by the actor. These
            # diagnostics do not introduce a second history-based penalty.
            with torch.no_grad():
                outgoing = (following - current.detach()) * compared
                incoming = (current.detach() - previous) * compared if previous is not None else torch.zeros_like(outgoing)
                covered = eligible if shared_space else torch.zeros_like(dimensions, dtype=torch.bool)
                def joint_average(rows):
                    return basket_balanced_mean(rows, data.observations, covered, task_groups)
                outgoing_energy = joint_average(outgoing.square().sum(-1))
                incoming_energy = joint_average(incoming.square().sum(-1))
                inner_product = joint_average((outgoing * incoming).sum(-1))
                reconstructed = outgoing_energy + incoming_energy - 2 * inner_product
                persistence = (inner_product / (outgoing_energy * incoming_energy).sqrt().clamp_min(1e-12)).clamp(-1, 1)
            diagnostics.update(
                crl_reference_is_forward=actions.new_tensor(1.),
                crl_flow_sampling_is_independent=actions.new_tensor(float(self.crl_flow_sampling == "independent")),
                crl_acceleration_valid_fraction=covered.float().mean(),
                crl_coupling_min=coupling, crl_coupling_max=coupling,
                crl_gated_acceleration_cost=coupling * acceleration_cost,
                crl_joint_velocity_cost=outgoing_energy, crl_previous_velocity_cost=incoming_energy,
                crl_velocity_inner_product=inner_product, crl_velocity_persistence=persistence,
                crl_reconstructed_acceleration_cost=reconstructed,
                crl_acceleration_identity_error=(reconstructed - acceleration_cost.detach()).abs())
        self._last_crl_diagnostics = {name: float(value.detach()) for name, value in diagnostics.items()}
        return smooth, diagnostics

    def _causal_crl_penalty(self, data, actions, task_groups=None):
        """Legacy v5 executed-history controller, for saved-objective compatibility.

        Velocity is proposal minus last executed target. Acceleration is its
        change from the last two executed targets. Neither term mixes a slow
        future policy/common latent with an independently sampled past action.
        """
        obs = data.observations
        mask = action_mask(obs)
        current = portfolio_weights(actions, mask)
        previous = obs["prev_action"].detach().to(current)
        previous_mask = obs.get("prev_mask")
        compared = mask
        if previous_mask is not None:
            available = torch.cat((torch.ones_like(previous_mask[:, :1]), previous_mask), -1).to(mask)
            known = previous_mask.sum(-1, keepdim=True) > 0
            compared = torch.where(known, mask * available, mask)
        dimensions = compared.sum(-1).clamp_min(1)
        def average(rows, valid=None):
            return basket_balanced_mean(rows, obs, valid, task_groups)
        velocity = (current - previous) * compared
        velocity_rows = velocity.square().sum(-1)
        velocity_cost = average(velocity_rows)
        velocity_violation = average(velocity_rows - dimensions * self.velocity_constraint_threshold)
        eligible = torch.zeros_like(dimensions, dtype=torch.bool)
        alignment = torch.zeros_like(eligible)
        acceleration_rows = torch.zeros_like(dimensions)
        previous_velocity = torch.zeros_like(velocity)
        history = obs.get("portfolio_reward_history")
        length, width = int(config.LONG_TERM_LOOKBACK), mask.shape[-1]
        if history is not None and length >= 2:
            history = history.detach().to(current)
            books = history[:, :length * width].reshape(-1, length, width)
            valid = history[:, length * width:length * width + length] > .5
            alignment = (books[:, -1] - previous).abs().amax(-1) <= 1e-4
            eligible = valid[:, -1] & valid[:, -2] & alignment
            previous_velocity = (previous - books[:, -2]) * compared
            acceleration_rows = (velocity - previous_velocity).square().sum(-1)
        acceleration_cost = average(acceleration_rows, eligible)
        acceleration_violation = (average(
            acceleration_rows - dimensions * self.acceleration_constraint_threshold, eligible)
            if bool(eligible.any()) else None)
        with torch.no_grad():
            if self.crl_coupling_version >= 2:
                # Coordinate the two constraints on the SAME replay decision.
                # A fast sample must not mute refinement of a slower reversal
                # elsewhere in the batch. Detach the gate, as in baseline TD3.
                coupling_rows = torch.sigmoid(-25.0 * (
                    velocity_rows.detach() - dimensions * self.velocity_constraint_threshold))
            else:
                coupling_rows = torch.ones_like(velocity_rows) * torch.sigmoid(-25.0 * velocity_violation)
            coupling = (average(coupling_rows, eligible) if acceleration_violation is not None
                        else average(coupling_rows))
        gated_acceleration_cost = average(coupling_rows * acceleration_rows, eligible)
        with torch.no_grad():
            joint_velocity_cost = average(velocity_rows, eligible)
            previous_velocity_cost = average(previous_velocity.square().sum(-1), eligible)
            velocity_inner_product = average((velocity * previous_velocity).sum(-1), eligible)
            reconstructed_acceleration = joint_velocity_cost + previous_velocity_cost - 2 * velocity_inner_product
            normalizer = (joint_velocity_cost * previous_velocity_cost).sqrt().clamp_min(1e-12)
            persistence = (velocity_inner_product / normalizer).clamp(-1, 1)
        active = self.use_crl and self._n_updates >= self.lambda_warmup_steps and acceleration_violation is not None
        acceleration_penalty = actions.new_zeros(())
        if self.use_crl:
            velocity_penalty = self.lambda_param_velocity.clamp_min(0) * velocity_cost
            if active:
                acceleration_penalty = self.lambda_param_acceleration.clamp_min(0) * gated_acceleration_cost
            self._update_lambdas(velocity_violation.detach(), acceleration_violation.detach() if active else None)
        else:
            velocity_penalty = self.action_reg_coef * velocity_cost
        smooth = velocity_penalty + acceleration_penalty
        diagnostics = {
            "crl_reference_is_forward": actions.new_zeros(()),
            "actor_velocity_penalty": velocity_penalty, "actor_acceleration_penalty": acceleration_penalty,
            "actor_smooth_loss": smooth, "velocity_cost": velocity_cost,
            "acceleration_cost": acceleration_cost, "velocity_violation": velocity_violation,
            "crl_coupling_factor": coupling, "crl_shared_subspace": actions.new_tensor(1.),
            "crl_acceleration_active": actions.new_tensor(float(active)),
            "crl_acceleration_valid_fraction": eligible.float().mean(),
            "crl_history_alignment_fraction": alignment.float().mean(),
            "crl_coupling_min": coupling_rows.min(), "crl_coupling_max": coupling_rows.max(),
            "crl_gated_acceleration_cost": gated_acceleration_cost,
            "crl_joint_velocity_cost": joint_velocity_cost,
            "crl_previous_velocity_cost": previous_velocity_cost,
            "crl_velocity_inner_product": velocity_inner_product,
            "crl_velocity_persistence": persistence,
            "crl_reconstructed_acceleration_cost": reconstructed_acceleration,
            "crl_acceleration_identity_error": (reconstructed_acceleration - acceleration_cost).abs(),
        }
        if acceleration_violation is not None:
            diagnostics.update(raw_acceleration_violation=acceleration_violation,
                               gated_acceleration_violation=average(
                                   (acceleration_rows - dimensions * self.acceleration_constraint_threshold)
                                   * (.05 + .95 * coupling_rows), eligible),
                               dynamic_gate=.05 + .95 * coupling)
        self._last_crl_diagnostics = {name: float(value.detach()) for name, value in diagnostics.items()}
        return smooth, diagnostics

    def _update_lambdas(self, velocity_violation, acceleration_violation):
        """Port of the stable TD3 EMA, dead bands, staged updates and ratio cap."""
        if not self.use_crl:
            return
        with torch.no_grad():
            for velocity, violation, parameter, rate, threshold in (
                    (True, velocity_violation, self.lambda_param_velocity,
                     self.velocity_lambda_lr, self.velocity_constraint_threshold),
                    (False, acceleration_violation, self.lambda_param_acceleration,
                     self.acceleration_lambda_lr, self.acceleration_constraint_threshold)):
                if violation is None or (not velocity and self._n_updates < self.lambda_warmup_steps):
                    continue
                ema = self._viol_ema_v if velocity else self._viol_ema_a
                beta = .4 if velocity else .6
                ema.mul_(1 - beta).add_(beta * violation)
                adaptive_rate = rate
                if not velocity:
                    magnitude = ema.abs()
                    adaptive_rate *= 2 if magnitude > 2 * threshold else 1.5 if magnitude > threshold else 1
                signal = ema.clamp(-1, 1)
                band = .05 * threshold
                if self._n_updates < self.lambda_warmup_steps:
                    if signal > band:
                        parameter.mul_((1 + rate * (signal - band)).clamp(1, 1.015))
                elif signal > band:
                    parameter.mul_((1 + adaptive_rate * (signal - band)).clamp(1, 1.015 if velocity else 1.005))
                elif signal < (-band if velocity else 0):
                    parameter.mul_((1 + adaptive_rate * (signal + (band if velocity else 0))).clamp(
                        .985 if velocity else .998, 1))
                if velocity:
                    parameter.clamp_(1e-18, 1e5)
                    self.max_velocity_lambda_reference = max(self.max_velocity_lambda_reference, float(parameter))
                else:
                    reference = max(float(self.lambda_param_velocity), self.max_velocity_lambda_reference)
                    parameter.clamp_(1e-18, reference * self.max_accel_to_velocity_lambda_ratio)
        self.velocity_constraint_history.append(float(velocity_violation) + self.velocity_constraint_threshold)
        self.velocity_lambda_history.append(float(self.lambda_param_velocity))
        if acceleration_violation is not None:
            self.acceleration_constraint_history.append(float(acceleration_violation) + self.acceleration_constraint_threshold)
            self.acceleration_lambda_history.append(float(self.lambda_param_acceleration))
        for kind in ("velocity", "acceleration"):
            if len(getattr(self, kind + "_constraint_history")) > 1000:
                for field in ("constraint_history", "lambda_history"):
                    setattr(self, kind + "_" + field, getattr(self, kind + "_" + field)[-500:])

    def get_crl_stats(self):
        """Use the same dual schema consumed by the TD3 CRL plotter."""
        if not self.use_crl:
            return {"crl_enabled": False}
        velocity, acceleration = float(self.lambda_param_velocity), float(self.lambda_param_acceleration)
        stats = dict(crl_enabled=True, crl_schema="dual_velocity_acceleration",
                     current_lambda_velocity=velocity, current_lambda_acceleration=acceleration,
                     velocity_constraint_threshold=self.velocity_constraint_threshold,
                     acceleration_constraint_threshold=self.acceleration_constraint_threshold,
                     velocity_lambda_lr=self.velocity_lambda_lr, acceleration_lambda_lr=self.acceleration_lambda_lr,
                     current_lambda=velocity, current_lambda_total=velocity + acceleration,
                     constraint_threshold=self.velocity_constraint_threshold + self.acceleration_constraint_threshold,
                     is_attention_policy=self.is_attention_policy, auto_adjust_enabled=self.auto_adjust_for_attention,
                     attention_exploration_bonus=self.attention_exploration_bonus)
        total, threshold = 0.0, 0.0
        observed = False
        for kind in ("velocity", "acceleration"):
            history = getattr(self, kind + "_constraint_history")
            if history:
                cost = float(np.mean(history[-10:]))
                if math.isfinite(cost):
                    limit = getattr(self, kind + "_constraint_threshold")
                    stats["avg_recent_" + kind + "_cost"] = cost
                    stats["avg_recent_" + kind + "_violation"] = cost - limit
                    total, threshold, observed = total + cost, threshold + limit, True
        if observed:
            stats.update(avg_recent_constraint_cost=total, avg_recent_violation=total - threshold)
        stats.update(getattr(self, "_last_crl_diagnostics", {}))
        return stats

    def _check_environment(self, env):
        if env is not None:
            try:
                mappings = env.get_attr("action_normalization")
            except AttributeError as exc:
                raise ValueError("FlowRL requires a joint-simplex PortfolioEnv") from exc
            if any(mapping != ACTION_NORMALIZATION for mapping in mappings):
                raise ValueError("FlowRL checkpoint/environment action normalization mismatch")

    def upgrade_cash_budget_architecture(self):
        """Attach zero-output v4 branches without changing a trained v3 function.

        Existing Parameter objects and Adam moments survive. Reorder the
        optimizer groups to the v4 module order so saving/reloading maps every
        old moment to the right parameter, including nested critic additions.
        """
        p = self.policy
        if p.architecture_version >= 4:
            return False
        if p.architecture_version != 3:
            raise ValueError("Cash-budget migration requires a FlowRL v3 learner")
        p.actor.enable_cash_budget()
        for name in ("critic", "behavior_critic"):
            getattr(p, name).enable_cash_budget()
            for online, target in zip(getattr(p, name).q_networks, getattr(p, name + "_target").q_networks):
                target.basket_opportunity_head.cash_marginal_head = deepcopy(
                    online.basket_opportunity_head.cash_marginal_head).requires_grad_(False)
            getattr(p, name + "_target").architecture_version = 4
        p.actor_target.cash_risk_budget_head = deepcopy(p.actor.cash_risk_budget_head).requires_grad_(False)
        p.actor_target.architecture_version = p.actor_target.state_encoder.architecture_version = 4
        for name in ("actor", "critic", "behavior_critic"):
            optimizer = getattr(p, name + "_optimizer")
            if len(optimizer.param_groups) != 1:
                raise ValueError("FlowRL cash migration requires the standard single-group Adam optimizer")
            optimizer.param_groups[0]["params"] = list(getattr(p, name).parameters())
        p.value.state_encoder.architecture_version = 4
        p.architecture_version = p.network_kwargs["architecture_version"] = 4
        self.policy_kwargs["architecture_version"] = self.flowrl_format_version = 4
        return True

    def set_env(self, env, force_reset=True):
        super().set_env(env, force_reset=force_reset)
        self._check_environment(self.env)

    def resolve_deterministic_flag(self, requested_mode="auto"):
        if requested_mode == "auto":
            return False  # Upstream evaluates the stochastic flow distribution.
        if requested_mode not in {"deterministic", "stochastic"}:
            raise ValueError(f"Unknown FlowRL action mode: {requested_mode}")
        return requested_mode == "deterministic"

    def configure_actor_objective(self, gradient_ratio, matching_floor, normalize_q, q_scale_floor,
                                  matching_mode=None):
        """Migrate optimization settings without retaining incompatible Adam moments."""
        requested = (None if gradient_ratio is None else float(gradient_ratio),
                     float(matching_floor), bool(normalize_q), float(q_scale_floor))
        if ((requested[0] is not None and requested[0] < 0)
                or not 0 <= requested[1] <= 1 or requested[3] <= 0):
            raise ValueError("Invalid FlowRL actor objective settings")
        previous = (self.actor_cfm_gradient_ratio, self.actor_cfm_scale_floor,
                    self.actor_q_normalization, self.actor_q_scale_floor)
        mode = self.actor_matching_mode if matching_mode is None else str(matching_mode)
        if mode not in {"q_only", "combined_control"}:
            raise ValueError("Unknown FlowRL actor matching mode")
        mode_changed = mode != self.actor_matching_mode
        changed = (previous[:3] != requested[:3]
                   or ((previous[2] or requested[2]) and previous[3] != requested[3]) or mode_changed)
        (self.actor_cfm_gradient_ratio, self.actor_cfm_scale_floor,
         self.actor_q_normalization, self.actor_q_scale_floor) = requested
        self.actor_matching_mode = mode
        if mode_changed:
            self._telemetry_loss_totals = {}
        if changed:
            self.policy.actor_optimizer.state.clear()
        return changed

    def configure_collection_exploration(self, sigma):
        """Apply current collection settings, including on explicit resume.

        Zero removes any saved post-flow noise object without changing the
        actor, flow latents, replay, optimizer or dual state. Positive sigma is
        an explicit opt-in retained for reproducing historical OU experiments.
        Loading alone preserves saved behavior; the runner applies its current
        setting before collection resumes.
        """
        sigma = float(sigma)
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("FlowRL collection exploration sigma must be finite and nonnegative")
        previous = self.exploration_noise
        had_noise = self.action_noise is not None
        self.exploration_noise = sigma
        if sigma == 0:
            self.action_noise = None
        elif previous != sigma or not had_noise:
            dimensions = self.action_space.shape[0]
            self.action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(dimensions), sigma=np.full(dimensions, sigma), theta=0.15)
        changed = previous != sigma or had_noise != (self.action_noise is not None)
        if changed:
            for name in ("exploration_raw_delta", "exploration_allocation_delta"):
                getattr(self, "_telemetry_loss_totals", {}).pop(name, None)
        if hasattr(self, "flowrl_data_provenance"):
            self.flowrl_data_provenance.update(
                collection_exploration_sigma=sigma,
                collection_exploration_mode=("stochastic_flow_only" if self.action_noise is None
                                             else "stochastic_flow_plus_ou"))
        return changed

    def _setup_learn(self, total_timesteps, callback=None, reset_num_timesteps=True,
                     tb_log_name="run", progress_bar=False):
        # SB3 resets noise on every learn() call. The portfolio runner calls
        # learn() in collection blocks, so preserve an ongoing noise process
        # across those blocks. A fresh/reset environment starts fresh noise.
        continuing = not reset_num_timesteps and self._last_obs is not None and self.num_timesteps > 0
        saved_noise = (deepcopy(self.action_noise) if continuing and self.policy.architecture_version >= 3
                       and self.action_noise is not None else None)
        result = super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)
        if saved_noise is not None:
            self.action_noise = saved_noise
        return result

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # The default executes the stochastic flow output itself. The optional
        # positive-sigma historical mode adds OU only after flow generation.
        # A zero setting also ignores a stale caller-supplied noise object.
        # Replay stores the exact execution, with canonical excluded coordinates.
        action, stored = super()._sample_action(learning_starts, None, n_envs)
        if self.policy.architecture_version < 3:
            return action, stored
        mask = np.concatenate((np.ones((n_envs, 1)), np.asarray(self._last_obs["mask"])), axis=1)
        base = np.where(mask, stored, -1)
        if self.exploration_noise > 0 and action_noise is not None and self.num_timesteps >= learning_starts:
            stored = np.clip(stored + action_noise(), -1, 1)
        stored = np.where(mask, stored, -1).astype(np.float32)
        if self.num_timesteps >= learning_starts:
            def normalized(values):
                scores = (values + 1) * .5 * mask
                total = scores.sum(-1, keepdims=True)
                return np.where(total > 1e-8, scores / total.clip(1e-8), mask / mask.sum(-1, keepdims=True))
            self._accumulate_metrics({
                "exploration_raw_delta": float((np.abs(stored - base) * mask).sum() / mask.sum()),
                "exploration_allocation_delta": float(np.abs(normalized(stored) - normalized(base)).sum(-1).mean()),
            })
        # This policy explicitly requires Box(-1, 1), so unscaling is the
        # identity. Avoid an unnecessary affine round trip before env.step.
        return stored.copy(), stored

    def _accumulate_metrics(self, values):
        if not hasattr(self, "_telemetry_loss_totals"):
            self._telemetry_loss_totals = {}
        for name, value in values.items():
            total, count = self._telemetry_loss_totals.get(name, (0.0, 0))
            self._telemetry_loss_totals[name] = (total + float(value), count + 1)

    def _candidate_value(self, critic, observations, actions, update_ema=False, include_reference=False):
        """TD3-style center-anchored budget/composition soft improvement.

        Candidate randomness explores allocations; Q decides their utility.
        Both actor and Bellman passes use the same physical-weight geometry.
        """
        candidates, families = portfolio_action_candidates(
            actions, action_mask(observations), self.candidate_count,
            self.target_candidate_budget_std if update_ema else self.candidate_budget_std,
            self.target_candidate_composition_std if update_ema else self.candidate_composition_std)
        if include_reference:
            held = weights_to_flow_actions(observations["prev_action"].float(), action_mask(observations))
            candidates = torch.cat((candidates, held[:, None]), 1)
            families = torch.cat((families, families.new_tensor([3])))
        twins = torch.stack(critic.evaluate_action_candidates(observations, candidates))
        conservative = twins.min(0).values
        value = conservative[:, :1]
        center = value
        if not hasattr(self, "_candidate_disagreement_ema"):
            self._candidate_disagreement_ema = {}
        for family in (1, 2):
            selected = families == family
            if not selected.any():
                continue
            q = conservative[:, selected]
            if q.shape[1] == 1:
                # log-mean-exp of a singleton is exactly its Q, independent
                # of temperature/disagreement. The default budget family has
                # one member; avoid a pointless host synchronization here.
                value = value + self.candidate_residual_weight * (q - center)
                continue
            disagreement = (twins[0, :, selected] - twins[1, :, selected]).abs().mean(-1)
            channel = str(family)
            previous = self._candidate_disagreement_ema.get(channel, float(disagreement.detach().mean()))
            if update_ema:
                previous = .99 * previous + .01 * float(disagreement.detach().mean())
                self._candidate_disagreement_ema[channel] = previous
            # Same scale-aware temperature and .01 floor ratio as 10/13.
            temperature = q.detach().std(-1, keepdim=True, unbiased=False).clamp(.0005, .05)
            amplification = 1 + (disagreement.detach()[:, None] / max(previous, 1e-6) - 1).relu()
            temperature = (temperature * amplification).clamp(.0005, .25)
            soft_value = temperature * (torch.logsumexp(q / temperature, -1, keepdim=True) - math.log(q.shape[1]))
            value = value + self.candidate_residual_weight * (soft_value - center)
        return (value, center, conservative[:, -1:]) if include_reference else (value, center)

    def _actor_objective_gradients(self, value_loss, matching_loss, smooth_loss=None):
        parameters = tuple(parameter for parameter in self.actor.parameters() if parameter.requires_grad)
        has_smoothness = smooth_loss is not None and smooth_loss.requires_grad
        losses = (value_loss, smooth_loss, matching_loss) if has_smoothness else (value_loss, matching_loss)
        gradients = [torch.autograd.grad(loss, parameters, retain_graph=index < len(losses) - 1,
                                         allow_unused=True) for index, loss in enumerate(losses)]
        smooth_gradients = gradients.pop(1) if has_smoothness else None
        norms = [torch.stack([gradient.detach().square().sum() for gradient in group if gradient is not None]).sum().sqrt()
                 for group in gradients]
        products = [left.detach().mul(right.detach()).sum() for left, right in zip(*gradients)
                    if left is not None and right is not None]
        dot = torch.stack(products).sum() if products else value_loss.new_zeros(())
        value_norm, matching_norm, dot = torch.stack((*norms, dot)).detach().cpu().tolist()
        if not all(math.isfinite(value) for value in (value_norm, matching_norm, dot)):
            raise FloatingPointError("Nonfinite FlowRL actor objective gradient")
        control_norm, control_dot = value_norm, dot
        smooth_norm = 0.0
        if smooth_gradients is not None:
            control = tuple(right if left is None else left if right is None else left + right
                            for left, right in zip(gradients[0], smooth_gradients))
            squared_smooth = torch.stack([g.detach().square().sum() for g in smooth_gradients if g is not None]).sum()
            squared_control = torch.stack([g.detach().square().sum() for g in control if g is not None]).sum()
            products = [a.detach().mul(b.detach()).sum() for a, b in zip(control, gradients[1])
                        if a is not None and b is not None]
            control_product = torch.stack(products).sum() if products else value_loss.new_zeros(())
            smooth_norm, control_norm, control_dot = torch.stack(
                (squared_smooth.sqrt(), squared_control.sqrt(), control_product)).cpu().tolist()
            gradients[0] = control
        # The joint cash head is an architectural choice, not permission to
        # change the October 29 optimization law. New training budgets CFM
        # against Q alone and adds the full CRL gradient independently.
        protected_control = self.actor_matching_mode == "combined_control"
        budget_norm = control_norm if protected_control else value_norm
        budget_scale = (1.0 if self.actor_cfm_gradient_ratio is None else
                        min(1.0, self.actor_cfm_gradient_ratio * budget_norm / max(matching_norm, 1e-12)))
        # The October 29 Q-only rule retains its matching-scale floor. Keep
        # flat-only recovery solely for the saved combined-control variant.
        recovery = protected_control and value_norm <= 1e-10 and control_norm <= 1e-10
        scale = max(self.actor_cfm_scale_floor, budget_scale) if not protected_control or recovery else budget_scale
        self._last_actor_gradient_diagnostics = {
            "actor_q_cfm_cosine": dot / max(value_norm * matching_norm, 1e-24),
            "actor_q_descent_margin": (1 + scale * dot / max(value_norm ** 2, 1e-24)) if value_norm else 0.0,
            "actor_matching_floor_active": float(scale > budget_scale),
            "actor_matching_recovery_active": float(recovery and scale > budget_scale),
            "actor_smooth_gradient_norm": smooth_norm, "actor_control_gradient_norm": control_norm,
            "actor_control_cfm_cosine": control_dot / max(control_norm * matching_norm, 1e-24),
            "actor_control_descent_margin": (1 + scale * control_dot / max(control_norm ** 2, 1e-24)) if control_norm else 0.,
            "actor_control_cfm_gradient_ratio": scale * matching_norm / max(control_norm, 1e-12),
        }
        return parameters, gradients, scale, value_norm, matching_norm

    def _apply_objective_gradients(self, optimizer, parameters, gradients, scale):
        """Combine measured gradients directly instead of backpropagating again."""
        optimizer.zero_grad(set_to_none=True)
        for parameter, first, second in zip(parameters, *gradients):
            parameter.grad = (None if second is None else second.mul(scale)) if first is None else (
                first if second is None else first.add(second, alpha=scale))
        torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()

    def _critic_sample_weights(self, observations):
        """Baseline volatility-aware critic focus, with excluded tape erased."""
        mask = observations["mask"].float()
        if not self.sigma_loss_focus:
            return torch.ones_like(mask[:, :1])
        indices = [index for index, name in enumerate(config.SCOLS) if name.endswith("_sigma")]
        if not indices:
            return torch.ones_like(mask[:, :1])
        features = observations["observations"].float().reshape(mask.shape[0], mask.shape[1], -1)
        with torch.no_grad():
            sigma = features[:, :, indices].abs().masked_fill(~mask.bool().unsqueeze(-1), 0)
            if config.LOCAL_CAUSAL_INPUTS_ENABLED:
                reference = torch.as_tensor(config.local_primary_references(config.SCOLS)[indices],
                                            device=features.device, dtype=features.dtype)
                sigma = torch.sinh(sigma) * reference
            intensity = sigma.sum((1, 2)) / (mask.sum(-1) * len(indices)).clamp_min(1)
            normalized = intensity / (intensity.mean() + 1e-6)
            return (1 + .75 * normalized).clamp(1, 3)[:, None]

    def _financial_gradient_scale(self, td_loss, auxiliary_loss, critic, system, measured=None):
        """Carry over TD3's shared-encoder teacher budget for each Q system.

        Increases are smoothed; a newly excessive teacher is reduced on the
        current batch. Task-only/Q-only parameters cannot inflate its budget.
        """
        if self.financial_aux_weight == 0:
            return 0.0, 0.0
        parameters = (tuple(parameter for parameter in critic.component_encoder_parameters() if parameter.requires_grad)
                      if hasattr(critic, "component_encoder_parameters") else
                      tuple(parameter for q in critic.q_networks for parameter in q.state_encoder.parameters() if parameter.requires_grad))
        if measured is None:
            td_gradients = torch.autograd.grad(td_loss, parameters, retain_graph=True, allow_unused=True)
            aux_gradients = torch.autograd.grad(auxiliary_loss, parameters, retain_graph=True, allow_unused=True)
        else:
            all_parameters, gradients = measured
            shared = {id(parameter) for parameter in parameters}
            selected = [(td, aux) for parameter, td, aux in zip(all_parameters, *gradients) if id(parameter) in shared]
            td_gradients, aux_gradients = zip(*selected)
        pairs = [(td, aux) for td, aux in zip(td_gradients, aux_gradients)
                 if td is not None and aux is not None]
        if not pairs:
            raise RuntimeError("FlowRL asset teacher and Bellman TD have no shared encoder gradient")
        td_norm = torch.stack([td.detach().square().sum() for td, _ in pairs]).sum().sqrt().item()
        aux_norm = torch.stack([aux.detach().square().sum() for _, aux in pairs]).sum().sqrt().item()
        ratio = float(config.COMPONENT_OUTCOME_TARGET_GRADIENT_RATIO) * self.financial_aux_weight
        raw_scale = ratio * td_norm / max(aux_norm, 1e-12)
        bounded = min(raw_scale, float(np.clip(raw_scale,
            config.COMPONENT_OUTCOME_GRADIENT_SCALE_MIN, config.COMPONENT_OUTCOME_GRADIENT_SCALE_MAX)))
        if not hasattr(self, "_financial_gradient_scales"):
            self._financial_gradient_scales = {}
        previous = self._financial_gradient_scales.get(system, bounded)
        decay = float(np.clip(config.COMPONENT_OUTCOME_GRADIENT_SCALE_EMA, 0, .9999))
        scale = min(bounded, decay * previous + (1 - decay) * bounded)
        self._financial_gradient_scales[system] = scale
        return scale, scale * aux_norm / max(td_norm, 1e-12)

    def train(self, gradient_steps, batch_size=256):
        self.policy.set_training_mode(True)
        optimizers = [getattr(self.policy, name + "_optimizer") for name in
                      ("actor", "critic", "behavior_critic", "value")]
        self._update_learning_rate(optimizers)
        if self.policy.architecture_version >= 3:
            for group in self.policy.actor_optimizer.param_groups:
                group["lr"] = self.actor_learning_rate
        metrics = {}
        def record(name, value):
            metrics.setdefault(name, []).append(float(value.detach()))
        def optimize(loss, optimizer):
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite FlowRL loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.policy.architecture_version >= 3:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in optimizer.param_groups for parameter in group["params"]],
                    self.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
        def optimize_critic(td_loss, auxiliary_loss, valid, critic, optimizer, system):
            if p.architecture_version >= 3 and valid and self.financial_aux_weight > 0:
                if not torch.isfinite(torch.stack((td_loss, auxiliary_loss))).all():
                    raise FloatingPointError("Nonfinite FlowRL critic objective")
                parameters = tuple(parameter for parameter in critic.parameters() if parameter.requires_grad)
                gradients = [torch.autograd.grad(loss, parameters, retain_graph=index == 0, allow_unused=True)
                             for index, loss in enumerate((td_loss, auxiliary_loss))]
                scale, ratio = self._financial_gradient_scale(
                    td_loss, auxiliary_loss, critic, system, measured=(parameters, gradients))
                self._apply_objective_gradients(optimizer, parameters, gradients, scale)
                return scale, ratio
            scale, ratio = (self._financial_gradient_scale(td_loss, auxiliary_loss, critic, system)
                            if valid else (0.0, 0.0))
            optimize(td_loss + scale * auxiliary_loss, optimizer)
            return scale, ratio
        for _ in range(gradient_steps):
            self._n_updates += 1
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            p = self.policy
            portfolio_objective = p.architecture_version >= 3
            task_groups = (torch.unique(data.observations["mask"], dim=0, return_inverse=True)[1]
                           if portfolio_objective else None)
            def objective_mean(values):
                return basket_balanced_mean(values, data.observations, task_groups=task_groups) if portfolio_objective else values.mean()
            with torch.no_grad():
                # October 29 compares one flow realization across adjacent
                # states, completing both integrations before CRL. Retain the
                # independent rule only when loading that saved variant.
                paired_noise = (torch.randn_like(data.actions).clamp(-1, 1)
                                if portfolio_objective and self.crl_reference_mode == "td3_forward_endpoints"
                                and self.crl_flow_sampling == "paired"
                                and (self.use_crl or self.action_reg_coef > 0) else None)
                next_actions = (p.actor_target if portfolio_objective else p.actor)(
                    data.next_observations, noise=paired_noise)
                next_q = (self._candidate_value(p.critic_target, data.next_observations, next_actions, update_ema=True)[0]
                          if portfolio_objective else
                          torch.cat(p.critic_target(data.next_observations, next_actions), 1).min(1, keepdim=True).values)
                target_q = data.rewards + self.gamma * (1 - data.dones) * next_q
                behavior_target = data.rewards + self.gamma * (1 - data.dones) * p.value(data.next_observations)
                expectile_target = torch.cat(p.behavior_critic_target(data.observations, data.actions), 1).min(1, keepdim=True).values
            def critic_components(critic):
                if portfolio_objective:
                    values, predictions, local, *_ = critic.financial_components(data.observations, data.actions)
                    return values, predictions, local
                values, predictions = critic(data.observations, data.actions, return_components=True)
                return values, predictions, None
            regression = F.smooth_l1_loss if portfolio_objective else F.mse_loss
            sample_weights = self._critic_sample_weights(data.observations) if portfolio_objective else 1.0
            critic_values, financial_predictions, local_predictions = critic_components(p.critic)
            critic_loss = sum(objective_mean(regression(q, target_q, reduction="none") * sample_weights) for q in critic_values)
            absolute_loss, relative_loss, financial_valid = financial_forecast_losses(financial_predictions, data, portfolio_objective, task_groups)
            financial_aux_loss = absolute_loss + float(config.BASKET_RELATIVE_AUX_WEIGHT) * relative_loss
            if local_predictions is not None:
                financial_aux_loss = financial_aux_loss + float(config.LOCAL_FINANCIAL_LOCAL_LOSS_WEIGHT) * financial_forecast_losses(local_predictions, data, True, task_groups)[0]
            financial_scale, financial_ratio = optimize_critic(
                critic_loss, financial_aux_loss, financial_valid, p.critic, p.critic_optimizer, "policy")
            behavior_values, behavior_predictions, behavior_local = critic_components(p.behavior_critic)
            behavior_q = torch.cat(behavior_values, 1).min(1, keepdim=True).values.detach()
            behavior_loss = sum(objective_mean(regression(q, behavior_target, reduction="none") * sample_weights) for q in behavior_values)
            behavior_absolute, behavior_relative, behavior_valid = financial_forecast_losses(behavior_predictions, data, portfolio_objective, task_groups)
            behavior_financial_loss = behavior_absolute + float(config.BASKET_RELATIVE_AUX_WEIGHT) * behavior_relative
            if behavior_local is not None:
                behavior_financial_loss = behavior_financial_loss + float(config.LOCAL_FINANCIAL_LOCAL_LOSS_WEIGHT) * financial_forecast_losses(behavior_local, data, True, task_groups)[0]
            behavior_scale, behavior_ratio = optimize_critic(
                behavior_loss, behavior_financial_loss, behavior_valid, p.behavior_critic, p.behavior_critic_optimizer, "behavior")
            delta = expectile_target - p.value(data.observations)
            value_loss = objective_mean(torch.where(delta > 0, self.expectile, 1 - self.expectile) * delta.square())
            optimize(value_loss, p.value_optimizer)
            record("critic_loss", critic_loss)
            record("behavior_critic_loss", behavior_loss)
            record("value_loss", value_loss)
            if portfolio_objective:
                record("critic_sigma_weight", sample_weights.mean())
                record("baskets_in_batch", data.rewards.new_tensor(torch.unique(data.observations["mask"], dim=0).shape[0]))
            if p.architecture_version >= 2:
                record("financial_aux_loss", financial_aux_loss)
                record("financial_relative_loss", relative_loss)
                record("behavior_financial_aux_loss", behavior_financial_loss)
                record("financial_aux_scale", data.rewards.new_tensor(financial_scale))
                record("financial_encoder_gradient_ratio", data.rewards.new_tensor(financial_ratio))
                record("behavior_financial_encoder_gradient_ratio", data.rewards.new_tensor(behavior_ratio))
                label_mask = getattr(data, "component_outcome_mask", None)
                record("financial_label_fraction", data.rewards.new_zeros(()) if label_mask is None else
                       label_mask.float().sum() / data.observations["mask"].sum().clamp_min(1))
            if self._n_updates % self.policy_delay == 0:
                p.critic.requires_grad_(False)
                try:
                    encoded = p.actor.encode_state(data.observations) if portfolio_objective else None
                    actions = p.actor(data.observations, noise=paired_noise, encoded=encoded)
                    if portfolio_objective:
                        actor_value, policy_q, held_q = self._candidate_value(p.critic, data.observations, actions, include_reference=True)
                    else:
                        policy_q = torch.cat(p.critic(data.observations, actions), 1).min(1, keepdim=True).values
                        actor_value = policy_q
                    weights = guided_flow_weights(behavior_q, policy_q, self.flow_coefficient)
                    cfm = p.actor.flow_matching_errors(data.observations, data.actions, encoded=encoded)
                    raw_value_objective = -objective_mean(actor_value)
                    # Value and velocity matching have different units. Keep
                    # the replay constraint intact and normalize only the
                    # actor's Q term, with a detached scale and finite floor.
                    # Critics, Bellman targets and reported policy Q retain
                    # the environment's original reward units.
                    q_scale = (objective_mean(policy_q.detach().abs()).clamp_min(self.actor_q_scale_floor)
                               if portfolio_objective and self.actor_q_normalization else actor_value.new_tensor(1.0))
                    value_objective = raw_value_objective / q_scale
                    matching_objective = objective_mean(weights * cfm)
                    smooth_loss = value_objective.new_zeros(())
                    if portfolio_objective and (self.use_crl or self.action_reg_coef > 0):
                        smooth_loss, crl_metrics = self._crl_penalty(data, actions, next_actions, task_groups)
                        for name, value in crl_metrics.items():
                            record(name, value)
                    control_objective = value_objective + smooth_loss
                    if portfolio_objective:
                        parameters, gradients, matching_scale, value_norm, matching_norm = self._actor_objective_gradients(
                            value_objective, matching_objective, smooth_loss)
                        actor_loss = control_objective.detach() + matching_scale * matching_objective.detach()
                        self._apply_objective_gradients(p.actor_optimizer, parameters, gradients, matching_scale)
                    else:
                        matching_scale, value_norm, matching_norm = 1.0, 0.0, 0.0
                        actor_loss = value_objective + matching_objective
                        optimize(actor_loss, p.actor_optimizer)
                finally:
                    p.critic.requires_grad_(True)
                polyak_update(p.critic.parameters(), p.critic_target.parameters(), self.tau)
                polyak_update(p.behavior_critic.parameters(), p.behavior_critic_target.parameters(), self.tau)
                if portfolio_objective:
                    polyak_update(p.actor.parameters(), p.actor_target.parameters(), self.tau)
                    record("actor_q_gradient_norm", data.rewards.new_tensor(value_norm))
                    record("actor_cfm_gradient_norm", data.rewards.new_tensor(matching_norm))
                    record("effective_actor_cfm_gradient_norm", data.rewards.new_tensor(matching_scale * matching_norm))
                    record("actor_cfm_gradient_ratio", data.rewards.new_tensor(matching_scale * matching_norm / max(value_norm, 1e-12)))
                    record("flow_regularizer_scale", data.rewards.new_tensor(matching_scale))
                    for name, value in self._last_actor_gradient_diagnostics.items():
                        record(name, data.rewards.new_tensor(value))
                    record("effective_flow_matching_loss", matching_scale * matching_objective)
                    record("actor_q_loss", value_objective)
                    record("actor_raw_q_loss", raw_value_objective)
                    record("actor_q_scale", q_scale)
                    record("actor_control_loss", control_objective)
                    if not (self.use_crl or self.action_reg_coef > 0):
                        record("actor_smooth_loss", smooth_loss)
                    actor_weights = portfolio_weights(actions.detach(), action_mask(data.observations))
                    record("actor_cash_weight", actor_weights[:, :1].mean())
                    if p.architecture_version == 4:
                        record("actor_cash_budget_velocity", encoded[3]["flow_cash_budget_velocity"].mean())
                    elif p.architecture_version >= 5:
                        record("actor_cash_marginal_score", encoded[3]["flow_cash_marginal_score"].mean())
                    record("actor_max_asset_weight", actor_weights[:, 1:].max(-1).values.mean())
                    active = action_mask(data.observations).bool()
                    record("actor_raw_saturation", (actions.detach()[active].abs() > .999).float().mean())
                    record("candidate_value_gain", (actor_value - policy_q).mean())
                    record("policy_vs_hold_advantage", (policy_q - held_q).mean())
                record("actor_loss", actor_loss)
                record("flow_matching_loss", cfm.mean())
                record("weighted_flow_matching_loss", (weights * cfm).mean())
                record("flow_weight", weights.mean())
                record("policy_q", policy_q.mean())
        for name, values in metrics.items():
            self.logger.record("train/" + name, np.mean(values))
            # train() normally runs once per transition. Preserve every update
            # until the outer runner flushes a complete telemetry interval.
            if not hasattr(self, "_telemetry_loss_totals"):
                self._telemetry_loss_totals = {}
            total, count = self._telemetry_loss_totals.get(name, (0.0, 0))
            self._telemetry_loss_totals[name] = (total + sum(values), count + len(values))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def drain_training_metrics(self):
        """Interval means, never fabricated zero losses during warmup."""
        totals = getattr(self, "_telemetry_loss_totals", {})
        names = ("actor_loss", "critic_loss", "behavior_critic_loss", "value_loss",
                 "flow_matching_loss", "weighted_flow_matching_loss", "flow_weight", "policy_q",
                 "financial_aux_loss", "financial_relative_loss", "behavior_financial_aux_loss", "financial_label_fraction",
                 "financial_aux_scale", "financial_encoder_gradient_ratio", "behavior_financial_encoder_gradient_ratio",
                 "actor_q_loss", "effective_flow_matching_loss", "flow_regularizer_scale",
                 "actor_q_gradient_norm", "actor_cfm_gradient_norm", "effective_actor_cfm_gradient_norm", "actor_cfm_gradient_ratio", "candidate_value_gain", "policy_vs_hold_advantage",
                 "actor_cash_weight", "actor_max_asset_weight", "actor_raw_saturation",
                 "actor_cash_budget_velocity", "actor_cash_marginal_score",
                 "actor_raw_q_loss", "actor_q_scale",
                 "actor_q_cfm_cosine", "actor_q_descent_margin", "actor_matching_floor_active",
                 "exploration_raw_delta", "exploration_allocation_delta")
        names += ("critic_sigma_weight", "baskets_in_batch")
        names += ("actor_control_loss", "actor_smooth_loss", "actor_velocity_penalty", "actor_acceleration_penalty",
                  "actor_smooth_gradient_norm", "actor_control_gradient_norm", "velocity_cost", "acceleration_cost",
                  "velocity_violation", "raw_acceleration_violation", "gated_acceleration_violation",
                  "crl_coupling_factor", "dynamic_gate", "crl_shared_subspace", "crl_acceleration_active")
        names += ("crl_acceleration_valid_fraction", "crl_history_alignment_fraction",
                  "crl_reference_is_forward",
                  "crl_flow_sampling_is_independent",
                  "actor_control_cfm_cosine", "actor_control_descent_margin", "actor_matching_recovery_active",
                  "actor_control_cfm_gradient_ratio")
        names += ("crl_coupling_min", "crl_coupling_max", "crl_gated_acceleration_cost",
                  "crl_joint_velocity_cost", "crl_previous_velocity_cost", "crl_velocity_inner_product",
                  "crl_velocity_persistence", "crl_reconstructed_acceleration_cost", "crl_acceleration_identity_error")
        result = {"train/" + name: totals[name][0] / totals[name][1]
                  if name in totals else float("nan") for name in names}
        result["train/actor_updates_in_interval"] = totals.get("actor_loss", (0, 0))[1]
        result["train/critic_updates_in_interval"] = totals.get("critic_loss", (0, 0))[1]
        result["train/n_updates"] = self._n_updates
        result["train/crl_coupling_version"] = self.crl_coupling_version
        result["train/actor_matching_is_q_only"] = int(self.actor_matching_mode == "q_only")
        result["train/collection_exploration_sigma"] = self.exploration_noise
        result["train/collection_ou_enabled"] = int(self.exploration_noise > 0 and self.action_noise is not None)
        for name in ("actor", "critic", "behavior_critic", "value"):
            result["train/" + name + "_lr"] = getattr(self.policy, name + "_optimizer").param_groups[0]["lr"]
        result["train/learning_rate"] = result["train/critic_lr"]
        for name, value in self.get_crl_stats().items():
            if isinstance(value, (int, float, bool)) and name not in getattr(self, "_last_crl_diagnostics", {}):
                result["train/" + name] = value
        self._telemetry_loss_totals = {}
        return result

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self):
        return (["policy"] + ["policy." + name + "_optimizer" for name in
                               ("actor", "critic", "behavior_critic", "value")],
                ["lambda_param_velocity", "lambda_param_acceleration", "_viol_ema_v", "_viol_ema_a"])

    def load_replay_buffer(self, path, truncate_last_traj=True):
        super().load_replay_buffer(path, truncate_last_traj=truncate_last_traj)
        if not hasattr(self.replay_buffer, "store_component_outcomes"):
            self.replay_buffer.store_component_outcomes = hasattr(self.replay_buffer, "component_outcomes")
        if getattr(self.replay_buffer, "action_normalization", None) != ACTION_NORMALIZATION:
            raise ValueError("FlowRL cannot reuse replay collected with another action mapping")
