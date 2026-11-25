# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# Adapted for the portfolio project; licensed under the Apache License, Version
# 2.0. See FlowRL_LICENSE.txt and http://www.apache.org/licenses/LICENSE-2.0.
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Portfolio FlowRL networks.

Flow integration and objectives are adapted from FlowRL-main (ByteDance,
Apache-2.0; see FlowRL_LICENSE.txt). The masked portfolio architecture is
specific to this project. Flow time is generation time, not market time.
"""
import torch
from torch import nn

import config
from attention_policy import (
    AttentionDeepSetEncoder, BasketConditionedOpportunityHead, JointPortfolioOpportunityHead, DeepSetCritic,
    PortfolioRewardHistoryEncoder, _resolve_token_scope_dims,
)

ACTION_NORMALIZATION = "flowrl_joint_simplex_v1"


def action_mask(observations):
    mask = observations["mask"].float()
    return torch.cat((torch.ones_like(mask[:, :1]), mask), dim=1)


def portfolio_weights(actions, mask):
    """Cash and active assets have identical coordinates; padding has no mass."""
    shifted = (actions.clamp(-1, 1) + 1) * 0.5 * mask
    total = shifted.sum(-1, keepdim=True)
    return torch.where(total > 1e-8, shifted / total.clamp_min(1e-8),
                       mask / mask.sum(-1, keepdim=True).clamp_min(1))


def weights_to_flow_actions(weights, mask):
    """A joint-simplex inverse; Q must be invariant to score magnitude."""
    weights = weights.clamp_min(0) * mask
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    return torch.where(mask.bool(), 2 * weights - 1, -torch.ones_like(weights))


def flow_to_baseline_actions(actions, mask):
    """Adapt joint traded weights to the unchanged TD3 critic's input API.

    This is an internal coordinate adapter only. The environment and replay
    continue to use joint cash/asset scores. Both normalizers produce exactly
    the same traded weights, including their action Jacobian.
    """
    weights = portfolio_weights(actions, mask)
    risky = weights[..., 1:]
    risky_total = risky.sum(-1, keepdim=True)
    uniform = mask[..., 1:] / mask[..., 1:].sum(-1, keepdim=True).clamp_min(1)
    relative = torch.where(risky_total > 1e-8, risky / risky_total.clamp_min(1e-8), uniform)
    return torch.cat((2 * weights[..., :1] - 1,
                      torch.where(mask[..., 1:].bool(), 2 * relative - 1, -torch.ones_like(relative))), -1)


def portfolio_action_candidates(actions, mask, count=5, budget_std=.05, composition_std=.04):
    """Center, cash-budget neighbors, and risky-composition neighbors.

    As in the TD3 ProExp path, changing budget retains risky relative shares;
    changing composition retains cash. All paths remain differentiable with
    respect to the proposed action, and excluded assets never receive mass.
    """
    if count < 1:
        raise ValueError("FlowRL candidate count must be positive")
    center = portfolio_weights(actions, mask)
    relative = center[:, 1:] / center[:, 1:].sum(-1, keepdim=True).clamp_min(1e-8)
    uniform = mask[:, 1:] / mask[:, 1:].sum(-1, keepdim=True).clamp_min(1)
    relative = torch.where(center[:, 1:].sum(-1, keepdim=True) > 1e-8, relative, uniform)
    budget_count = max(1, min(count - 2, round((count - 1) * .34))) if count > 2 else 0
    composition_count = count - 1 - budget_count
    candidates, families = [actions[:, None]], [0]
    if budget_count:
        noise = torch.randn(actions.shape[0], budget_count, 1, device=actions.device, dtype=actions.dtype)
        cash = (center[:, None, :1] + noise.clamp(-2.5, 2.5) * budget_std).clamp(0, 1)
        weights = torch.cat((cash, (1 - cash) * relative[:, None]), -1)
        candidates.append(weights_to_flow_actions(weights, mask[:, None]))
        families.extend([1] * budget_count)
    if composition_count:
        noise = torch.randn(actions.shape[0], composition_count, actions.shape[1] - 1,
                            device=actions.device, dtype=actions.dtype).clamp(-2.5, 2.5) * composition_std
        shares = (relative[:, None] + noise).clamp_min(0) * mask[:, None, 1:]
        total = shares.sum(-1, keepdim=True)
        shares = torch.where(total > 1e-8, shares / total.clamp_min(1e-8), uniform[:, None])
        cash = center[:, None, :1].expand(-1, composition_count, -1)
        weights = torch.cat((cash, shares * (1 - cash)), -1)
        candidates.append(weights_to_flow_actions(weights, mask[:, None]))
        families.extend([2] * composition_count)
    return torch.cat(candidates, 1), torch.tensor(families, device=actions.device)


def mlp(input_dim, hidden, output_dim):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), nn.LayerNorm(width), nn.ELU()))
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class CashRiskBudgetHead(nn.Module):
    """A causal set-conditioned cash marginal shared across the risky sleeve.

    Absolute opportunity level, dispersion, held exposures and market/history
    context inform one cash-versus-risk decision. Zero output preserves the
    existing policy and Q exactly when this branch is attached on resume.
    """
    def __init__(self, model_dim, control_dim):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(2 * control_dim + 4 * model_dim + 6, model_dim),
                                  nn.SiLU(), nn.Linear(model_dim, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, local, relational, fused, pool_cls, mask, previous_weights,
                asset_scores, portfolio_context=None):
        active = mask.bool()
        count = mask.sum(-1, keepdim=True).clamp_min(1)
        def mean(tokens):
            return tokens.masked_fill(~active.unsqueeze(-1), 0).sum(1) / count
        scores = asset_scores.masked_fill(~active, 0)
        score_mean = scores.sum(-1, keepdim=True) / count
        score_variance = ((scores - score_mean).square() * mask).sum(-1, keepdim=True) / count
        highest = asset_scores.masked_fill(~active, -torch.inf).max(-1, keepdim=True).values
        lowest = asset_scores.masked_fill(~active, torch.inf).min(-1, keepdim=True).values
        nonempty = active.any(-1, keepdim=True)
        highest = torch.where(nonempty, highest, torch.zeros_like(highest))
        lowest = torch.where(nonempty, lowest, torch.zeros_like(lowest))
        held = previous_weights[:, 1:] * mask
        held_fused = (fused.masked_fill(~active.unsqueeze(-1), 0) * held.unsqueeze(-1)).sum(1)
        held_score = (scores * held).sum(-1, keepdim=True)
        context = torch.zeros_like(pool_cls) if portfolio_context is None else portfolio_context
        summary = torch.cat((mean(local), mean(relational), mean(fused), held_fused, pool_cls, context,
                             score_mean, score_variance, highest, lowest, held_score,
                             previous_weights[:, :1]), -1)
        return self.head(summary)


class PortfolioStateEncoder(nn.Module):
    """Version 2+ uses the baseline's identity, pool and availability paths.

    Every learner owns its encoder. Version 1 is retained solely to read
    existing checkpoints without silently changing their inference behavior.
    """
    def __init__(self, observation_space, model_dim=256, net_arch=(512, 384),
                 n_heads=8, n_layers=2, use_market_attention=False,
                 use_global_pooling=True, architecture_version=1):
        super().__init__()
        self.architecture_version = int(architecture_version)
        if self.architecture_version not in (1, 2, 3, 4, 5):
            raise ValueError("Unsupported FlowRL architecture version")
        self.max_coins = observation_space["mask"].shape[0]
        features = observation_space["observations"].shape[0] // self.max_coins
        local, position, comparative, context = _resolve_token_scope_dims(
            features, None, None, None, None)
        self.model_dim = model_dim
        self.features = features
        identity_dim = (int(config.GLOBAL_POOL_IDENTITY_DIM)
                        if self.architecture_version >= 2 and config.EARNMORE_GLOBAL_POOL_ENABLED else 0)
        self.asset_identity_embedding = (nn.Embedding(self.max_coins + 1, identity_dim,
                                                     padding_idx=self.max_coins) if identity_dim else None)
        self.identity_dim = identity_dim
        self.encoder = AttentionDeepSetEncoder(
            features_per_coin=features, local_feature_dim=local,
            position_feature_dim=position, comparative_feature_dim=comparative,
            context_feature_dim=context, model_dim=model_dim, net_arch=list(net_arch),
            n_heads=n_heads, n_layers=n_layers, dropout=0.0,
            use_market_attention=use_market_attention, use_global_pooling=use_global_pooling,
            asset_identity_dim=identity_dim,
            relation_uses_context=self.architecture_version >= 2,
            basket_context_dim=observation_space["basket_context"].shape[0] if "basket_context" in observation_space.spaces else 0,
            basket_membership_dim=observation_space["basket_membership"].shape[0] if "basket_membership" in observation_space.spaces else 0,
            fam_enabled="local_feature_history" in observation_space.spaces,
            fam_feature_dim=int(getattr(config, "MDAEN_FAM_FEATURE_COUNT", 0)),
            fam_history_length=int(getattr(config, "MDAEN_FAM_HISTORY_LENGTH", 8)),
            fam_kernel_sizes=tuple(getattr(config, "MDAEN_FAM_KERNEL_SIZES", (3, 5))),
            fam_asset_vector_dim=model_dim,
        )
        # FlowRL's Bellman/actor losses own these representations, even if the
        # unrelated TD3 foundation experiment is enabled in global config.
        self.encoder.strict_semantic_gradient_ownership = False
        self.history = PortfolioRewardHistoryEncoder(self.max_coins, model_dim)
        self.cash_token = nn.Parameter(torch.zeros(model_dim))
        self.book_projection = nn.Linear(1, model_dim)
        self.book_context = mlp(model_dim + 1, (model_dim,), model_dim)
        if self.architecture_version >= 2:
            self.task_mask_projection = nn.Linear(self.max_coins, model_dim, bias=False)
            self.task_pool_projection = nn.Linear(model_dim, model_dim, bias=False)
            self.task_norm = nn.LayerNorm(model_dim)

    def forward(self, obs):
        mask = obs["mask"].float()
        features = obs["observations"].float().reshape(-1, self.max_coins, self.features)
        # Erase excluded evidence before any learned computation.
        features = features.masked_fill(~mask.bool().unsqueeze(-1), 0)
        history = None
        if self.encoder.fam_enabled:
            history = torch.cat((obs["local_feature_history"].float(),
                                 obs["episode_feature_history"].float()), dim=-1)
            history = history.masked_fill(~mask.bool()[..., None, None], 0)
        if self.architecture_version >= 2:
            identity = (self.asset_identity_embedding(obs["asset_ids"].long())
                        if self.asset_identity_embedding is not None else None)
            components = self.encoder.encode_components(
                features, mask, asset_identity=identity, basket_context=obs.get("basket_context"),
                basket_membership=obs.get("basket_membership"), feature_history=history,
                feature_history_mask=obs.get("feature_history_mask"), control_only=True)
            tokens, pooled = components["fused"], components["pooled"]
            components["asset_identity"] = identity
        else:
            tokens, pooled, _ = self.encoder(
                features, mask, basket_context=obs.get("basket_context"),
                basket_membership=obs.get("basket_membership"),
                feature_history=history, feature_history_mask=obs.get("feature_history_mask"))
        portfolio_context = self.history(obs.get("portfolio_reward_history"), mask)
        pooled = pooled + portfolio_context
        if self.architecture_version >= 2:
            task = self.task_norm(self.task_mask_projection(mask)
                                  + self.task_pool_projection(components["pool_cls"]) + portfolio_context)
            pooled = pooled + task
            components.update(portfolio_context=portfolio_context, task=task,
                              asset_mask=mask, previous_weights=obs["prev_action"].float() * action_mask(obs))
        cash = pooled.unsqueeze(1) + self.cash_token
        tokens = torch.cat((cash, tokens), dim=1)
        previous = obs["prev_action"].float() * action_mask(obs)
        full_mask = action_mask(obs)
        book = self.book_context(torch.cat((tokens, previous.unsqueeze(-1)), -1))
        pooled = pooled + (book * full_mask.unsqueeze(-1)).sum(1) / full_mask.sum(1, keepdim=True)
        tokens = tokens + self.book_projection(previous.unsqueeze(-1))
        return ((tokens, pooled, full_mask, components) if self.architecture_version >= 2
                else (tokens, pooled, full_mask))


class FlowActor(nn.Module):
    def __init__(self, observation_space, flow_steps=1, **encoder_kwargs):
        super().__init__()
        if flow_steps < 1:
            raise ValueError("flow_steps must be positive")
        self.flow_steps = int(flow_steps)
        self.state_encoder = PortfolioStateEncoder(observation_space, **encoder_kwargs)
        width = self.state_encoder.model_dim
        hidden = encoder_kwargs.get("net_arch", (512, 384))
        self.noisy_action_encoder = mlp(width + 1, (width,), width)
        self.velocity_head = mlp(3 * width + 2, hidden, 1)
        self.architecture_version = self.state_encoder.architecture_version
        if self.architecture_version >= 2:
            head_class = JointPortfolioOpportunityHead if self.architecture_version >= 5 else BasketConditionedOpportunityHead
            self.basket_opportunity_head = head_class(
                self.state_encoder.max_coins, width, self.state_encoder.encoder.control_stream_dim)
            if self.architecture_version < 5:
                self.cash_opportunity_head = nn.Linear(width + 1, 1)
                nn.init.xavier_uniform_(self.cash_opportunity_head.weight, gain=0.1)
                nn.init.zeros_(self.cash_opportunity_head.bias)
        if self.architecture_version == 4:
            self.enable_cash_budget()

    def enable_cash_budget(self):
        if not hasattr(self, "cash_risk_budget_head"):
            reference = next(self.parameters())
            with torch.random.fork_rng(devices=[]):
                self.cash_risk_budget_head = CashRiskBudgetHead(
                    self.state_encoder.model_dim, self.state_encoder.encoder.control_stream_dim).to(reference)
        self.architecture_version = self.state_encoder.architecture_version = 4

    def _opportunity_velocity(self, components, pooled):
        if self.architecture_version >= 5:
            scores = self.basket_opportunity_head.joint_scores(
                components["local_control"], components["relational_control"], components["fused"],
                components["pool_cls"], components["asset_mask"], components["previous_weights"],
                portfolio_context=components["portfolio_context"])
            components["flow_cash_marginal_score"] = scores[:, :1]
            return scores
        scores = self.basket_opportunity_head(
            components["local_control"], components["relational_control"], components["fused"],
            components["pool_cls"], components["asset_mask"], components["previous_weights"],
            portfolio_context=components["portfolio_context"])
        cash = self.cash_opportunity_head(torch.cat((pooled, components["previous_weights"][:, :1]), -1))
        if self.architecture_version == 4:
            budget = self.cash_risk_budget_head(
                components["local_control"], components["relational_control"], components["fused"],
                components["pool_cls"], components["asset_mask"], components["previous_weights"],
                scores, components["portfolio_context"])
            # Positive budget transports toward cash and away from every
            # active risky coordinate together, before tanh/joint normalization.
            cash, scores = cash + budget, scores - budget
            components["flow_cash_budget_velocity"] = budget
        return torch.cat((cash, scores), -1)

    def encode_state(self, observations):
        """Share market evidence across flow steps and matching on one batch."""
        encoded = self.state_encoder(observations)
        if self.architecture_version >= 3:
            tokens, pooled, mask, components = encoded
            # This tensor remains differentiable. It depends on the market
            # and held book, never on flow time or the transported action.
            components["flow_opportunity_velocity"] = self._opportunity_velocity(components, pooled)
        return encoded

    def velocity(self, encoded, noisy_actions, time):
        tokens, pooled, mask = encoded[:3]
        noisy_actions = noisy_actions * mask
        joint = self.noisy_action_encoder(torch.cat((tokens, noisy_actions.unsqueeze(-1)), -1))
        joint = (joint * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        count = tokens.shape[1]
        inputs = torch.cat((tokens, pooled[:, None].expand(-1, count, -1),
                            joint[:, None].expand(-1, count, -1),
                            noisy_actions.unsqueeze(-1), time[:, None].expand(-1, count, -1)), -1)
        velocity = self.velocity_head(inputs).squeeze(-1)
        if self.architecture_version >= 2:
            c = encoded[3]
            opportunity = c.get("flow_opportunity_velocity")
            if opportunity is None:
                opportunity = self._opportunity_velocity(c, pooled)
            velocity = velocity + opportunity
        return velocity * mask

    def forward(self, observations, deterministic=False, noise=None, *, encoded=None):
        encoded = self.encode_state(observations) if encoded is None else encoded
        mask = encoded[2]
        if noise is None:
            noise = torch.zeros_like(mask) if deterministic else torch.randn_like(mask).clamp(-1, 1)
        x = noise * mask
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            time = x.new_full((x.shape[0], 1), step * dt)
            midpoint = x + 0.5 * dt * self.velocity(encoded, x, time)
            x = x + dt * self.velocity(encoded, midpoint, time + 0.5 * dt)
        # Match upstream tanh-bounded flow. Inactive raw actions are canonical -1.
        return torch.where(mask.bool(), torch.tanh(x), -torch.ones_like(x))

    def flow_matching_errors(self, observations, actions, noise=None, time=None, *, encoded=None):
        encoded = self.encode_state(observations) if encoded is None else encoded
        mask = encoded[2]
        noise = torch.randn_like(actions).clamp(-1, 1) if noise is None else noise
        time = torch.rand_like(actions[:, :1]) if time is None else time
        # Generation applies tanh after integration. Version 3 transports to
        # the inverse-squashed endpoint, so perfect matching reconstructs the
        # action actually recorded in replay rather than tanh(action) again.
        endpoint = torch.atanh(actions.clamp(-1 + 1e-4, 1 - 1e-4)) if self.architecture_version >= 3 else actions
        interpolated = (time * endpoint + (1 - time) * noise) * mask
        target = (endpoint - noise) * mask
        error = (self.velocity(encoded, interpolated, time) - target).square() * mask
        # Reduce coordinates only; retain one loss per replay transition.
        return error.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)


class PortfolioQ(nn.Module):
    def __init__(self, observation_space, **encoder_kwargs):
        super().__init__()
        self.state_encoder = PortfolioStateEncoder(observation_space, **encoder_kwargs)
        width = self.state_encoder.model_dim
        self.action_encoder = mlp(width + 1, (width,), width)
        self.head = mlp(2 * width, encoder_kwargs.get("net_arch", (512, 384)), 1)
        self.architecture_version = self.state_encoder.architecture_version
        if self.architecture_version >= 2:
            control_dim = self.state_encoder.encoder.control_stream_dim
            self.basket_opportunity_head = BasketConditionedOpportunityHead(
                self.state_encoder.max_coins, width, control_dim, zero_output=True)
            self.financial_return_head = mlp(control_dim + self.state_encoder.identity_dim, (width,), 1)
            nn.init.zeros_(self.financial_return_head[-1].weight)
            nn.init.zeros_(self.financial_return_head[-1].bias)

    def forward(self, obs, actions, return_components=False):
        encoded = self.state_encoder(obs)
        tokens, pooled, mask = encoded[:3]
        weights = portfolio_weights(actions, mask)
        joint = self.action_encoder(torch.cat((tokens, weights.unsqueeze(-1)), -1))
        joint = (joint * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        value = self.head(torch.cat((pooled, joint), -1))
        predictions = None
        if self.architecture_version >= 2:
            c = encoded[3]
            local = c["local_control"]
            if c["asset_identity"] is not None:
                local = torch.cat((local, c["asset_identity"]), -1)
            predictions = self.financial_return_head(local) * c["asset_mask"].unsqueeze(-1)
            scores = self.basket_opportunity_head(
                c["local_control"], c["relational_control"], c["fused"], c["pool_cls"],
                c["asset_mask"], c["previous_weights"], portfolio_context=c["portfolio_context"])
            delta = weights[:, 1:] - c["previous_weights"][:, 1:]
            marginal_value = (delta * scores).sum(-1, keepdim=True)
            # The local predictor learns only from causal unit-return labels.
            # Its detached values still provide a direct action gradient.
            financial_base = (weights[:, 1:] * predictions.detach().squeeze(-1)).sum(-1, keepdim=True)
            value = value + marginal_value + financial_base
        return (value, predictions) if return_components else value


class TwinPortfolioQ(nn.Module):
    def __init__(self, observation_space, **encoder_kwargs):
        super().__init__()
        self.q_networks = nn.ModuleList([PortfolioQ(observation_space, **encoder_kwargs) for _ in range(2)])

    def forward(self, obs, actions, return_components=False):
        outputs = tuple(q(obs, actions, return_components=return_components) for q in self.q_networks)
        if return_components:
            return tuple(value[0] for value in outputs), tuple(value[1] for value in outputs)
        return outputs


class FlowPortfolioCritic(DeepSetCritic):
    """The established portfolio critic, with joint-simplex action coordinates.

    Reuse the complete local/current/delta value path, allocation interaction,
    task conditioning, opportunity heads and local-plus-peer financial teacher.
    The small input adapter keeps TD3's existing geometry and checkpoints intact.
    """
    def __init__(self, observation_space, action_space, model_dim=256, net_arch=(512, 384),
                 n_heads=8, n_layers=2, use_market_attention=False, use_global_pooling=True,
                 architecture_version=3):
        assets = observation_space["mask"].shape[0]
        features = observation_space["observations"].shape[0] // assets
        local, position, comparative, context = _resolve_token_scope_dims(features, None, None, None, None)
        super().__init__(
            observation_space, action_space, max_coins=assets, features_per_coin=features,
            set_hidden_dim=model_dim, net_arch=list(net_arch), n_heads=n_heads, n_layers=n_layers,
            dropout=0.0, use_market_attention=use_market_attention, use_global_pooling=use_global_pooling,
            local_feature_dim=local, position_feature_dim=position,
            comparative_feature_dim=comparative, context_feature_dim=context,
            num_asset_ids=assets, asset_identity_dim=int(config.GLOBAL_POOL_IDENTITY_DIM),
            basket_context_dim=observation_space["basket_context"].shape[0] if "basket_context" in observation_space.spaces else 0,
            basket_membership_dim=observation_space["basket_membership"].shape[0] if "basket_membership" in observation_space.spaces else 0,
            component_outcome_aux_enabled=True, component_outcome_dim=1, component_outcome_unit_return_targets=True)
        self.architecture_version = int(architecture_version)
        for q in self.q_networks:
            q.token_encoder.strict_semantic_gradient_ownership = False
            if not q.local_financial_handoff_enabled or not q.basket_conditioned_allocation_enabled:
                raise ValueError("FlowRL v3 requires the causal basket-opportunity/local-financial portfolio configuration")
        if self.architecture_version >= 5:
            for q in self.q_networks:
                old = q.basket_opportunity_head
                with torch.random.fork_rng(devices=[]):
                    joint = JointPortfolioOpportunityHead(
                        assets, model_dim, q.token_encoder.control_stream_dim, zero_output=True)
                joint.load_state_dict(old.state_dict(), strict=False)
                q.basket_opportunity_head = joint
        elif self.architecture_version == 4:
            self.enable_cash_budget()

    def enable_cash_budget(self):
        for q in self.q_networks:
            opportunity = q.basket_opportunity_head
            if not hasattr(opportunity, "cash_marginal_head"):
                reference = next(q.parameters())
                with torch.random.fork_rng(devices=[]):
                    opportunity.cash_marginal_head = CashRiskBudgetHead(
                        opportunity.pool_projection.out_features, q.token_encoder.control_stream_dim).to(reference)
        self.architecture_version = 4

    def forward(self, observations, actions, return_components=False):
        adapted = flow_to_baseline_actions(actions, action_mask(observations))
        if return_components:
            return super().forward_with_component_outcomes(observations, adapted)
        return super().forward(observations, adapted)

    def financial_components(self, observations, actions):
        adapted = flow_to_baseline_actions(actions, action_mask(observations))
        return super().forward_with_component_outcomes(observations, adapted, return_financial_decomposition=True)

    def forward_with_component_outcomes(self, observations, actions, *, return_financial_decomposition=False):
        adapted = flow_to_baseline_actions(actions, action_mask(observations))
        return super().forward_with_component_outcomes(
            observations, adapted, return_financial_decomposition=return_financial_decomposition)

    def evaluate_action_candidates(self, observations, candidates):
        adapted = flow_to_baseline_actions(candidates, action_mask(observations)[:, None])
        return super().evaluate_action_candidates(observations, adapted)

    def forward_with_control_decomposition(self, observations, actions):
        adapted = flow_to_baseline_actions(actions, action_mask(observations))
        return super().forward_with_control_decomposition(observations, adapted)

    def q1_forward(self, observations, actions):
        adapted = flow_to_baseline_actions(actions, action_mask(observations))
        return super().q1_forward(observations, adapted)


class PortfolioValue(nn.Module):
    def __init__(self, observation_space, **encoder_kwargs):
        super().__init__()
        self.state_encoder = PortfolioStateEncoder(observation_space, **encoder_kwargs)
        self.head = mlp(self.state_encoder.model_dim, encoder_kwargs.get("net_arch", (512, 384)), 1)

    def forward(self, obs):
        return self.head(self.state_encoder(obs)[1])
