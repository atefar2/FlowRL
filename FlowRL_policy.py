"""SB3 policy container; version 1 defaults preserve legacy checkpoint loading.

New FlowRL learners explicitly request version 5: a complete portfolio critic,
consistent flow endpoints, and a separate slow flow actor for target evaluation.
Version 5 uses one marginal scorer for cash and all risky alternatives.
Versions 1 through 4 retain their original network shapes and inference behavior.
Adaptive CRL lives in the algorithm: new training differentiates final joint
portfolio weights through this actor using TD3's previous/current/next-target
path. New training restores October 29's shared initializer for the current and
target flow, each of which completes its own integration. Actor matching and
CRL sampling are saved separately from this joint marginal network. The target
actor also supplies next-state Bellman actions.
Dual multipliers are training state, not actor or critic network parameters.
"""
from copy import deepcopy
import numpy as np
import torch
from stable_baselines3.common.policies import BasePolicy

from FlowRL_model import FlowActor, TwinPortfolioQ, FlowPortfolioCritic, PortfolioValue


class FlowRLPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 model_dim=256, net_arch=(512, 384), n_heads=8, n_layers=2,
                 use_market_attention=False, use_global_pooling=True, flow_steps=1,
                 normalize_images=True, architecture_version=1):
        super().__init__(observation_space, action_space, squash_output=True, normalize_images=normalize_images)
        if not (np.all(action_space.low == -1) and np.all(action_space.high == 1)):
            raise ValueError("FlowRL portfolio actions require Box(-1, 1)")
        if action_space.shape != (observation_space["mask"].shape[0] + 1,):
            raise ValueError("FlowRL needs one cash coordinate plus one coordinate per asset slot")
        self.network_kwargs = dict(model_dim=model_dim, net_arch=tuple(net_arch), n_heads=n_heads,
                                   n_layers=n_layers, use_market_attention=use_market_attention,
                                   use_global_pooling=use_global_pooling, architecture_version=int(architecture_version))
        self.architecture_version = int(architecture_version)
        self.flow_steps = flow_steps
        self.actor = FlowActor(observation_space, flow_steps=flow_steps, **self.network_kwargs)
        critic_class = FlowPortfolioCritic if self.architecture_version >= 3 else TwinPortfolioQ
        critic_kwargs = dict(self.network_kwargs)
        if self.architecture_version >= 3:
            critic_kwargs["action_space"] = action_space
        self.critic = critic_class(observation_space, **critic_kwargs)
        self.behavior_critic = critic_class(observation_space, **critic_kwargs)
        self.value = PortfolioValue(observation_space, **self.network_kwargs)
        self.critic_target = deepcopy(self.critic).requires_grad_(False)
        self.behavior_critic_target = deepcopy(self.behavior_critic).requires_grad_(False)
        if self.architecture_version >= 3:
            self.actor_target = deepcopy(self.actor).requires_grad_(False)
        for name in ("actor", "critic", "behavior_critic", "value"):
            setattr(self, name + "_optimizer", torch.optim.Adam(getattr(self, name).parameters(), lr=lr_schedule(1)))

    def _predict(self, observation, deterministic=False):
        # Deterministic means zero initial latent, not the mean or MAP action.
        return self.actor(observation, deterministic=deterministic)

    def forward(self, observation, deterministic=False):
        return self._predict(observation, deterministic)

    def set_training_mode(self, mode):
        self.train(mode)
        self.critic_target.eval()
        self.behavior_critic_target.eval()
        if self.architecture_version >= 3:
            self.actor_target.eval()

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), **self.network_kwargs,
                "flow_steps": self.flow_steps, "lr_schedule": self._dummy_schedule}
