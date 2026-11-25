import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Type, Union, Optional
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.td3.policies import TD3Policy, Actor as TD3Actor
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.policies import ContinuousCritic
import config


def local_market_token_view(tokens, primary_dim, position_dim, comparative_dim):
    """Remove handcrafted basket fields without mutating replay tensors.

    Keep primary causal fields, portfolio position and the three asset-scale
    anchors. Zero reserved basket-scale and episode-relative slots. The caller
    owns the mode so a constructed policy has a fixed information contract.
    """
    keep = int(primary_dim) + int(position_dim) + min(3, int(comparative_dim))
    return torch.cat((tokens[..., :keep], torch.zeros_like(tokens[..., keep:])), dim=-1)


def expectile_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    expectile: float,
) -> torch.Tensor:
    """Per-sample asymmetric squared loss used for an in-sample state value.

    The target is expected to be detached by the caller.  Values above the
    prediction receive ``expectile`` weight; values below it receive
    ``1-expectile``.  An expectile above one half therefore summarizes the
    better replay actions without querying an unseen action.
    """
    expectile = float(expectile)
    if not 0.0 < expectile < 1.0:
        raise ValueError("expectile must lie strictly between zero and one")
    residual = target - prediction
    weight = torch.where(
        residual >= 0.0,
        residual.new_tensor(expectile),
        residual.new_tensor(1.0 - expectile),
    )
    return weight * residual.square()


def value_equivalent_bellman_backup(
    reward: torch.Tensor,
    done: torch.Tensor,
    continuation_value: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Construct the sampled Bellman quantity preserved by the latent model.

    The terminal transition keeps its observed reward but deliberately removes
    continuation value.  Keeping this operation in one small function makes
    the value-equivalence contract shared by model predictions and empirical
    targets, and makes terminal semantics independently testable.
    """
    if reward.shape != done.shape or reward.shape != continuation_value.shape:
        raise ValueError(
            "reward, done, and continuation_value must have identical shapes"
        )
    return reward + float(gamma) * (1.0 - done.clamp(0.0, 1.0)) * continuation_value


def bounded_advantage_weights(
    advantage: torch.Tensor,
    *,
    temperature: float,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Return detached mean-one AWR weights while preserving hard bounds.

    A scalar bisection finds the normalization multiplier *inside* the clamp.
    This avoids the common bug where clipping and then dividing by the mean
    silently violates the configured floor or ceiling.
    """
    temperature = float(temperature)
    minimum = float(minimum)
    maximum = float(maximum)
    if temperature < 0.0:
        raise ValueError("advantage temperature must be non-negative")
    if not 0.0 < minimum <= 1.0 <= maximum:
        raise ValueError("advantage bounds must satisfy 0 < min <= 1 <= max")
    with torch.no_grad():
        raw = torch.exp(
            (temperature * advantage.detach()).clamp(min=-20.0, max=20.0)
        )
        lower = raw.new_tensor(0.0)
        upper = raw.new_tensor(maximum / max(float(raw.min().item()), 1e-12))
        for _ in range(48):
            midpoint = (lower + upper) * 0.5
            mean = (raw * midpoint).clamp(minimum, maximum).mean()
            if float(mean.item()) < 1.0:
                lower = midpoint
            else:
                upper = midpoint
        weights = (raw * ((lower + upper) * 0.5)).clamp(minimum, maximum)
    return weights


class TwoHotSymlogHead(nn.Module):
    """Distributional scalar head with TD-MPC2-style symlog two-hot targets.

    Cross entropy over a bounded transformed support is materially better
    behaved than raw squared regression when one model sees quiet equities,
    leveraged products, rewards, and long-horizon values on different scales.
    Two adjacent bins preserve sub-bin information, including the very small
    financial increments used by this environment.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        bins: int = 101,
        symlog_bound: float = 2.0,
    ):
        super().__init__()
        input_dim = int(input_dim)
        bins = int(bins)
        symlog_bound = float(symlog_bound)
        if input_dim <= 0:
            raise ValueError("distributional head input_dim must be positive")
        if bins < 3 or bins % 2 == 0:
            raise ValueError("distributional head requires an odd bins >= 3")
        if symlog_bound <= 0.0:
            raise ValueError("symlog bound must be positive")
        self.bins = bins
        self.symlog_bound = symlog_bound
        self.logits = nn.Linear(input_dim, bins)
        nn.init.xavier_uniform_(self.logits.weight, gain=0.01)
        nn.init.zeros_(self.logits.bias)
        self.register_buffer(
            "support",
            torch.linspace(-symlog_bound, symlog_bound, bins),
        )

    @staticmethod
    def symlog(value: torch.Tensor) -> torch.Tensor:
        return torch.sign(value) * torch.log1p(value.abs())

    @staticmethod
    def symexp(value: torch.Tensor) -> torch.Tensor:
        return torch.sign(value) * torch.expm1(value.abs())

    def encode_target(self, target: torch.Tensor) -> torch.Tensor:
        transformed = self.symlog(target.float()).clamp(
            -self.symlog_bound, self.symlog_bound
        )
        transformed = transformed.squeeze(-1)
        coordinate = (
            (transformed + self.symlog_bound)
            * float(self.bins - 1)
            / (2.0 * self.symlog_bound)
        )
        lower = coordinate.floor().long().clamp(0, self.bins - 1)
        upper = (lower + 1).clamp(0, self.bins - 1)
        upper_weight = coordinate - lower.to(coordinate.dtype)
        lower_weight = 1.0 - upper_weight
        encoded = target.new_zeros(*transformed.shape, self.bins)
        encoded.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
        encoded.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
        return encoded

    def loss(self, hidden: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_distribution = self.encode_target(target).detach()
        log_probabilities = F.log_softmax(self.logits(hidden), dim=-1)
        return -(
            target_distribution * log_probabilities
        ).sum(dim=-1, keepdim=True)

    def mean(self, hidden: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(self.logits(hidden), dim=-1)
        transformed_mean = (
            probabilities * self.support.to(probabilities.dtype)
        ).sum(dim=-1, keepdim=True)
        return self.symexp(transformed_mean)


def _module_on_cuda(module: nn.Module) -> bool:
    try:
        first_param = next(module.parameters())
    except StopIteration:
        return False
    return first_param.device.type == "cuda"


def _torch_compile_settings() -> Dict[str, object]:
    return {
        "backend": str(getattr(config, "TORCH_COMPILE_BACKEND", "inductor")),
        "mode": str(getattr(config, "TORCH_COMPILE_MODE", "default")),
        "dynamic": bool(getattr(config, "TORCH_COMPILE_DYNAMIC", False)),
    }


def _maybe_compile_module(
    module: Optional[nn.Module],
    label: str,
    *,
    allow_cpu: bool = False,
    backend_override: Optional[str] = None,
    mode_override: Optional[str] = None,
    dynamic_override: Optional[bool] = None,
    fullgraph_override: Optional[bool] = None,
) -> bool:
    """Compile one stable region in place without changing its state_dict.

    ``nn.Module.compile`` preserves the module hierarchy and parameter names,
    unlike replacing a child with an ``OptimizedModule`` wrapper.  Production
    compilation remains CUDA-only; ``allow_cpu`` exists solely for the focused
    eager-backend verification test.
    """
    if module is None:
        return False
    if not bool(getattr(config, "TORCH_COMPILE_ENABLED", False)):
        return False
    if not allow_cpu and not _module_on_cuda(module):
        return False
    if getattr(module, "_torch_compile_region_enabled", False):
        return True
    compile_method = getattr(module, "compile", None)
    if not callable(compile_method):
        return False
    settings = _torch_compile_settings()
    if backend_override is not None:
        settings["backend"] = str(backend_override)
    if mode_override is not None:
        settings["mode"] = str(mode_override)
    if dynamic_override is not None:
        settings["dynamic"] = bool(dynamic_override)
    try:
        compile_method(
            **settings,
            fullgraph=bool(
                getattr(config, "TORCH_COMPILE_FULLGRAPH_REGIONS", True)
                if fullgraph_override is None
                else fullgraph_override
            ),
        )
        setattr(module, "_torch_compile_region_enabled", True)
        print(f"⚙️ torch.compile regional kernel enabled for {label}")
        return True
    except Exception as exc:
        print(f"⚠️ torch.compile regional kernel skipped for {label}: {exc}")
        return False


def _maybe_compile_method(
    module: nn.Module,
    method_name: str,
    label: str,
    *,
    mode_override: Optional[str] = None,
    dynamic_override: Optional[bool] = None,
    fullgraph: bool = False,
) -> None:
    compile_fn = getattr(torch, "compile", None)
    if (
        compile_fn is None
        or not bool(getattr(config, "TORCH_COMPILE_ENABLED", False))
        or not _module_on_cuda(module)
    ):
        return

    compiled_flag = f"_compiled_{method_name}"
    if getattr(module, compiled_flag, False):
        return

    method = getattr(module, method_name, None)
    if method is None or not callable(method):
        return

    settings = _torch_compile_settings()
    if mode_override is not None:
        settings["mode"] = str(mode_override)
    if dynamic_override is not None:
        settings["dynamic"] = bool(dynamic_override)
    try:
        setattr(
            module,
            method_name,
            compile_fn(method, fullgraph=bool(fullgraph), **settings),
        )
        setattr(module, compiled_flag, True)
        print(f"⚙️ torch.compile enabled for {label}.{method_name}")
    except Exception as exc:
        print(f"⚠️ torch.compile skipped for {label}.{method_name}: {exc}")


class BasketCurriculumValueModel(nn.Module):
    """Teacher-only distributional value model for basket curriculum pacing.

    This module is intentionally separate from the TD3 critic.  It learns from
    completed-trajectory return-to-go targets and never participates in actor,
    critic, ProExp, forecast-bridge, or CRL gradients.  The input is a compact
    causal set state.  A shared token encoder followed by masked mean/max
    pooling makes the estimate invariant to slot order and prevents ticker or
    exact-membership identity from serving as the competence shortcut.

    For each target the model emits monotone lower, median, and upper quantiles.
    The curriculum uses the lower quantile as a conservative competence value;
    the interval width remains available as an uncertainty diagnostic.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        target_count: int = 2,
        max_assets: int = 0,
        token_dim: int = 0,
        global_dim: int = 0,
        snapshot_token_dim: int = 0,
        history_length: int = 0,
        history_feature_dim: int = 0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.target_count = int(target_count)
        self.max_assets = int(max_assets)
        self.token_dim = int(token_dim)
        self.global_dim = int(global_dim)
        self.snapshot_token_dim = int(snapshot_token_dim)
        self.history_length = int(history_length)
        self.history_feature_dim = int(history_feature_dim)
        self.set_mode = (
            self.max_assets > 0 and self.token_dim > 0 and self.global_dim >= 0
        )
        if self.input_dim <= 0:
            raise ValueError("curriculum value input_dim must be positive")
        if self.target_count <= 0:
            raise ValueError("curriculum value target_count must be positive")

        hidden_dims = tuple(int(value) for value in hidden_dims)
        if not hidden_dims or any(value <= 0 for value in hidden_dims):
            raise ValueError("curriculum value hidden dimensions must be positive")
        if self.set_mode:
            expected = (
                self.max_assets * self.token_dim
                + self.max_assets
                + self.global_dim
            )
            if expected != self.input_dim:
                raise ValueError(
                    "set-value input contract mismatch: expected "
                    f"{expected}, received {self.input_dim}"
                )
            token_hidden = hidden_dims[0]
            history_width = self.history_length * self.history_feature_dim
            if (
                self.snapshot_token_dim <= 0
                or self.snapshot_token_dim + history_width != self.token_dim
                or self.global_dim < self.history_length
            ):
                raise ValueError("invalid curriculum temporal set-value contract")
            self.token_encoder = nn.Sequential(
                nn.LayerNorm(self.snapshot_token_dim),
                nn.Linear(self.snapshot_token_dim, token_hidden),
                nn.SiLU(),
                nn.Linear(token_hidden, token_hidden),
                nn.SiLU(),
            )
            if self.history_length > 0 and self.history_feature_dim > 0:
                self.temporal_feature_attention = CausalFeatureAttentionModule(
                    feature_dim=self.history_feature_dim,
                    kernel_sizes=tuple(
                        getattr(config, "MDAEN_FAM_KERNEL_SIZES", (3, 5))
                    ),
                )
                self.temporal_asset_embedding = SharedAssetMapEmbedding(
                    history_length=self.history_length,
                    map_feature_dim=(
                        len(
                            tuple(
                                getattr(config, "MDAEN_FAM_KERNEL_SIZES", (3, 5))
                            )
                        )
                        * self.history_feature_dim
                    ),
                    asset_vector_dim=token_hidden,
                )
            else:
                self.temporal_feature_attention = None
                self.temporal_asset_embedding = None
            self.snapshot_residual_scale = nn.Parameter(
                torch.tensor(
                    float(
                        getattr(
                            config,
                            "MDAEN_POST_AAM_SNAPSHOT_INITIAL_RESIDUAL_SCALE",
                            0.05,
                        )
                    )
                )
            )
            self.temporal_token_norm = nn.LayerNorm(token_hidden)
            self.value_global_state_dim = self.global_dim - self.history_length
            self.global_encoder = (
                nn.Sequential(
                    nn.LayerNorm(self.value_global_state_dim),
                    nn.Linear(self.value_global_state_dim, token_hidden),
                    nn.SiLU(),
                )
                if self.value_global_state_dim > 0
                else None
            )
            layers: List[nn.Module] = [nn.LayerNorm(3 * token_hidden)]
            last_dim = 3 * token_hidden
            remaining_hidden_dims = hidden_dims[1:]
        else:
            self.token_encoder = None
            self.temporal_feature_attention = None
            self.temporal_asset_embedding = None
            self.snapshot_residual_scale = None
            self.temporal_token_norm = None
            self.value_global_state_dim = self.global_dim
            self.global_encoder = None
            layers = [nn.LayerNorm(self.input_dim)]
            last_dim = self.input_dim
            remaining_hidden_dims = hidden_dims
        for hidden_dim in remaining_hidden_dims:
            hidden_dim = int(hidden_dim)
            linear = nn.Linear(last_dim, hidden_dim)
            nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.SiLU()))
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.quantile_head = nn.Linear(last_dim, self.target_count * 3)
        nn.init.orthogonal_(self.quantile_head.weight, gain=0.05)
        nn.init.zeros_(self.quantile_head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                "curriculum value feature width mismatch: expected "
                f"{self.input_dim}, received {features.shape[-1]}"
            )
        if self.set_mode:
            token_width = self.max_assets * self.token_dim
            tokens = features[:, :token_width].view(
                features.shape[0], self.max_assets, self.token_dim
            )
            mask_start = token_width
            mask = features[:, mask_start:mask_start + self.max_assets].float()
            global_features = features[:, mask_start + self.max_assets:]
            snapshot_tokens = tokens[..., :self.snapshot_token_dim]
            history = None
            history_mask = None
            if self.history_length > 0:
                history = tokens[..., self.snapshot_token_dim:].view(
                    features.shape[0],
                    self.max_assets,
                    self.history_length,
                    self.history_feature_dim,
                )
                history_mask = global_features[:, -self.history_length:]
            global_state = global_features[:, :self.value_global_state_dim]
            snapshot_encoded = self.token_encoder(snapshot_tokens)
            if self.temporal_feature_attention is not None:
                temporal_map, _ = self.temporal_feature_attention(
                    history,
                    mask,
                    history_mask,
                )
                temporal_encoded = self.temporal_asset_embedding(temporal_map, mask)
                snapshot_scale = torch.clamp(
                    self.snapshot_residual_scale,
                    min=0.0,
                    max=float(
                        getattr(
                            config,
                            "MDAEN_POST_AAM_SNAPSHOT_MAX_RESIDUAL_SCALE",
                            0.25,
                        )
                    ),
                ).to(snapshot_encoded.dtype)
                encoded = self.temporal_token_norm(
                    temporal_encoded + snapshot_scale * snapshot_encoded
                ) * mask.unsqueeze(-1)
            else:
                encoded = self.temporal_token_norm(snapshot_encoded) * mask.unsqueeze(-1)
            denominator = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled_mean = encoded.sum(dim=1) / denominator
            masked_for_max = encoded.masked_fill(
                ~mask.bool().unsqueeze(-1), -torch.inf
            )
            pooled_max = masked_for_max.max(dim=1).values
            pooled_max = torch.where(
                torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max)
            )
            global_embedding = (
                self.global_encoder(global_state)
                if self.global_encoder is not None
                else torch.zeros_like(pooled_mean)
            )
            backbone_input = torch.cat(
                (pooled_mean, pooled_max, global_embedding), dim=-1
            )
        else:
            backbone_input = features
        raw = self.quantile_head(self.backbone(backbone_input)).view(
            features.shape[0], self.target_count, 3
        )
        median = raw[..., 1]
        lower = median - F.softplus(raw[..., 0])
        upper = median + F.softplus(raw[..., 2])
        return torch.stack((lower, median, upper), dim=-1)

    @staticmethod
    def quantile_huber_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantiles: Tuple[float, float, float] = (0.2, 0.5, 0.8),
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Huber-smoothed pinball loss for monotone three-quantile outputs."""
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        if predictions.shape[:-1] != targets.shape:
            raise ValueError(
                "curriculum value target shape mismatch: predictions "
                f"{tuple(predictions.shape)}, targets {tuple(targets.shape)}"
            )
        quantile_tensor = predictions.new_tensor(quantiles).view(1, 1, 3)
        errors = targets.unsqueeze(-1) - predictions
        abs_errors = errors.abs()
        beta = max(float(beta), 1e-6)
        huber = torch.where(
            abs_errors <= beta,
            0.5 * errors.pow(2) / beta,
            abs_errors - 0.5 * beta,
        )
        weights = torch.abs(
            quantile_tensor - (errors.detach() < 0.0).to(predictions.dtype)
        )
        return (weights * huber).mean()


class VariableCoinAttentionExtractor(BaseFeaturesExtractor):
    """
    Advanced attention extractor for variable portfolio sizes with masking.

    Multi-head attention: Yes. Two places. (1) TransformerEncoder is a stack of
    TransformerEncoderLayers; each layer is multi-head self-attention (Q,K,V all
    from the same sequence of coin embeddings) plus a feed-forward block. (2)
    market_attention is a separate nn.MultiheadAttention: again Q,K,V all from
    the transformed sequence (transformed, transformed, transformed). So each
    position attends over all (unmasked) positions; heads split the embedding
    dimension for parallel attention subspaces.

    Relation to the other extractors: Takes from CoinAttention the coin-wise
    structure (reshape to [batch, max_coins, features_per_coin], per-coin
    projection, transformer + market attention, masking). Takes from MultiHead the
    use of stacked multi-head layers. It is distinct in that it preserves
    per-slot output (flatten only, no projection to a single features_dim) and
    is Dict+mask-first for variable portfolios.

    Handles Dict observations with 'observations' and 'mask' keys and applies
    proper attention masking to ignore inactive coins.

    Optional use_global_pooling: when True, we compute a masked mean over the
    active coins, pass it through a small MLP (global_pooling), and
    concatenate [flattened_slots | pooled_mean]. For variable portfolios, a
    mean summary is a better regime descriptor than a max-pooled "most extreme
    asset" summary.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        max_coins: int = 5,
        features_per_coin: int = 14,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_market_attention: bool = True,
        use_global_pooling: bool = False,
    ):
        # Calculate total features for the Dict observation space
        total_features = observation_space['observations'].shape[0] + observation_space['mask'].shape[0]
        # Initialize with provisional features_dim; we will override to slot-aware combined dim below
        super().__init__(observation_space, features_dim)
        
        self.max_coins = max_coins
        self.features_per_coin = features_per_coin
        self.d_model = features_dim // 2  # Use half for coin embeddings
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_market_attention = use_market_attention
        self.use_global_pooling = use_global_pooling

        # Project each coin's features to d_model
        self.coin_projection = nn.Linear(self.features_per_coin, self.d_model)

        # Learnable coin position embeddings (different from coin type embeddings)
        self.position_embeddings = nn.Parameter(torch.randn(max_coins, self.d_model))

        # Transformer layers with proper masking support (each layer = multi-head self-attention + FFN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Optional second multi-head attention over coins (cross-asset context); skip for faster training
        self.market_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        ) if use_market_attention else None

        # Slot-aware output: flatten masked coin embeddings; optionally add global summaries
        self.flatten_dim = max_coins * self.d_model
        self.output_projection = nn.Identity()

        # Optional: learned global context from the masked mean over coins.
        if use_global_pooling:
            self.global_pooling = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self._features_dim = self.flatten_dim + self.d_model  # flattened + mean_pooled
        else:
            self.global_pooling = None
            self._features_dim = self.flatten_dim

    @property
    def features_dim(self) -> int:
        """Output size of forward(); must match so actor/critic get correct input dim."""
        return self._features_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Extract observations and mask
        obs = observations['observations']  # [batch_size, max_coins * features_per_coin]
        mask = observations['mask']  # [batch_size, max_coins]
        
        batch_size = obs.shape[0]
        
        # Reshape observations to [batch_size, max_coins, features_per_coin]
        coin_features = obs.view(batch_size, self.max_coins, self.features_per_coin)
        
        # Project each coin's features to d_model
        coin_embeddings = self.coin_projection(coin_features)  # [batch_size, max_coins, d_model]
        
        # Add position embeddings
        coin_embeddings = coin_embeddings + self.position_embeddings.unsqueeze(0)
        
        # Create attention mask for transformer (True = ignore, False = attend)
        # Transformer expects True for positions to ignore
        attention_mask = ~mask.bool()  # [batch_size, max_coins]
        
        # Apply transformer with masking
        # Note: src_key_padding_mask masks out entire sequences (coins)
        # Suppress nested tensor warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*nested tensors.*")
            transformed = self.transformer(
                coin_embeddings, 
                src_key_padding_mask=attention_mask
            )  # [batch_size, max_coins, d_model]
        
        # Optional second multi-head attention (Q,K,V = transformed) for cross-coin context
        if self.market_attention is not None:
            market_context, _ = self.market_attention(
                transformed, transformed, transformed,
                key_padding_mask=attention_mask
            )
            enhanced_features = transformed + market_context
        else:
            enhanced_features = transformed

        # Mask out inactive coins then flatten (slot-preserving)
        mask_expanded = mask.unsqueeze(-1).expand_as(enhanced_features)  # [batch_size, max_coins, d_model]
        masked_features = enhanced_features * mask_expanded
        flattened = masked_features.reshape(batch_size, self.flatten_dim)

        if self.global_pooling is not None:
            # Mean over (masked) coins, then learned summary; concat with flattened for global context
            valid_coins = mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean_pooled = masked_features.sum(dim=1) / valid_coins  # [batch_size, d_model]
            mean_summary = self.global_pooling(mean_pooled)
            flattened = torch.cat([flattened, mean_summary], dim=1)

        # Debug attention output
        if torch.rand(1).item() < 0.001:  # 0.1% of the time
            with torch.no_grad():
                print(f"🔍 Attention Debug:")
                print(f"   Masked features shape: {masked_features.shape}")
                print(f"   Flattened shape: {flattened.shape}")
                print(f"   Expected flatten_dim: {self.flatten_dim}")
                print(f"   First coin features: {masked_features[0, 0, :5].detach().cpu().numpy()}")
                print(f"   Second coin features: {masked_features[0, 1, :5].detach().cpu().numpy()}")
                print(f"   Third coin features: {masked_features[0, 2, :5].detach().cpu().numpy()}")
                print(f"   Flattened first 10: {flattened[0, :10].detach().cpu().numpy()}")
        
        return flattened


def _build_mlp_layers(input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Sequential:
    """Build a simple MLP used by the Deep Sets actor/critic blocks."""
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
        ])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def normalize_raw_actions_to_portfolio_weights(actions: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Match the environment's hierarchical action normalization.

    The policy emits raw logits in [-1, 1]. The environment first decides cash,
    then normalizes asset logits relative to one another inside the active asset
    subset. The critic must value those traded weights, not the pre-normalized
    logits, otherwise it learns Q(s, a_raw) instead of Q(s, w(a_raw)).
    """
    actions = actions.float()
    mask = mask.float()

    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    if mask.shape[1] == actions.shape[1]:
        asset_mask = mask[:, 1:]
    elif mask.shape[1] == actions.shape[1] - 1:
        asset_mask = mask
    else:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} is incompatible with actions shape {tuple(actions.shape)}."
        )

    cash_logits = actions[:, 0:1]
    asset_logits = actions[:, 1:]

    cash_weight = (cash_logits + 1.0) * 0.5
    asset_shifted = (asset_logits + 1.0) * 0.5
    asset_masked = asset_shifted * asset_mask
    asset_sum = asset_masked.sum(dim=1, keepdim=True)
    uniform_active = asset_mask / asset_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    asset_weights_relative = torch.where(
        asset_sum > eps, asset_masked / asset_sum.clamp(min=eps), uniform_active
    )
    equity_allocation = 1.0 - cash_weight
    final_asset_weights = asset_weights_relative * equity_allocation

    return torch.cat([cash_weight, final_asset_weights], dim=1)


class MaskedTemporalPool(nn.Module):
    """Learn which observed history rows matter instead of hand-picking three.

    The pool is shared across assets, so permuting asset slots permutes the
    resulting asset representations.  It is deliberately small: QWM needs the
    temporal information already present in replay, not a second full policy
    backbone inside every tree node.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.score = nn.Linear(int(feature_dim), 1, bias=False)

    def forward(
        self, history: torch.Tensor, history_mask: torch.Tensor
    ) -> torch.Tensor:
        if history.ndim != 4:
            raise ValueError("QWM history must be [batch, asset, time, feature]")
        if history_mask.ndim != 2:
            raise ValueError("QWM history mask must be [batch, time]")
        scores = self.score(history).squeeze(-1)
        valid = history_mask[:, None, :].to(dtype=torch.bool)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * valid.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return (history * weights.unsqueeze(-1)).sum(dim=2)


class PortfolioWorldModel(nn.Module):
    """Stochastic, permutation-equivariant residual model for QWM search.

    The model consumes the same dictionary observation used by the production
    actor and critics.  It predicts only fields that can change within an
    episode, copies basket identity/masks exactly, and constructs the next
    ``prev_action`` with the environment's allocation normalization.  It is a
    decision-time model: its predictions are never replay rows or TD targets.

    A conditional latent variable represents the part of the next market state
    that is not determined by the causal observation and candidate allocation.
    During fitting a posterior sees the realized transition; during search the
    learned prior produces K coherent alternatives for the same state-action
    pair.  One latent is shared by every active asset and by the reward head, so
    each sample is a joint market/portfolio scenario rather than independent
    feature-wise noise.
    """

    required_keys = (
        "observations",
        "mask",
        "prev_action",
        "prev_mask",
        "basket_context",
        "local_feature_history",
        "episode_feature_history",
        "feature_history_mask",
    )

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_dim: int,
        hidden_dim: int = 256,
        max_normalized_residual: float = 0.50,
        latent_dim: int = 32,
        kl_weight: float = 1e-3,
        kl_free_nats: float = 0.02,
        reward_target_scale: float = 100.0,
        antithetic_samples: bool = True,
        prior_reconstruction_weight: float = 1.0,
        controlled_observation_indices: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("PortfolioWorldModel requires a Dict observation space")
        missing = [key for key in self.required_keys if key not in observation_space.spaces]
        if missing:
            raise KeyError(f"PortfolioWorldModel observation keys missing: {missing}")

        self.action_dim = int(action_dim)
        self.max_coins = int(observation_space["mask"].shape[0])
        if self.action_dim != self.max_coins + 1:
            raise ValueError(
                "QWM action width must be cash plus one coordinate per asset: "
                f"received {self.action_dim}, expected {self.max_coins + 1}"
            )
        flat_observation_dim = int(observation_space["observations"].shape[0])
        if flat_observation_dim % self.max_coins != 0:
            raise ValueError("Flat portfolio observation is not divisible by max_coins")
        self.features_per_coin = flat_observation_dim // self.max_coins

        local_shape = observation_space["local_feature_history"].shape
        episode_shape = observation_space["episode_feature_history"].shape
        if local_shape[:2] != episode_shape[:2] or local_shape[0] != self.max_coins:
            raise ValueError("Local and episode histories must share asset/time axes")
        self.history_length = int(local_shape[1])
        self.local_feature_dim = int(local_shape[2])
        self.episode_feature_dim = int(episode_shape[2])
        self.basket_context_dim = int(observation_space["basket_context"].shape[0])
        self.hidden_dim = int(hidden_dim)
        self.max_normalized_residual = float(max_normalized_residual)
        self.latent_dim = int(latent_dim)
        self.kl_weight = float(kl_weight)
        self.kl_free_nats = float(kl_free_nats)
        self.reward_target_scale = float(reward_target_scale)
        self.antithetic_samples = bool(antithetic_samples)
        self.prior_reconstruction_weight = float(prior_reconstruction_weight)
        if self.hidden_dim <= 0:
            raise ValueError("QWM hidden_dim must be positive")
        if self.latent_dim <= 0:
            raise ValueError("QWM latent_dim must be positive")
        if not 0.0 < self.max_normalized_residual <= 2.0:
            raise ValueError("QWM max_normalized_residual must be in (0, 2]")
        if self.kl_weight < 0.0 or self.kl_free_nats < 0.0:
            raise ValueError("QWM KL settings must be non-negative")
        if not math.isfinite(self.reward_target_scale) or self.reward_target_scale <= 0.0:
            raise ValueError("QWM reward_target_scale must be finite and positive")
        if self.prior_reconstruction_weight < 0.0:
            raise ValueError("QWM prior reconstruction weight must be non-negative")

        if controlled_observation_indices is None:
            configured_index = int(
                getattr(config, "COST_BASIS_TOKEN_INDEX", -1)
            )
            controlled_observation_indices = (
                (configured_index,)
                if 0 <= configured_index < self.features_per_coin
                else ()
            )
        controlled_indices = tuple(
            sorted(
                {
                    int(index)
                    for index in controlled_observation_indices
                    if 0 <= int(index) < self.features_per_coin
                }
            )
        )
        controlled_feature_mask = torch.zeros(
            self.features_per_coin, dtype=torch.float32
        )
        if controlled_indices:
            controlled_feature_mask[list(controlled_indices)] = 1.0
        self.controlled_observation_indices = controlled_indices
        self.register_buffer(
            "controlled_observation_feature_mask", controlled_feature_mask
        )

        observation_low = torch.as_tensor(
            observation_space["observations"].low, dtype=torch.float32
        ).reshape(self.max_coins, self.features_per_coin)
        observation_high = torch.as_tensor(
            observation_space["observations"].high, dtype=torch.float32
        ).reshape(self.max_coins, self.features_per_coin)
        context_low = torch.as_tensor(
            observation_space["basket_context"].low, dtype=torch.float32
        )
        context_high = torch.as_tensor(
            observation_space["basket_context"].high, dtype=torch.float32
        )
        self.register_buffer("observation_mid", 0.5 * (observation_low + observation_high))
        self.register_buffer(
            "observation_scale",
            (0.5 * (observation_high - observation_low)).clamp(min=1e-3),
        )
        self.register_buffer("context_mid", 0.5 * (context_low + context_high))
        self.register_buffer(
            "context_scale",
            (0.5 * (context_high - context_low)).clamp(min=1e-3),
        )
        history_bound = max(
            abs(float(observation_space["local_feature_history"].low.min())),
            abs(float(observation_space["local_feature_history"].high.max())),
            abs(float(observation_space["episode_feature_history"].low.min())),
            abs(float(observation_space["episode_feature_history"].high.max())),
            1.0,
        )
        self.register_buffer("history_scale", torch.tensor(history_bound, dtype=torch.float32))

        self.local_temporal_pool = MaskedTemporalPool(self.local_feature_dim)
        self.episode_temporal_pool = MaskedTemporalPool(self.episode_feature_dim)
        token_input_dim = (
            self.features_per_coin
            + 4 * self.local_feature_dim  # latest, mean, trend, learned pool
            + 4 * self.episode_feature_dim
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(token_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        global_input_dim = self.hidden_dim + self.basket_context_dim
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.prior_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.latent_dim),
        )
        action_token_input_dim = self.hidden_dim + 4 + self.basket_context_dim
        self.action_token_encoder = nn.Sequential(
            nn.Linear(action_token_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        action_global_input_dim = 2 * self.hidden_dim + 2
        self.action_global_encoder = nn.Sequential(
            nn.Linear(action_global_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        posterior_token_input_dim = (
            self.features_per_coin
            + self.local_feature_dim
            + self.episode_feature_dim
        )
        self.posterior_token_encoder = nn.Sequential(
            nn.Linear(posterior_token_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        posterior_input_dim = 2 * self.hidden_dim + self.basket_context_dim
        self.posterior_head = nn.Sequential(
            nn.Linear(posterior_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.latent_dim),
        )

        transition_input_dim = 2 * self.hidden_dim + self.latent_dim
        self.observation_delta = nn.Linear(transition_input_dim, self.features_per_coin)
        self.controlled_observation_delta = nn.Linear(
            transition_input_dim, self.features_per_coin
        )
        self.local_delta = nn.Linear(transition_input_dim, self.local_feature_dim)
        self.episode_delta = nn.Linear(transition_input_dim, self.episode_feature_dim)
        global_transition_dim = self.hidden_dim + self.latent_dim
        self.context_delta = nn.Linear(global_transition_dim, self.basket_context_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(global_transition_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        # An identity transition is the safest state prior before supervised
        # replay fitting.  The reward prediction starts at zero.
        for head in (
            self.observation_delta,
            self.controlled_observation_delta,
            self.local_delta,
            self.episode_delta,
            self.context_delta,
            self.reward_head[-1],
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        # Start both latent distributions at N(0, I). Reconstruction teaches
        # the posterior which innovations matter; direct prior reconstruction
        # plus KL makes the distribution used by search learn those dynamics.
        for distribution_head in (self.prior_head[-1], self.posterior_head[-1]):
            nn.init.zeros_(distribution_head.weight)
            nn.init.zeros_(distribution_head.bias)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float().unsqueeze(-1)
        return (values * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def _normalized_observation_tokens(self, observation: torch.Tensor) -> torch.Tensor:
        tokens = observation.float().reshape(
            observation.shape[0], self.max_coins, self.features_per_coin
        )
        return (tokens - self.observation_mid) / self.observation_scale

    def _normalized_context(self, context: torch.Tensor) -> torch.Tensor:
        return (context.float() - self.context_mid) / self.context_scale

    def _prepare_state(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the action-independent half of world-model encoding once."""
        mask = observations["mask"].float()
        previous_weights = observations["prev_action"].float()
        observation_tokens = self._normalized_observation_tokens(
            observations["observations"]
        )
        local_history = observations["local_feature_history"].float() / self.history_scale
        episode_history = observations["episode_feature_history"].float() / self.history_scale
        temporal_mask = observations["feature_history_mask"].float()[:, None, :, None]
        temporal_count = temporal_mask.sum(dim=2).clamp(min=1.0)
        local_latest = local_history[:, :, -1, :]
        episode_latest = episode_history[:, :, -1, :]
        local_mean = (local_history * temporal_mask).sum(dim=2) / temporal_count
        episode_mean = (episode_history * temporal_mask).sum(dim=2) / temporal_count
        local_trend = local_latest - local_mean
        episode_trend = episode_latest - episode_mean
        local_learned = self.local_temporal_pool(
            local_history, observations["feature_history_mask"].float()
        )
        episode_learned = self.episode_temporal_pool(
            episode_history, observations["feature_history_mask"].float()
        )
        context = self._normalized_context(observations["basket_context"])
        repeated_context = context.unsqueeze(1).expand(-1, self.max_coins, -1)
        previous_cash = previous_weights[:, :1].unsqueeze(1).expand(
            -1, self.max_coins, -1
        )
        state_token_input = torch.cat(
            (
                observation_tokens,
                local_latest,
                local_mean,
                local_trend,
                local_learned,
                episode_latest,
                episode_mean,
                episode_trend,
                episode_learned,
            ),
            dim=-1,
        )
        return {
            "mask": mask,
            "previous_weights": previous_weights,
            "observation_tokens": observation_tokens,
            "context": context,
            "repeated_context": repeated_context,
            "previous_cash": previous_cash,
            "state_token_input": state_token_input,
        }

    @staticmethod
    def _repeat_prepared_state(
        prepared: Dict[str, torch.Tensor], repeats: int
    ) -> Dict[str, torch.Tensor]:
        return {
            key: value.unsqueeze(1).expand(
                value.shape[0], repeats, *value.shape[1:]
            ).reshape(value.shape[0] * repeats, *value.shape[1:])
            for key, value in prepared.items()
        }

    def _encode_prepared(
        self,
        prepared: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            market_token_hidden,
            market_transition_hidden,
            market_global_hidden,
        ) = self._encode_market_prepared(prepared)
        (
            action_transition_hidden,
            action_global_hidden,
            action_weights,
        ) = self._encode_actions_from_market(
            prepared,
            actions,
            market_token_hidden,
            market_global_hidden,
        )
        return (
            market_transition_hidden,
            action_transition_hidden,
            market_global_hidden,
            action_global_hidden,
            action_weights,
            prepared["observation_tokens"],
            prepared["context"],
        )

    def _encode_market_prepared(
        self, prepared: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode action-independent market state once per tree node."""
        mask = prepared["mask"]
        context = prepared["context"]
        market_token_hidden = self.token_encoder(
            prepared["state_token_input"]
        ) * mask.unsqueeze(-1)
        market_pooled = self._masked_mean(market_token_hidden, mask)
        market_global_hidden = self.global_encoder(
            torch.cat((market_pooled, context), dim=-1)
        )
        market_transition_hidden = torch.cat(
            (
                market_token_hidden,
                market_global_hidden.unsqueeze(1).expand(-1, self.max_coins, -1),
            ),
            dim=-1,
        )
        return (
            market_token_hidden,
            market_transition_hidden,
            market_global_hidden,
        )

    def _encode_actions_from_market(
        self,
        prepared: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        market_token_hidden: torch.Tensor,
        market_global_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attach candidate allocation effects to a cached market encoding."""
        mask = prepared["mask"]
        previous_weights = prepared["previous_weights"]
        action_weights = normalize_raw_actions_to_portfolio_weights(actions, mask)
        candidate_cash = action_weights[:, :1].unsqueeze(1).expand(
            -1, self.max_coins, -1
        )
        action_token_input = torch.cat(
            (
                market_token_hidden,
                action_weights[:, 1:].unsqueeze(-1),
                previous_weights[:, 1:].unsqueeze(-1),
                candidate_cash,
                prepared["previous_cash"],
                prepared["repeated_context"],
            ),
            dim=-1,
        )
        action_token_hidden = self.action_token_encoder(
            action_token_input
        ) * mask.unsqueeze(-1)
        action_pooled = self._masked_mean(action_token_hidden, mask)
        action_global_hidden = self.action_global_encoder(
            torch.cat(
                (
                    action_pooled,
                    market_global_hidden,
                    action_weights[:, :1],
                    previous_weights[:, :1],
                ),
                dim=-1,
            )
        )
        action_transition_hidden = torch.cat(
            (
                action_token_hidden,
                action_global_hidden.unsqueeze(1).expand(-1, self.max_coins, -1),
            ),
            dim=-1,
        )
        return action_transition_hidden, action_global_hidden, action_weights

    def _encode(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return self._encode_prepared(
            self._prepare_state(observations), actions
        )

    @staticmethod
    def _latent_parameters(parameters: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_variance = parameters.chunk(2, dim=-1)
        return mean, log_variance.clamp(min=-8.0, max=4.0)

    def _prior_parameters(
        self, global_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._latent_parameters(self.prior_head(global_hidden))

    def _posterior_parameters(
        self,
        observations: Dict[str, torch.Tensor],
        next_observations: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        market_global_hidden: torch.Tensor,
        observation_tokens: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer the realized innovation without leaking it into search."""
        mask = observations["mask"].float()
        target_observation = self._normalized_observation_tokens(
            next_observations["observations"]
        )
        current_local = (
            observations["local_feature_history"].float()[:, :, -1, :]
            / self.history_scale
        )
        target_local = (
            next_observations["local_feature_history"].float()[:, :, -1, :]
            / self.history_scale
        )
        current_episode = (
            observations["episode_feature_history"].float()[:, :, -1, :]
            / self.history_scale
        )
        target_episode = (
            next_observations["episode_feature_history"].float()[:, :, -1, :]
            / self.history_scale
        )
        observation_delta = target_observation - observation_tokens
        if self.controlled_observation_indices:
            observation_delta = observation_delta * (
                1.0 - self.controlled_observation_feature_mask
            )
        realized_token_delta = torch.cat(
            (
                observation_delta,
                target_local - current_local,
                target_episode - current_episode,
            ),
            dim=-1,
        )
        future_hidden = self.posterior_token_encoder(realized_token_delta)
        future_hidden = future_hidden * mask.unsqueeze(-1)
        future_pooled = self._masked_mean(future_hidden, mask)
        target_context = self._normalized_context(
            next_observations["basket_context"]
        )
        posterior_input = torch.cat(
            (
                market_global_hidden,
                future_pooled,
                target_context - context,
            ),
            dim=-1,
        )
        return self._latent_parameters(self.posterior_head(posterior_input))

    @staticmethod
    def _expand_samples(value: torch.Tensor, samples: int) -> torch.Tensor:
        return value.unsqueeze(1).expand(
            value.shape[0], int(samples), *value.shape[1:]
        )

    def _decode_latents(
        self,
        observations: Dict[str, torch.Tensor],
        market_transition_hidden: torch.Tensor,
        action_transition_hidden: torch.Tensor,
        market_global_hidden: torch.Tensor,
        action_global_hidden: torch.Tensor,
        action_weights: torch.Tensor,
        observation_tokens: torch.Tensor,
        context: torch.Tensor,
        latents: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Decode all K scenarios in one vectorized CUDA-friendly pass."""
        if latents.ndim != 3 or latents.shape[-1] != self.latent_dim:
            raise ValueError("QWM latents must have shape [batch, K, latent_dim]")
        batch_size, samples, _ = latents.shape
        mask = observations["mask"].float()
        residual_scale = self.max_normalized_residual

        token_latents = latents.unsqueeze(2).expand(
            -1, -1, self.max_coins, -1
        )
        market_tokens = torch.cat(
            (self._expand_samples(market_transition_hidden, samples), token_latents),
            dim=-1,
        )
        action_tokens = torch.cat(
            (self._expand_samples(action_transition_hidden, samples), token_latents),
            dim=-1,
        )
        market_global = torch.cat(
            (self._expand_samples(market_global_hidden, samples), latents), dim=-1
        )
        action_global = torch.cat(
            (self._expand_samples(action_global_hidden, samples), latents), dim=-1
        )

        exogenous_observation = (
            self._expand_samples(observation_tokens, samples)
            + residual_scale * torch.tanh(self.observation_delta(market_tokens))
        )
        controlled_observation = (
            self._expand_samples(observation_tokens, samples)
            + residual_scale * torch.tanh(
                self.controlled_observation_delta(action_tokens)
            )
        )
        controlled_mask = self.controlled_observation_feature_mask.view(
            1, 1, 1, -1
        )
        next_observation_normalized = torch.clamp(
            exogenous_observation * (1.0 - controlled_mask)
            + controlled_observation * controlled_mask,
            -1.0,
            1.0,
        )
        next_observation_tokens = (
            next_observation_normalized * self.observation_scale + self.observation_mid
        ) * mask[:, None, :, None]

        local_history = observations["local_feature_history"].float()
        episode_history = observations["episode_feature_history"].float()
        next_local = torch.clamp(
            local_history[:, None, :, -1, :] / self.history_scale
            + residual_scale * torch.tanh(self.local_delta(market_tokens)),
            -1.0,
            1.0,
        ) * self.history_scale
        next_episode = torch.clamp(
            episode_history[:, None, :, -1, :] / self.history_scale
            + residual_scale * torch.tanh(self.episode_delta(market_tokens)),
            -1.0,
            1.0,
        ) * self.history_scale
        next_local = next_local * mask[:, None, :, None]
        next_episode = next_episode * mask[:, None, :, None]

        next_context_normalized = torch.clamp(
            self._expand_samples(context, samples)
            + residual_scale * torch.tanh(self.context_delta(market_global)),
            -1.0,
            1.0,
        )
        next_context = (
            next_context_normalized * self.context_scale + self.context_mid
        )

        flat_rows = batch_size * samples
        next_observations = {
            key: (
                self._expand_samples(value, samples).reshape(
                    flat_rows, *value.shape[1:]
                )
                if torch.is_tensor(value)
                else value
            )
            for key, value in observations.items()
        }
        next_observations["observations"] = next_observation_tokens.reshape(
            flat_rows, -1
        )
        expanded_local = self._expand_samples(local_history, samples)
        expanded_episode = self._expand_samples(episode_history, samples)
        next_observations["local_feature_history"] = torch.cat(
            (expanded_local[:, :, :, 1:, :], next_local.unsqueeze(3)), dim=3
        ).reshape(flat_rows, self.max_coins, self.history_length, self.local_feature_dim)
        next_observations["episode_feature_history"] = torch.cat(
            (expanded_episode[:, :, :, 1:, :], next_episode.unsqueeze(3)), dim=3
        ).reshape(flat_rows, self.max_coins, self.history_length, self.episode_feature_dim)
        history_mask = observations["feature_history_mask"].float()
        expanded_history_mask = self._expand_samples(history_mask, samples)
        next_observations["feature_history_mask"] = torch.cat(
            (
                expanded_history_mask[:, :, 1:],
                torch.ones_like(expanded_history_mask[:, :, :1]),
            ),
            dim=2,
        ).reshape(flat_rows, self.history_length)
        next_observations["basket_context"] = next_context.reshape(
            flat_rows, self.basket_context_dim
        )
        next_observations["_qwm_older_action"] = self._expand_samples(
            observations["prev_action"].float(), samples
        ).reshape(flat_rows, self.action_dim)
        next_observations["prev_action"] = self._expand_samples(
            action_weights, samples
        ).reshape(flat_rows, self.action_dim)
        next_observations["prev_mask"] = self._expand_samples(
            mask, samples
        ).reshape(flat_rows, self.max_coins)
        predicted_reward = (
            self.reward_head(action_global).squeeze(-1)
            / self.reward_target_scale
        )
        return next_observations, predicted_reward

    def _sample_latent_noise(
        self,
        batch_size: int,
        samples: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if samples <= 0:
            raise ValueError("QWM model sample count must be positive")
        if not self.antithetic_samples or samples == 1:
            return torch.randn(
                batch_size,
                samples,
                self.latent_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        pairs = samples // 2
        positive = torch.randn(
            batch_size,
            pairs,
            self.latent_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        paired = torch.cat((positive, -positive), dim=1)
        if samples % 2:
            zero = torch.zeros(
                batch_size, 1, self.latent_dim, device=device, dtype=dtype
            )
            paired = torch.cat((zero, paired), dim=1)
        return paired

    def sample(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        samples: int,
        *,
        generator: Optional[torch.Generator] = None,
        latent_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Draw K joint next-state/reward scenarios from the learned prior."""
        encoded = self._encode(observations, actions.float())
        (
            market_transition_hidden,
            action_transition_hidden,
            market_global_hidden,
            action_global_hidden,
            action_weights,
            observation_tokens,
            context,
        ) = encoded
        prior_mean, prior_log_variance = self._prior_parameters(
            market_global_hidden
        )
        samples = int(samples)
        if latent_noise is None:
            latent_noise = self._sample_latent_noise(
                actions.shape[0],
                samples,
                device=prior_mean.device,
                dtype=prior_mean.dtype,
                generator=generator,
            )
        expected_shape = (actions.shape[0], samples, self.latent_dim)
        if tuple(latent_noise.shape) != expected_shape:
            raise ValueError(
                f"QWM latent noise shape {tuple(latent_noise.shape)} != {expected_shape}"
            )
        prior_std = torch.exp(0.5 * prior_log_variance).unsqueeze(1)
        latents = prior_mean.unsqueeze(1) + prior_std * latent_noise
        return self._decode_latents(
            observations,
            market_transition_hidden,
            action_transition_hidden,
            market_global_hidden,
            action_global_hidden,
            action_weights,
            observation_tokens,
            context,
            latents,
        )

    def sample_action_candidates(
        self,
        observations: Dict[str, torch.Tensor],
        candidate_actions: torch.Tensor,
        samples: int,
        *,
        generator: Optional[torch.Generator] = None,
        latent_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Sample K futures for N actions while preparing the state once.

        The returned contract is identical to flattening the candidate axis,
        repeating every observation N times, and calling :meth:`sample`.
        Expensive normalization and temporal summary construction are shared
        across those actions before the encoded state is expanded.
        """
        if candidate_actions.ndim != 3:
            raise ValueError(
                "QWM candidate actions must be [batch, candidates, action]"
            )
        batch_size, candidate_count, action_dim = candidate_actions.shape
        if action_dim != self.action_dim:
            raise ValueError("QWM candidate action width mismatch")
        base_prepared = self._prepare_state(observations)
        (
            base_market_token_hidden,
            base_market_transition_hidden,
            base_market_global_hidden,
        ) = self._encode_market_prepared(base_prepared)
        prepared = self._repeat_prepared_state(
            base_prepared, candidate_count
        )
        repeated_observations = {
            key: (
                value.unsqueeze(1).expand(
                    batch_size, candidate_count, *value.shape[1:]
                ).reshape(batch_size * candidate_count, *value.shape[1:])
                if torch.is_tensor(value)
                else value
            )
            for key, value in observations.items()
        }
        flat_actions = candidate_actions.float().reshape(
            batch_size * candidate_count, action_dim
        )
        market_token_hidden = base_market_token_hidden.unsqueeze(1).expand(
            -1, candidate_count, -1, -1
        ).reshape(
            batch_size * candidate_count, self.max_coins, self.hidden_dim
        )
        market_transition_hidden = (
            base_market_transition_hidden.unsqueeze(1).expand(
                -1, candidate_count, -1, -1
            ).reshape(
                batch_size * candidate_count,
                self.max_coins,
                2 * self.hidden_dim,
            )
        )
        market_global_hidden = base_market_global_hidden.unsqueeze(1).expand(
            -1, candidate_count, -1
        ).reshape(batch_size * candidate_count, self.hidden_dim)
        (
            action_transition_hidden,
            action_global_hidden,
            action_weights,
        ) = self._encode_actions_from_market(
            prepared,
            flat_actions,
            market_token_hidden,
            market_global_hidden,
        )
        observation_tokens = prepared["observation_tokens"]
        context = prepared["context"]
        prior_mean, prior_log_variance = self._prior_parameters(
            market_global_hidden
        )
        samples = int(samples)
        if latent_noise is None:
            shared_noise = self._sample_latent_noise(
                batch_size,
                samples,
                device=prior_mean.device,
                dtype=prior_mean.dtype,
                generator=generator,
            )
            latent_noise = shared_noise.unsqueeze(1).expand(
                -1, candidate_count, -1, -1
            ).reshape(batch_size * candidate_count, samples, self.latent_dim)
        elif tuple(latent_noise.shape) == (
            batch_size, samples, self.latent_dim
        ):
            latent_noise = latent_noise.unsqueeze(1).expand(
                -1, candidate_count, -1, -1
            ).reshape(batch_size * candidate_count, samples, self.latent_dim)
        expected_flat_shape = (
            batch_size * candidate_count, samples, self.latent_dim
        )
        if tuple(latent_noise.shape) != expected_flat_shape:
            raise ValueError(
                f"QWM latent noise shape {tuple(latent_noise.shape)} != "
                f"shared {(batch_size, samples, self.latent_dim)} or "
                f"flat {expected_flat_shape}"
            )
        prior_std = torch.exp(0.5 * prior_log_variance).unsqueeze(1)
        latents = prior_mean.unsqueeze(1) + prior_std * latent_noise
        return self._decode_latents(
            repeated_observations,
            market_transition_hidden,
            action_transition_hidden,
            market_global_hidden,
            action_global_hidden,
            action_weights,
            observation_tokens,
            context,
            latents,
        )

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Return the prior-mean transition for diagnostics and compatibility."""
        encoded = self._encode(observations, actions.float())
        (
            market_transition_hidden,
            action_transition_hidden,
            market_global_hidden,
            action_global_hidden,
            action_weights,
            observation_tokens,
            context,
        ) = encoded
        prior_mean, _ = self._prior_parameters(market_global_hidden)
        next_observations, predicted_reward = self._decode_latents(
            observations,
            market_transition_hidden,
            action_transition_hidden,
            market_global_hidden,
            action_global_hidden,
            action_weights,
            observation_tokens,
            context,
            prior_mean.unsqueeze(1),
        )
        return next_observations, predicted_reward[:, :1]

    def _reconstruction_losses(
        self,
        observations: Dict[str, torch.Tensor],
        predicted_next: Dict[str, torch.Tensor],
        predicted_reward: torch.Tensor,
        next_observations: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Measure one decoded transition against the real replay successor."""
        mask = observations["mask"].float().unsqueeze(-1)
        predicted_observation = self._normalized_observation_tokens(
            predicted_next["observations"]
        )
        target_observation = self._normalized_observation_tokens(
            next_observations["observations"]
        )
        observation_loss = (
            (predicted_observation - target_observation).square() * mask
        ).sum() / (mask.sum().clamp(min=1.0) * self.features_per_coin)

        predicted_local = (
            predicted_next["local_feature_history"][:, :, -1, :]
            / self.history_scale
        )
        target_local = (
            next_observations["local_feature_history"].float()[:, :, -1, :]
            / self.history_scale
        )
        local_loss = ((predicted_local - target_local).square() * mask).sum() / (
            mask.sum().clamp(min=1.0) * self.local_feature_dim
        )
        predicted_episode = (
            predicted_next["episode_feature_history"][:, :, -1, :]
            / self.history_scale
        )
        target_episode = (
            next_observations["episode_feature_history"].float()[:, :, -1, :]
            / self.history_scale
        )
        episode_loss = (
            (predicted_episode - target_episode).square() * mask
        ).sum() / (mask.sum().clamp(min=1.0) * self.episode_feature_dim)
        context_loss = F.mse_loss(
            self._normalized_context(predicted_next["basket_context"]),
            self._normalized_context(next_observations["basket_context"]),
        )
        state_loss = 0.25 * (
            observation_loss + local_loss + episode_loss + context_loss
        )
        scaled_prediction = predicted_reward * self.reward_target_scale
        scaled_reward = rewards.float().reshape(-1, 1) * self.reward_target_scale
        reward_loss = F.smooth_l1_loss(scaled_prediction, scaled_reward)
        return {
            "state_loss": state_loss,
            "reward_loss": reward_loss,
            "observation_loss": observation_loss,
            "local_history_loss": local_loss,
            "episode_history_loss": episode_loss,
            "context_loss": context_loss,
        }

    def loss(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: Dict[str, torch.Tensor],
        *,
        state_weight: float = 1.0,
        reward_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        encoded = self._encode(observations, actions.float())
        (
            market_transition_hidden,
            action_transition_hidden,
            market_global_hidden,
            action_global_hidden,
            action_weights,
            observation_tokens,
            context,
        ) = encoded
        prior_mean, prior_log_variance = self._prior_parameters(
            market_global_hidden
        )
        posterior_mean, posterior_log_variance = self._posterior_parameters(
            observations,
            next_observations,
            rewards,
            market_global_hidden,
            observation_tokens,
            context,
        )
        posterior_std = torch.exp(0.5 * posterior_log_variance)
        posterior_latent = posterior_mean + posterior_std * torch.randn_like(
            posterior_mean
        )
        # Decode posterior and prior mean in one batched pass.  This preserves
        # direct prior supervision without doubling the decoder/kernel-launch
        # work of every online world-model update.
        combined_next, combined_reward = self._decode_latents(
            observations,
            market_transition_hidden,
            action_transition_hidden,
            market_global_hidden,
            action_global_hidden,
            action_weights,
            observation_tokens,
            context,
            torch.stack((posterior_latent, prior_mean), dim=1),
        )
        batch_size = actions.shape[0]
        predicted_next = {
            key: value.reshape(batch_size, 2, *value.shape[1:])[:, 0]
            if torch.is_tensor(value)
            else value
            for key, value in combined_next.items()
        }
        prior_predicted_next = {
            key: value.reshape(batch_size, 2, *value.shape[1:])[:, 1]
            if torch.is_tensor(value)
            else value
            for key, value in combined_next.items()
        }
        predicted_reward = combined_reward[:, 0:1]
        prior_predicted_reward = combined_reward[:, 1:2]
        posterior_losses = self._reconstruction_losses(
            observations,
            predicted_next,
            predicted_reward,
            next_observations,
            rewards,
        )

        # Search never sees the posterior.  Directly supervising the prior mean
        # closes the former train/search mismatch instead of relying on a tiny
        # KL term to transfer all transition information into the search model.
        prior_losses = self._reconstruction_losses(
            observations,
            prior_predicted_next,
            prior_predicted_reward,
            next_observations,
            rewards,
        )

        prior_variance = torch.exp(prior_log_variance)
        posterior_variance = torch.exp(posterior_log_variance)
        kl_per_dimension = 0.5 * (
            prior_log_variance
            - posterior_log_variance
            + (
                posterior_variance
                + (posterior_mean - prior_mean).square()
            )
            / prior_variance.clamp(min=1e-6)
            - 1.0
        )
        kl_loss = kl_per_dimension.clamp(min=self.kl_free_nats).mean()
        posterior_reconstruction_loss = (
            float(state_weight) * posterior_losses["state_loss"]
            + float(reward_weight) * posterior_losses["reward_loss"]
        )
        prior_reconstruction_loss = (
            float(state_weight) * prior_losses["state_loss"]
            + float(reward_weight) * prior_losses["reward_loss"]
        )
        prior_weight = self.prior_reconstruction_weight
        reconstruction_loss = (
            posterior_reconstruction_loss
            + prior_weight * prior_reconstruction_loss
        ) / (1.0 + prior_weight)
        total_loss = reconstruction_loss + self.kl_weight * kl_loss
        # Telemetry is not part of the learning objective.  Detaching before
        # sqrt matters at initialization, where the two means are equal:
        # sqrt(x) has an infinite derivative at x=0, and an AOT multi-output
        # backward may otherwise form 0 * inf for this zero-cotangent output.
        latent_mean_gap = (
            posterior_mean.detach() - prior_mean.detach()
        ).square().mean().sqrt()
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "posterior_reconstruction_loss": posterior_reconstruction_loss,
            "prior_reconstruction_loss": prior_reconstruction_loss,
            "state_loss": posterior_losses["state_loss"],
            "reward_loss": posterior_losses["reward_loss"],
            "prior_state_loss": prior_losses["state_loss"],
            "prior_reward_loss": prior_losses["reward_loss"],
            "kl_loss": kl_loss,
            "prior_std_mean": torch.exp(0.5 * prior_log_variance).mean(),
            "posterior_std_mean": posterior_std.mean(),
            "latent_mean_gap": latent_mean_gap,
            "observation_loss": posterior_losses["observation_loss"],
            "local_history_loss": posterior_losses["local_history_loss"],
            "episode_history_loss": posterior_losses["episode_history_loss"],
            "context_loss": posterior_losses["context_loss"],
            "prior_observation_loss": prior_losses["observation_loss"],
            "prior_local_history_loss": prior_losses["local_history_loss"],
            "prior_episode_history_loss": prior_losses["episode_history_loss"],
            "prior_context_loss": prior_losses["context_loss"],
            "predicted_reward": predicted_reward,
            "predicted_next_observations": predicted_next,
            "prior_predicted_reward": prior_predicted_reward,
            "prior_predicted_next_observations": prior_predicted_next,
        }


def _masked_mean_pool(values: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a smooth masked mean pool and active counts over the set dimension."""
    if mask.dim() == values.dim() - 1:
        mask = mask.unsqueeze(-1)
    mask_f = mask.float()
    counts = mask_f.sum(dim=1).clamp(min=1.0)
    pooled = (values * mask_f).sum(dim=1) / counts
    return pooled, counts


def _masked_weighted_pool(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a masked weighted mean pool and total active weight."""
    masked_weights = weights.float() * mask.float()
    weight_sums = masked_weights.sum(dim=1, keepdim=True)
    pooled = (values * masked_weights.unsqueeze(-1)).sum(dim=1) / weight_sums.clamp(min=eps)
    return pooled, weight_sums


def _resolve_attention_heads(model_dim: int, requested_heads: int) -> int:
    """Return the largest valid attention head count that divides model_dim."""
    max_heads = max(1, min(int(requested_heads), int(model_dim)))
    for n_heads in range(max_heads, 0, -1):
        if model_dim % n_heads == 0:
            return n_heads
    return 1

class LinearAttentionEncoderLayer(nn.Module):
    """
    Production Assets-Attention Module (AAM) relation block.

    Its input is the ``[batch, assets=m, asset_vector=q]`` tensor produced by
    ``SharedAssetMapEmbedding`` from FAM's complete ``n x Kf`` maps.  Like the
    Transformer Encoder used by MDAEN, the block constructs cross-asset Q/K/V
    associations, applies residual feed-forward processing, respects the asset
    padding mask, and returns exactly the same ``[batch, m, q]`` shape.

    Conventional softmax attention is deliberately not used here. Exponentiated
    query/key scores previously made a few assets dominate the relational state
    and were followed by portfolios concentrated in two or three assets plus
    cash. The replacement remains kernelized and linear in the active-set
    dimension, but fixes the opposite failure of the former positive-kernel
    path: near-uniform routing, a vanishing query/key gradient, and hard-clamped
    relation gates that could die permanently.

    The relation source is centered over the selected basket. Signed tanh
    queries and keys can represent both agreement and opposition, which matters
    because a peer may be a substitute or a hedge. The associative K^T V
    calculation is normalized by an absolute-mass upper bound rather than an
    exponential softmax. By the triangle inequality, the sum of the absolute
    peer coefficients for a query is at most one. Self-contributions are
    removed because the full-strength local path already owns token-local
    information.

    For portfolio encoding we therefore split the layer into two paths:
    1. A token-local path that carries each asset's own observable state.
    2. A small relational path where kernel attention only modulates the token
       as a low-gain residual.

    A bounded sigmoid gate then lets relational evidence become learnable
    without either a dead hard-clamp region or winner-take-most routing.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        num_layers: int = 1,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.num_layers = max(1, num_layers)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.local_proj = nn.Linear(d_model, d_model)

        # Keep the token-local path at full strength. This is the primary
        # constructor of each token representation, while the relational and
        # feed-forward branches are the low-gain corrective residuals.
        self.local_path_scale = nn.Parameter(torch.tensor(1.0))
        self.ff_path_scale = nn.Parameter(torch.tensor(0.10))
        self.max_local_path_scale = 1.0
        self.max_relational_path_scale = 0.25
        self.max_ff_path_scale = 0.30

        # Construct the new gate outside the caller's RNG stream. The gate is
        # intentionally zero-weighted and begins at an effective gain of 0.10,
        # so its random constructor draws have no semantic value and must not
        # silently reseed the retained transformer, fusion, or twin-Q heads.
        with torch.random.fork_rng(devices=[]):
            self.relational_gate = nn.Linear(d_model, 1)
            nn.init.zeros_(self.relational_gate.weight)
            initial_gate_fraction = 0.10 / self.max_relational_path_scale
            initial_gate_logit = math.log(
                initial_gate_fraction / (1.0 - initial_gate_fraction)
            )
            nn.init.constant_(self.relational_gate.bias, initial_gate_logit)

        # GTrXL-style identity map reordering: normalize branch inputs, not the
        # post-residual sum, so the skip path stays clean.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff_in = nn.Linear(d_model, dim_feedforward)
        self.ff_out = nn.Linear(dim_feedforward, d_model)
        self._reset_residual_parameters()

    def _reset_residual_parameters(self) -> None:
        """
        Fixup-style small initialization on residual branch endpoints.

        The goal is not to kill the branches, but to keep their initial updates
        proportional to depth so the attention stack begins near a stable
        identity map.
        """
        branch_scale = 1.0 / math.sqrt(float(self.num_layers))

        with torch.no_grad():
            # Absolute-mass normalization and the 0.25-bounded relation gate
            # already control this branch. Keep the depth correction, but do
            # not apply the old extra 0.25 shrink that made Q/K gradients four
            # orders of magnitude weaker than the local path.
            self.out_proj.weight.mul_(branch_scale)
            if self.out_proj.bias is not None:
                self.out_proj.bias.mul_(branch_scale)

            self.ff_out.weight.mul_(0.25 * branch_scale)
            if self.ff_out.bias is not None:
                self.ff_out.bias.mul_(0.25 * branch_scale)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        *,
        memory: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply signed linear self-attention or decoder cross-attention.

        ``memory=None`` preserves the production AAM self-attention exactly.
        Supplying ``memory`` keeps ``src`` as the residual/query sequence while
        constructing keys and values from a separate sequence.  EarnMore's
        reconstruction decoder needs this distinction: restored mask-token
        queries attend to a full, unmasked stock-pool encoder pass.
        """
        B, N, D = src.shape
        H, d = self.nhead, self.head_dim
        cross_attention = memory is not None
        if memory is None:
            memory = src
            memory_key_padding_mask = src_key_padding_mask
        if memory.dim() != 3 or memory.shape[0] != B or memory.shape[2] != D:
            raise ValueError(
                "LinearAttentionEncoderLayer memory must be [batch, tokens, d_model]"
            )
        memory_tokens = memory.shape[1]
        active_mask = (
            (~src_key_padding_mask).to(src.dtype)
            if src_key_padding_mask is not None
            else src.new_ones((B, N))
        )
        active_mask_expanded = active_mask.unsqueeze(-1)
        memory_active_mask = (
            (~memory_key_padding_mask).to(memory.dtype)
            if memory_key_padding_mask is not None
            else memory.new_ones((B, memory_tokens))
        )
        memory_active_expanded = memory_active_mask.unsqueeze(-1)
        memory_counts = memory_active_expanded.sum(dim=1, keepdim=True).clamp(min=1.0)

        residual_src = src
        local_input = self.norm1(src)
        local_update = self.local_proj(local_input)

        # Normalize the original evidence before extracting peer differences.
        # Normalizing a near-zero centered residual would instead magnify its
        # direction and its input Jacobian. The local path remains unchanged;
        # query and memory use the same normalized-memory coordinate system.
        # Self-attention reuses the normalization already needed locally.
        normalized_memory = self.norm1(memory) if cross_attention else local_input
        normalized_memory_mean = (
            (normalized_memory * memory_active_expanded).sum(dim=1, keepdim=True)
            / memory_counts
        )
        branch_input = (local_input - normalized_memory_mean) * active_mask_expanded
        memory_input = (normalized_memory - normalized_memory_mean) * memory_active_expanded

        Q = torch.tanh(self.q_proj(branch_input).view(B, N, H, d))
        K = torch.tanh(
            self.k_proj(memory_input).view(B, memory_tokens, H, d)
        )
        V = self.v_proj(memory_input).view(B, memory_tokens, H, d)

        if memory_key_padding_mask is not None:
            key_mask = (
                memory_active_mask.to(dtype=K.dtype)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )  # [B, memory_tokens, 1, 1]
            K = K * key_mask
            V = V * key_mask

        # Express the two contractions as explicit strided batched GEMMs. This
        # maps directly to cuBLAS and gives TorchInductor a simpler fusion
        # boundary than the equivalent generic einsum equations.
        q_heads = Q.permute(0, 2, 1, 3)  # [B, H, N, d]
        k_heads = K.permute(0, 2, 3, 1)  # [B, H, d, N]
        v_heads = V.permute(0, 2, 1, 3)  # [B, H, N, d]
        KV = torch.matmul(k_heads, v_heads)  # [B, H, d, d]
        attn_out = torch.matmul(q_heads, KV).permute(0, 2, 1, 3)

        absolute_key_mass = K.abs().sum(dim=1)  # [B, H, d]
        if cross_attention:
            # A hidden-stock mask query must be allowed to read the same stock's
            # full teacher memory.  Removing the diagonal here would defeat the
            # full-unmasked K/V path used by EarnMore's reconstruction decoder.
            normalizer = (
                Q.abs() * absolute_key_mass.unsqueeze(1)
            ).sum(dim=-1)
        else:
            # The local residual already carries the query token during normal
            # self-attention, so remove its own K/V message from the relation
            # stream and from the conservative absolute-mass normalization.
            self_score = (Q * K).sum(dim=-1)
            attn_out = attn_out - self_score.unsqueeze(-1) * V
            peer_absolute_key_mass = (
                absolute_key_mass.unsqueeze(1) - K.abs()
            ).clamp(min=0.0)
            normalizer = (Q.abs() * peer_absolute_key_mass).sum(dim=-1)
        attn_out = attn_out / normalizer.unsqueeze(-1).clamp(min=1e-6)
        attn_out = attn_out * active_mask.unsqueeze(-1).unsqueeze(-1)

        attn_out = attn_out.reshape(B, N, D)
        relational_update = self.out_proj(attn_out)
        local_scale = torch.clamp(self.local_path_scale, min=0.0, max=self.max_local_path_scale).to(src.dtype)
        relational_scale = (
            self.max_relational_path_scale
            * torch.sigmoid(self.relational_gate(branch_input))
            * active_mask_expanded
        ).to(src.dtype)
        ff_scale = torch.clamp(self.ff_path_scale, min=0.0, max=self.max_ff_path_scale).to(src.dtype)

        src = residual_src + local_scale * local_update + relational_scale * relational_update
        ff_input = self.norm2(src)
        ff_update = self.ff_out(F.relu(self.ff_in(ff_input)))
        src = src + ff_scale * ff_update
        return src * active_mask_expanded


def _resolve_token_scope_dims(
    features_per_coin: int,
    local_feature_dim: Optional[int] = None,
    position_feature_dim: Optional[int] = None,
    context_feature_dim: Optional[int] = None,
    comparative_feature_dim: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """Resolve primary | position | comparative | episode-context scopes."""
    if (
        local_feature_dim is None
        and position_feature_dim is None
        and context_feature_dim is None
        and comparative_feature_dim is None
    ):
        configured = (
            getattr(config, "LOCAL_FEATURE_COUNT", None),
            getattr(config, "POSITION_FEATURE_COUNT", None),
            getattr(config, "COMPARATIVE_FEATURE_COUNT", 0),
            getattr(config, "EPISODE_RELATIVE_FEATURE_COUNT", None),
        )
        if all(value is not None for value in configured) and sum(configured) == features_per_coin:
            (
                local_feature_dim,
                position_feature_dim,
                comparative_feature_dim,
                context_feature_dim,
            ) = configured
        else:
            # Small standalone tests and legacy callers may not have the scoped
            # production config. Treat their complete token as market input.
            local_feature_dim, position_feature_dim, comparative_feature_dim, context_feature_dim = (
                features_per_coin,
                0,
                0,
                0,
            )
    else:
        local_feature_dim = int(local_feature_dim or 0)
        position_feature_dim = int(position_feature_dim or 0)
        context_feature_dim = int(context_feature_dim or 0)
        if comparative_feature_dim is None:
            # Training callers written before the comparative branch explicitly
            # pass local/position/context. Infer the newly inserted block from
            # the remaining token width so saved CLI plumbing stays compatible.
            comparative_feature_dim = (
                int(features_per_coin)
                - local_feature_dim
                - position_feature_dim
                - context_feature_dim
            )
        comparative_feature_dim = int(comparative_feature_dim)

    resolved = (
        int(local_feature_dim),
        int(position_feature_dim),
        int(comparative_feature_dim),
        int(context_feature_dim),
    )
    if resolved[0] <= 0 or any(value < 0 for value in resolved):
        raise ValueError(f"Invalid token scope dimensions: {resolved}")
    if sum(resolved) != int(features_per_coin):
        raise ValueError(
            "Token scope mismatch: local + position + comparative + context "
            "must equal "
            f"features_per_coin, received {resolved} for {features_per_coin}."
        )
    return resolved


class BasketContextFiLM(nn.Module):
    """UVFA-style conditioning: modulate market tokens by continuous g(B).

    Starts as near-identity (zero-init) so training can learn routing without
    immediately rewriting the April-style market stem.
    """

    def __init__(self, context_dim: int, model_dim: int):
        super().__init__()
        self.context_dim = int(context_dim)
        self.model_dim = int(model_dim)
        if self.context_dim <= 0 or self.model_dim <= 0:
            raise ValueError("BasketContextFiLM requires positive dims")
        self.to_scale_shift = nn.Linear(self.context_dim, 2 * self.model_dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(
        self, embeddings: torch.Tensor, basket_context: torch.Tensor
    ) -> torch.Tensor:
        if basket_context.dim() == 1:
            basket_context = basket_context.unsqueeze(0)
        scale_shift = self.to_scale_shift(basket_context)
        gamma, beta = scale_shift.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return embeddings * (1.0 + gamma) + beta


class BasketMembershipFiLM(nn.Module):
    """Privileged membership conditioner z(B): live (nonzero) at cold start.

    Unlike regime BasketContextFiLM, membership identifies which combinatorial
    MDP is active. Zero-init would recreate task-blind early training.
    """

    def __init__(self, membership_dim: int, model_dim: int, gain: float = 0.1):
        super().__init__()
        self.membership_dim = int(membership_dim)
        self.model_dim = int(model_dim)
        if self.membership_dim <= 0 or self.model_dim <= 0:
            raise ValueError("BasketMembershipFiLM requires positive dims")
        self.to_scale_shift = nn.Linear(self.membership_dim, 2 * self.model_dim)
        nn.init.xavier_uniform_(self.to_scale_shift.weight, gain=float(gain))
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(
        self, embeddings: torch.Tensor, basket_membership: torch.Tensor
    ) -> torch.Tensor:
        if basket_membership.dim() == 1:
            basket_membership = basket_membership.unsqueeze(0)
        scale_shift = self.to_scale_shift(basket_membership)
        gamma, beta = scale_shift.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return embeddings * (1.0 + gamma) + beta


class SoftPredictiveExpertLayer(nn.Module):
    """Soft compositional experts whose latent is part of the control path.

    Every active asset token is routed through every expert. The router emits
    a convex mixture rather than a hard top-k decision, keeping gradients dense
    while the 8-of-13 curriculum is still discovering useful factors. The
    bounded residual is consumed by the ordinary fusion encoder, so predictive
    learning changes the representation seen by the actor and critic rather
    than creating a teacher-only side embedding.

    The transition model preserves three control-relevant outcomes first:
    causal financial reward, target-policy continuation value, and their
    sampled Bellman composition.  Low-weight raw/FAM targets retain empirical
    grounding, while EMA fused and downstream latents retain a decoder-free
    temporal coordinate.  This makes prediction a value-equivalent teaching
    signal rather than an observation-reconstruction authority.
    """

    def __init__(
        self,
        model_dim: int,
        features_per_coin: int,
        basket_context_dim: int,
        basket_membership_dim: int,
        num_experts: int = 4,
        expert_hidden_dim: int = 64,
        router_temperature: float = 1.0,
        max_residual_abs: float = 0.10,
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.features_per_coin = int(features_per_coin)
        self.basket_context_dim = int(basket_context_dim)
        self.basket_membership_dim = int(basket_membership_dim)
        self.num_experts = int(num_experts)
        self.expert_hidden_dim = int(expert_hidden_dim)
        self.router_temperature = float(router_temperature)
        self.max_residual_abs = float(max_residual_abs)
        if self.num_experts < 2:
            raise ValueError("predictive MoE requires at least two experts")
        if self.expert_hidden_dim <= 0:
            raise ValueError("predictive expert_hidden_dim must be positive")
        if self.router_temperature <= 0.0:
            raise ValueError("predictive router_temperature must be positive")
        if self.max_residual_abs <= 0.0:
            raise ValueError("predictive max_residual_abs must be positive")

        router_input_dim = (
            2 * self.model_dim
            + self.basket_context_dim
            + self.basket_membership_dim
        )
        self.router = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, self.expert_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.expert_hidden_dim, self.num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.model_dim),
                    nn.Linear(self.model_dim, self.expert_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.expert_hidden_dim, self.model_dim),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.control_projection = nn.Linear(
            self.model_dim, self.model_dim, bias=False
        )
        self.control_gate_logit = nn.Parameter(torch.tensor(-2.0))
        # Cold-start control behavior is bitwise unchanged. Prediction still
        # sees the live expert mixture and can begin organizing it immediately.
        nn.init.zeros_(self.control_projection.weight)

        # fused token + pooled basket + expert mixture + executed-action roles
        prediction_input_dim = (3 * self.model_dim) + 3
        self.predictor = nn.Sequential(
            nn.LayerNorm(prediction_input_dim),
            nn.Linear(prediction_input_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
        )
        self.feature_delta_head = nn.Linear(
            self.model_dim, self.features_per_coin
        )
        self.fam_delta_head = nn.Linear(self.model_dim, self.model_dim)
        self.market_delta_head = nn.Linear(self.model_dim, self.model_dim)
        self.relation_delta_head = nn.Linear(self.model_dim, self.model_dim)
        self.fused_delta_head = nn.Linear(self.model_dim, self.model_dim)
        for head in (
            self.feature_delta_head,
            self.fam_delta_head,
            self.market_delta_head,
            self.relation_delta_head,
            self.fused_delta_head,
        ):
            nn.init.xavier_uniform_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        outcome_bins = int(
            getattr(config, "VALUE_EQUIVALENT_DISTRIBUTION_BINS", 101)
        )
        outcome_bound = float(
            getattr(config, "VALUE_EQUIVALENT_SYMLOG_BOUND", 2.0)
        )
        # The reward head reads the action-conditioned transition hidden.  The
        # state-value head reads only the pooled set state, so V cannot smuggle
        # the replay action into its baseline.  Both heads are private model
        # parameters; only their representation gradients are admitted to the
        # control path by the one-sided financial gradient guard.
        self.reward_head = TwoHotSymlogHead(
            self.model_dim,
            bins=outcome_bins,
            symlog_bound=outcome_bound,
        )
        self.state_value_head = TwoHotSymlogHead(
            self.model_dim,
            bins=outcome_bins,
            symlog_bound=outcome_bound,
        )

    @staticmethod
    def _masked_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (tokens * mask.unsqueeze(-1)).sum(dim=1) / counts

    def compose(
        self,
        relational_tokens: torch.Tensor,
        mask: torch.Tensor,
        basket_context: Optional[torch.Tensor],
        basket_membership: Optional[torch.Tensor],
        detach_control_adapter: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = mask.float()
        pooled = self._masked_pool(relational_tokens, mask)
        pooled_expanded = pooled.unsqueeze(1).expand_as(relational_tokens)
        router_parts = [relational_tokens, pooled_expanded]
        if self.basket_context_dim > 0:
            if basket_context is None:
                raise ValueError("predictive MoE requires basket_context")
            router_parts.append(
                basket_context.unsqueeze(1).expand(
                    -1, relational_tokens.shape[1], -1
                )
            )
        if self.basket_membership_dim > 0:
            if basket_membership is None:
                raise ValueError("predictive MoE requires basket_membership")
            router_parts.append(
                basket_membership.unsqueeze(1).expand(
                    -1, relational_tokens.shape[1], -1
                )
            )
        router_logits = self.router(torch.cat(router_parts, dim=-1))
        routing = torch.softmax(
            router_logits / self.router_temperature, dim=-1
        ) * mask.unsqueeze(-1)
        expert_outputs = torch.stack(
            [expert(relational_tokens) for expert in self.experts], dim=2
        )
        expert_outputs = expert_outputs * mask.unsqueeze(-1).unsqueeze(-1)
        mixture = (routing.unsqueeze(-1) * expert_outputs).sum(dim=2)
        gate = self.max_residual_abs * torch.sigmoid(self.control_gate_logit)
        residual = (
            gate
            * torch.tanh(self.control_projection(mixture))
        ) * mask.unsqueeze(-1)
        if detach_control_adapter:
            # Prediction may organize the expert library and every downstream
            # representation that consumes this residual, but it may not grant
            # itself more steering authority. Financial actor/Bellman losses
            # remain the sole owners of the adapter and gate.
            residual = residual.detach()
        return relational_tokens + residual, routing, mixture, expert_outputs

    def forward(
        self,
        relational_tokens: torch.Tensor,
        mask: torch.Tensor,
        basket_context: Optional[torch.Tensor],
        basket_membership: Optional[torch.Tensor],
        detach_control_adapter: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compiled control-path entry point for soft expert composition."""
        return self.compose(
            relational_tokens,
            mask,
            basket_context,
            basket_membership,
            detach_control_adapter=detach_control_adapter,
        )

    def predict_next(
        self,
        current_features: torch.Tensor,
        components: Dict[str, torch.Tensor],
        action_roles: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pooled_expanded = components["pooled"].unsqueeze(1).expand(
            -1, current_features.shape[1], -1
        )
        prediction_input = torch.cat(
            [
                components["fused"],
                pooled_expanded,
                components["predictive_expert_mixture"],
                action_roles,
            ],
            dim=-1,
        )
        hidden = self.predictor(prediction_input)
        return {
            "outcome_hidden": hidden,
            "features": current_features + self.feature_delta_head(hidden),
            "fam": components["fam"] + self.fam_delta_head(hidden),
            "market": components["market"] + self.market_delta_head(hidden),
            "relational": (
                components["relational"] + self.relation_delta_head(hidden)
            ),
            "fused": components["fused"] + self.fused_delta_head(hidden),
        }

    def predict_rollout(
        self,
        current_features: torch.Tensor,
        current_components: Dict[str, torch.Tensor],
        action_roles: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Iteratively predict every latent horizon using replay actions.

        Horizon one uses the complete observed control representation. Later
        horizons consume only the preceding predicted features and latents.
        The current soft expert routing is reused over the short five-minute
        rollout while expert outputs are recomputed from each predicted
        relational state. This retains object/expert composition without
        leaking any future observation into the transition model.
        """
        if action_roles.dim() == 3:
            action_roles = action_roles.unsqueeze(1)
        if action_roles.dim() != 4:
            raise ValueError(
                "predictive action roles must be [batch, horizon, assets, 3]"
            )
        features = current_features
        components = current_components
        routing = current_components["predictive_routing"]
        predictions = []
        horizon = action_roles.shape[1]
        for step in range(horizon):
            predicted = self.predict_next(
                features, components, action_roles[:, step]
            )
            predictions.append(predicted)
            if step + 1 == horizon:
                # No later prediction consumes a successor expert mixture.
                # At the production one-step horizon, recomputing all experts
                # here built an unused autograd graph for the actor and both
                # critics on every predictive SSL update.
                break
            features = predicted["features"]
            relational = predicted["relational"]
            expert_outputs = torch.stack(
                [expert(relational) for expert in self.experts], dim=2
            ) * mask.unsqueeze(-1).unsqueeze(-1)
            mixture = (routing.unsqueeze(-1) * expert_outputs).sum(dim=2)
            components = {
                "fam": predicted["fam"],
                "market": predicted["market"],
                "relational": relational,
                "fused": predicted["fused"],
                "pooled": self._masked_pool(predicted["fused"], mask),
                "predictive_expert_mixture": mixture,
                "predictive_expert_outputs": expert_outputs,
                "predictive_routing": routing,
            }
        return predictions

    def state_value(self, components: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Action-free in-sample value of an observed or predicted set state."""
        return self.state_value_head.mean(components["pooled"])


def collect_basket_amortization_diagnostics(
    observations: Dict[str, torch.Tensor],
    actor: Optional[nn.Module] = None,
    critic: Optional[nn.Module] = None,
    actions: Optional[torch.Tensor] = None,
    *,
    include_sensitivity: bool = False,
    sensitivity_delta: float = 0.75,
) -> Dict[str, float]:
    """Cheap Rank-2 amortization monitors for the SB3 logger.

    Always reports replay-batch geometry of g(B) and parameter norms for FiLM /
    basket projections when present. Optionally measures actor/critic response
    to a bounded g-swap (more expensive; call on an interval).
    """
    stats: Dict[str, float] = {}
    if not isinstance(observations, dict):
        return stats

    g = observations.get("basket_context")
    if torch.is_tensor(g):
        g = g.detach().float()
        if g.dim() == 1:
            g = g.unsqueeze(0)
        if g.numel() == 0 or g.shape[-1] == 0:
            g = None
    else:
        g = None

    z = observations.get("basket_membership")
    if torch.is_tensor(z):
        z = z.detach().float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.numel() == 0 or z.shape[-1] == 0:
            z = None
    else:
        z = None

    m = observations.get("mask")
    if torch.is_tensor(m):
        m = m.detach().float()
        if m.dim() == 1:
            m = m.unsqueeze(0)
        if m.numel() == 0 or m.shape[-1] == 0:
            m = None
    else:
        m = None

    if g is None and z is None and m is None:
        return stats

    batch = int(
        g.shape[0]
        if g is not None
        else z.shape[0]
        if z is not None
        else m.shape[0]
    )
    if g is not None:
        g_norm = g.norm(dim=-1)
        stats["basket_context_norm_mean"] = float(g_norm.mean().item())
        stats["basket_context_norm_std"] = float(g_norm.std(unbiased=False).item())
        stats["basket_context_abs_mean"] = float(g.abs().mean().item())
        stats["basket_context_dim"] = float(g.shape[-1])
        if batch >= 2:
            # The full cdist matrix stores every pair twice plus a diagonal
            # that is immediately discarded. pdist returns each unordered
            # pair once; mean and maximum are therefore exactly unchanged.
            off = torch.pdist(g, p=2.0)
            stats["basket_context_pairwise_l2_mean"] = float(off.mean().item())
            stats["basket_context_pairwise_l2_max"] = float(off.max().item())
        else:
            stats["basket_context_pairwise_l2_mean"] = 0.0
            stats["basket_context_pairwise_l2_max"] = 0.0

    if z is not None:
        unique_z, counts = torch.unique(z, dim=0, return_counts=True)
        stats["basket_membership_dim"] = float(z.shape[-1])
        stats["basket_membership_hamming_weight_mean"] = float(
            z.sum(dim=-1).mean().item()
        )
        stats["basket_membership_unique_keys"] = float(unique_z.shape[0])
        stats["basket_membership_dominant_fraction"] = float(
            counts.max().float().div(float(max(1, batch))).item()
        )
        stats["basket_membership_covariance_trace"] = float(
            z.var(dim=0, unbiased=False).sum().item()
        )
        if batch >= 2:
            off_z = torch.pdist(z, p=1.0)
            # Keep logger-visible names below SB3 HumanOutputFormat's default
            # 36-character limit (which includes three indentation spaces).
            # Otherwise the mean/max suffixes are truncated to the same key.
            stats["basket_membership_hamming_mean"] = float(
                off_z.mean().item()
            )
            stats["basket_membership_hamming_max"] = float(
                off_z.max().item()
            )
        else:
            stats["basket_membership_hamming_mean"] = 0.0
            stats["basket_membership_hamming_max"] = 0.0

    def _resolve_deepset_leaf(module: Optional[nn.Module]) -> Optional[nn.Module]:
        if module is None:
            return None
        if hasattr(module, "token_encoder"):
            return module
        q_networks = getattr(module, "q_networks", None)
        if q_networks is not None and len(q_networks) > 0:
            return q_networks[0]
        return module

    def _record_module_norms(module: Optional[nn.Module], prefix: str) -> None:
        leaf = _resolve_deepset_leaf(module)
        if leaf is None:
            return
        encoder = getattr(leaf, "token_encoder", None)
        film = getattr(encoder, "basket_context_film", None) if encoder is not None else None
        if film is not None and g is not None:
            weight = film.to_scale_shift.weight.detach()
            bias = film.to_scale_shift.bias.detach()
            stats[f"{prefix}_film_weight_norm"] = float(weight.norm().item())
            stats[f"{prefix}_film_bias_norm"] = float(bias.norm().item())
            with torch.no_grad():
                scale_shift = film.to_scale_shift(g)
                gamma, beta = scale_shift.chunk(2, dim=-1)
                stats[f"{prefix}_film_gamma_abs_mean"] = float(gamma.abs().mean().item())
                stats[f"{prefix}_film_beta_abs_mean"] = float(beta.abs().mean().item())
        proj = getattr(leaf, "basket_context_proj", None)
        if proj is not None and g is not None:
            stats[f"{prefix}_proj_weight_norm"] = float(proj.weight.detach().norm().item())
            with torch.no_grad():
                feat = proj(g)
                stats[f"{prefix}_feat_norm_mean"] = float(feat.norm(dim=-1).mean().item())

        # Make the new semantic-control contract observable in training logs.
        # A run can now reveal whether local/relational heads are learning and
        # whether the legacy fused residual is growing toward its safety cap.
        for name in (
            "local_action_head",
            "relational_action_head",
            "cross_action_head",
            "local_control_value",
            "relational_control_value",
            "cross_control_value",
        ):
            control_head = getattr(leaf, name, None)
            if control_head is not None:
                stats[f"{prefix}_{name}_w_norm"] = float(
                    control_head.weight.detach().norm().item()
                )
        for gate_name, maximum_name, metric_name in (
            (
                "fused_action_gate_logit",
                "fused_action_max_scale",
                "fused_action_scale",
            ),
            (
                "fused_value_gate_logit",
                "fused_value_max_scale",
                "fused_value_scale",
            ),
        ):
            gate = getattr(leaf, gate_name, None)
            if gate is not None:
                typed_control_enabled = bool(
                    getattr(leaf, "foundation_typed_control_enabled", False)
                )
                if typed_control_enabled:
                    maximum = float(getattr(leaf, maximum_name))
                    effective_scale = float(
                        (maximum * torch.sigmoid(gate.detach())).item()
                    )
                else:
                    effective_scale = 1.0
                stats[f"{prefix}_{metric_name}"] = effective_scale
                stats[f"{prefix}_foundation_typed_control"] = float(
                    typed_control_enabled
                )
        if encoder is not None:
            snapshot_gate = getattr(encoder, "snapshot_residual_scale", None)
            if snapshot_gate is not None:
                stats[f"{prefix}_snapshot_context_scale"] = float(
                    torch.clamp(
                        snapshot_gate.detach(),
                        min=0.0,
                        max=float(encoder.snapshot_max_residual_scale),
                    ).item()
                )

        membership_film = (
            getattr(encoder, "basket_membership_film", None)
            if encoder is not None
            else None
        )
        if membership_film is not None and z is not None:
            weight = membership_film.to_scale_shift.weight.detach()
            bias = membership_film.to_scale_shift.bias.detach()
            stats[f"{prefix}_membership_film_weight_norm"] = float(
                weight.norm().item()
            )
            stats[f"{prefix}_membership_film_bias_norm"] = float(
                bias.norm().item()
            )
            with torch.no_grad():
                scale_shift = membership_film.to_scale_shift(z)
                gamma, beta = scale_shift.chunk(2, dim=-1)
                stats[f"{prefix}_membership_film_gamma_abs_mean"] = float(
                    gamma.abs().mean().item()
                )
                stats[f"{prefix}_membership_film_beta_abs_mean"] = float(
                    beta.abs().mean().item()
                )
        membership_proj = getattr(leaf, "basket_membership_proj", None)
        if membership_proj is not None and z is not None:
            stats[f"{prefix}_membership_proj_weight_norm"] = float(
                membership_proj.weight.detach().norm().item()
            )
            with torch.no_grad():
                membership_feat = membership_proj(z)
                stats[f"{prefix}_membership_feat_norm_mean"] = float(
                    membership_feat.norm(dim=-1).mean().item()
                )

        pool_mask_proj = getattr(leaf, "pool_task_mask_projection", None)
        pool_cls_proj = getattr(leaf, "pool_task_cls_projection", None)
        if pool_mask_proj is not None and m is not None:
            stats[f"{prefix}_pool_task_mask_w_norm"] = float(
                pool_mask_proj.weight.detach().norm().item()
            )
            with torch.no_grad():
                mask_feat = pool_mask_proj(m)
                stats[f"{prefix}_pool_task_mask_feat_norm"] = float(
                    mask_feat.norm(dim=-1).mean().item()
                )
        if pool_cls_proj is not None:
            stats[f"{prefix}_pool_task_cls_w_norm"] = float(
                pool_cls_proj.weight.detach().norm().item()
            )
        if hasattr(leaf, "global_head_uses_direct_task"):
            stats[f"{prefix}_pool_task_global_head"] = float(
                bool(leaf.global_head_uses_direct_task)
            )

        # The low-rank UVFA branch is critic-only.  These compact diagnostics
        # distinguish a dormant zero-start path from one that has learned
        # reusable local factors and a bounded task-dependent gain.
        uvfa_asset_proj = getattr(leaf, "uvfa_asset_factor_projection", None)
        uvfa_task_proj = getattr(leaf, "uvfa_task_factor_projection", None)
        uvfa_gate = getattr(leaf, "uvfa_interaction_gate_logit", None)
        if uvfa_asset_proj is not None:
            stats[f"{prefix}_uvfa_asset_w_norm"] = float(
                uvfa_asset_proj.weight.detach().norm().item()
            )
        if uvfa_task_proj is not None and g is not None:
            stats[f"{prefix}_uvfa_task_w_norm"] = float(
                uvfa_task_proj.weight.detach().norm().item()
            )
            with torch.no_grad():
                task_factors = torch.tanh(uvfa_task_proj(g))
                stats[f"{prefix}_uvfa_task_abs_mean"] = float(
                    task_factors.abs().mean().item()
                )
        if uvfa_gate is not None:
            gate = torch.sigmoid(uvfa_gate.detach())
            stats[f"{prefix}_uvfa_gate"] = float(gate.item())

    _record_module_norms(actor, "actor")
    _record_module_norms(critic, "critic")

    if not include_sensitivity:
        return stats
    if actor is None and critic is None:
        return stats

    with torch.no_grad():
        if g is not None:
            delta = float(max(0.0, sensitivity_delta))
            if delta > 0.0:
                obs_alt = dict(observations)
                obs_alt["basket_context"] = g + delta
                if actor is not None:
                    a0 = actor(observations)
                    a1 = actor(obs_alt)
                    stats["actor_g_swap_abs_max"] = float((a0 - a1).abs().max().item())
                    stats["actor_g_swap_abs_mean"] = float((a0 - a1).abs().mean().item())
                if critic is not None and actions is not None:
                    q_fn = getattr(critic, "q1_forward", None)
                    if callable(q_fn):
                        q0 = q_fn(observations, actions)
                        q1 = q_fn(obs_alt, actions)
                    else:
                        out0 = critic(observations, actions)
                        out1 = critic(obs_alt, actions)
                        q0 = out0[0] if isinstance(out0, (tuple, list)) else out0
                        q1 = out1[0] if isinstance(out1, (tuple, list)) else out1
                    stats["critic_g_swap_abs_max"] = float((q0 - q1).abs().max().item())
                    stats["critic_g_swap_abs_mean"] = float((q0 - q1).abs().mean().item())

        if z is not None and batch >= 2 and torch.unique(z, dim=0).shape[0] > 1:
            z_alt = torch.roll(z, shifts=1, dims=0)
            same = (z_alt == z).all(dim=1)
            if same.any():
                for shift in range(2, batch + 1):
                    replacement = torch.roll(z, shifts=shift, dims=0)
                    replace_rows = same & ~((replacement == z).all(dim=1))
                    z_alt[replace_rows] = replacement[replace_rows]
                    same = (z_alt == z).all(dim=1)
                    if not same.any():
                        break
            obs_z_alt = dict(observations)
            obs_z_alt["basket_membership"] = z_alt
            if actor is not None:
                a0_z = actor(observations)
                a1_z = actor(obs_z_alt)
                stats["actor_z_swap_abs_max"] = float(
                    (a0_z - a1_z).abs().max().item()
                )
                stats["actor_z_swap_abs_mean"] = float(
                    (a0_z - a1_z).abs().mean().item()
                )
            if critic is not None and actions is not None:
                q_fn = getattr(critic, "q1_forward", None)
                if callable(q_fn):
                    q0_z = q_fn(observations, actions)
                    q1_z = q_fn(obs_z_alt, actions)
                else:
                    out0_z = critic(observations, actions)
                    out1_z = critic(obs_z_alt, actions)
                    q0_z = out0_z[0] if isinstance(out0_z, (tuple, list)) else out0_z
                    q1_z = out1_z[0] if isinstance(out1_z, (tuple, list)) else out1_z
                stats["critic_z_swap_abs_max"] = float(
                    (q0_z - q1_z).abs().max().item()
                )
                stats["critic_z_swap_abs_mean"] = float(
                    (q0_z - q1_z).abs().mean().item()
                )
    return stats


class CausalFeatureAttentionModule(nn.Module):
    """MDAEN feature attention ending at the full per-asset feature map.

    With the batch axis made explicit, the paper's input ``X in R[m,n,f]`` is
    represented as ``[batch, assets=m, history=n, features=f]``.  Kernel-three
    depthwise convolutions operate only along the ordered time axis and
    independently for every feature of every asset.  A pooled feature-wise
    gate selects among those temporal scales, their weighted maps are
    concatenated, and a 1x1 skip projection preserves an unfiltered path.  For
    ``K`` kernels the result remains ``[batch, m, n, Kf]``; temporal compression
    does *not* belong to FAM and is deliberately handled by AAM's embedding.

    No asset identity, slot embedding, basket-membership code, action, reward,
    or critic value participates in this module.
    """

    def __init__(
        self,
        feature_dim: int,
        kernel_sizes: Tuple[int, ...] = (3, 5),
        local_feature_dim: Optional[int] = None,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.kernel_sizes = tuple(int(value) for value in kernel_sizes)
        if self.feature_dim <= 0:
            raise ValueError("FAM feature_dim must be positive")
        if len(self.kernel_sizes) < 2 or any(value <= 0 for value in self.kernel_sizes):
            raise ValueError("FAM requires at least two positive temporal kernels")
        if len(set(self.kernel_sizes)) != len(self.kernel_sizes):
            raise ValueError("FAM temporal kernels must be distinct")
        self.branch_count = len(self.kernel_sizes)
        self.max_kernel_size = max(self.kernel_sizes)
        self.map_feature_dim = self.branch_count * self.feature_dim
        self.local_feature_dim = int(
            self.feature_dim if local_feature_dim is None else local_feature_dim
        )
        self.episode_feature_dim = self.feature_dim - self.local_feature_dim
        if not 0 < self.local_feature_dim <= self.feature_dim:
            raise ValueError(
                "FAM local_feature_dim must be in [1, feature_dim]"
            )
        self.temporal_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.feature_dim,
                    self.feature_dim,
                    kernel_size=kernel_size,
                    groups=self.feature_dim,
                    bias=True,
                )
                for kernel_size in self.kernel_sizes
            ]
        )
        # The local and episode skips are separate on purpose.  A dense f->Kf
        # skip would reintroduce the exact peer-to-local route removed from the
        # selective-kernel gate even though the temporal convolution is
        # depthwise.  The expensive causal convolution remains fused once.
        self.local_skip_projection = nn.Conv1d(
            self.local_feature_dim,
            self.branch_count * self.local_feature_dim,
            kernel_size=1,
            bias=True,
        )
        self.episode_skip_projection = (
            nn.Conv1d(
                self.episode_feature_dim,
                self.branch_count * self.episode_feature_dim,
                kernel_size=1,
                bias=True,
            )
            if self.episode_feature_dim > 0
            else None
        )
        self.local_feature_gate = nn.Linear(
            self.local_feature_dim,
            self.branch_count * self.local_feature_dim,
        )
        self.episode_feature_gate = (
            nn.Linear(
                self.episode_feature_dim,
                self.branch_count * self.episode_feature_dim,
            )
            if self.episode_feature_dim > 0
            else None
        )
        for convolution in self.temporal_convs:
            nn.init.xavier_uniform_(convolution.weight, gain=0.25)
            nn.init.zeros_(convolution.bias)
        # Begin with a norm-preserving symmetric copy of every raw feature into
        # all K blocks of the skip map.  The projection remains trainable.
        def _initialize_skip(projection, width):
            if projection is None:
                return
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            with torch.no_grad():
                identity_scale = 1.0 / math.sqrt(float(self.branch_count))
                for branch_index in range(self.branch_count):
                    for feature_index in range(width):
                        projection.weight[
                            branch_index * width + feature_index,
                            feature_index,
                            0,
                        ] = identity_scale

        _initialize_skip(self.local_skip_projection, self.local_feature_dim)
        _initialize_skip(self.episode_skip_projection, self.episode_feature_dim)
        nn.init.xavier_uniform_(self.local_feature_gate.weight, gain=0.25)
        nn.init.zeros_(self.local_feature_gate.bias)
        if self.episode_feature_gate is not None:
            nn.init.xavier_uniform_(self.episode_feature_gate.weight, gain=0.25)
            nn.init.zeros_(self.episode_feature_gate.bias)

    @property
    def short_conv(self) -> nn.Conv1d:
        """Compatibility/readability alias for the shortest temporal branch."""
        return self.temporal_convs[0]

    @property
    def long_conv(self) -> nn.Conv1d:
        """Compatibility/readability alias for the longest temporal branch."""
        return self.temporal_convs[-1]

    @staticmethod
    def _causal_depthwise(
        inputs: torch.Tensor, convolution: nn.Conv1d
    ) -> torch.Tensor:
        padded = F.pad(inputs, (int(convolution.kernel_size[0]) - 1, 0))
        return convolution(padded)

    def _fused_causal_depthwise_branches(
        self, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate all causal branches in one grouped depth-wise kernel.

        The FAM branches have different temporal widths, so the old forward
        path padded and launched one convolution per width.  All branches are
        depth-wise over the same feature channels, however.  Right-aligning
        each shorter kernel inside the longest causal kernel makes one grouped
        convolution mathematically identical to the separate launches.  The
        branch modules remain the registered parameters, so checkpoints and
        optimizer state keep their existing names and gradients still flow to
        every original branch weight.
        """
        feature_count = int(self.feature_dim)
        branch_count = len(self.temporal_convs)
        max_kernel = int(self.max_kernel_size)
        weights = [
            F.pad(
                convolution.weight,
                (max_kernel - int(convolution.kernel_size[0]), 0),
            )
            for convolution in self.temporal_convs
        ]
        # ``groups=feature_count`` expects all branch filters for feature 0,
        # followed by all branch filters for feature 1, and so on.  Stacking
        # on the branch axis before flattening produces that interleaving.
        fused_weight = torch.stack(weights, dim=1).reshape(
            branch_count * feature_count, 1, max_kernel
        )
        fused_bias = torch.stack(
            [convolution.bias for convolution in self.temporal_convs], dim=1
        ).reshape(-1)
        fused = F.conv1d(
            F.pad(inputs, (max_kernel - 1, 0)),
            fused_weight,
            fused_bias,
            groups=feature_count,
        )
        # Grouped-convolution channels are ordered feature-major then branch.
        # Expose a view [batch*assets, time, branch, feature] so forward can
        # pool and gate every branch without constructing a Python list,
        # stacking that list, and concatenating it again.
        return fused.reshape(
            inputs.shape[0], feature_count, branch_count, inputs.shape[-1]
        ).permute(0, 3, 2, 1)

    def forward(
        self,
        feature_history: torch.Tensor,
        asset_mask: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if feature_history.dim() != 4:
            raise ValueError(
                "feature_history must have [batch, assets, time, features]"
            )
        batch, assets, history, features = feature_history.shape
        if features != self.feature_dim:
            raise ValueError(
                f"FAM feature width mismatch: expected {self.feature_dim}, "
                f"received {features}"
            )
        if asset_mask.shape != (batch, assets):
            raise ValueError("FAM asset mask shape mismatch")
        if history_mask.shape != (batch, history):
            raise ValueError("FAM history mask shape mismatch")

        values = feature_history.float()
        temporal_valid = history_mask.float().unsqueeze(1).unsqueeze(-1)
        asset_valid = asset_mask.float().unsqueeze(-1).unsqueeze(-1)
        valid = temporal_valid * asset_valid
        values = values * valid
        convolution_input = values.permute(0, 1, 3, 2).reshape(
            batch * assets, features, history
        )
        # [batch, assets, time, branch, feature]. This is a view of the single
        # grouped convolution until masking creates the one live branch tensor.
        branches = self._fused_causal_depthwise_branches(
            convolution_input
        ).reshape(batch, assets, history, self.branch_count, features)
        branches = branches * valid.unsqueeze(-2)

        local_branches = branches[..., : self.local_feature_dim]
        local_denominator = valid.sum(dim=2).clamp(min=1.0)
        local_pooled = local_branches.sum(dim=3).sum(dim=2) / local_denominator
        local_gate = torch.softmax(
            self.local_feature_gate(local_pooled).view(
                batch, assets, self.local_feature_dim, self.branch_count
            ),
            dim=-1,
        )
        if self.episode_feature_dim > 0:
            episode_branches = branches[..., self.local_feature_dim :]
            episode_denominator = valid.sum(dim=(1, 2)).clamp(min=1.0)
            episode_pooled = (
                episode_branches.sum(dim=3).sum(dim=(1, 2))
                / episode_denominator
            )
            episode_gate_shared = torch.softmax(
                self.episode_feature_gate(episode_pooled).view(
                    batch, self.episode_feature_dim, self.branch_count
                ),
                dim=-1,
            )
            episode_gate = episode_gate_shared.unsqueeze(1).expand(
                -1, assets, -1, -1
            )
            gate = torch.cat((local_gate, episode_gate), dim=2)
        else:
            episode_branches = None
            gate = local_gate
        weighted = (
            branches
            * gate.permute(0, 1, 3, 2).unsqueeze(2)
        ).reshape(batch, assets, history, self.map_feature_dim)
        # A 1x1 Conv1d is a shared linear map at every time position.  Calling
        # the equivalent GEMM avoids a second small convolution launch while
        # keeping ``skip_projection`` as the registered Conv1d parameter for
        # checkpoint compatibility.
        local_values = convolution_input[:, : self.local_feature_dim].transpose(1, 2)
        local_skip = F.linear(
            local_values,
            self.local_skip_projection.weight.squeeze(-1),
            self.local_skip_projection.bias,
        ).reshape(
            batch,
            assets,
            history,
            self.branch_count,
            self.local_feature_dim,
        )
        if self.episode_skip_projection is not None:
            episode_values = convolution_input[:, self.local_feature_dim :].transpose(1, 2)
            episode_skip = F.linear(
                episode_values,
                self.episode_skip_projection.weight.squeeze(-1),
                self.episode_skip_projection.bias,
            ).reshape(
                batch,
                assets,
                history,
                self.branch_count,
                self.episode_feature_dim,
            )
            skip = torch.cat((local_skip, episode_skip), dim=-1).reshape(
                batch, assets, history, self.map_feature_dim
            )
        else:
            skip = local_skip.reshape(batch, assets, history, self.map_feature_dim)
        feature_map = (weighted + skip) * valid
        return feature_map, gate


class SharedAssetMapEmbedding(nn.Module):
    """AAM embedding: one shared FC compression for every asset map.

    The paper describes ``m`` applications of an FC layer whose weights are
    shared across the ``m`` assets.  A single ``nn.Linear`` applied to the last
    dimension of ``[batch, m, n*Kf]`` is exactly that operation: every asset
    keeps its ordered full FAM map, every slot uses the same parameters, and no
    ticker-specific lookup can enter the compression.  ``asset_vector_dim`` is
    the paper's ``q``.  The production encoder deliberately keeps this compact
    (q=32), then makes one shared q-to-model adapter before the wider TD3 AAM.
    That preserves the paper's map-to-vector bottleneck and avoids making the
    flattened 30-minute map projection scale linearly with the critic width.
    """

    def __init__(
        self,
        history_length: int,
        map_feature_dim: int,
        asset_vector_dim: int,
    ):
        super().__init__()
        self.history_length = int(history_length)
        self.map_feature_dim = int(map_feature_dim)
        self.asset_vector_dim = int(asset_vector_dim)
        if min(
            self.history_length,
            self.map_feature_dim,
            self.asset_vector_dim,
        ) <= 0:
            raise ValueError("AAM asset-map embedding dimensions must be positive")
        self.flattened_map_dim = self.history_length * self.map_feature_dim
        self.input_norm = nn.LayerNorm(self.flattened_map_dim)
        self.projection = nn.Linear(
            self.flattened_map_dim, self.asset_vector_dim
        )
        self.output_norm = nn.LayerNorm(self.asset_vector_dim)
        nn.init.xavier_uniform_(self.projection.weight, gain=0.25)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        feature_map: torch.Tensor,
        asset_mask: torch.Tensor,
    ) -> torch.Tensor:
        if feature_map.dim() != 4:
            raise ValueError(
                "AAM feature map must have [batch, assets, history, map_features]"
            )
        batch, assets, history, map_features = feature_map.shape
        if history != self.history_length or map_features != self.map_feature_dim:
            raise ValueError(
                "AAM feature-map shape mismatch: expected history/map width "
                f"({self.history_length}, {self.map_feature_dim}), received "
                f"({history}, {map_features})"
            )
        if asset_mask.shape != (batch, assets):
            raise ValueError("AAM asset mask shape mismatch")
        flattened = feature_map.reshape(batch, assets, self.flattened_map_dim)
        vectors = self.output_norm(
            self.projection(self.input_norm(flattened))
        )
        return vectors * asset_mask.float().unsqueeze(-1)


class AttentionDeepSetEncoder(nn.Module):
    """
    Production MDAEN/DeepSet state encoder shared in form by actor and critics.

    When FAM is enabled, the primary path is explicit temporal tensor -> FAM
    ``n x Kf`` asset maps -> shared AAM map compression -> cross-asset AAM.  A
    bounded post-AAM flat-snapshot residual preserves current-state nonlinear
    detail without entering the cross-asset attention scores.
    Position and candidate allocation are deliberately composed only after the
    action-independent market relation has been constructed.
    """

    def __init__(
        self,
        features_per_coin: int,
        local_feature_dim: int,
        position_feature_dim: int,
        context_feature_dim: int,
        comparative_feature_dim: int,
        model_dim: int,
        net_arch: List[int],
        n_heads: int,
        n_layers: int,
        dropout: float,
        use_market_attention: bool,
        use_global_pooling: bool,
        relation_aux_dim: int = 0,
        initialization_rng_compat_relation_aux_dim: int = 0,
        relation_uses_comparative: bool = True,
        relation_uses_context: bool = False,
        relation_uses_position: bool = False,
        asset_identity_dim: int = 0,
        basket_context_dim: int = 0,
        basket_membership_dim: int = 0,
        predictive_moe_enabled: bool = False,
        predictive_moe_num_experts: int = 4,
        predictive_moe_expert_hidden_dim: int = 64,
        predictive_moe_router_temperature: float = 1.0,
        predictive_moe_max_residual_abs: float = 0.10,
        fam_enabled: bool = False,
        fam_feature_dim: int = 0,
        fam_history_length: int = 8,
        fam_kernel_sizes: Tuple[int, ...] = (3, 5),
        fam_asset_vector_dim: Optional[int] = None,
        fam_initial_residual_scale: float = 0.05,
        fam_max_residual_scale: float = 0.25,
    ):
        super().__init__()
        self.local_causal_inputs_enabled = config.LOCAL_CAUSAL_INPUTS_ENABLED
        del dropout
        self.features_per_coin = int(features_per_coin)
        self.local_feature_dim = int(local_feature_dim)
        self.position_feature_dim = int(position_feature_dim)
        self.comparative_feature_dim = int(comparative_feature_dim)
        self.context_feature_dim = int(context_feature_dim)
        self.asset_identity_dim = int(asset_identity_dim)
        self.pool_level_enabled = bool(config.EARNMORE_POOL_LEVEL_ENABLED and self.asset_identity_dim > 0)
        if self.asset_identity_dim < 0:
            raise ValueError("asset_identity_dim must be non-negative")
        # Stable stock positions enter portfolio relations, not the causal FAM.
        self.asset_position_projection = (
            nn.Linear(self.asset_identity_dim, model_dim, bias=False)
            if self.asset_identity_dim > 0 and config.EARNMORE_GLOBAL_POOL_ENABLED else None
        )
        self.basket_context_dim = int(basket_context_dim)
        if self.basket_context_dim < 0:
            raise ValueError("basket_context_dim must be non-negative")
        self.basket_membership_dim = int(basket_membership_dim)
        if self.basket_membership_dim < 0:
            raise ValueError("basket_membership_dim must be non-negative")
        self.relation_aux_dim = int(relation_aux_dim)
        self.initialization_rng_compat_relation_aux_dim = int(
            initialization_rng_compat_relation_aux_dim
        )
        if self.initialization_rng_compat_relation_aux_dim < 0:
            raise ValueError(
                "initialization_rng_compat_relation_aux_dim must be non-negative"
            )
        if (
            self.relation_aux_dim > 0
            and self.initialization_rng_compat_relation_aux_dim > 0
        ):
            raise ValueError(
                "RNG compatibility is only valid when relation_aux_dim is zero"
            )
        # Kept as constructor attributes for saved policy-configuration
        # compatibility. Comparative and episode-context fields participate in
        # the joint market token. Position is intentionally not allowed to
        # define the market-relation attention seed.
        self.relation_uses_comparative = bool(relation_uses_comparative)
        self.relation_uses_context = bool(relation_uses_context)
        self.relation_uses_position = bool(relation_uses_position)
        self.model_dim = model_dim
        self.use_global_pooling = use_global_pooling
        self.attention_dropout = 0.0
        self.predictive_moe_enabled = bool(predictive_moe_enabled)
        self.predictive_expert_layer: Optional[SoftPredictiveExpertLayer] = None
        self.fam_enabled = bool(fam_enabled)
        self.fam_feature_dim = int(fam_feature_dim)
        self.fam_history_length = int(fam_history_length)
        self.fam_branch_count = len(tuple(fam_kernel_sizes))
        self.fam_map_feature_dim = self.fam_branch_count * self.fam_feature_dim
        configured_local_width = int(
            getattr(config, "MDAEN_LOCAL_FEATURE_COUNT", self.fam_feature_dim)
        )
        self.fam_local_feature_dim = (
            configured_local_width
            if self.fam_feature_dim
            == int(getattr(config, "MDAEN_FAM_FEATURE_COUNT", -1))
            else self.fam_feature_dim
        )
        self.fam_episode_feature_dim = (
            self.fam_feature_dim - self.fam_local_feature_dim
        )
        self.fam_asset_vector_dim = int(
            model_dim if fam_asset_vector_dim is None else fam_asset_vector_dim
        )
        self.strict_semantic_gradient_ownership_configured = bool(
            getattr(
                config,
                "FOUNDATION_STRICT_SEMANTIC_GRADIENT_OWNERSHIP",
                True,
            )
        )
        # This detach boundary belongs to the foundation-learning contract:
        # foundation explicitly owns the peer-independent temporal code and
        # protects it from being rewritten by contextual control losses.  When
        # foundation is disabled (as in the EarnMore experiment), leaving the
        # boundary active would let masked reconstruction change the encoder
        # while preventing much of TD3's relational, cross-asset, allocation,
        # and UVFA feedback from teaching that encoder what matters for return.
        self.strict_semantic_gradient_ownership = bool(
            getattr(config, "FOUNDATION_LEARNING_ENABLED", False)
            and self.strict_semantic_gradient_ownership_configured
        )
        # The 81-wide flat token is a selected-basket snapshot, not a valid
        # asset-local coordinate system.  In schema-v2 the first comparative
        # fields are already present in the peer-independent temporal stream;
        # exclude those duplicated local anchors from the contextual snapshot
        # residual instead of mixing them back together with episode-relative
        # features in one unnamed input stem.
        configured_local_anchors = len(
            tuple(getattr(config, "MDAEN_LOCAL_BEHAVIORAL_FEATURES", ()))
        )
        self.snapshot_local_anchor_count = (
            min(configured_local_anchors, self.comparative_feature_dim)
            if self.fam_enabled
            else 0
        )
        self.snapshot_max_residual_scale = float(fam_max_residual_scale)
        if self.fam_enabled and (
            self.fam_feature_dim <= 0 or self.fam_history_length <= 0
        ):
            raise ValueError(
                "enabled FAM requires positive feature and history widths"
            )
        if self.fam_enabled and self.fam_history_length < max(fam_kernel_sizes):
            raise ValueError(
                "FAM history must cover its longest temporal kernel: "
                f"history={self.fam_history_length}, kernels={fam_kernel_sizes}"
            )
        if self.fam_enabled and self.fam_asset_vector_dim <= 0:
            raise ValueError("enabled FAM requires a positive AAM q width")

        if (
            self.local_feature_dim
            + self.position_feature_dim
            + self.comparative_feature_dim
            + self.context_feature_dim
            != self.features_per_coin
        ):
            raise ValueError("AttentionDeepSetEncoder received an invalid token scope")

        branch_arch = [model_dim]
        self.market_feature_dim = (
            self.local_feature_dim
            + self.comparative_feature_dim
            - self.snapshot_local_anchor_count
            + self.context_feature_dim
            + self.asset_identity_dim
        )
        # Preserve the useful nonlinear flat-snapshot path, but classify it
        # correctly: these selected-basket-normalized primary and relative
        # fields are contextual evidence.  In the production FAM path this MLP
        # is added only after cross-asset attention and is bounded twice—at
        # representation fusion and at the final actor/Q control heads.
        self.market_encoder = _build_mlp_layers(
            self.market_feature_dim, net_arch, model_dim
        )
        self.market_token_norm = nn.LayerNorm(model_dim)
        # FAM is a new representation branch.  Its depth-wise kernels and
        # feature gate must not consume the main constructor RNG stream: doing
        # so changes the retained transformer, fusion, actor, and twin-critic
        # initializations even when the FAM residual is initially small.  That
        # presents as an inverted cold-start actor loss before TD has supplied
        # any financial evidence.  The branch keeps its own initialized
        # parameters, while downstream legacy-compatible modules see the same
        # RNG sequence as the preceding architecture.
        self.feature_attention = None
        if self.fam_enabled:
            with torch.random.fork_rng(devices=[]):
                self.feature_attention = CausalFeatureAttentionModule(
                    feature_dim=self.fam_feature_dim,
                    kernel_sizes=fam_kernel_sizes,
                    local_feature_dim=self.fam_local_feature_dim,
                )
        # This shared projection is the embedding layer owned by AAM in the
        # paper: every complete n x Kf asset map is compressed into one q-wide
        # vector with exactly the same weights before cross-asset attention.
        #
        # Construct both schema-v2 stream projections inside a private RNG
        # boundary.  Their dimensions are new and intentionally incompatible
        # with legacy checkpoints, but their initialization must not move the
        # retained relation/fusion/control modules merely because the temporal
        # representation gained a second semantic branch.
        self.fam_asset_embedding = None
        self.fam_local_asset_embedding = None
        self.fam_episode_asset_embedding = None
        self.fam_to_model_projection = None
        if self.fam_enabled:
            with torch.random.fork_rng(devices=[]):
                self.fam_local_asset_embedding = SharedAssetMapEmbedding(
                    history_length=self.fam_history_length,
                    map_feature_dim=(
                        self.fam_branch_count * self.fam_local_feature_dim
                    ),
                    asset_vector_dim=self.fam_asset_vector_dim,
                )
                if self.fam_episode_feature_dim > 0:
                    self.fam_episode_asset_embedding = SharedAssetMapEmbedding(
                        history_length=self.fam_history_length,
                        map_feature_dim=(
                            self.fam_branch_count
                            * self.fam_episode_feature_dim
                        ),
                        asset_vector_dim=self.fam_asset_vector_dim,
                    )
                # MDAEN supplies q-wide asset vectors to AAM.  The established
                # TD3 relation stack is wider, so a *shared across assets*
                # adapter expands q only after the complete FAM map has already
                # been compressed.  This is not a second source of asset
                # identity; it is the bridge between the paper's q=32 embedding
                # and the existing wider attention path.
                self.fam_to_model_projection = nn.Sequential(
                    nn.LayerNorm(
                        self.fam_asset_vector_dim
                        * (1 + int(self.fam_episode_feature_dim > 0))
                    ),
                    nn.Linear(
                        self.fam_asset_vector_dim
                        * (1 + int(self.fam_episode_feature_dim > 0)),
                        model_dim,
                    ),
                    nn.SiLU(),
                    nn.LayerNorm(model_dim),
                )
                nn.init.xavier_uniform_(
                    self.fam_to_model_projection[1].weight, gain=0.25
                )
                nn.init.zeros_(self.fam_to_model_projection[1].bias)
        self.snapshot_residual_scale = (
            nn.Parameter(torch.tensor(float(fam_initial_residual_scale)))
            if self.fam_enabled
            else None
        )
        self.post_aam_snapshot_norm = (
            nn.LayerNorm(model_dim) if self.fam_enabled else None
        )
        self.basket_context_film = (
            BasketContextFiLM(self.basket_context_dim, model_dim)
            if self.basket_context_dim > 0
            else None
        )
        # Membership FiLM is constructed in fork_rng after the shared encoder
        # stack so its Xavier draws cannot reseed transformer / fusion / twin Q2.
        self.basket_membership_film = None

        # A one-value cost basis must not be promoted into an equally normalized
        # full-width branch.  This small tanh projection is bounded, is exactly
        # zero when cost basis is zero, and conditions decisions only after the
        # market opportunity has been represented.
        self.position_latent_dim = (
            max(1, min(16, model_dim // 16))
            if self.position_feature_dim > 0
            else 0
        )
        self.position_projection = (
            nn.Linear(self.position_feature_dim, self.position_latent_dim)
            if self.position_feature_dim > 0
            else None
        )
        if self.position_projection is not None:
            nn.init.xavier_uniform_(self.position_projection.weight, gain=0.25)
            nn.init.zeros_(self.position_projection.bias)

        self.relation_aux_encoder = (
            _build_mlp_layers(self.relation_aux_dim, branch_arch, model_dim)
            if self.relation_aux_dim > 0
            else None
        )
        if self.initialization_rng_compat_relation_aux_dim > 0:
            # Removing the critic's allocation-conditioned relation changed the
            # global PyTorch RNG position for every retained module constructed
            # below it.  Consume the exact legacy auxiliary-encoder draws without
            # registering or using that rejected module.  This preserves the
            # established seeded initialization of the relation/fusion/value
            # paths while keeping candidate allocation completely out of the
            # opportunity-set encoder.
            _build_mlp_layers(
                self.initialization_rng_compat_relation_aux_dim,
                branch_arch,
                model_dim,
            )
        self.relation_aux_norm = nn.LayerNorm(model_dim)

        relation_parts = 1 + int(self.relation_aux_dim > 0)
        if relation_parts == 1:
            if self.initialization_rng_compat_relation_aux_dim > 0:
                # The legacy two-part relation seed also consumed parameters
                # before the retained transformer and value heads were built.
                # Burn that initialization stream for seed compatibility only;
                # the production relation seed remains an identity mapping of
                # the action-independent market representation.
                _build_mlp_layers(2 * model_dim, branch_arch, model_dim)
            self.relation_seed_projection = nn.Identity()
        else:
            self.relation_seed_projection = _build_mlp_layers(
                relation_parts * model_dim, branch_arch, model_dim
            )
        self.relation_seed_norm = nn.LayerNorm(model_dim)
        self.relation_output_norm = nn.LayerNorm(model_dim)

        fusion_input_dim = (2 * model_dim) + self.position_latent_dim
        self.fusion_encoder = _build_mlp_layers(
            fusion_input_dim, branch_arch, model_dim
        )
        self.fusion_norm = nn.LayerNorm(model_dim)

        # Keep the semantic streams alive after the general-purpose fusion.
        # These compact lanes are the stable contract consumed directly by
        # the actor and both scalar-Q networks.  ``fused`` remains available as
        # a bounded residual, but it is no longer the only route to control.
        self.control_stream_dim = int(
            self.fam_asset_vector_dim
            if self.fam_enabled
            else min(32, model_dim)
        )
        with torch.random.fork_rng(devices=[]):
            self.local_control_projection = (
                nn.Identity()
                if self.fam_enabled
                and self.fam_asset_vector_dim == self.control_stream_dim
                else nn.Linear(
                    self.fam_asset_vector_dim
                    if self.fam_enabled
                    else self.local_feature_dim,
                    self.control_stream_dim,
                )
            )
            self.relational_control_projection = nn.Linear(
                model_dim, self.control_stream_dim
            )
            if isinstance(self.local_control_projection, nn.Linear):
                nn.init.xavier_uniform_(
                    self.local_control_projection.weight, gain=0.25
                )
                nn.init.zeros_(self.local_control_projection.bias)
            nn.init.xavier_uniform_(
                self.relational_control_projection.weight, gain=0.25
            )
            nn.init.zeros_(self.relational_control_projection.bias)
        self.local_control_norm = nn.LayerNorm(self.control_stream_dim)
        self.relational_control_norm = nn.LayerNorm(self.control_stream_dim)

        # Direct, causally labelled foundation heads.  They live on every
        # production encoder (actor, Q1, and Q2) and remain trainable during
        # TD3; there is no exported or frozen encoder.  The heads are private
        # teachers and never replace the actor logits or Bellman Q scalar.
        foundation_horizon_count = len(
            tuple(getattr(config, "FOUNDATION_HORIZONS", (1, 5, 15, 30)))
        )
        foundation_risk_count = 2 * max(0, foundation_horizon_count - 1)
        self.foundation_local_reconstruction_head = nn.Linear(
            self.fam_asset_vector_dim,
            self.fam_history_length * self.fam_local_feature_dim,
        ) if self.fam_enabled else None
        self.foundation_local_return_head = nn.Linear(
            self.fam_asset_vector_dim,
            foundation_horizon_count,
        ) if self.fam_enabled else None
        self.foundation_local_risk_head = nn.Linear(
            self.fam_asset_vector_dim,
            foundation_risk_count,
        ) if self.fam_enabled else None
        self.foundation_episode_rank_head = nn.Linear(
            model_dim, foundation_horizon_count
        )
        self.foundation_episode_risk_head = nn.Linear(
            model_dim, 4
        )
        for foundation_head in (
            self.foundation_local_reconstruction_head,
            self.foundation_local_return_head,
            self.foundation_local_risk_head,
            self.foundation_episode_rank_head,
            self.foundation_episode_risk_head,
        ):
            if foundation_head is not None:
                nn.init.xavier_uniform_(foundation_head.weight, gain=0.10)
                nn.init.zeros_(foundation_head.bias)

        # Each actor/critic owns a decoder over the same pool encoder used for
        # financial decisions. Whole-stock mode restores learned missing-token
        # queries with identities and decodes using our signed attention.
        # The older coordinate-corruption decoder remains an explicit ablation.
        self.earnmore_masked_reconstruction_enabled = bool(
            self.fam_enabled
            and getattr(config, "EARNMORE_MASKED_RECONSTRUCTION_ENABLED", False)
            and (self.asset_identity_dim == 0 or config.EARNMORE_GLOBAL_POOL_ENABLED)
        )
        self.earnmore_mask_token = None
        self.earnmore_reconstruction_decoder = None
        # Missing-token restoration belongs to the financial pool architecture,
        # even when the independent reconstruction objective is disabled.
        if self.pool_level_enabled or self.earnmore_masked_reconstruction_enabled:
            with torch.random.fork_rng(devices=[]):
                self.earnmore_mask_token = nn.Parameter(torch.zeros(model_dim))
                nn.init.normal_(self.earnmore_mask_token, mean=0.0, std=0.02)
        if self.earnmore_masked_reconstruction_enabled:
            decoder_input_dim = model_dim if self.pool_level_enabled else self.fam_asset_vector_dim + model_dim
            with torch.random.fork_rng(devices=[]):
                self.earnmore_reconstruction_decoder = nn.Sequential(
                    nn.LayerNorm(decoder_input_dim),
                    nn.Linear(decoder_input_dim, model_dim),
                    nn.SiLU(),
                    nn.Linear(
                        model_dim,
                        self.fam_history_length * self.fam_local_feature_dim,
                    ),
                )
                nn.init.xavier_uniform_(
                    self.earnmore_reconstruction_decoder[1].weight, gain=0.25
                )
                nn.init.zeros_(self.earnmore_reconstruction_decoder[1].bias)
                nn.init.xavier_uniform_(
                    self.earnmore_reconstruction_decoder[-1].weight, gain=0.10
                )
                nn.init.zeros_(self.earnmore_reconstruction_decoder[-1].bias)

        resolved_heads = _resolve_attention_heads(model_dim, n_heads)
        ff_dim = max(
            model_dim * 4,
            net_arch[0] if len(net_arch) > 0 else model_dim * 4,
        )
        self.transformer = nn.ModuleList([
            LinearAttentionEncoderLayer(
                d_model=model_dim,
                nhead=resolved_heads,
                dim_feedforward=ff_dim,
                dropout=self.attention_dropout,
                batch_first=True,
                num_layers=n_layers,
            )
            for _ in range(n_layers)
        ])
        if self.pool_level_enabled:
            if not self.fam_enabled:
                raise ValueError("Whole-stock pool learning requires FAM")
            if not self.local_causal_inputs_enabled:
                raise ValueError("Whole-stock pool learning requires local causal inputs")
            # Reuse our signed, non-softmax attention for visible encoding,
            # restored-state reading, and historical reconstruction.
            self.pool_cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            nn.init.normal_(self.pool_cls_token, std=0.02)
            self.pool_readout = LinearAttentionEncoderLayer(
                model_dim, resolved_heads, ff_dim, num_layers=1)
            self.pool_decoder_attention = LinearAttentionEncoderLayer(
                model_dim, resolved_heads, ff_dim, num_layers=1)
        self.market_norm = nn.LayerNorm(model_dim)
        self.market_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=resolved_heads,
            dropout=self.attention_dropout,
            batch_first=True,
        ) if use_market_attention else None
        self.market_path_scale = nn.Parameter(torch.tensor(0.05))
        self.max_market_path_scale = 0.20
        self._reset_market_attention_parameters()

        # Live membership FiLM (nonzero) lives off the main RNG stream used to
        # build the retained April-compatible encoder and twin critics.
        with torch.random.fork_rng(devices=[]):
            if self.basket_membership_dim > 0:
                self.basket_membership_film = BasketMembershipFiLM(
                    self.basket_membership_dim, model_dim, gain=0.1
                )
            if self.predictive_moe_enabled:
                self.predictive_expert_layer = SoftPredictiveExpertLayer(
                    model_dim=model_dim,
                    features_per_coin=self.features_per_coin,
                    basket_context_dim=self.basket_context_dim,
                    basket_membership_dim=self.basket_membership_dim,
                    num_experts=predictive_moe_num_experts,
                    expert_hidden_dim=predictive_moe_expert_hidden_dim,
                    router_temperature=predictive_moe_router_temperature,
                    max_residual_abs=predictive_moe_max_residual_abs,
                )

    def _reset_market_attention_parameters(self) -> None:
        if self.market_attention is None:
            return
        branch_scale = 1.0 / math.sqrt(float(max(1, len(self.transformer))))
        with torch.no_grad():
            self.market_attention.out_proj.weight.mul_(0.25 * branch_scale)
            if self.market_attention.out_proj.bias is not None:
                self.market_attention.out_proj.bias.mul_(0.25 * branch_scale)

    def _encode_components(
        self,
        coin_features: torch.Tensor,
        mask: torch.Tensor,
        relation_aux: Optional[torch.Tensor] = None,
        asset_identity: Optional[torch.Tensor] = None,
        basket_context: Optional[torch.Tensor] = None,
        basket_membership: Optional[torch.Tensor] = None,
        feature_history: Optional[torch.Tensor] = None,
        feature_history_mask: Optional[torch.Tensor] = None,
        detach_predictive_control_adapter: bool = False,
        control_only: bool = False,
        representation_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if coin_features.shape[-1] != self.features_per_coin:
            raise ValueError(
                "Scoped encoder token width mismatch: expected "
                f"{self.features_per_coin}, received {coin_features.shape[-1]}."
            )
        mask = mask.float()
        mask_expanded = mask.unsqueeze(-1)
        if self.pool_level_enabled:
            # Replay carries global historical targets in this tensor. Erase
            # excluded evidence before any learned operation, including FAM.
            coin_features = coin_features.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
            if feature_history is not None:
                feature_history = feature_history.masked_fill(
                    ~mask.bool().unsqueeze(-1).unsqueeze(-1), 0.0)
        local_end = self.local_feature_dim
        position_end = local_end + self.position_feature_dim
        comparative_end = position_end + self.comparative_feature_dim

        if self.local_causal_inputs_enabled:
            coin_features = local_market_token_view(
                coin_features, local_end, self.position_feature_dim,
                self.comparative_feature_dim,
            )
            basket_context = None if basket_context is None else torch.zeros_like(basket_context)
            basket_membership = None if basket_membership is None else torch.zeros_like(basket_membership)
            relation_aux = None if relation_aux is None else torch.zeros_like(relation_aux)
            if feature_history is not None:
                feature_history = torch.cat((
                    feature_history[..., :self.fam_local_feature_dim],
                    torch.zeros_like(feature_history[..., self.fam_local_feature_dim:]),
                ), dim=-1)

        primary_input = coin_features[..., :local_end]
        position_input = coin_features[..., local_end:position_end]
        comparative_input = coin_features[..., position_end:comparative_end]
        context_input = coin_features[..., comparative_end:]

        # Local mode supplies fixed asset-local primary coordinates here;
        # historical mode supplies selected-basket-normalized coordinates.
        # Per-asset behavioral anchors already live in local_feature_history.
        # Preserve the snapshot branch and let subsequent attention learn
        # relationships from these inputs.
        snapshot_comparative = comparative_input[
            ..., self.snapshot_local_anchor_count :
        ]
        market_parts = [primary_input, snapshot_comparative, context_input]
        if self.asset_identity_dim > 0:
            if asset_identity is None:
                raise ValueError(
                    "AttentionDeepSetEncoder requires asset_identity when "
                    f"asset_identity_dim={self.asset_identity_dim}"
                )
            if asset_identity.shape[-1] != self.asset_identity_dim:
                raise ValueError(
                    "asset_identity width mismatch: expected "
                    f"{self.asset_identity_dim}, received {asset_identity.shape[-1]}."
                )
            market_parts.append(asset_identity)
        market_input = torch.cat(market_parts, dim=-1)
        snapshot_embeddings = self.market_token_norm(
            self.market_encoder(market_input)
        ) * mask_expanded
        market_embeddings = snapshot_embeddings
        fam_embeddings = snapshot_embeddings.new_zeros(snapshot_embeddings.shape)
        local_temporal = snapshot_embeddings.new_zeros(
            (
                snapshot_embeddings.shape[0],
                snapshot_embeddings.shape[1],
                self.fam_asset_vector_dim if self.fam_enabled else 0,
            )
        )
        episode_context = snapshot_embeddings.new_zeros(
            (
                snapshot_embeddings.shape[0],
                snapshot_embeddings.shape[1],
                self.fam_asset_vector_dim
                if self.fam_enabled and self.fam_episode_feature_dim > 0
                else 0,
            )
        )
        fam_compressed = torch.cat((local_temporal, episode_context), dim=-1)
        fam_feature_map = snapshot_embeddings.new_zeros(
            (
                snapshot_embeddings.shape[0],
                snapshot_embeddings.shape[1],
                self.fam_history_length if self.fam_enabled else 0,
                self.fam_map_feature_dim if self.fam_enabled else 0,
            )
        )
        fam_gate_weights = snapshot_embeddings.new_zeros(
            (
                snapshot_embeddings.shape[0],
                snapshot_embeddings.shape[1],
                self.fam_feature_dim,
                self.fam_branch_count,
            )
        )
        if self.feature_attention is not None:
            if feature_history is None or feature_history_mask is None:
                raise ValueError(
                    "enabled MDAEN FAM requires feature_history and "
                    "feature_history_mask"
                )
            fam_feature_map, fam_gate_weights = self.feature_attention(
                feature_history,
                mask,
                feature_history_mask,
            )
            branch_maps = fam_feature_map.reshape(
                fam_feature_map.shape[0],
                fam_feature_map.shape[1],
                fam_feature_map.shape[2],
                self.fam_branch_count,
                self.fam_feature_dim,
            )
            local_map = branch_maps[..., : self.fam_local_feature_dim].reshape(
                fam_feature_map.shape[0],
                fam_feature_map.shape[1],
                fam_feature_map.shape[2],
                self.fam_branch_count * self.fam_local_feature_dim,
            )
            local_temporal = self.fam_local_asset_embedding(
                local_map,
                mask,
            )
            if self.fam_episode_asset_embedding is not None:
                episode_map = branch_maps[..., self.fam_local_feature_dim :].reshape(
                    fam_feature_map.shape[0],
                    fam_feature_map.shape[1],
                    fam_feature_map.shape[2],
                    self.fam_branch_count * self.fam_episode_feature_dim,
                )
                episode_context = self.fam_episode_asset_embedding(
                    episode_map,
                    mask,
                )
            fam_compressed = torch.cat((local_temporal, episode_context), dim=-1)
            # AAM may read the local code but the basket-relative objective may
            # not redefine it.  Detach preserves the complete forward value,
            # while local heads and local foundation losses retain exclusive
            # ownership of the peer-independent temporal encoder.
            relational_local_temporal = (
                local_temporal.detach()
                if self.strict_semantic_gradient_ownership
                else local_temporal
            )
            relational_fam_compressed = torch.cat(
                (relational_local_temporal, episode_context), dim=-1
            )
            fam_embeddings = self.fam_to_model_projection(
                relational_fam_compressed
            ) * mask_expanded
            # AAM's association coordinate comes only from the shared FC
            # compression of FAM's complete maps.  The flat snapshot is fused
            # later and therefore cannot alter AAM queries, keys, or values.
            market_embeddings = fam_embeddings
        if representation_mask is not None and not self.pool_level_enabled:
            if self.earnmore_mask_token is None:
                raise RuntimeError(
                    "representation_mask requires EarnMore masked reconstruction"
                )
            representation_mask = representation_mask.bool()
            if representation_mask.shape != mask.shape:
                raise ValueError(
                    "EarnMore representation mask must match the real asset mask"
                )
            if torch.any(representation_mask & ~mask.bool()):
                raise ValueError(
                    "EarnMore cannot synthetically mask an unavailable asset"
                )
            # The shared token marks missing local evidence while retaining the
            # basket-relative episode context.  Retaining that context makes a
            # hidden asset distinguishable by its temporal situation rather
            # than by an impermissible ticker/slot lookup.
            market_embeddings = market_embeddings + (
                representation_mask.to(market_embeddings.dtype).unsqueeze(-1)
                * self.earnmore_mask_token.view(1, 1, -1)
            )
            market_embeddings = market_embeddings * mask_expanded
        if self.asset_position_projection is not None:
            market_embeddings = (
                market_embeddings + self.asset_position_projection(asset_identity)
            ) * mask_expanded
        if self.basket_context_film is not None:
            if basket_context is None:
                raise ValueError(
                    "AttentionDeepSetEncoder requires basket_context when "
                    f"basket_context_dim={self.basket_context_dim}"
                )
            if basket_context.shape[-1] != self.basket_context_dim:
                raise ValueError(
                    "basket_context width mismatch: expected "
                    f"{self.basket_context_dim}, received {basket_context.shape[-1]}."
                )
            market_embeddings = (
                self.basket_context_film(market_embeddings, basket_context)
                * mask_expanded
            )
        if self.basket_membership_film is not None:
            if basket_membership is None:
                raise ValueError(
                    "AttentionDeepSetEncoder requires basket_membership when "
                    f"basket_membership_dim={self.basket_membership_dim}"
                )
            if basket_membership.shape[-1] != self.basket_membership_dim:
                raise ValueError(
                    "basket_membership width mismatch: expected "
                    f"{self.basket_membership_dim}, received "
                    f"{basket_membership.shape[-1]}."
                )
            market_embeddings = (
                self.basket_membership_film(market_embeddings, basket_membership)
                * mask_expanded
            )

        if self.position_projection is not None:
            position_embeddings = torch.tanh(
                self.position_projection(position_input)
            ) * mask_expanded
        else:
            position_embeddings = coin_features.new_zeros(
                (*coin_features.shape[:-1], 0)
            )

        relation_parts = [market_embeddings]
        if self.relation_aux_encoder is not None:
            if relation_aux is None or relation_aux.shape[-1] != self.relation_aux_dim:
                raise ValueError(
                    f"relation_aux must end in {self.relation_aux_dim} fields"
                )
            aux_embeddings = self.relation_aux_norm(
                self.relation_aux_encoder(relation_aux)
            ) * mask_expanded
            relation_parts.append(aux_embeddings)

        relation_seed = self.relation_seed_norm(
            self.relation_seed_projection(torch.cat(relation_parts, dim=-1))
        ) * mask_expanded
        # This is the exact [batch, m, q] tensor consumed by the Assets-
        # Attention Module.  Exposing it in the component dictionary makes the
        # FAM -> shared FC -> AAM contract directly testable.
        aam_input = relation_seed
        attention_mask = ~mask.bool()
        relational_embeddings = relation_seed
        pool_restored = relation_seed
        pool_cls = relation_seed.new_zeros((relation_seed.shape[0], self.model_dim))
        if self.pool_level_enabled:
            pool_restored, pool_cls, relational_embeddings = self._encode_whole_stock_pool(
                relation_seed, mask, asset_identity)
        else:
            for layer in self.transformer:
                relational_embeddings = layer(
                    relational_embeddings, src_key_padding_mask=attention_mask
                )

        if self.market_attention is not None and not self.pool_level_enabled:
            market_attn_input = self.market_norm(relational_embeddings)
            market_context, _ = self.market_attention(
                market_attn_input,
                market_attn_input,
                market_attn_input,
                key_padding_mask=attention_mask,
            )
            market_scale = torch.clamp(
                self.market_path_scale,
                min=0.0,
                max=self.max_market_path_scale,
            ).to(relational_embeddings.dtype)
            relational_embeddings = relational_embeddings + market_scale * market_context

        relational_embeddings = (
            self.relation_output_norm(relational_embeddings)
            * (torch.ones_like(mask_expanded) if self.pool_level_enabled else mask_expanded)
        )
        predictive_routing = relational_embeddings.new_zeros(
            (*relational_embeddings.shape[:-1], 0)
        )
        predictive_expert_mixture = relational_embeddings.new_zeros(
            relational_embeddings.shape
        )
        predictive_expert_outputs = relational_embeddings.new_zeros(
            (*relational_embeddings.shape[:-1], 0, relational_embeddings.shape[-1])
        )
        if self.predictive_expert_layer is not None:
            (
                relational_embeddings,
                predictive_routing,
                predictive_expert_mixture,
                predictive_expert_outputs,
            ) = self.predictive_expert_layer(
                relational_embeddings,
                mask,
                basket_context,
                basket_membership,
                detach_control_adapter=detach_predictive_control_adapter,
            )
            relational_embeddings = relational_embeddings * mask_expanded
        # The general-purpose fused branch is a reader of the semantic streams,
        # not their owner.  In particular, its basket-dependent snapshot may
        # not send a context-specific gradient back into FAM/AAM.  The values
        # are unchanged; only the backward graph is cut at this boundary.
        fusion_market_source = (
            market_embeddings.detach()
            if self.strict_semantic_gradient_ownership
            else market_embeddings
        )
        fusion_relational_source = (
            relational_embeddings.detach()
            if self.strict_semantic_gradient_ownership
            else relational_embeddings
        )
        fusion_market_embeddings = fusion_market_source
        if self.feature_attention is not None:
            snapshot_scale = torch.clamp(
                self.snapshot_residual_scale,
                min=0.0,
                max=self.snapshot_max_residual_scale,
            ).to(snapshot_embeddings.dtype)
            fusion_market_embeddings = self.post_aam_snapshot_norm(
                fusion_market_source + snapshot_scale * snapshot_embeddings
            ) * mask_expanded
        fusion_parts = [fusion_market_embeddings, fusion_relational_source]
        if self.position_projection is not None:
            fusion_parts.append(position_embeddings)
        fused_embeddings = self.fusion_norm(
            self.fusion_encoder(torch.cat(fusion_parts, dim=-1))
        ) * (torch.ones_like(mask_expanded) if self.pool_level_enabled else mask_expanded)
        pooled_mean, valid_counts = _masked_mean_pool(fused_embeddings, mask)
        if self.pool_level_enabled:
            # All restored coordinates inform the state, while action masks
            # still exclude missing stocks. CLS also reaches the cash head.
            pooled_mean = fused_embeddings.mean(dim=1) + pool_cls
        local_control_source = (
            local_temporal if self.feature_attention is not None else primary_input
        )
        local_control = self.local_control_norm(
            self.local_control_projection(local_control_source)
        ) * mask_expanded
        relational_control = self.relational_control_norm(
            self.relational_control_projection(relational_embeddings)
        ) * mask_expanded
        local_set, _ = _masked_mean_pool(local_control, mask)
        relational_set, _ = _masked_mean_pool(relational_control, mask)
        if self.pool_level_enabled:
            relational_set = self.relational_control_norm(
                self.relational_control_projection(pool_cls))
        if control_only:
            return {
                "market": market_embeddings,
                "local_temporal": local_temporal,
                "episode_context": episode_context,
                "relational": relational_embeddings,
                "local_control": local_control,
                "relational_control": relational_control,
                "local_set": local_set,
                "relational_set": relational_set,
                "position": position_embeddings,
                "fused": fused_embeddings,
                "pooled": pooled_mean,
                "valid_counts": valid_counts,
                "pool_restored": pool_restored,
                "pool_cls": pool_cls,
            }
        return {
            "primary_input": primary_input * mask_expanded,
            "comparative_input": comparative_input * mask_expanded,
            "context_input": context_input * mask_expanded,
            "market": market_embeddings,
            "fam": fam_embeddings,
            "local_temporal": local_temporal,
            "episode_context": episode_context,
            "fam_compressed": fam_compressed,
            "fam_feature_map": fam_feature_map,
            "aam_input": aam_input,
            "snapshot_context": snapshot_embeddings,
            # Compatibility name for diagnostics written before the snapshot
            # was explicitly classified as contextual.
            "snapshot_market": snapshot_embeddings,
            "fusion_market": fusion_market_embeddings,
            "position": position_embeddings,
            "relational": relational_embeddings,
            "local_control": local_control,
            "relational_control": relational_control,
            "local_set": local_set,
            "relational_set": relational_set,
            "fused": fused_embeddings,
            "pooled": pooled_mean,
            "valid_counts": valid_counts,
            "predictive_routing": predictive_routing,
            "predictive_expert_mixture": predictive_expert_mixture,
            "predictive_expert_outputs": predictive_expert_outputs,
            "fam_gate_weights": fam_gate_weights,
            "pool_restored": pool_restored,
            "pool_cls": pool_cls,
        }

    def _encode_whole_stock_pool(self, seeds, visible_mask, asset_identity):
        """Visible tokens -> signed attention -> global mask filling -> control.

        No hidden market token enters the encoder. Padding handles different
        visible counts in auxiliary minibatches; CLS is always visible.
        """
        batch, assets, width = seeds.shape
        counts = visible_mask.bool().sum(dim=1)
        kept_count = max(1, int(counts.max().item()))
        order = torch.argsort(visible_mask, dim=1, descending=True, stable=True)
        kept = order[:, :kept_count]
        valid = torch.arange(kept_count, device=seeds.device)[None] < counts[:, None]
        packed = seeds.gather(1, kept.unsqueeze(-1).expand(-1, -1, width))
        packed = packed.masked_fill(~valid.unsqueeze(-1), 0.0)
        encoded = torch.cat((self.pool_cls_token.expand(batch, -1, -1), packed), dim=1)
        padding = torch.cat((torch.zeros_like(valid[:, :1]), ~valid), dim=1)
        for layer in self.transformer:
            encoded = layer(encoded, src_key_padding_mask=padding)
        positions = self.asset_position_projection(asset_identity)
        restored = self.earnmore_mask_token.view(1, 1, width) + positions
        scattered = torch.zeros_like(seeds).scatter(
            1, kept.unsqueeze(-1).expand(-1, -1, width), encoded[:, 1:] * valid.unsqueeze(-1))
        restored = torch.where(visible_mask.bool().unsqueeze(-1), scattered, restored)
        # Control consumes rho, not decoder predictions. This existing-style
        # attention reader lets both visible and missing positions affect Q.
        state = self.pool_readout(torch.cat((encoded[:, :1], restored), dim=1))
        return restored, state[:, 0], state[:, 1:]

    def encode_components(
        self,
        coin_features: torch.Tensor,
        mask: torch.Tensor,
        relation_aux: Optional[torch.Tensor] = None,
        asset_identity: Optional[torch.Tensor] = None,
        basket_context: Optional[torch.Tensor] = None,
        basket_membership: Optional[torch.Tensor] = None,
        feature_history: Optional[torch.Tensor] = None,
        feature_history_mask: Optional[torch.Tensor] = None,
        detach_predictive_control_adapter: bool = False,
        control_only: bool = False,
        representation_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self._encode_components(
            coin_features,
            mask,
            relation_aux,
            asset_identity,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
            detach_predictive_control_adapter,
            control_only,
            representation_mask,
        )

    def sample_earnmore_asset_mask(
        self,
        asset_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample the paper-style truncated-normal mask over real assets.

        The returned boolean mask is synthetic corruption.  It never changes
        the environment availability mask or the portfolio action mask.
        """
        active = asset_mask.bool()
        active_counts = active.sum(dim=1)
        if torch.any(active_counts < 2):
            raise ValueError(
                "EarnMore reconstruction requires at least two active assets"
            )
        ratio_min = float(getattr(config, "EARNMORE_MASK_RATIO_MIN", 0.60))
        ratio_max = float(getattr(config, "EARNMORE_MASK_RATIO_MAX", 0.80))
        ratio_mean = float(getattr(config, "EARNMORE_MASK_RATIO_MEAN", 0.70))
        ratio_std = float(getattr(config, "EARNMORE_MASK_RATIO_STD", 0.10))
        if not 0.0 < ratio_min <= ratio_max < 1.0:
            raise ValueError("EarnMore mask ratio bounds must satisfy 0 < min <= max < 1")
        if ratio_std <= 0.0:
            ratios = asset_mask.new_full(
                (asset_mask.shape[0],),
                fill_value=min(max(ratio_mean, ratio_min), ratio_max),
            )
        else:
            # Inverse-CDF sampling gives an actual truncated Gaussian rather
            # than clipping a Gaussian and accumulating mass at the bounds.
            sqrt_two = math.sqrt(2.0)
            low_z = (ratio_min - ratio_mean) / ratio_std
            high_z = (ratio_max - ratio_mean) / ratio_std
            low_cdf = 0.5 * (1.0 + math.erf(low_z / sqrt_two))
            high_cdf = 0.5 * (1.0 + math.erf(high_z / sqrt_two))
            uniform = torch.rand(
                asset_mask.shape[0],
                device=asset_mask.device,
                dtype=asset_mask.dtype,
            )
            probability = low_cdf + uniform * (high_cdf - low_cdf)
            z_value = sqrt_two * torch.erfinv(
                (2.0 * probability - 1.0).clamp(-0.999999, 0.999999)
            )
            ratios = (ratio_mean + ratio_std * z_value).clamp(
                ratio_min, ratio_max
            )
        masked_counts = torch.round(
            active_counts.to(ratios.dtype) * ratios
        ).to(torch.long)
        masked_counts = torch.minimum(
            torch.maximum(masked_counts, torch.ones_like(masked_counts)),
            active_counts - 1,
        )
        random_priority = torch.rand(
            active.shape,
            device=asset_mask.device,
            dtype=asset_mask.dtype,
        ).masked_fill(~active, -1.0)
        order = random_priority.argsort(dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1,
            order,
            torch.arange(
                active.shape[1], device=asset_mask.device
            ).view(1, -1).expand_as(order),
        )
        synthetic_mask = active & (ranks < masked_counts.unsqueeze(1))
        return synthetic_mask, ratios

    def earnmore_masked_reconstruction_loss(
        self,
        coin_features: torch.Tensor,
        asset_mask: torch.Tensor,
        *,
        asset_identity: Optional[torch.Tensor] = None,
        basket_context: Optional[torch.Tensor],
        basket_membership: Optional[torch.Tensor],
        feature_history: torch.Tensor,
        feature_history_mask: torch.Tensor,
        synthetic_mask: Optional[torch.Tensor] = None,
        reconstruction_mask: Optional[torch.Tensor] = None,
        pool_history_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct hidden local histories from temporal and peer context."""
        if self.pool_level_enabled:
            return self._whole_stock_reconstruction_loss(
                coin_features, asset_mask, asset_identity, feature_history,
                feature_history_mask, pool_history_valid, synthetic_mask)
        if not self.earnmore_masked_reconstruction_enabled:
            zero = coin_features.sum() * 0.0
            return {
                "loss": zero,
                "masked_asset_fraction": zero.detach(),
                "sampled_mask_ratio": zero.detach(),
                "masked_local_coordinate_fraction": zero.detach(),
                "masked_assets_per_sample": zero.detach(),
                "visible_assets_per_sample": asset_mask.sum(dim=1).float().mean().detach(),
            }
        if synthetic_mask is None:
            synthetic_mask, sampled_ratios = self.sample_earnmore_asset_mask(
                asset_mask
            )
        else:
            synthetic_mask = synthetic_mask.bool()
            sampled_ratios = (
                synthetic_mask.sum(dim=1).float()
                / asset_mask.sum(dim=1).float().clamp(min=1.0)
            )
        if synthetic_mask.shape != asset_mask.shape:
            raise ValueError("synthetic_mask must match asset_mask")
        if torch.any(synthetic_mask & ~asset_mask.bool()):
            raise ValueError("synthetic_mask may contain only active assets")

        local_width = self.fam_local_feature_dim
        # With no ticker/slot identity, removing an entire asset leaves no
        # principled query that can say which hidden target should be decoded.
        # Keep a partial causal history and mask coordinates only inside the
        # selected stock tokens.  This retains EarnMore's stock-level
        # mask law while making the reconstruction problem identifiable from
        # temporal evidence rather than identity.
        local_mask_probability = float(
            getattr(config, "EARNMORE_LOCAL_HISTORY_MASK_PROBABILITY", 0.50)
        )
        if not 0.0 < local_mask_probability < 1.0:
            raise ValueError(
                "EARNMORE_LOCAL_HISTORY_MASK_PROBABILITY must be in (0, 1)"
            )
        valid_history = feature_history_mask.bool().unsqueeze(1).unsqueeze(-1)
        valid_history = valid_history.expand(
            -1, feature_history.shape[1], -1, local_width
        )
        # The flat current observation contains the final history row.  Do not
        # score that row as reconstruction because it would be a direct target
        # copy rather than temporal representation learning.
        historical_only = torch.arange(
            feature_history.shape[-2], device=feature_history.device
        ) < (feature_history.shape[-2] - 1)
        valid_history = valid_history & historical_only.view(1, 1, -1, 1)
        eligible = valid_history & synthetic_mask.unsqueeze(-1).unsqueeze(-1)
        flat_eligible = eligible.reshape(-1, eligible.shape[-2] * eligible.shape[-1])
        valid_counts = flat_eligible.sum(dim=1)
        coordinate_counts = torch.round(
            valid_counts.float() * local_mask_probability
        ).to(torch.long)
        coordinate_counts = torch.where(
            valid_counts >= 2,
            torch.minimum(
                torch.maximum(coordinate_counts, torch.ones_like(coordinate_counts)),
                valid_counts - 1,
            ),
            valid_counts,
        )
        coordinate_priority = torch.rand(
            flat_eligible.shape,
            device=feature_history.device,
            dtype=feature_history.dtype,
        ).masked_fill(~flat_eligible, -1.0)
        coordinate_order = coordinate_priority.argsort(dim=1, descending=True)
        coordinate_ranks = torch.empty_like(coordinate_order)
        coordinate_ranks.scatter_(
            1,
            coordinate_order,
            torch.arange(
                flat_eligible.shape[1], device=feature_history.device
            ).view(1, -1).expand_as(coordinate_order),
        )
        sampled_reconstruction_mask = (
            flat_eligible & (coordinate_ranks < coordinate_counts.unsqueeze(1))
        ).reshape_as(eligible)
        if reconstruction_mask is None:
            reconstruction_mask = sampled_reconstruction_mask
        else:
            reconstruction_mask = reconstruction_mask.bool()
            if reconstruction_mask.shape != eligible.shape:
                raise ValueError(
                    "reconstruction_mask must match the local-history target"
                )
            if torch.any(reconstruction_mask & ~eligible):
                raise ValueError(
                    "reconstruction_mask may contain only eligible historical "
                    "coordinates of synthetically masked assets"
                )

        corrupted_features = coin_features
        corrupted_history = feature_history.clone()
        corrupted_history[..., :local_width] = (
            corrupted_history[..., :local_width].masked_fill(
                reconstruction_mask, 0.0
            )
        )
        components = self.encode_components(
            corrupted_features,
            asset_mask,
            asset_identity=asset_identity,
            basket_context=basket_context,
            basket_membership=basket_membership,
            feature_history=corrupted_history,
            feature_history_mask=feature_history_mask,
            representation_mask=synthetic_mask,
        )
        decoder_input = torch.cat(
            (components["local_temporal"], components["relational"]), dim=-1
        )
        target = feature_history[..., :local_width].detach()
        prediction = self.earnmore_reconstruction_decoder(decoder_input).reshape_as(
            target
        )
        valid = reconstruction_mask
        squared_error = F.mse_loss(prediction, target, reduction="none")
        valid_float = valid.to(squared_error.dtype)
        loss = (squared_error * valid_float).sum() / valid_float.sum().clamp(min=1.0)
        active_count = asset_mask.sum(dim=1).float().clamp(min=1.0)
        masked_count = synthetic_mask.sum(dim=1).float()
        eligible_count = eligible.sum().to(loss.dtype).clamp(min=1.0)
        return {
            "loss": loss,
            "masked_asset_fraction": (masked_count / active_count).mean().detach(),
            "sampled_mask_ratio": sampled_ratios.mean().detach(),
            "masked_local_coordinate_fraction": (
                reconstruction_mask.sum().to(loss.dtype) / eligible_count
            ).detach(),
            "masked_assets_per_sample": masked_count.mean().detach(),
            "visible_assets_per_sample": (active_count - masked_count).mean().detach(),
            "prediction": prediction,
            "target": target,
            "synthetic_mask": synthetic_mask,
            "reconstruction_mask": reconstruction_mask,
        }

    def _whole_stock_reconstruction_loss(self, features, real_mask, identity,
                                         history, time_mask, history_valid, hidden):
        if history_valid is None:
            raise ValueError("Whole-stock reconstruction requires global pool_history_valid targets")
        valid = history_valid.bool() & time_mask.bool().unsqueeze(1)
        eligible = valid.any(dim=-1)
        if hidden is None:
            hidden, _ = self.sample_earnmore_asset_mask(eligible.float())
            use_real = torch.rand((features.shape[0], 1), device=features.device) < config.EARNMORE_REAL_POOL_MASK_FRACTION
            hidden = torch.where(use_real, ~real_mask.bool() & eligible, hidden)
        hidden = hidden.bool()
        if hidden.shape != real_mask.shape or torch.any(hidden & ~eligible):
            raise ValueError("Whole-stock mask must select assets with historical targets")
        visible = eligible & ~hidden
        if torch.any(~visible.any(dim=1)) or torch.any(~hidden.any(dim=1)):
            raise ValueError("Whole-stock reconstruction needs visible and hidden assets")
        # EarnMore uses two reconstruction paths.  Queries come from the masked
        # pool and therefore contain learned mask/identity tokens at hidden stock
        # positions.  Keys and values come from a second, fully unmasked encoder
        # pass over complete stock histories.  This is whole-stock masking, not
        # temporal masking: no time coordinate inside a stock is removed.
        # Keep the visible and full passes separate.  Their compact token
        # sequences have different widths (eight versus thirteen stocks in the
        # production setup).  Concatenating them forces the visible pass to
        # carry full-pool padding through every attention layer, which both
        # adds work and changes the numerical learning path.  Stateless replay
        # preparation is shared outside this method instead.
        masked_components = self.encode_components(
            features,
            visible.float(),
            asset_identity=identity,
            basket_context=features.new_zeros(
                (features.shape[0], self.basket_context_dim)
            ),
            basket_membership=features.new_zeros(
                (features.shape[0], self.basket_membership_dim)
            ),
            feature_history=history,
            feature_history_mask=time_mask,
        )
        full_components = self.encode_components(
            features,
            eligible.float(),
            asset_identity=identity,
            basket_context=features.new_zeros(
                (features.shape[0], self.basket_context_dim)
            ),
            basket_membership=features.new_zeros(
                (features.shape[0], self.basket_membership_dim)
            ),
            feature_history=history,
            feature_history_mask=time_mask,
        )
        masked_pool_cls = masked_components["pool_cls"]
        masked_pool_restored = masked_components["pool_restored"]
        full_pool_cls = full_components["pool_cls"]
        full_pool_restored = full_components["pool_restored"]
        decoder_tokens = torch.cat(
            (masked_pool_cls.unsqueeze(1), masked_pool_restored),
            dim=1,
        )
        decoder_memory = torch.cat(
            (
                full_pool_cls.unsqueeze(1),
                full_pool_restored,
            ),
            dim=1,
        )
        decoder_padding = torch.cat(
            (
                torch.zeros(
                    (features.shape[0], 1),
                    dtype=torch.bool,
                    device=features.device,
                ),
                ~eligible,
            ),
            dim=1,
        )
        decoded = self.pool_decoder_attention(
            decoder_tokens,
            src_key_padding_mask=decoder_padding,
            memory=decoder_memory,
            memory_key_padding_mask=decoder_padding,
        )[:, 1:]
        target = history[..., :self.fam_local_feature_dim].detach()
        prediction = self.earnmore_reconstruction_decoder(decoded).reshape_as(target)
        scored = (hidden.unsqueeze(-1) & valid).unsqueeze(-1).expand_as(target)
        error = (prediction - target).square()
        loss = error.masked_fill(~scored, 0.0).sum() / scored.sum().clamp(min=1)
        fraction = hidden.sum(dim=1).float() / eligible.sum(dim=1).clamp(min=1)
        return dict(loss=loss, masked_asset_fraction=fraction.mean().detach(),
                    sampled_mask_ratio=fraction.mean().detach(),
                    masked_local_coordinate_fraction=loss.new_tensor(1.),
                    masked_assets_per_sample=hidden.sum(1).float().mean().detach(),
                    visible_assets_per_sample=visible.sum(1).float().mean().detach(),
                    prediction=prediction, target=target, synthetic_mask=hidden,
                    reconstruction_mask=scored,
                    pool_restored=masked_pool_restored,
                    decoder_memory=decoder_memory)

    @staticmethod
    def _align_next_tokens(
        tensor: torch.Tensor,
        current_asset_ids: torch.Tensor,
        next_asset_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align a next-state token tensor to the current canonical asset IDs."""
        matches = current_asset_ids.unsqueeze(-1).eq(
            next_asset_ids.unsqueeze(1)
        )
        has_match = matches.any(dim=-1)
        next_index = matches.to(torch.int64).argmax(dim=-1)
        gather_shape = [*next_index.shape, *tensor.shape[2:]]
        gather_index = next_index.view(
            *next_index.shape, *([1] * (tensor.dim() - 2))
        ).expand(*gather_shape)
        return tensor.gather(1, gather_index), has_match

    @staticmethod
    def _masked_scalar_mean(
        values: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        while token_mask.dim() < values.dim():
            token_mask = token_mask.unsqueeze(-1)
        expanded_mask = token_mask.expand_as(values).to(values.dtype)
        return (values * expanded_mask).sum() / expanded_mask.sum().clamp(min=1.0)

    def foundation_representation_loss(
        self,
        components: Dict[str, torch.Tensor],
        feature_history_target: torch.Tensor,
        asset_mask: torch.Tensor,
        future_unit_returns: torch.Tensor,
        future_valid: torch.Tensor,
        peer_local_temporal: Optional[torch.Tensor] = None,
        shared_peer_mask: Optional[torch.Tensor] = None,
        local_reconstruction_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Dense local and relational supervision for a live encoder.

        Future outcomes are labels only.  They are never concatenated into the
        state representation.  ``future_unit_returns`` uses unscaled simple
        per-asset returns with shape [batch, steps, assets].
        """
        if self.foundation_local_return_head is None:
            zero = components["fused"].sum() * 0.0
            return {"loss": zero, "local_loss": zero, "episode_loss": zero}
        returns = future_unit_returns.float()
        if returns.dim() == 4 and returns.shape[-1] == 1:
            returns = returns[..., 0]
        if returns.dim() != 3:
            raise ValueError(
                "future_unit_returns must have [batch, steps, assets]"
            )
        batch, steps, assets = returns.shape
        if assets != asset_mask.shape[1]:
            raise ValueError("future return assets must match asset_mask")
        valid = future_valid.bool()
        if valid.dim() == 2:
            valid = valid.unsqueeze(-1).expand(-1, -1, assets)
        if valid.shape != returns.shape:
            raise ValueError("future_valid must broadcast to future returns")

        safe_returns = torch.where(valid, returns, torch.zeros_like(returns))
        safe_returns = safe_returns.clamp(min=-0.95, max=5.0)
        cumulative = torch.cumprod(1.0 + safe_returns, dim=1) - 1.0
        horizons = tuple(int(v) for v in getattr(
            config, "FOUNDATION_HORIZONS", (1, 5, 15, 30)
        ))
        if steps < max(horizons):
            raise ValueError(
                f"foundation sequence has {steps} steps but requires {max(horizons)}"
            )
        scale = float(getattr(config, "FINANCIAL_REWARD_SCALE", 10.0))
        return_targets = torch.stack(
            [cumulative[:, horizon - 1] * scale for horizon in horizons],
            dim=-1,
        ).clamp(
            min=-float(getattr(config, "COMPONENT_OUTCOME_TARGET_CLIP", 5.0)),
            max=float(getattr(config, "COMPONENT_OUTCOME_TARGET_CLIP", 5.0)),
        )
        horizon_valid = torch.stack(
            [valid[:, :horizon].all(dim=1) for horizon in horizons], dim=-1
        ) & asset_mask.bool().unsqueeze(-1)

        local_code = components["local_temporal"]
        local_return_prediction = self.foundation_local_return_head(local_code)
        local_return_loss = self._masked_scalar_mean(
            F.smooth_l1_loss(
                local_return_prediction, return_targets, reduction="none"
            ),
            horizon_valid,
        )
        local_history_target = feature_history_target[
            ..., : self.fam_local_feature_dim
        ].float()
        reconstruction_prediction = self.foundation_local_reconstruction_head(
            local_code
        ).reshape_as(local_history_target)
        if local_reconstruction_mask is None:
            reconstruction_valid = (
                asset_mask.bool().unsqueeze(-1).unsqueeze(-1)
            ).expand_as(local_history_target)
        else:
            reconstruction_valid = (
                local_reconstruction_mask.bool()
                & asset_mask.bool().unsqueeze(-1).unsqueeze(-1)
            )
        reconstruction_loss = self._masked_scalar_mean(
            F.smooth_l1_loss(
                reconstruction_prediction,
                local_history_target,
                reduction="none",
            ),
            reconstruction_valid,
        )

        risk_targets = []
        risk_valid = []
        for horizon in horizons[1:]:
            window = safe_returns[:, :horizon]
            window_valid = valid[:, :horizon].all(dim=1) & asset_mask.bool()
            volatility = window.std(dim=1, unbiased=False) * scale
            downside = torch.sqrt(
                torch.square(torch.minimum(window, torch.zeros_like(window)))
                .mean(dim=1)
                .clamp(min=1e-12)
            ) * scale
            risk_targets.extend((volatility, downside))
            risk_valid.extend((window_valid, window_valid))
        risk_target = torch.stack(risk_targets, dim=-1)
        risk_target_valid = torch.stack(risk_valid, dim=-1)
        local_risk_prediction = self.foundation_local_risk_head(local_code)
        local_risk_loss = self._masked_scalar_mean(
            F.smooth_l1_loss(
                local_risk_prediction, risk_target, reduction="none"
            ),
            risk_target_valid,
        )

        consistency_loss = local_return_loss.new_zeros(())
        if peer_local_temporal is not None:
            consistency_mask = (
                asset_mask.bool()
                if shared_peer_mask is None
                else shared_peer_mask.bool() & asset_mask.bool()
            )
            consistency_loss = self._masked_scalar_mean(
                1.0 - F.cosine_similarity(
                    local_code, peer_local_temporal.detach(), dim=-1
                ),
                consistency_mask,
            )
        local_loss = torch.stack(
            (
                reconstruction_loss,
                local_return_loss,
                local_risk_loss,
                consistency_loss,
            )
        ).mean()

        relational_code = components["relational"]
        rank_prediction = self.foundation_episode_rank_head(relational_code)
        listwise_losses = []
        pairwise_losses = []
        pair_indices = torch.triu_indices(
            assets, assets, offset=1, device=returns.device
        )
        for horizon_index in range(len(horizons)):
            active = horizon_valid[..., horizon_index]
            target_logits = return_targets[..., horizon_index] * 5.0
            target_logits = target_logits.masked_fill(~active, -1e4)
            predicted_logits = rank_prediction[..., horizon_index].masked_fill(
                ~active, -1e4
            )
            target_distribution = torch.softmax(target_logits, dim=1).detach()
            # Use listwise KL rather than raw cross entropy.  Cross entropy
            # contains the target distribution's irreducible entropy (about
            # log(number of active assets) for small return spreads), which
            # makes a perfectly learned ranker report a large non-zero loss
            # and distorts equal group authority.  KL retains the same optimum
            # and gradients while measuring only prediction error.
            target_log_probability = torch.log(
                target_distribution.clamp(min=1e-8)
            )
            listwise = (
                target_distribution
                * (
                    target_log_probability
                    - torch.log_softmax(predicted_logits, dim=1)
                )
            ).sum(dim=1)
            listwise_losses.append(
                listwise[active.any(dim=1)].mean()
                if active.any()
                else listwise.sum() * 0.0
            )
            left, right = pair_indices
            pair_valid = active[:, left] & active[:, right]
            target_preference = (
                return_targets[:, left, horizon_index]
                > return_targets[:, right, horizon_index]
            ).to(rank_prediction.dtype)
            predicted_difference = (
                rank_prediction[:, left, horizon_index]
                - rank_prediction[:, right, horizon_index]
            )
            pair_error = F.binary_cross_entropy_with_logits(
                predicted_difference, target_preference, reduction="none"
            )
            pairwise_losses.append(
                pair_error[pair_valid].mean()
                if pair_valid.any()
                else pair_error.sum() * 0.0
            )
        listwise_loss = torch.stack(listwise_losses).mean()
        pairwise_loss = torch.stack(pairwise_losses).mean()
        episode_horizon = max(horizons)
        episode_window = safe_returns[:, :episode_horizon]
        active_float = asset_mask.float()
        basket_return = (
            episode_window * active_float.unsqueeze(1)
        ).sum(dim=2) / active_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        asset_volatility = episode_window.std(dim=1, unbiased=False)
        asset_downside = torch.sqrt(
            torch.minimum(episode_window, torch.zeros_like(episode_window))
            .square()
            .mean(dim=1)
            .clamp(min=1e-12)
        )
        peer_volatility = []
        peer_downside = []
        for asset_index in range(assets):
            peer_mask = asset_mask.bool().clone()
            peer_mask[:, asset_index] = False
            peer_volatility.append(
                asset_volatility.masked_fill(~peer_mask, float("nan"))
                .nanmedian(dim=1)
                .values
            )
            peer_downside.append(
                asset_downside.masked_fill(~peer_mask, float("nan"))
                .nanmedian(dim=1)
                .values
            )
        peer_volatility = torch.stack(peer_volatility, dim=1)
        peer_downside = torch.stack(peer_downside, dim=1)
        relative_volatility_target = torch.log(
            (asset_volatility + 1e-6) / (peer_volatility + 1e-6)
        ).clamp(-3.0, 3.0)
        relative_downside_target = (
            (asset_downside - peer_downside) * scale
        ).clamp(-3.0, 3.0)
        centered_asset = episode_window - episode_window.mean(
            dim=1, keepdim=True
        )
        centered_basket = basket_return - basket_return.mean(
            dim=1, keepdim=True
        )
        covariance = (
            centered_asset * centered_basket.unsqueeze(-1)
        ).mean(dim=1)
        basket_variance = centered_basket.square().mean(dim=1).unsqueeze(-1)
        asset_standard_deviation = torch.sqrt(
            centered_asset.square().mean(dim=1).clamp(min=1e-12)
        )
        basket_standard_deviation = torch.sqrt(
            basket_variance.clamp(min=1e-12)
        )
        correlation_target = (
            covariance
            / (asset_standard_deviation * basket_standard_deviation + 1e-8)
        ).clamp(-1.0, 1.0)
        beta_target = (covariance / (basket_variance + 1e-8)).clamp(-3.0, 3.0)
        episode_risk_target = torch.stack(
            (
                relative_volatility_target,
                relative_downside_target,
                correlation_target,
                beta_target,
            ),
            dim=-1,
        )
        episode_risk_valid = (
            valid[:, :episode_horizon].all(dim=1)
            & asset_mask.bool()
        ).unsqueeze(-1).expand_as(episode_risk_target)
        episode_risk_prediction = self.foundation_episode_risk_head(
            relational_code
        )
        episode_risk_loss = self._masked_scalar_mean(
            F.smooth_l1_loss(
                episode_risk_prediction,
                episode_risk_target,
                reduction="none",
            ),
            episode_risk_valid,
        )
        episode_loss = torch.stack(
            (listwise_loss, pairwise_loss, episode_risk_loss)
        ).mean()
        total = 0.5 * (local_loss + episode_loss)
        return {
            "loss": total,
            "local_loss": local_loss,
            "episode_loss": episode_loss,
            "reconstruction_loss": reconstruction_loss,
            "local_return_loss": local_return_loss,
            "local_risk_loss": local_risk_loss,
            "local_consistency_loss": consistency_loss,
            "listwise_loss": listwise_loss,
            "pairwise_loss": pairwise_loss,
            "episode_risk_loss": episode_risk_loss,
            "return_prediction": local_return_prediction,
            "return_target": return_targets,
            "rank_prediction": rank_prediction,
            "horizon_valid": horizon_valid,
        }

    def predictive_loss(
        self,
        current_features: torch.Tensor,
        current_components: Dict[str, torch.Tensor],
        current_asset_ids: torch.Tensor,
        current_mask: torch.Tensor,
        next_features: torch.Tensor,
        next_components: Dict[str, torch.Tensor],
        next_asset_ids: torch.Tensor,
        next_mask: torch.Tensor,
        action_roles: torch.Tensor,
        transition_valid: Optional[torch.Tensor] = None,
        outcome_valid_steps: Optional[torch.Tensor] = None,
        future_rewards: Optional[torch.Tensor] = None,
        future_dones: Optional[torch.Tensor] = None,
        continuation_value_targets: Optional[torch.Tensor] = None,
        current_q_targets: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        feature_weight: float = 1.0,
        fam_weight: float = 0.0,
        market_weight: float = 0.25,
        relation_weight: float = 0.50,
        fused_weight: float = 1.0,
        load_balance_weight: float = 0.01,
        diversity_weight: float = 0.01,
        routing_entropy_weight: float = 0.01,
        horizon_discount: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Predict a value-equivalent multi-step transition in the control latent.

        Targets are produced by the slowly moving TD3 target encoder and are
        detached by the caller. Terminal rewards remain genuine model targets;
        terminal reset observations and their continuation values are excluded
        because they are not physical market transitions. Every nonterminal
        replay token remains available to the low-weight empirical auxiliaries.
        """
        layer = self.predictive_expert_layer
        if layer is None:
            zero = current_features.sum() * 0.0
            return {"loss": zero, "valid_fraction": zero.detach()}

        # Backward-compatible one-step callers are lifted into a horizon axis.
        if next_features.dim() == current_features.dim():
            next_features = next_features.unsqueeze(1)
            next_asset_ids = next_asset_ids.unsqueeze(1)
            next_mask = next_mask.unsqueeze(1)
            next_components = {
                name: value.unsqueeze(1)
                for name, value in next_components.items()
            }
        if action_roles.dim() == 3:
            action_roles = action_roles.unsqueeze(1)
        horizon = int(next_features.shape[1])
        if int(action_roles.shape[1]) != horizon:
            raise ValueError("predictive action/target horizons do not match")
        if transition_valid is None:
            transition_valid = torch.ones(
                (current_features.shape[0], horizon),
                dtype=torch.bool,
                device=current_features.device,
            )
        else:
            transition_valid = transition_valid.bool().reshape(
                current_features.shape[0], horizon
            )
        if outcome_valid_steps is None:
            outcome_valid_steps = transition_valid
        else:
            outcome_valid_steps = outcome_valid_steps.bool().reshape(
                current_features.shape[0], horizon
            )
        horizon_discount = float(horizon_discount)
        if not 0.0 < horizon_discount <= 1.0:
            raise ValueError("predictive horizon_discount must be in (0, 1]")

        rollout = layer.predict_rollout(
            current_features,
            current_components,
            action_roles,
            current_mask.float(),
        )
        feature_numerator = current_features.new_zeros(())
        latent_numerators = {
            name: current_features.new_zeros(())
            for name in ("fam", "market", "relational", "fused")
        }
        weight_total = current_features.new_zeros(())
        valid_token_total = current_features.new_zeros(())
        possible_token_total = current_features.new_zeros(())
        per_horizon_losses = []
        value_equivalent_enabled = bool(
            getattr(config, "VALUE_EQUIVALENT_MODEL_ENABLED", True)
            and future_rewards is not None
            and future_dones is not None
            and continuation_value_targets is not None
            and current_q_targets is not None
        )
        if value_equivalent_enabled:
            future_rewards = future_rewards.float().reshape(
                current_features.shape[0], horizon, 1
            )
            future_dones = future_dones.float().reshape(
                current_features.shape[0], horizon, 1
            )
            continuation_value_targets = (
                continuation_value_targets.detach().float().reshape(
                    current_features.shape[0], horizon, 1
                )
            )
            current_value = layer.state_value(current_components)
            detached_q = current_q_targets.detach().float().reshape(-1, 1)
            expectile_loss = expectile_regression_loss(
                current_value,
                detached_q,
                expectile=float(
                    getattr(config, "VALUE_EQUIVALENT_EXPECTILE", 0.80)
                ),
            ).mean()
            advantage = detached_q - current_value.detach()
            advantage_weights = bounded_advantage_weights(
                advantage,
                temperature=float(
                    getattr(
                        config,
                        "VALUE_EQUIVALENT_ADVANTAGE_TEMPERATURE",
                        25.0,
                    )
                ),
                minimum=float(
                    getattr(
                        config,
                        "VALUE_EQUIVALENT_ADVANTAGE_WEIGHT_MIN",
                        0.25,
                    )
                ),
                maximum=float(
                    getattr(
                        config,
                        "VALUE_EQUIVALENT_ADVANTAGE_WEIGHT_MAX",
                        4.0,
                    )
                ),
            )
        else:
            expectile_loss = current_features.new_zeros(())
            current_value = current_features.new_zeros(
                (current_features.shape[0], 1)
            )
            advantage = current_features.new_zeros(
                (current_features.shape[0], 1)
            )
            advantage_weights = torch.ones_like(advantage)
        reward_numerator = current_features.new_zeros(())
        continuation_numerator = current_features.new_zeros(())
        bellman_numerator = current_features.new_zeros(())
        outcome_denominator = current_features.new_zeros(())
        continuation_denominator = current_features.new_zeros(())
        per_horizon_value_losses: List[torch.Tensor] = []
        for step, predictions in enumerate(rollout):
            step_next_ids = next_asset_ids[:, step]
            step_next_mask = next_mask[:, step]
            aligned_next_features, has_match = self._align_next_tokens(
                next_features[:, step], current_asset_ids, step_next_ids
            )
            aligned_next_mask, _ = self._align_next_tokens(
                step_next_mask.unsqueeze(-1),
                current_asset_ids,
                step_next_ids,
            )
            valid = (
                current_mask.bool()
                & has_match
                & aligned_next_mask.squeeze(-1).bool()
                & transition_valid[:, step].view(-1, 1)
            )
            valid_float = valid.to(current_features.dtype)
            valid_count = valid_float.sum()
            possible_token_total = possible_token_total + current_mask.sum()
            valid_token_total = valid_token_total + valid_count
            step_weight = current_features.new_tensor(
                horizon_discount ** step
            ) * (valid_count > 0).to(current_features.dtype)

            aligned_targets: Dict[str, torch.Tensor] = {
                "features": aligned_next_features,
            }
            for name in ("fam", "market", "relational", "fused"):
                aligned_targets[name], _ = self._align_next_tokens(
                    next_components[name][:, step],
                    current_asset_ids,
                    step_next_ids,
                )

            # Reserved contextual slots are not prediction tasks in local mode.
            # Fitting their zeros would dilute useful feature supervision and
            # retain a redundant objective for removed input semantics.
            feature_count = (
                self.local_feature_dim + self.position_feature_dim
                + min(3, self.comparative_feature_dim)
                if self.local_causal_inputs_enabled else self.features_per_coin
            )
            step_feature_loss = self._masked_scalar_mean(
                F.smooth_l1_loss(
                    predictions["features"][..., :feature_count],
                    aligned_targets["features"][..., :feature_count],
                    reduction="none",
                ),
                valid_float,
            )
            step_latent_losses = {}
            for name in ("fam", "market", "relational", "fused"):
                cosine_error = 1.0 - F.cosine_similarity(
                    predictions[name], aligned_targets[name], dim=-1
                )
                step_latent_losses[name] = self._masked_scalar_mean(
                    cosine_error, valid_float
                )
                latent_numerators[name] = (
                    latent_numerators[name]
                    + step_weight * step_latent_losses[name]
                )
            feature_numerator = (
                feature_numerator + step_weight * step_feature_loss
            )
            weight_total = weight_total + step_weight
            per_horizon_losses.append(
                step_feature_loss
                + step_latent_losses["market"]
                + step_latent_losses["relational"]
                + step_latent_losses["fused"]
            )
            if value_equivalent_enabled:
                outcome_mask = outcome_valid_steps[:, step].float().view(-1, 1)
                continuation_mask = transition_valid[:, step].float().view(-1, 1)
                temporal_weight = current_features.new_tensor(
                    horizon_discount ** step
                )
                weighted_outcome_mask = (
                    temporal_weight * outcome_mask * advantage_weights
                )
                weighted_continuation_mask = (
                    temporal_weight * continuation_mask * advantage_weights
                )
                pooled_outcome_hidden = layer._masked_pool(
                    predictions["outcome_hidden"], current_mask.float()
                )
                predicted_reward = layer.reward_head.mean(
                    pooled_outcome_hidden
                )
                per_sample_reward_loss = layer.reward_head.loss(
                    pooled_outcome_hidden, future_rewards[:, step]
                )
                predicted_next_components = {
                    "pooled": layer._masked_pool(
                        predictions["fused"], current_mask.float()
                    )
                }
                predicted_next_value = layer.state_value(
                    predicted_next_components
                )
                per_sample_continuation_loss = layer.state_value_head.loss(
                    predicted_next_components["pooled"],
                    continuation_value_targets[:, step],
                )
                sampled_backup = value_equivalent_bellman_backup(
                    future_rewards[:, step],
                    future_dones[:, step],
                    continuation_value_targets[:, step],
                    gamma=gamma,
                )
                predicted_backup = value_equivalent_bellman_backup(
                    predicted_reward,
                    future_dones[:, step],
                    predicted_next_value,
                    gamma=gamma,
                )
                per_sample_bellman_loss = F.smooth_l1_loss(
                    predicted_backup,
                    sampled_backup,
                    reduction="none",
                )
                reward_numerator = reward_numerator + (
                    per_sample_reward_loss * weighted_outcome_mask
                ).sum()
                continuation_numerator = continuation_numerator + (
                    per_sample_continuation_loss
                    * weighted_continuation_mask
                ).sum()
                bellman_numerator = bellman_numerator + (
                    per_sample_bellman_loss * weighted_outcome_mask
                ).sum()
                outcome_denominator = outcome_denominator + (
                    weighted_outcome_mask.sum()
                )
                continuation_denominator = continuation_denominator + (
                    weighted_continuation_mask.sum()
                )
                per_horizon_value_losses.append(
                    (
                        per_sample_reward_loss
                        + per_sample_bellman_loss
                    ).mul(outcome_mask).sum()
                    / outcome_mask.sum().clamp(min=1.0)
                )

        denominator = weight_total.clamp(min=1.0)
        feature_loss = feature_numerator / denominator
        latent_losses = {
            name: value / denominator
            for name, value in latent_numerators.items()
        }
        # MoE regularizers describe the observed starting representation and
        # therefore enter once, rather than becoming five times stronger merely
        # because the transition model predicts farther into the future.
        valid_float = (
            current_mask.bool()
            & transition_valid[:, 0].view(-1, 1)
        ).to(current_features.dtype)

        routing = current_components["predictive_routing"]
        raw_route_counts = valid_float.sum()
        has_valid_transition = (raw_route_counts > 0).to(current_features.dtype)
        route_counts = raw_route_counts.clamp(min=1.0)
        mean_route = (
            routing * valid_float.unsqueeze(-1)
        ).sum(dim=(0, 1)) / route_counts
        load_balance = (
            float(layer.num_experts) * mean_route.pow(2).sum() - 1.0
        ) * has_valid_transition
        route_entropy = -(
            routing.clamp(min=1e-8).log() * routing
        ).sum(dim=-1)
        route_entropy = self._masked_scalar_mean(route_entropy, valid_float)
        route_entropy = route_entropy / math.log(float(layer.num_experts))

        expert_outputs = F.normalize(
            current_components["predictive_expert_outputs"], dim=-1
        )
        expert_gram = torch.matmul(
            expert_outputs, expert_outputs.transpose(-1, -2)
        )
        off_diagonal = 1.0 - torch.eye(
            layer.num_experts,
            dtype=expert_gram.dtype,
            device=expert_gram.device,
        )
        diversity = self._masked_scalar_mean(
            expert_gram.pow(2) * off_diagonal,
            valid_float,
        ) * has_valid_transition

        reward_loss = reward_numerator / outcome_denominator.clamp(min=1.0)
        continuation_value_loss = (
            continuation_numerator
            / continuation_denominator.clamp(min=1.0)
        )
        bellman_equivalence_loss = (
            bellman_numerator / outcome_denominator.clamp(min=1.0)
        )

        total = (
            float(feature_weight) * feature_loss
            + float(fam_weight) * latent_losses["fam"]
            + float(market_weight) * latent_losses["market"]
            + float(relation_weight) * latent_losses["relational"]
            + float(fused_weight) * latent_losses["fused"]
            + float(load_balance_weight) * load_balance
            + float(diversity_weight) * diversity
            + float(routing_entropy_weight) * route_entropy
            + float(
                getattr(config, "VALUE_EQUIVALENT_REWARD_WEIGHT", 1.0)
            )
            * reward_loss
            + float(
                getattr(
                    config, "VALUE_EQUIVALENT_CONTINUATION_WEIGHT", 0.5
                )
            )
            * continuation_value_loss
            + float(
                getattr(config, "VALUE_EQUIVALENT_BELLMAN_WEIGHT", 1.0)
            )
            * bellman_equivalence_loss
            + float(
                getattr(config, "VALUE_EQUIVALENT_EXPECTILE_WEIGHT", 0.25)
            )
            * expectile_loss
        )
        result = {
            "loss": total,
            "feature_loss": feature_loss,
            "fam_loss": latent_losses["fam"],
            "market_loss": latent_losses["market"],
            "relation_loss": latent_losses["relational"],
            "fused_loss": latent_losses["fused"],
            "reward_loss": reward_loss,
            "continuation_value_loss": continuation_value_loss,
            "bellman_equivalence_loss": bellman_equivalence_loss,
            "expectile_value_loss": expectile_loss,
            "state_value_mean": (
                current_value.mean()
                if value_equivalent_enabled
                else current_features.new_zeros(())
            ),
            "advantage_mean": advantage.mean(),
            "advantage_positive_fraction": (advantage > 0.0).float().mean(),
            "advantage_weight_mean": advantage_weights.mean(),
            "advantage_weight_min": advantage_weights.min(),
            "advantage_weight_max": advantage_weights.max(),
            "load_balance_loss": load_balance,
            "expert_diversity_loss": diversity,
            "routing_entropy": route_entropy,
            "expert_min_usage": mean_route.min(),
            "expert_max_usage": mean_route.max(),
            "valid_fraction": (
                valid_token_total / possible_token_total.clamp(min=1.0)
            ),
            "horizon_loss_mean": torch.stack(per_horizon_losses).mean(),
            "valid_horizon_fraction": transition_valid.float().mean(),
            "control_gate": (
                layer.max_residual_abs
                * torch.sigmoid(layer.control_gate_logit)
            ),
            "control_projection_norm": layer.control_projection.weight.norm(),
        }
        for step, step_loss in enumerate(per_horizon_losses, start=1):
            result[f"horizon_{step}_loss"] = step_loss
        for step, step_loss in enumerate(per_horizon_value_losses, start=1):
            result[f"value_horizon_{step}_loss"] = step_loss
        return result

    def forward(
        self,
        coin_features: torch.Tensor,
        mask: torch.Tensor,
        relation_aux: Optional[torch.Tensor] = None,
        asset_identity: Optional[torch.Tensor] = None,
        basket_context: Optional[torch.Tensor] = None,
        basket_membership: Optional[torch.Tensor] = None,
        feature_history: Optional[torch.Tensor] = None,
        feature_history_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        components = self._encode_components(
            coin_features,
            mask,
            relation_aux,
            asset_identity,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
            control_only=True,
        )
        return components["fused"], components["pooled"], components["valid_counts"]


class PortfolioRewardHistoryEncoder(nn.Module):
    """Read the ordered causal book/reward state at the decision interface.

    Each actor and Q network owns its encoder. It does not enter the absolute
    asset-return predictor: changing a held book changes a decision, not the
    physical next return of an asset.
    """

    def __init__(self, max_coins: int, model_dim: int):
        super().__init__()
        self.input_dim = config.portfolio_reward_history_dim(max_coins)
        self.enabled = bool(config.PORTFOLIO_REWARD_HISTORY_ENABLED)
        self.model_dim = model_dim
        self.max_abs = float(getattr(config, 'PORTFOLIO_HISTORY_MAX_ABS', 0.25))
        output_gain = float(getattr(config, 'PORTFOLIO_HISTORY_INITIAL_OUTPUT_GAIN', 0.01))
        if self.max_abs <= 0.0 or output_gain <= 0.0:
            raise ValueError('Portfolio history bound and initial output gain must be positive')
        # This module precedes the retained actor/critic layers. Its random
        # draws must not change those layers' initialization, even when a
        # history field is absent or the feature is disabled in an ablation.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Sequential(
                nn.Linear(self.input_dim, 128), nn.SiLU(), nn.Linear(128, model_dim)
            ) if self.enabled else None
            if self.projection is not None:
                nn.init.xavier_uniform_(self.projection[0].weight, gain=0.25)
                nn.init.zeros_(self.projection[0].bias)
                nn.init.xavier_uniform_(self.projection[-1].weight, gain=output_gain)
                nn.init.zeros_(self.projection[-1].bias)

    def forward(self, history: Optional[torch.Tensor], reference: torch.Tensor):
        if not self.enabled or history is None:
            return reference.new_zeros((reference.shape[0], self.model_dim))
        if history.shape != (reference.shape[0], self.input_dim):
            raise ValueError(f'Portfolio reward history must be [batch, {self.input_dim}]')
        projected = self.projection(history.float())
        # Linear near zero, so the complete financial gradient stays live.
        # Raw causal histories and reward semantics are unchanged; only this
        # learned conditioning residual is bounded before it reaches control.
        return self.max_abs * torch.tanh(projected / self.max_abs)


class BasketConditionedOpportunityHead(nn.Module):
    """One compositional marginal score for any available asset set.

    The thirteen mask coordinates name stable assets, while active and held
    pooled tokens describe this minute and the current book.  The same small
    per-asset network is applied to every active slot.  Actor and critic own
    separate instances so Bellman gradients cannot train the actor directly.
    """

    def __init__(
        self, max_coins: int, model_dim: int, control_dim: int, *,
        zero_output: bool = False,
    ):
        super().__init__()
        self.max_coins = int(max_coins)
        self.mask_projection = nn.Linear(self.max_coins, model_dim, bias=False)
        self.active_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.held_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.pool_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.cash_projection = nn.Linear(1, model_dim, bias=False)
        self.context_norm = nn.LayerNorm(model_dim)
        self.asset_score = nn.Sequential(
            nn.LayerNorm(2 * control_dim + 2 * model_dim + 1),
            nn.Linear(2 * control_dim + 2 * model_dim + 1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 1),
        )
        for projection in (
            self.mask_projection, self.active_projection,
            self.held_projection, self.pool_projection,
            self.cash_projection,
        ):
            nn.init.xavier_uniform_(projection.weight, gain=0.25)
        if zero_output:
            nn.init.zeros_(self.asset_score[-1].weight)
        else:
            nn.init.xavier_uniform_(self.asset_score[-1].weight, gain=0.10)
        nn.init.zeros_(self.asset_score[-1].bias)

    def _context(self, fused, pool_cls, active, previous_weights, portfolio_context):
        held = previous_weights[:, 1 : self.max_coins + 1].float() * active
        active_mean = (fused * active.unsqueeze(-1)).sum(dim=1) / (
            active.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        held_mean = (fused * held.unsqueeze(-1)).sum(dim=1)
        if portfolio_context is None:
            portfolio_context = torch.zeros_like(pool_cls)
        return self.context_norm(
            self.mask_projection(active)
            + self.active_projection(active_mean)
            + self.held_projection(held_mean)
            + self.pool_projection(pool_cls)
            + self.cash_projection(previous_weights[:, :1].float())
            + portfolio_context
        )

    def forward(
        self, local: torch.Tensor, relational: torch.Tensor,
        fused: torch.Tensor, pool_cls: torch.Tensor, mask: torch.Tensor,
        previous_weights: torch.Tensor,
        portfolio_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask.ndim != 2 or mask.shape[1] != self.max_coins:
            raise ValueError("Basket opportunity mask must be [batch, max_coins]")
        active = mask.float()
        held = previous_weights[:, 1 : self.max_coins + 1].float() * active
        context = self._context(fused, pool_cls, active, previous_weights, portfolio_context)
        per_asset = torch.cat(
            (
                local, relational, fused,
                context.unsqueeze(1).expand(-1, self.max_coins, -1),
                held.unsqueeze(-1),
            ), dim=-1,
        )
        scores = self.asset_score(per_asset).squeeze(-1) * active
        # FlowRL can attach an explicit cash marginal without changing TD3's
        # existing head or its parameter names. Budget conservation makes
        # sum(delta_asset * (asset_score - cash_score)) the full cash-inclusive
        # marginal value, since delta_cash = -sum(delta_asset).
        cash_head = getattr(self, "cash_marginal_head", None)
        if cash_head is not None:
            cash_score = cash_head(local, relational, fused, pool_cls, active,
                                   previous_weights, scores, portfolio_context)
            scores = (scores - cash_score) * active
        return scores


class JointPortfolioOpportunityHead(BasketConditionedOpportunityHead):
    """Cash and risky assets are choices evaluated by one shared marginal rule.

    Cash has zero local market-return/covariance features, a learned cash role
    token, the same opportunity-set context, and its actual held weight. No
    independent cash MLP or sleeve-wide budget shift overrides these scores.
    The critic API returns asset-minus-cash scores by budget conservation.
    """
    def __init__(self, max_coins, model_dim, control_dim, *, zero_output=False):
        super().__init__(max_coins, model_dim, control_dim, zero_output=zero_output)
        self.cash_token = nn.Parameter(torch.zeros(model_dim))

    def joint_scores(self, local, relational, fused, pool_cls, mask,
                     previous_weights, portfolio_context=None):
        active = mask.float()
        full_mask = torch.cat((torch.ones_like(active[:, :1]), active), -1)
        context = self._context(fused, pool_cls, active, previous_weights, portfolio_context)
        # One scorer sees every choice in the same feature schema. Market
        # evidence of excluded assets is erased before any set aggregation.
        all_local = torch.cat((torch.zeros_like(local[:, :1]), local), 1)
        all_relational = torch.cat((torch.zeros_like(relational[:, :1]), relational), 1)
        all_fused = torch.cat(((pool_cls + self.cash_token)[:, None], fused), 1)
        inputs = torch.cat((all_local, all_relational, all_fused,
                            context[:, None].expand(-1, self.max_coins + 1, -1),
                            (previous_weights * full_mask).unsqueeze(-1)), -1)
        scores = self.asset_score(inputs).squeeze(-1) * full_mask
        common = scores.sum(-1, keepdim=True) / full_mask.sum(-1, keepdim=True).clamp_min(1)
        # Remove the unidentifiable common score level rather than driving all
        # pre-squash coordinates toward the same bound together.
        return (scores - common) * full_mask

    def forward(self, local, relational, fused, pool_cls, mask,
                previous_weights, portfolio_context=None):
        scores = self.joint_scores(local, relational, fused, pool_cls, mask,
                                   previous_weights, portfolio_context)
        return (scores[:, 1:] - scores[:, :1]) * mask


class DeepSetActor(nn.Module):
    """
    Permutation-equivariant actor for variable portfolios.

    Each coin token is encoded with shared weights, pooled with a masked mean,
    then translated into action tokens. Cash and asset action tokens share the
    same backbone and scalar head, and then pass through one final masked
    action-space attention block so the last-stage logits are conditioned on the
    whole proposed portfolio move rather than scored independently token by
    token. Previous allocations (prev_cash, prev_assets) are fused into the
    action head so the actor can anchor around current exposure while CRL
    remains the dominant smoothing mechanism. A second, low-gain global context
    summarizes the previous full portfolio after encoding so phase 2 can
    coordinate portfolio-wide slowdown without injecting previous weights into
    the shared encoder itself.
    """

    expects_precomputed_features = False

    def __init__(
        self,
        max_coins: int,
        features_per_coin: int,
        action_dim: int,
        set_hidden_dim: int,
        net_arch: List[int],
        n_heads: int,
        n_layers: int,
        dropout: float,
        use_market_attention: bool,
        use_global_pooling: bool,
        use_action_mixer: bool = False,  # Off by default. When enabled, keep it as a small corrective residual rather than a second dominant action constructor.
        local_feature_dim: Optional[int] = None,
        position_feature_dim: Optional[int] = None,
        context_feature_dim: Optional[int] = None,
        comparative_feature_dim: Optional[int] = None,
        num_asset_ids: Optional[int] = None,
        asset_identity_dim: int = 16,
        basket_context_dim: int = 0,
        basket_membership_dim: int = 0,
        predictive_moe_enabled: bool = False,
        predictive_moe_num_experts: int = 4,
        predictive_moe_expert_hidden_dim: int = 64,
        predictive_moe_router_temperature: float = 1.0,
        predictive_moe_max_residual_abs: float = 0.10,
        require_portfolio_reward_history: bool = False,
    ):
        super().__init__()
        self.max_coins = max_coins
        self.features_per_coin = features_per_coin
        self.action_dim = action_dim
        self.portfolio_history_encoder = PortfolioRewardHistoryEncoder(max_coins, set_hidden_dim)
        self.require_portfolio_reward_history = bool(require_portfolio_reward_history)
        self.use_global_pooling = use_global_pooling
        self.use_action_mixer = use_action_mixer
        self.predictive_moe_enabled = bool(predictive_moe_enabled)
        self.foundation_typed_control_configured = bool(
            getattr(config, "FOUNDATION_TYPED_CONTROL_ENABLED", True)
        )
        self.foundation_typed_control_enabled = bool(
            getattr(config, "FOUNDATION_LEARNING_ENABLED", False)
            and self.foundation_typed_control_configured
        )
        self.local_financial_handoff_enabled = bool(
            getattr(config, "LOCAL_FINANCIAL_HANDOFF_ENABLED", False)
        )
        self.basket_conditioned_allocation_enabled = bool(
            getattr(config, "BASKET_CONDITIONED_ALLOCATION_ENABLED", False)
            and self.local_financial_handoff_enabled
            and asset_identity_dim > 0
            and use_global_pooling
            and not self.foundation_typed_control_enabled
        )
        self.local_financial_actor_context_max_logit = float(
            getattr(config, "LOCAL_FINANCIAL_ACTOR_CONTEXT_MAX_LOGIT", 0.25)
        )
        if self.local_financial_actor_context_max_logit <= 0.0:
            raise ValueError("Local-financial actor context bound must be positive")
        (
            self.local_feature_dim,
            self.position_feature_dim,
            self.comparative_feature_dim,
            self.context_feature_dim,
        ) = _resolve_token_scope_dims(
            features_per_coin,
            local_feature_dim,
            position_feature_dim,
            context_feature_dim,
            comparative_feature_dim,
        )

        # Asset ids remain in the observation for replay alignment and
        # curriculum bookkeeping, but Path 2 deliberately removes them from
        # the control representation.  A zero identity width therefore means
        # "no learned ticker lookup", not an invalid architecture.
        resolved_num_asset_ids = int(
            max_coins if num_asset_ids is None else num_asset_ids
        )
        if resolved_num_asset_ids <= 0:
            raise ValueError("num_asset_ids must be positive")
        self.asset_padding_id = resolved_num_asset_ids
        self.asset_identity_dim = int(asset_identity_dim)
        if self.asset_identity_dim < 0:
            raise ValueError("asset_identity_dim must be non-negative")
        self.basket_context_dim = int(basket_context_dim)
        if self.basket_context_dim < 0:
            raise ValueError("basket_context_dim must be non-negative")
        self.basket_membership_dim = int(basket_membership_dim)
        if self.basket_membership_dim < 0:
            raise ValueError("basket_membership_dim must be non-negative")
        self.task_feature_enabled = (
            self.basket_context_dim > 0 or self.basket_membership_dim > 0
        )
        # fused + pooled + prev weight + active fraction + optional task code
        # (regime g and/or membership z share one set_hidden_dim slot via sum)
        action_token_input_dim = (2 * set_hidden_dim) + 2 + (
            set_hidden_dim if self.task_feature_enabled else 0
        )

        self.asset_identity_embedding = (
            nn.Embedding(
                resolved_num_asset_ids + 1,
                self.asset_identity_dim,
                padding_idx=self.asset_padding_id,
            )
            if self.asset_identity_dim > 0
            else None
        )
        self.token_encoder = AttentionDeepSetEncoder(
            features_per_coin=features_per_coin,
            local_feature_dim=self.local_feature_dim,
            position_feature_dim=self.position_feature_dim,
            context_feature_dim=self.context_feature_dim,
            comparative_feature_dim=self.comparative_feature_dim,
            model_dim=set_hidden_dim,
            net_arch=net_arch,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_market_attention=use_market_attention,
            use_global_pooling=use_global_pooling,
            relation_aux_dim=0,
            relation_uses_comparative=True,
            # Episode-relative fields are the selected-basket replacement for
            # the old market-relative fields, so they participate in the same
            # shared market relation used to construct opportunity-aware
            # actions. Cost basis remains a post-attention position condition.
            relation_uses_context=True,
            relation_uses_position=False,
            asset_identity_dim=self.asset_identity_dim,
            basket_context_dim=self.basket_context_dim,
            basket_membership_dim=self.basket_membership_dim,
            predictive_moe_enabled=predictive_moe_enabled,
            predictive_moe_num_experts=predictive_moe_num_experts,
            predictive_moe_expert_hidden_dim=predictive_moe_expert_hidden_dim,
            predictive_moe_router_temperature=(
                predictive_moe_router_temperature
            ),
            predictive_moe_max_residual_abs=predictive_moe_max_residual_abs,
            fam_enabled=bool(getattr(config, "MDAEN_FAM_ENABLED", False)),
            fam_feature_dim=int(
                getattr(config, "MDAEN_FAM_FEATURE_COUNT", 0)
            ),
            fam_history_length=int(
                getattr(config, "MDAEN_FAM_HISTORY_LENGTH", 8)
            ),
            fam_kernel_sizes=tuple(
                getattr(config, "MDAEN_FAM_KERNEL_SIZES", (3, 5))
            ),
            fam_asset_vector_dim=int(
                getattr(config, "MDAEN_FAM_ASSET_VECTOR_DIM", set_hidden_dim)
            ),
            fam_initial_residual_scale=float(
                getattr(
                    config,
                    "MDAEN_POST_AAM_SNAPSHOT_INITIAL_RESIDUAL_SCALE",
                    getattr(
                        config,
                        "MDAEN_AAM_SNAPSHOT_INITIAL_RESIDUAL_SCALE",
                        getattr(config, "MDAEN_FAM_INITIAL_RESIDUAL_SCALE", 0.05),
                    ),
                )
            ),
            fam_max_residual_scale=float(
                getattr(
                    config,
                    "MDAEN_POST_AAM_SNAPSHOT_MAX_RESIDUAL_SCALE",
                    getattr(
                        config,
                        "MDAEN_AAM_SNAPSHOT_MAX_RESIDUAL_SCALE",
                        getattr(config, "MDAEN_FAM_MAX_RESIDUAL_SCALE", 0.25),
                    ),
                )
            ),
        )
        self.basket_context_proj = (
            nn.Linear(self.basket_context_dim, set_hidden_dim)
            if self.basket_context_dim > 0
            else None
        )
        if self.basket_context_proj is not None:
            nn.init.xavier_uniform_(self.basket_context_proj.weight, gain=0.25)
            nn.init.zeros_(self.basket_context_proj.bias)
        # Membership projection is built in fork_rng after the action backbone so
        # its construction cannot reseed cash/asset action MLPs.
        self.basket_membership_proj = None
        self.cash_token_embedding = nn.Parameter(torch.zeros(set_hidden_dim))
        self.action_token_backbone = _build_mlp_layers(action_token_input_dim, net_arch, set_hidden_dim)
        self.action_token_norm = nn.LayerNorm(set_hidden_dim)
        if use_action_mixer:
            resolved_heads = _resolve_attention_heads(set_hidden_dim, n_heads)
            self.action_mixer = nn.MultiheadAttention(
                embed_dim=set_hidden_dim,
                num_heads=resolved_heads,
                dropout=0.0,
                batch_first=True,
            )
            # Give the actor a small amount of portfolio-wide coordination at
            # the final action stage. This is the closest DeepSet analogue to
            # the flat CombinedExtractor MLP's global mixing, but kept low-gain
            # so phase 1 remains market-led rather than pre-smoothed.
            self.action_mixer_scale = nn.Parameter(torch.tensor(0.02))
            # Normalize only the mixer branch before scaling so the residual
            # gain means what it says. Normalizing the whole sum made the
            # supposedly small mixer residual much more influential early.
            self.action_mixer_norm = nn.LayerNorm(set_hidden_dim)
            self._reset_action_mixer_parameters()
        else:
            self.action_mixer = None
            self.action_mixer_scale = None
            self.action_mixer_norm = None
        self.action_logit_head = nn.Linear(set_hidden_dim, 1)
        # Persistent typed action lanes.  They consume the explicit local and
        # relational control representations instead of asking one fused MLP
        # to rediscover that distinction from sparse TD gradients.  The old
        # fused action path is retained only as a bounded corrective residual.
        control_dim = self.token_encoder.control_stream_dim
        local_control_input_dim = control_dim + 2
        relational_control_input_dim = (2 * control_dim) + 2 + (
            set_hidden_dim if self.task_feature_enabled else 0
        )
        cross_control_input_dim = (2 * control_dim) + 2
        with torch.random.fork_rng(devices=[]):
            self.local_action_head = nn.Linear(local_control_input_dim, 1)
            self.relational_action_head = nn.Linear(
                relational_control_input_dim, 1
            )
            self.cross_action_head = nn.Linear(cross_control_input_dim, 1)
            for head in (
                self.local_action_head,
                self.relational_action_head,
                self.cross_action_head,
            ):
                nn.init.xavier_uniform_(head.weight, gain=0.10)
                nn.init.zeros_(head.bias)
        self.fused_action_max_scale = float(
            getattr(config, "FOUNDATION_FUSED_CONTROL_MAX_SCALE", 0.25)
        )
        self.fused_action_gate_logit = nn.Parameter(torch.tensor(0.0))
        self.predictive_action_delta_head = None
        self.basket_opportunity_head = None
        if self.basket_conditioned_allocation_enabled:
            with torch.random.fork_rng(devices=[]):
                self.basket_opportunity_head = BasketConditionedOpportunityHead(
                    max_coins, set_hidden_dim, control_dim
                )

        with torch.random.fork_rng(devices=[]):
            if self.basket_membership_dim > 0:
                self.basket_membership_proj = nn.Linear(
                    self.basket_membership_dim, set_hidden_dim
                )
                # Match weather proj gain scale so cold-start Q is not dominated
                # by membership while remaining nonzero/live for z-routing.
                nn.init.xavier_uniform_(self.basket_membership_proj.weight, gain=0.25)
                nn.init.zeros_(self.basket_membership_proj.bias)
            if self.predictive_moe_enabled:
                self.predictive_action_delta_head = nn.Sequential(
                    nn.LayerNorm(set_hidden_dim + 3),
                    nn.Linear(set_hidden_dim + 3, set_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(set_hidden_dim, set_hidden_dim),
                )
                nn.init.xavier_uniform_(
                    self.predictive_action_delta_head[-1].weight, gain=0.01
                )
                nn.init.zeros_(self.predictive_action_delta_head[-1].bias)

    def _reset_action_mixer_parameters(self) -> None:
        if self.action_mixer is None:
            return

        # Same principle as the encoder residuals: the mixer should begin as a
        # quiet corrective branch, not as a competing portfolio constructor.
        with torch.no_grad():
            self.action_mixer.out_proj.weight.mul_(0.25)
            if self.action_mixer.out_proj.bias is not None:
                self.action_mixer.out_proj.bias.mul_(0.25)

    def _parse_observation(
        self, observations: Dict[str, torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not isinstance(observations, dict):
            raise ValueError(
                "DeepSetActor expects dict observations with "
                "observations/asset_ids/mask/prev_action/prev_mask "
                "(and basket_context / basket_membership when enabled)."
            )

        if self.require_portfolio_reward_history and 'portfolio_reward_history' not in observations:
            raise ValueError('Actor requires portfolio reward history from the current observation contract')
        obs = observations["observations"].float()
        asset_ids = observations.get("asset_ids")
        mask = observations["mask"].float()
        prev_action = observations.get("prev_action")
        prev_mask = observations.get("prev_mask")
        basket_context = observations.get("basket_context")
        basket_membership = observations.get("basket_membership")
        if "feature_history" in observations:
            raise RuntimeError(
                "Foundation schema v2 rejects legacy feature_history. "
                "Start a fresh run with local_feature_history and "
                "episode_feature_history."
            )
        local_feature_history = observations.get("local_feature_history")
        episode_feature_history = observations.get("episode_feature_history")
        feature_history = None
        feature_history_mask = observations.get("feature_history_mask")

        batch_size = obs.shape[0]
        expected_width = self.max_coins * self.features_per_coin
        if obs.ndim != 2 or obs.shape[1] != expected_width:
            raise ValueError(
                "DeepSetActor observation width mismatch: expected "
                f"{self.max_coins} × {self.features_per_coin} = "
                f"{expected_width}, received shape {tuple(obs.shape)}."
            )
        coin_features = obs.view(batch_size, self.max_coins, self.features_per_coin)
        encoder = self.token_encoder
        if encoder.local_causal_inputs_enabled:
            coin_features = local_market_token_view(
                coin_features, encoder.local_feature_dim, encoder.position_feature_dim,
                encoder.comparative_feature_dim,
            )
            basket_context = None if basket_context is None else torch.zeros_like(basket_context)
            basket_membership = None if basket_membership is None else torch.zeros_like(basket_membership)
            if episode_feature_history is not None:
                episode_feature_history = torch.zeros_like(episode_feature_history)

        if asset_ids is None:
            asset_ids = torch.arange(
                self.max_coins, device=obs.device
            ).unsqueeze(0).expand(batch_size, -1)
            padding_fill = torch.full_like(asset_ids, self.asset_padding_id)
            asset_ids = torch.where(mask > 0, asset_ids, padding_fill)
        else:
            asset_ids = asset_ids.long()
            if asset_ids.dim() == 1:
                asset_ids = asset_ids.unsqueeze(0)
            asset_ids = asset_ids.clamp(min=0, max=self.asset_padding_id)

        if prev_action is None:
            prev_action = obs.new_zeros((batch_size, self.max_coins + 1))
        else:
            prev_action = prev_action.float()

        if prev_mask is None:
            prev_mask = mask
        else:
            prev_mask = prev_mask.float()

        if self.basket_context_dim > 0:
            if basket_context is None:
                basket_context = obs.new_zeros((batch_size, self.basket_context_dim))
            else:
                basket_context = basket_context.float()
                if basket_context.dim() == 1:
                    basket_context = basket_context.unsqueeze(0)
                if basket_context.shape[-1] != self.basket_context_dim:
                    raise ValueError(
                        "DeepSetActor basket_context width mismatch: expected "
                        f"{self.basket_context_dim}, received {basket_context.shape[-1]}."
                    )
        else:
            basket_context = obs.new_zeros((batch_size, 0))

        if self.basket_membership_dim > 0:
            if basket_membership is None:
                basket_membership = obs.new_zeros(
                    (batch_size, self.basket_membership_dim)
                )
            else:
                basket_membership = basket_membership.float()
                if basket_membership.dim() == 1:
                    basket_membership = basket_membership.unsqueeze(0)
                if basket_membership.shape[-1] != self.basket_membership_dim:
                    raise ValueError(
                        "DeepSetActor basket_membership width mismatch: expected "
                        f"{self.basket_membership_dim}, received "
                        f"{basket_membership.shape[-1]}."
                    )
        else:
            basket_membership = obs.new_zeros((batch_size, 0))

        if self.token_encoder.fam_enabled:
            expected_local_shape = (
                batch_size,
                self.max_coins,
                self.token_encoder.fam_history_length,
                self.token_encoder.fam_local_feature_dim,
            )
            expected_episode_shape = (
                batch_size,
                self.max_coins,
                self.token_encoder.fam_history_length,
                self.token_encoder.fam_episode_feature_dim,
            )
            if local_feature_history is None:
                local_feature_history = obs.new_zeros(expected_local_shape)
            else:
                local_feature_history = local_feature_history.float()
                if tuple(local_feature_history.shape) != expected_local_shape:
                    raise ValueError(
                        "DeepSetActor local_feature_history shape mismatch: expected "
                        f"{expected_local_shape}, received "
                        f"{tuple(local_feature_history.shape)}."
                    )
            if episode_feature_history is None:
                episode_feature_history = obs.new_zeros(expected_episode_shape)
            else:
                episode_feature_history = episode_feature_history.float()
                if tuple(episode_feature_history.shape) != expected_episode_shape:
                    raise ValueError(
                        "DeepSetActor episode_feature_history shape mismatch: expected "
                        f"{expected_episode_shape}, received "
                        f"{tuple(episode_feature_history.shape)}."
                    )
            feature_history = torch.cat(
                (local_feature_history, episode_feature_history), dim=-1
            )
            expected_history_mask_shape = (
                batch_size,
                expected_local_shape[2],
            )
            if feature_history_mask is None:
                feature_history_mask = obs.new_zeros(expected_history_mask_shape)
            else:
                feature_history_mask = feature_history_mask.float()
                if tuple(feature_history_mask.shape) != expected_history_mask_shape:
                    raise ValueError(
                        "DeepSetActor feature_history_mask shape mismatch: "
                        f"expected {expected_history_mask_shape}, received "
                        f"{tuple(feature_history_mask.shape)}."
                    )
        else:
            feature_history = obs.new_zeros((batch_size, self.max_coins, 0, 0))
            feature_history_mask = obs.new_zeros((batch_size, 0))

        return (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        )

    def _project_task_features(
        self,
        basket_context: torch.Tensor,
        basket_membership: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.token_encoder.local_causal_inputs_enabled:
            basket_context = None if basket_context is None else torch.zeros_like(basket_context)
            basket_membership = None if basket_membership is None else torch.zeros_like(basket_membership)
        feat = None
        if self.basket_context_proj is not None:
            feat = self.basket_context_proj(basket_context)
        if self.basket_membership_proj is not None:
            membership_feat = self.basket_membership_proj(basket_membership)
            feat = membership_feat if feat is None else feat + membership_feat
        return feat

    def _encode_state(
        self,
        coin_features: torch.Tensor,
        asset_ids: torch.Tensor,
        mask: torch.Tensor,
        prev_mask: torch.Tensor,
        prev_assets: torch.Tensor,
        basket_context: torch.Tensor,
        basket_membership: torch.Tensor,
        feature_history: torch.Tensor,
        feature_history_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del prev_mask, prev_assets
        asset_identity = (
            self.asset_identity_embedding(asset_ids)
            if self.asset_identity_embedding is not None
            else None
        )
        return self.token_encoder(
            coin_features,
            mask,
            asset_identity=asset_identity,
            basket_context=basket_context if self.basket_context_dim > 0 else None,
            basket_membership=(
                basket_membership if self.basket_membership_dim > 0 else None
            ),
            feature_history=(
                feature_history if self.token_encoder.fam_enabled else None
            ),
            feature_history_mask=(
                feature_history_mask if self.token_encoder.fam_enabled else None
            ),
        )

    def _asset_action_features_from_components(
        self,
        components: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        prev_action: torch.Tensor,
        basket_context: torch.Tensor,
        basket_membership: torch.Tensor,
    ) -> torch.Tensor:
        pooled = components["pooled"]
        valid_counts = components["valid_counts"]
        previous_assets = prev_action[:, 1:1 + self.max_coins]
        active_fraction = valid_counts / float(self.max_coins)
        pooled_expanded = pooled.unsqueeze(1).expand(-1, self.max_coins, -1)
        active_expanded = active_fraction.unsqueeze(1).expand(
            -1, self.max_coins, -1
        )
        task_feature = self._project_task_features(
            basket_context, basket_membership
        )
        parts = [
            components["fused"],
            pooled_expanded,
            previous_assets.unsqueeze(-1),
            active_expanded,
        ]
        if task_feature is not None:
            parts.append(
                task_feature.unsqueeze(1).expand(-1, self.max_coins, -1)
            )
        return self.action_token_norm(
            self.action_token_backbone(torch.cat(parts, dim=-1))
        ) * mask.unsqueeze(-1)

    def encode_state(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        ) = self._parse_observation(observations)
        prev_assets = prev_action[:, 1:1 + self.max_coins]
        return self._encode_state(
            coin_features,
            asset_ids,
            mask,
            prev_mask,
            prev_assets,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        )

    def earnmore_masked_reconstruction_loss(
        self,
        observations: Dict[str, torch.Tensor],
        *,
        synthetic_mask: Optional[torch.Tensor] = None,
        reconstruction_mask: Optional[torch.Tensor] = None,
        prepared_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train the pool state from masked historical evidence."""
        if prepared_inputs is None:
            prepared_inputs = self.prepare_earnmore_reconstruction_inputs(
                observations
            )
        coin_features = prepared_inputs["coin_features"]
        asset_ids = prepared_inputs["asset_ids"]
        mask = prepared_inputs["mask"]
        basket_context = prepared_inputs["basket_context"]
        basket_membership = prepared_inputs["basket_membership"]
        feature_history = prepared_inputs["feature_history"]
        feature_history_mask = prepared_inputs["feature_history_mask"]
        return self.token_encoder.earnmore_masked_reconstruction_loss(
            coin_features,
            mask,
            asset_identity=(self.asset_identity_embedding(asset_ids).detach()
                            if self.asset_identity_embedding is not None else None),
            basket_context=(
                basket_context if self.basket_context_dim > 0 else None
            ),
            basket_membership=(
                basket_membership if self.basket_membership_dim > 0 else None
            ),
            feature_history=feature_history,
            feature_history_mask=feature_history_mask,
            synthetic_mask=synthetic_mask,
            reconstruction_mask=reconstruction_mask,
            pool_history_valid=prepared_inputs.get("pool_history_valid"),
        )

    def prepare_earnmore_reconstruction_inputs(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Parse immutable reconstruction inputs once for actor and critics."""
        (
            coin_features,
            asset_ids,
            mask,
            _,
            _,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        ) = self._parse_observation(observations)
        prepared = {
            "coin_features": coin_features,
            "asset_ids": asset_ids,
            "mask": mask,
            "basket_context": basket_context,
            "basket_membership": basket_membership,
            "feature_history": feature_history,
            "feature_history_mask": feature_history_mask,
        }
        pool_history_valid = observations.get("pool_history_valid")
        if pool_history_valid is not None:
            prepared["pool_history_valid"] = pool_history_valid
        return prepared

    def foundation_loss(
        self,
        observations: Dict[str, torch.Tensor],
        future_unit_returns: torch.Tensor,
        future_valid: torch.Tensor,
        candidate_actions: torch.Tensor,
        candidate_utilities: torch.Tensor,
        peer_observations: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train the live actor representation and a hindsight action prior."""
        (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            _,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        ) = self._parse_observation(observations)
        asset_identity = (
            self.asset_identity_embedding(asset_ids)
            if self.asset_identity_embedding is not None
            else None
        )
        components = self.token_encoder.encode_components(
            coin_features,
            mask,
            asset_identity=asset_identity,
            basket_context=(
                basket_context if self.basket_context_dim > 0 else None
            ),
            basket_membership=(
                basket_membership if self.basket_membership_dim > 0 else None
            ),
            feature_history=feature_history,
            feature_history_mask=feature_history_mask,
        )
        feature_history_target = feature_history
        local_target = observations.get("foundation_local_history_target")
        if local_target is not None:
            feature_history_target = torch.cat(
                (
                    local_target.float(),
                    feature_history[..., self.token_encoder.fam_local_feature_dim :],
                ),
                dim=-1,
            )
        peer_local_temporal = None
        shared_peer_mask = None
        if peer_observations is not None:
            peer_parsed = self._parse_observation(peer_observations)
            peer_components = self.token_encoder.encode_components(
                peer_parsed[0],
                peer_parsed[2],
                asset_identity=None,
                basket_context=(
                    peer_parsed[5] if self.basket_context_dim > 0 else None
                ),
                basket_membership=(
                    peer_parsed[6] if self.basket_membership_dim > 0 else None
                ),
                feature_history=peer_parsed[7],
                feature_history_mask=peer_parsed[8],
                control_only=True,
            )
            correspondence = (
                asset_ids.unsqueeze(2) == peer_parsed[1].unsqueeze(1)
            ) & peer_parsed[2].bool().unsqueeze(1)
            peer_local_temporal = torch.einsum(
                "bij,bjd->bid",
                correspondence.to(peer_components["local_temporal"].dtype),
                peer_components["local_temporal"],
            )
            shared_peer_mask = correspondence.any(dim=2) & mask.bool()
        representation = self.token_encoder.foundation_representation_loss(
            components,
            feature_history_target,
            mask,
            future_unit_returns,
            future_valid,
            peer_local_temporal=peer_local_temporal,
            shared_peer_mask=shared_peer_mask,
            local_reconstruction_mask=observations.get(
                "foundation_local_reconstruction_mask"
            ),
        )

        raw_actor_action = self.forward(observations)
        actor_weights = normalize_raw_actions_to_portfolio_weights(
            raw_actor_action, mask
        )
        batch, candidate_count, action_dim = candidate_actions.shape
        candidate_weights = candidate_actions.float().clamp(min=0.0)
        candidate_weights = torch.cat(
            (
                candidate_weights[..., :1],
                candidate_weights[..., 1:] * mask.unsqueeze(1),
            ),
            dim=-1,
        )
        candidate_weights = candidate_weights / candidate_weights.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        utility_center = candidate_utilities.mean(dim=1, keepdim=True)
        utility_scale = candidate_utilities.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp(min=1e-6)
        standardized_utility = (
            candidate_utilities - utility_center
        ) / utility_scale
        temperature = float(
            getattr(config, "FOUNDATION_ACTOR_AWR_TEMPERATURE", 0.5)
        )
        candidate_probability = torch.softmax(
            standardized_utility / max(temperature, 1e-6), dim=1
        )
        target_weights = (
            candidate_probability.unsqueeze(-1) * candidate_weights.detach()
        ).sum(dim=1)
        awr_loss = F.smooth_l1_loss(actor_weights, target_weights)
        total = 0.5 * (representation["loss"] + awr_loss)
        result = dict(representation)
        result.update(
            {
                "loss": total,
                "representation_loss": representation["loss"],
                "action_loss": awr_loss,
                "actor_regret": (
                    actor_weights - target_weights.detach()
                ).abs().mean(),
            }
        )
        return result

    def predictive_representation_loss(
        self,
        observations: Dict[str, torch.Tensor],
        next_observations: Dict[str, torch.Tensor],
        replay_actions: torch.Tensor,
        target_actor: "DeepSetActor",
        dones: Optional[torch.Tensor] = None,
        *,
        future_observations: Optional[Dict[str, torch.Tensor]] = None,
        future_actions: Optional[torch.Tensor] = None,
        future_valid_steps: Optional[torch.Tensor] = None,
        future_outcome_valid_steps: Optional[torch.Tensor] = None,
        future_rewards: Optional[torch.Tensor] = None,
        future_dones: Optional[torch.Tensor] = None,
        future_continuation_value_targets: Optional[torch.Tensor] = None,
        current_q_targets: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        horizon_discount: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Train actor-facing representations with a recursive SPR rollout.

        The one-step arguments remain valid for old checkpoints and focused
        tests.  Production passes a contiguous ``future_observations`` tensor
        and the behavior actions that generated it.  Only the online model's
        own previous prediction is available after horizon one; every future
        observation is encoded under ``no_grad`` by the slowly moving target
        actor and is used solely as an empirical target.
        """
        (
            current_features,
            current_ids,
            current_mask,
            current_prev_action,
            _,
            current_context,
            current_membership,
            current_history,
            current_history_mask,
        ) = self._parse_observation(observations)

        if future_observations is None:
            future_observations = {
                name: value.unsqueeze(1)
                for name, value in next_observations.items()
            }
        if future_actions is None:
            future_actions = replay_actions.unsqueeze(1)
        if future_actions.dim() != 3:
            raise ValueError("actor SPR actions must be [batch, horizon, action]")
        batch_size, horizon = future_actions.shape[:2]
        flat_future_observations = {
            name: value.reshape(batch_size * horizon, *value.shape[2:])
            for name, value in future_observations.items()
        }
        flat_next = target_actor._parse_observation(flat_future_observations)
        (
            flat_next_features,
            flat_next_ids,
            flat_next_mask,
            flat_next_prev_action,
            _,
            flat_next_context,
            flat_next_membership,
            flat_next_history,
            flat_next_history_mask,
        ) = flat_next

        def _horizon_view(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch_size, horizon, *value.shape[1:])

        next_features = _horizon_view(flat_next_features)
        next_ids = _horizon_view(flat_next_ids)
        next_mask = _horizon_view(flat_next_mask)
        next_prev_action = _horizon_view(flat_next_prev_action)
        next_context = _horizon_view(flat_next_context)
        next_membership = _horizon_view(flat_next_membership)

        current_identity = (
            self.asset_identity_embedding(current_ids)
            if self.asset_identity_embedding is not None
            else None
        )
        current_components = self.token_encoder.encode_components(
            current_features,
            current_mask,
            asset_identity=current_identity,
            basket_context=(
                current_context if self.basket_context_dim > 0 else None
            ),
            basket_membership=(
                current_membership
                if self.basket_membership_dim > 0
                else None
            ),
            feature_history=current_history,
            feature_history_mask=current_history_mask,
            detach_predictive_control_adapter=bool(
                getattr(
                    config,
                    "PREDICTIVE_SSL_CONTROL_ADAPTER_FINANCIAL_ONLY",
                    True,
                )
            ),
        )
        with torch.no_grad():
            flat_next_identity = (
                target_actor.asset_identity_embedding(flat_next_ids)
                if target_actor.asset_identity_embedding is not None
                else None
            )
            flat_next_components = target_actor.token_encoder.encode_components(
                flat_next_features,
                flat_next_mask,
                asset_identity=flat_next_identity,
                basket_context=(
                    flat_next_context
                    if target_actor.basket_context_dim > 0
                    else None
                ),
                basket_membership=(
                    flat_next_membership
                    if target_actor.basket_membership_dim > 0
                    else None
                ),
                feature_history=flat_next_history,
                feature_history_mask=flat_next_history_mask,
                detach_predictive_control_adapter=bool(
                    getattr(
                        config,
                        "PREDICTIVE_SSL_CONTROL_ADAPTER_FINANCIAL_ONLY",
                        True,
                    )
                ),
            )
            next_components = {
                name: _horizon_view(value)
                for name, value in flat_next_components.items()
            }
            target_predictive_layer = (
                target_actor.token_encoder.predictive_expert_layer
            )
            if future_continuation_value_targets is not None:
                continuation_value_targets = (
                    future_continuation_value_targets.detach().float().reshape(
                        batch_size, horizon, 1
                    )
                )
            elif bool(getattr(config, "VALUE_EQUIVALENT_MODEL_ENABLED", True)):
                continuation_value_targets = (
                    _horizon_view(
                        target_predictive_layer.state_value(flat_next_components)
                    )
                    if target_predictive_layer is not None
                    else current_features.new_zeros((batch_size, horizon, 1))
                )
            else:
                continuation_value_targets = current_features.new_zeros(
                    (batch_size, horizon, 1)
                )

        traded_weights = normalize_raw_actions_to_portfolio_weights(
            future_actions.float().reshape(batch_size * horizon, -1),
            current_mask.unsqueeze(1)
            .expand(-1, horizon, -1)
            .reshape(batch_size * horizon, -1),
        ).reshape(batch_size, horizon, -1)
        current_assets = traded_weights[:, :, 1:1 + self.max_coins]
        initial_previous_assets = current_prev_action[
            :, 1:1 + self.max_coins
        ].unsqueeze(1)
        previous_assets = torch.cat(
            [initial_previous_assets, current_assets[:, :-1]], dim=1
        )
        action_roles = torch.stack(
            (
                previous_assets,
                current_assets,
                current_assets - previous_assets,
            ),
            dim=-1,
        )
        if future_valid_steps is not None:
            transition_valid = future_valid_steps.bool().reshape(
                batch_size, horizon
            )
        elif dones is not None:
            transition_valid = (dones.view(-1) < 0.5).unsqueeze(1)
        else:
            transition_valid = torch.ones(
                (batch_size, horizon),
                dtype=torch.bool,
                device=current_features.device,
            )
        stats = self.token_encoder.predictive_loss(
            current_features=current_features,
            current_components=current_components,
            current_asset_ids=current_ids,
            current_mask=current_mask,
            next_features=next_features.detach(),
            next_components=next_components,
            next_asset_ids=next_ids,
            next_mask=next_mask,
            action_roles=action_roles,
            transition_valid=transition_valid,
            outcome_valid_steps=future_outcome_valid_steps,
            future_rewards=future_rewards,
            future_dones=future_dones,
            continuation_value_targets=continuation_value_targets,
            current_q_targets=current_q_targets,
            gamma=gamma,
            feature_weight=float(
                getattr(config, "PREDICTIVE_SSL_FEATURE_WEIGHT", 1.0)
            ),
            fam_weight=float(
                getattr(config, "PREDICTIVE_SSL_FAM_WEIGHT", 0.0)
            ),
            market_weight=float(
                getattr(config, "PREDICTIVE_SSL_MARKET_WEIGHT", 0.25)
            ),
            relation_weight=float(
                getattr(config, "PREDICTIVE_SSL_RELATION_WEIGHT", 0.50)
            ),
            fused_weight=float(
                getattr(config, "PREDICTIVE_SSL_FUSED_WEIGHT", 1.0)
            ),
            load_balance_weight=float(
                getattr(config, "PREDICTIVE_MOE_LOAD_BALANCE_WEIGHT", 0.01)
            ),
            diversity_weight=float(
                getattr(config, "PREDICTIVE_MOE_DIVERSITY_WEIGHT", 0.01)
            ),
            routing_entropy_weight=float(
                getattr(
                    config,
                    "PREDICTIVE_MOE_ROUTING_ENTROPY_WEIGHT",
                    0.01,
                )
            ),
            horizon_discount=horizon_discount,
        )
        if self.predictive_action_delta_head is None:
            return stats

        current_action_features = self._asset_action_features_from_components(
            current_components,
            current_mask,
            current_prev_action,
            current_context,
            current_membership,
        )
        with torch.no_grad():
            flat_next_action_features = (
                target_actor._asset_action_features_from_components(
                    flat_next_components,
                    flat_next_mask,
                    flat_next_prev_action,
                    flat_next_context,
                    flat_next_membership,
                )
            )
            next_action_features = _horizon_view(
                flat_next_action_features
            )

        predicted_action_features = current_action_features
        downstream_numerator = current_action_features.new_zeros(())
        downstream_denominator = current_action_features.new_zeros(())
        for step in range(horizon):
            predicted_action_features = (
                predicted_action_features
                + self.predictive_action_delta_head(
                    torch.cat(
                        [predicted_action_features, action_roles[:, step]],
                        dim=-1,
                    )
                )
            )
            aligned_target, has_match = self.token_encoder._align_next_tokens(
                next_action_features[:, step],
                current_ids,
                next_ids[:, step],
            )
            aligned_mask, _ = self.token_encoder._align_next_tokens(
                next_mask[:, step].unsqueeze(-1),
                current_ids,
                next_ids[:, step],
            )
            valid = (
                current_mask.bool()
                & has_match
                & aligned_mask.squeeze(-1).bool()
                & transition_valid[:, step].view(-1, 1)
            )
            error = 1.0 - F.cosine_similarity(
                predicted_action_features, aligned_target, dim=-1
            )
            step_loss = self.token_encoder._masked_scalar_mean(
                error, valid.to(error.dtype)
            )
            step_weight = current_action_features.new_tensor(
                float(horizon_discount) ** step
            ) * valid.any().to(current_action_features.dtype)
            downstream_numerator = (
                downstream_numerator + step_weight * step_loss
            )
            downstream_denominator = downstream_denominator + step_weight
        downstream_loss = downstream_numerator / downstream_denominator.clamp(
            min=1.0
        )
        downstream_weight = float(
            getattr(config, "PREDICTIVE_SSL_DOWNSTREAM_WEIGHT", 1.0)
        )
        stats["loss"] = stats["loss"] + downstream_weight * downstream_loss
        stats["downstream_loss"] = downstream_loss
        return stats

    def _forward_with_control_components(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        ) = self._parse_observation(observations)

        prev_cash = prev_action[:, 0:1]
        prev_assets = prev_action[:, 1:1 + self.max_coins]
        asset_identity = (
            self.asset_identity_embedding(asset_ids)
            if self.asset_identity_embedding is not None
            else None
        )
        components = self.token_encoder.encode_components(
            coin_features,
            mask,
            asset_identity=asset_identity,
            basket_context=(
                basket_context if self.basket_context_dim > 0 else None
            ),
            basket_membership=(
                basket_membership if self.basket_membership_dim > 0 else None
            ),
            feature_history=(
                feature_history if self.token_encoder.fam_enabled else None
            ),
            feature_history_mask=(
                feature_history_mask if self.token_encoder.fam_enabled else None
            ),
            control_only=True,
        )
        fused_action_embeddings = components["fused"]
        portfolio_context = self.portfolio_history_encoder(
            observations.get('portfolio_reward_history'), mask)
        # Cash must read the same reward-relevant book history as risky scores.
        pooled_mean = components["pooled"] + portfolio_context
        valid_counts = components["valid_counts"]

        active_fraction = valid_counts / float(self.max_coins)
        pooled_mean_expanded = pooled_mean.unsqueeze(1).expand(-1, self.max_coins, -1)
        active_fraction_expanded = active_fraction.unsqueeze(1).expand(-1, self.max_coins, -1)
        basket_feat = self._project_task_features(basket_context, basket_membership)
        if basket_feat is not None:
            basket_feat_expanded = basket_feat.unsqueeze(1).expand(
                -1, self.max_coins, -1
            )
        else:
            basket_feat_expanded = None

        # Opportunity scores already own risky allocation. When the optional
        # action mixer is absent, only cash reads the old fused action MLP.
        # Its per-token MLP/LayerNorm has no dependence on the unused risky
        # tokens, so skip their much wider backbone rather than discard them.
        cash_only_fused = (
            self.basket_conditioned_allocation_enabled
            and self.action_mixer is None
        )
        if not cash_only_fused:
            asset_parts = [
                fused_action_embeddings,
                pooled_mean_expanded,
                prev_assets.unsqueeze(-1),
                active_fraction_expanded,
            ]
            if basket_feat_expanded is not None:
                asset_parts.append(basket_feat_expanded)
            asset_action_tokens = torch.cat(asset_parts, dim=-1)
            asset_action_features = self.action_token_norm(self.action_token_backbone(asset_action_tokens))

        cash_local_feature = self.cash_token_embedding.unsqueeze(0).expand(pooled_mean.shape[0], -1)
        cash_parts = [
            cash_local_feature,
            pooled_mean,
            prev_cash,
            active_fraction,
        ]
        if basket_feat is not None:
            cash_parts.append(basket_feat)
        cash_action_token = torch.cat(cash_parts, dim=1)
        cash_action_feature = self.action_token_norm(self.action_token_backbone(cash_action_token))

        cash_mask = mask.new_ones((mask.shape[0], 1))
        joint_action_mask = torch.cat([cash_mask, mask], dim=1)
        if cash_only_fused:
            fused_logits = self.action_logit_head(cash_action_feature)
        else:
            joint_action_tokens = torch.cat([cash_action_feature.unsqueeze(1), asset_action_features], dim=1)
            if self.action_mixer is not None and self.action_mixer_norm is not None:
                action_padding_mask = ~joint_action_mask.bool()
                mixer_input = joint_action_tokens.detach()
                mixed_action_tokens, _ = self.action_mixer(
                    mixer_input,
                    mixer_input,
                    mixer_input,
                    key_padding_mask=action_padding_mask,
                )
                mixed_action_tokens = self.action_mixer_norm(mixed_action_tokens)
                mixer_scale = torch.clamp(self.action_mixer_scale, min=0.0, max=0.15).to(joint_action_tokens.dtype)
                joint_action_tokens = joint_action_tokens + mixer_scale * mixed_action_tokens
            joint_action_tokens = joint_action_tokens * joint_action_mask.unsqueeze(-1)
            fused_logits = self.action_logit_head(joint_action_tokens).squeeze(-1)

        if self.basket_conditioned_allocation_enabled:
            # Form the principal score directly from asset evidence, its
            # available peers, and the held book. Historical local/cross heads
            # below are not part of this policy; do not compute them first.
            basket_logits = self.basket_opportunity_head(
                components["local_control"], components["relational_control"],
                fused_action_embeddings, components["pool_cls"], mask, prev_action,
                portfolio_context=portfolio_context,
            )
            cash_logit = torch.tanh(fused_logits[:, :1])
            action = torch.cat(
                (cash_logit, torch.tanh(basket_logits) * mask), dim=1
            )
            main_logits = torch.cat((fused_logits[:, :1], basket_logits), dim=1)
            return action, {
                "local": torch.zeros_like(action),
                "relational": torch.cat(
                    (torch.zeros_like(cash_logit), basket_logits), dim=1
                ),
                "cross": torch.zeros_like(action),
                "fused": main_logits,
                "total_logits": main_logits,
                "fused_scale": basket_logits.new_tensor(1.0),
            }

        # The typed local/relational/cross heads are the control contract taught
        # by Foundation warmup.  When Foundation is absent, asking these small
        # cold-start heads to construct the action while limiting the complete
        # fused policy to a 25% residual discards the established TD3 control
        # path.  EarnMore teaches the representation with reconstruction; TD3
        # must then be allowed to read that complete representation directly.
        if not self.foundation_typed_control_enabled and not self.local_financial_handoff_enabled:
            raw_logits = fused_logits * joint_action_mask
            cash_logit = torch.tanh(raw_logits[:, 0:1])
            asset_logits = torch.tanh(raw_logits[:, 1:]) * mask
            action = torch.cat([cash_logit, asset_logits], dim=1)
            inactive = torch.zeros_like(raw_logits)
            return action, {
                "local": inactive,
                "relational": inactive,
                "cross": inactive,
                "fused": raw_logits,
                "total_logits": raw_logits,
                "fused_scale": raw_logits.new_tensor(1.0),
            }

        local_control = components["local_control"]
        relational_control = components["relational_control"]
        local_set = components["local_set"]
        relational_set = components["relational_set"]
        relational_set_expanded = relational_set.unsqueeze(1).expand(
            -1, self.max_coins, -1
        )

        local_asset_input = torch.cat(
            (
                local_control,
                prev_assets.unsqueeze(-1),
                active_fraction_expanded,
            ),
            dim=-1,
        )
        local_cash_input = torch.cat(
            (local_set, prev_cash, active_fraction), dim=-1
        )
        local_input = torch.cat(
            (local_cash_input.unsqueeze(1), local_asset_input), dim=1
        )
        local_logits = self.local_action_head(local_input).squeeze(-1)

        relational_asset_parts = [
            relational_control,
            relational_set_expanded,
            prev_assets.unsqueeze(-1),
            active_fraction_expanded,
        ]
        relational_cash_parts = [
            relational_set,
            relational_set,
            prev_cash,
            active_fraction,
        ]
        if basket_feat_expanded is not None and basket_feat is not None:
            relational_asset_parts.append(basket_feat_expanded)
            relational_cash_parts.append(basket_feat)
        relational_asset_input = torch.cat(relational_asset_parts, dim=-1)
        relational_cash_input = torch.cat(relational_cash_parts, dim=-1)
        relational_input = torch.cat(
            (relational_cash_input.unsqueeze(1), relational_asset_input), dim=1
        )
        relational_logits = self.relational_action_head(
            relational_input
        ).squeeze(-1)

        contextual_local_control = (
            local_control.detach()
            if self.token_encoder.strict_semantic_gradient_ownership
            else local_control
        )
        contextual_local_set = (
            local_set.detach()
            if self.token_encoder.strict_semantic_gradient_ownership
            else local_set
        )
        cross_asset_input = torch.cat(
            (
                contextual_local_control,
                relational_control,
                prev_assets.unsqueeze(-1),
                active_fraction_expanded,
            ),
            dim=-1,
        )
        cross_cash_input = torch.cat(
            (
                contextual_local_set,
                relational_set,
                prev_cash,
                active_fraction,
            ),
            dim=-1,
        )
        cross_input = torch.cat(
            (cross_cash_input.unsqueeze(1), cross_asset_input), dim=1
        )
        cross_logits = self.cross_action_head(cross_input).squeeze(-1)

        if not self.foundation_typed_control_enabled and self.local_financial_handoff_enabled:
            # The same asset-local FAM code enters the same local head in every
            # basket.  Peer evidence may redistribute risky preference, but
            # cannot become an unbounded alternate route around that head.
            contextual_asset_logits = (
                relational_logits[:, 1:]
                + cross_logits[:, 1:]
                + fused_logits[:, 1:]
            )
            contextual_center = (
                (contextual_asset_logits * mask).sum(dim=1, keepdim=True)
                / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            )
            centered_context = (contextual_asset_logits - contextual_center) * mask
            context_scale = self.local_financial_actor_context_max_logit * torch.sigmoid(
                self.fused_action_gate_logit
            )
            bounded_context = torch.tanh(centered_context) * mask
            bounded_context = bounded_context - (
                (bounded_context * mask).sum(dim=1, keepdim=True)
                / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            ) * mask
            context_correction = 0.5 * context_scale * bounded_context
            # The cash logit is a market-wide budget decision; it is not an
            # asset-specific identity and may read the complete basket.
            cash_logit = torch.tanh(fused_logits[:, :1])
            risky_logits = local_logits[:, 1:] + context_correction
            asset_logits = torch.tanh(risky_logits) * mask
            action = torch.cat((cash_logit, asset_logits), dim=1)
            cash_and_risky_local = torch.cat(
                (torch.zeros_like(cash_logit), local_logits[:, 1:] * mask), dim=1
            )
            cash_and_risky_context = torch.cat(
                (fused_logits[:, :1], context_correction), dim=1
            )
            return action, {
                "local": cash_and_risky_local,
                "relational": torch.cat(
                    (torch.zeros_like(cash_logit),
                     relational_logits[:, 1:] * mask), dim=1
                ),
                "cross": torch.cat(
                    (torch.zeros_like(cash_logit), cross_logits[:, 1:] * mask), dim=1
                ),
                "fused": cash_and_risky_context,
                "total_logits": torch.cat((fused_logits[:, :1], risky_logits), dim=1),
                "fused_scale": context_scale,
            }

        fused_scale = self.fused_action_max_scale * torch.sigmoid(
            self.fused_action_gate_logit
        )
        fused_contribution = fused_scale * fused_logits
        raw_logits = (
            local_logits
            + relational_logits
            + cross_logits
            + fused_contribution
        )
        raw_logits = raw_logits * joint_action_mask
        cash_logit = torch.tanh(raw_logits[:, 0:1])
        asset_logits = torch.tanh(raw_logits[:, 1:]) * mask
        action = torch.cat([cash_logit, asset_logits], dim=1)
        return action, {
            "local": local_logits * joint_action_mask,
            "relational": relational_logits * joint_action_mask,
            "cross": cross_logits * joint_action_mask,
            "fused": fused_contribution * joint_action_mask,
            "total_logits": raw_logits,
            "fused_scale": fused_scale,
        }

    def forward_with_control_decomposition(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return the action and inspectable typed logit contributions."""
        return self._forward_with_control_components(observations)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        action, _ = self._forward_with_control_components(observations)
        return action

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


class DeepSetQNetwork(nn.Module):
    """
    Single permutation-invariant Q-network over coin/action tokens.

    The critic intentionally values normalized traded weights, not pre-
    normalized logits, because the CRL costs and constraints live in that
    traded-weight geometry. The shared encoder constructs one market relation
    for the replay state, while the local/global value heads score turnover and
    candidate allocation directly in normalized weight space.

    A bounded low-rank adapter gives the value stream an explicit behavioral
    state × allocation interaction without adding an action-weighted shortcut
    directly to Q. The explicit previous/current/delta fields in the shared
    local value head remain the live base action path. The adapter is
    token-local, shared across assets, masked before pooling, and norm-bounded
    by a dimension-aware gate so its initial action Jacobian cannot dominate
    the CRL actor update.
    """

    def __init__(
        self,
        max_coins: int,
        features_per_coin: int,
        set_hidden_dim: int,
        net_arch: List[int],
        n_heads: int,
        n_layers: int,
        dropout: float,
        use_market_attention: bool,
        use_global_pooling: bool,
        local_feature_dim: Optional[int] = None,
        position_feature_dim: Optional[int] = None,
        context_feature_dim: Optional[int] = None,
        comparative_feature_dim: Optional[int] = None,
        allocation_interaction_dim: Optional[int] = None,
        allocation_interaction_max_norm: float = 0.25,
        uvfa_local_interaction_enabled: bool = False,
        uvfa_local_interaction_rank: int = 8,
        uvfa_local_interaction_max_abs: float = 0.10,
        component_outcome_aux_enabled: bool = False,
        component_outcome_dim: int = 1,
        component_outcome_unit_return_targets: bool = False,
        component_forecast_bridge_enabled: bool = False,
        component_forecast_bridge_return_reference: float = 0.01,
        component_forecast_bridge_max_abs: float = 0.10,
        component_forecast_bridge_local_only: bool = False,
        num_asset_ids: Optional[int] = None,
        asset_identity_dim: int = 16,
        basket_context_dim: int = 0,
        basket_membership_dim: int = 0,
        predictive_moe_enabled: bool = False,
        predictive_moe_num_experts: int = 4,
        predictive_moe_expert_hidden_dim: int = 64,
        predictive_moe_router_temperature: float = 1.0,
        predictive_moe_max_residual_abs: float = 0.10,
    ):
        super().__init__()
        self.max_coins = max_coins
        self.portfolio_history_encoder = PortfolioRewardHistoryEncoder(max_coins, set_hidden_dim)
        self.features_per_coin = features_per_coin
        self.use_global_pooling = use_global_pooling
        self.predictive_moe_enabled = bool(predictive_moe_enabled)
        self.foundation_typed_control_configured = bool(
            getattr(config, "FOUNDATION_TYPED_CONTROL_ENABLED", True)
        )
        self.foundation_typed_control_enabled = bool(
            getattr(config, "FOUNDATION_LEARNING_ENABLED", False)
            and self.foundation_typed_control_configured
        )
        self.local_financial_handoff_enabled = bool(
            getattr(config, "LOCAL_FINANCIAL_HANDOFF_ENABLED", False)
        )
        self.basket_conditioned_allocation_enabled = bool(
            getattr(config, "BASKET_CONDITIONED_ALLOCATION_ENABLED", False)
            and self.local_financial_handoff_enabled
            and asset_identity_dim > 0
            and use_global_pooling
            and not self.foundation_typed_control_enabled
        )
        self.financial_forecast_identity_enabled = bool(
            getattr(config, "FINANCIAL_FORECAST_IDENTITY_ENABLED", False)
            and self.local_financial_handoff_enabled
            and asset_identity_dim > 0
        )
        self.local_financial_context_max_abs = float(
            getattr(config, "LOCAL_FINANCIAL_CONTEXT_MAX_ABS", 0.02)
        )
        if self.local_financial_context_max_abs <= 0.0:
            raise ValueError("Local-financial return correction bound must be positive")
        (
            self.local_feature_dim,
            self.position_feature_dim,
            self.comparative_feature_dim,
            self.context_feature_dim,
        ) = _resolve_token_scope_dims(
            features_per_coin,
            local_feature_dim,
            position_feature_dim,
            context_feature_dim,
            comparative_feature_dim,
        )

        # Construct one action-independent market relation for the replay
        # state, then compose candidate allocation in the shared per-asset
        # value path. This gives every ProExp candidate a common opportunity-set
        # coordinate system instead of allowing each candidate to rewrite the
        # critic's attention queries, keys, and values. Cost basis is
        # post-attention position state.
        # Asset ids are metadata for transition alignment, not learned state.
        # Path 2 accepts a zero identity width so the same Q rule can be
        # applied to observable asset roles in an unseen 8-of-N universe.
        resolved_num_asset_ids = int(
            max_coins if num_asset_ids is None else num_asset_ids
        )
        if resolved_num_asset_ids <= 0:
            raise ValueError("num_asset_ids must be positive")
        self.asset_padding_id = resolved_num_asset_ids
        self.asset_identity_dim = int(asset_identity_dim)
        if self.asset_identity_dim < 0:
            raise ValueError("asset_identity_dim must be non-negative")
        self.basket_context_dim = int(basket_context_dim)
        if self.basket_context_dim < 0:
            raise ValueError("basket_context_dim must be non-negative")
        self.basket_membership_dim = int(basket_membership_dim)
        if self.basket_membership_dim < 0:
            raise ValueError("basket_membership_dim must be non-negative")
        # The availability mask is an observed control condition, not an
        # episode-relative market feature.  Keep it and the learned pool CLS
        # summary explicit in the critic even when local-causal mode zeros the
        # legacy basket feature lanes.
        self.pool_task_conditioning_enabled = bool(
            getattr(config, "CRITIC_POOL_TASK_CONDITIONING_ENABLED", False)
            and getattr(config, "EARNMORE_POOL_LEVEL_ENABLED", False)
            and self.asset_identity_dim > 0
        )
        self.centered_advantage_enabled = bool(
            getattr(config, "CRITIC_CENTERED_ADVANTAGE_ENABLED", False)
            and self.pool_task_conditioning_enabled
        )
        self.bellman_composition = str(
            getattr(config, "CRITIC_BELLMAN_COMPOSITION", "separate_v")
        )
        if self.bellman_composition not in {"direct_q", "separate_v"}:
            raise ValueError("Unknown critic Bellman composition")
        self.centered_bellman_enabled = (
            self.centered_advantage_enabled
            and self.bellman_composition == "separate_v"
        )
        self.task_feature_enabled = (
            self.basket_context_dim > 0
            or self.basket_membership_dim > 0
            or self.pool_task_conditioning_enabled
        )
        # Fused critic token + prev/current/delta weights, plus optional task code
        # (regime g and/or membership z share one set_hidden_dim slot via sum).
        # Local utilities must see task codes directly: regime FiLM starts at
        # identity, while membership FiLM/proj are live at cold start.
        local_value_input_dim = set_hidden_dim + 3 + (
            set_hidden_dim if self.task_feature_enabled else 0
        )
        base_global_head_input_dim = set_hidden_dim + 4
        global_head_input_dim = base_global_head_input_dim + (
            set_hidden_dim if self.pool_task_conditioning_enabled else 0
        )
        legacy_head_input_dim = base_global_head_input_dim + (
            set_hidden_dim if self.task_feature_enabled else 0
        )

        self.asset_identity_embedding = (
            nn.Embedding(
                resolved_num_asset_ids + 1,
                self.asset_identity_dim,
                padding_idx=self.asset_padding_id,
            )
            if self.asset_identity_dim > 0
            else None
        )
        self.token_encoder = AttentionDeepSetEncoder(
            features_per_coin=features_per_coin,
            local_feature_dim=self.local_feature_dim,
            position_feature_dim=self.position_feature_dim,
            context_feature_dim=self.context_feature_dim,
            comparative_feature_dim=self.comparative_feature_dim,
            model_dim=set_hidden_dim,
            net_arch=net_arch,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_market_attention=use_market_attention,
            use_global_pooling=use_global_pooling,
            relation_aux_dim=0,
            # The rejected critic relation used [previous, current, delta]
            # allocation fields.  Preserve only the random draws its two MLPs
            # consumed so removing it cannot silently reseed the retained Q
            # function and invert the cold-start actor-loss sign.
            initialization_rng_compat_relation_aux_dim=3,
            relation_uses_comparative=True,
            relation_uses_context=True,
            # Market context belongs to the state relation. Candidate
            # allocation belongs to the value composition below.
            relation_uses_position=False,
            asset_identity_dim=self.asset_identity_dim,
            basket_context_dim=self.basket_context_dim,
            basket_membership_dim=self.basket_membership_dim,
            predictive_moe_enabled=predictive_moe_enabled,
            predictive_moe_num_experts=predictive_moe_num_experts,
            predictive_moe_expert_hidden_dim=predictive_moe_expert_hidden_dim,
            predictive_moe_router_temperature=(
                predictive_moe_router_temperature
            ),
            predictive_moe_max_residual_abs=predictive_moe_max_residual_abs,
            fam_enabled=bool(getattr(config, "MDAEN_FAM_ENABLED", False)),
            fam_feature_dim=int(
                getattr(config, "MDAEN_FAM_FEATURE_COUNT", 0)
            ),
            fam_history_length=int(
                getattr(config, "MDAEN_FAM_HISTORY_LENGTH", 8)
            ),
            fam_kernel_sizes=tuple(
                getattr(config, "MDAEN_FAM_KERNEL_SIZES", (3, 5))
            ),
            fam_asset_vector_dim=int(
                getattr(config, "MDAEN_FAM_ASSET_VECTOR_DIM", set_hidden_dim)
            ),
            fam_initial_residual_scale=float(
                getattr(
                    config,
                    "MDAEN_POST_AAM_SNAPSHOT_INITIAL_RESIDUAL_SCALE",
                    getattr(
                        config,
                        "MDAEN_AAM_SNAPSHOT_INITIAL_RESIDUAL_SCALE",
                        getattr(config, "MDAEN_FAM_INITIAL_RESIDUAL_SCALE", 0.05),
                    ),
                )
            ),
            fam_max_residual_scale=float(
                getattr(
                    config,
                    "MDAEN_POST_AAM_SNAPSHOT_MAX_RESIDUAL_SCALE",
                    getattr(
                        config,
                        "MDAEN_AAM_SNAPSHOT_MAX_RESIDUAL_SCALE",
                        getattr(config, "MDAEN_FAM_MAX_RESIDUAL_SCALE", 0.25),
                    ),
                )
            ),
        )
        self.basket_context_proj = (
            nn.Linear(self.basket_context_dim, set_hidden_dim)
            if self.basket_context_dim > 0
            else None
        )
        if self.basket_context_proj is not None:
            nn.init.xavier_uniform_(self.basket_context_proj.weight, gain=0.25)
            nn.init.zeros_(self.basket_context_proj.bias)
        # Membership projection is created inside fork_rng below so Linear/Xavier
        # draws cannot reseed local_value_backbone, asset_value_head, global head,
        # or twin Q2 (the historical actor_q_loss sign-flip failure mode).
        self.basket_membership_proj = None
        self.pool_task_mask_projection = None
        self.pool_task_cls_projection = None
        self.pool_task_norm = None

        # Keep the critic local-value head as a plain remapping MLP. At this
        # stage the encoder has already stabilized representation depth; the
        # value head needs curvature more than identity preservation.
        #
        # RNG / cold-start actor_q_loss compatibility: before local g(B) concat,
        # local_value_input_dim was set_hidden_dim + 3. Replacing that MLP with a
        # wider one on the main stream re-seeds asset_value_head, the global head,
        # and twin Q2 (flipping actor_q_loss sign). Rebuilding the full wider MLP
        # even inside fork_rng still replaces the legacy local weights and can
        # change Q1 at g=0. So: build the legacy-width backbone on the main
        # stream, then expand only the first Linear's new g(B) columns inside
        # fork_rng while copying the legacy block bit-exactly.
        legacy_local_value_input_dim = set_hidden_dim + 3
        self.local_value_backbone = _build_mlp_layers(
            legacy_local_value_input_dim, net_arch, set_hidden_dim
        )
        self.asset_value_head = nn.Linear(set_hidden_dim, 1)
        # Build the historical wide head first so every retained parameter and
        # twin-Q RNG position stays matched to the established critic.  Legacy
        # handcrafted task columns remain excluded from the global head.  The
        # causal global-pool task state is different: it is the exact observed
        # availability mask plus the learned pool CLS summary and therefore
        # remains explicit all the way into scalar Q.
        self.head = _build_mlp_layers(legacy_head_input_dim, net_arch, 1)
        if self.task_feature_enabled and not self.pool_task_conditioning_enabled:
            old_head_linear = self.head[0]
            if not isinstance(old_head_linear, nn.Linear):
                raise TypeError(
                    "Expected head[0] to be Linear before removing the direct "
                    f"task columns, got {type(old_head_linear)}"
                )
            with torch.random.fork_rng(devices=[]):
                narrowed_head_linear = nn.Linear(
                    global_head_input_dim,
                    old_head_linear.out_features,
                    bias=old_head_linear.bias is not None,
                )
            with torch.no_grad():
                narrowed_head_linear.weight.copy_(
                    old_head_linear.weight[:, :global_head_input_dim]
                )
                if narrowed_head_linear.bias is not None:
                    if old_head_linear.bias is None:
                        narrowed_head_linear.bias.zero_()
                    else:
                        narrowed_head_linear.bias.copy_(old_head_linear.bias)
            head_layers = list(self.head)
            head_layers[0] = narrowed_head_linear
            self.head = nn.Sequential(*head_layers)
        self.global_head_uses_direct_task = self.pool_task_conditioning_enabled
        self.state_value_head = None
        if self.centered_advantage_enabled:
            # Historical separate-V mode receives the opportunity-set mean, the market
            # representation currently held by the portfolio, cash, active
            # fraction, and the exact mask+CLS task coordinate.  It never sees
            # the proposed action. It remains available for controlled legacy
            # experiments; direct-Q mode obtains V_ref from the live Q instead.
            state_value_input_dim = (
                2 * set_hidden_dim
                + 2
                + (set_hidden_dim if self.task_feature_enabled else 0)
            )
            # Keep the historical module for explicit separate-V experiments.
            # In direct-Q mode it is dormant and cannot reseed the retained
            # action heads or Q2: introducing V consumed those random draws
            # on September 20 even though the actor had not changed.
            with torch.random.fork_rng(
                devices=[], enabled=self.bellman_composition == "direct_q"
            ):
                self.state_value_head = _build_mlp_layers(
                    state_value_input_dim, net_arch, 1
                )
            # A neutral baseline makes the initial absolute Q level explicit:
            # Bellman evidence must earn it, while the existing action scorer
            # can immediately express differences around the current portfolio.
            final_value_layer = self.state_value_head[-1]
            if not isinstance(final_value_layer, nn.Linear):
                raise TypeError("state_value_head must terminate in Linear")
            nn.init.zeros_(final_value_layer.weight)
            nn.init.zeros_(final_value_layer.bias)

        # The 8-of-N critic needs an efficient way to express "this observable
        # behavior under this proposed allocation".  Keep that interaction in
        # the shared per-asset value stream instead of constructing a direct
        # portfolio utility sum. The full-width allocation-conditioned relation
        # has been removed so this bounded path has clear ownership of learned
        # behavior × allocation interactions.
        default_interaction_dim = max(4, min(16, set_hidden_dim // 8))
        self.allocation_interaction_dim = int(
            default_interaction_dim
            if allocation_interaction_dim is None
            else allocation_interaction_dim
        )
        if self.allocation_interaction_dim <= 0:
            raise ValueError("allocation_interaction_dim must be positive")
        self.allocation_interaction_max_norm = float(
            allocation_interaction_max_norm
        )
        if self.allocation_interaction_max_norm <= 0.0:
            raise ValueError("allocation_interaction_max_norm must be positive")

        # The adapter survives the joint-stem correction. It now composes the
        # proposed allocation with one coherent market basis and, when present,
        # the small bounded position code. It does not receive three separately
        # normalized full-width feature-family embeddings.
        self.allocation_basis_component_names = [
            "local_control",
            "relational_control",
        ]
        if self.position_feature_dim > 0:
            self.allocation_basis_component_names.append("position")
        self.allocation_basis_component_names = tuple(
            self.allocation_basis_component_names
        )
        control_dim = self.token_encoder.control_stream_dim
        self.basket_opportunity_head = None
        if self.basket_conditioned_allocation_enabled:
            with torch.random.fork_rng(devices=[]):
                self.basket_opportunity_head = BasketConditionedOpportunityHead(
                    max_coins, set_hidden_dim, control_dim, zero_output=True
                )
        state_scope_dim = (
            (2 * control_dim) + self.token_encoder.position_latent_dim
        )
        interaction_role_dim = 4 * self.allocation_interaction_dim

        self.uvfa_local_interaction_enabled = bool(
            uvfa_local_interaction_enabled
        )
        self.uvfa_local_interaction_rank = int(uvfa_local_interaction_rank)
        self.uvfa_local_interaction_max_abs = float(
            uvfa_local_interaction_max_abs
        )
        # Keep the explicit compositional factors ticker-neutral even while the
        # established encoder/local value path retains its soft identity
        # residual.  Every input here is observable for a future unseen asset.
        self.uvfa_asset_factor_input_dim = control_dim + 3
        if self.uvfa_local_interaction_enabled:
            if self.basket_context_dim <= 0:
                raise ValueError(
                    "The low-rank local UVFA interaction requires continuous "
                    "basket_context; membership-only conditioning is not a "
                    "transferable task coordinate."
                )
            if self.uvfa_local_interaction_rank <= 0:
                raise ValueError("uvfa_local_interaction_rank must be positive")
            if self.uvfa_local_interaction_max_abs <= 0.0:
                raise ValueError(
                    "uvfa_local_interaction_max_abs must be positive"
                )
        self.uvfa_asset_factor_projection = None
        self.uvfa_task_factor_projection = None
        self.uvfa_relational_factor_projection = None
        self.uvfa_interaction_gate_logit = None

        # Keep the adapter's random initialization outside the RNG stream used
        # to construct the original twin critics.  Without this isolation, the
        # adapter attached to Q1 consumes random numbers before Q2 is built, so
        # a matched training seed silently starts from a different Q2 and thus
        # a different clipped-double-Q target even when the adapter is gated
        # down.  fork_rng restores the caller's CPU RNG state on exit while the
        # registered modules retain the parameters initialized inside it.
        # Local task-feature column expansion lives in this isolated stream.
        # Membership projection uses a separate fork so its Xavier draws cannot
        # shift the expand/adapter stream relative to the no-membership build.
        with torch.random.fork_rng(devices=[]):
            if self.task_feature_enabled:
                old_linear = self.local_value_backbone[0]
                if not isinstance(old_linear, nn.Linear):
                    raise TypeError(
                        "Expected local_value_backbone[0] to be Linear before "
                        f"g(B) expand, got {type(old_linear)}"
                    )
                if old_linear.in_features != legacy_local_value_input_dim:
                    raise ValueError(
                        "local_value_backbone legacy width mismatch before g(B) "
                        f"expand: expected {legacy_local_value_input_dim}, got "
                        f"{old_linear.in_features}"
                    )
                expanded = nn.Linear(
                    local_value_input_dim,
                    old_linear.out_features,
                    bias=old_linear.bias is not None,
                )
                with torch.no_grad():
                    expanded.weight.zero_()
                    expanded.weight[:, :legacy_local_value_input_dim].copy_(
                        old_linear.weight
                    )
                    nn.init.xavier_uniform_(
                        expanded.weight[:, legacy_local_value_input_dim:],
                        gain=0.25,
                    )
                    if expanded.bias is not None:
                        if old_linear.bias is None:
                            expanded.bias.zero_()
                        else:
                            expanded.bias.copy_(old_linear.bias)
                backbone_layers = list(self.local_value_backbone)
                backbone_layers[0] = expanded
                self.local_value_backbone = nn.Sequential(*backbone_layers)
            self.allocation_basis_projection = nn.Linear(
                state_scope_dim,
                self.allocation_interaction_dim,
            )
            self.allocation_interaction_up = nn.Linear(
                interaction_role_dim,
                set_hidden_dim,
            )
            # The scalar-Q trunk receives explicit local, relational, and
            # local×relational action lanes.  The established fused trunk is a
            # bounded residual so it can add detail without becoming a route
            # around the foundation representation contract.
            self.local_control_value = nn.Linear(
                control_dim + 3, set_hidden_dim
            )
            self.relational_control_value = nn.Linear(
                control_dim
                + 3
                + (set_hidden_dim if self.task_feature_enabled else 0),
                set_hidden_dim,
            )
            self.cross_control_value = nn.Linear(
                (2 * control_dim) + 3, set_hidden_dim
            )
            for projection in (
                self.local_control_value,
                self.relational_control_value,
                self.cross_control_value,
            ):
                nn.init.xavier_uniform_(projection.weight, gain=0.10)
                nn.init.zeros_(projection.bias)
            nn.init.xavier_uniform_(
                self.allocation_basis_projection.weight,
                gain=0.25,
            )
            nn.init.zeros_(self.allocation_basis_projection.bias)

            # The adapter must add no unearned value or action gradient at
            # initialization.  A zero output projection makes its residual
            # exactly zero, while gradients from the first TD error still
            # reach this projection. The explicit scalar allocation fields in
            # the local value head remain fully live throughout this cold start.
            nn.init.zeros_(self.allocation_interaction_up.weight)
            nn.init.zeros_(self.allocation_interaction_up.bias)

            if self.uvfa_local_interaction_enabled:
                # p_i(s, a_i): reusable observable-state × allocation factors.
                # The projection is zero at cold start so the complete UVFA
                # correction and its action Jacobian are exactly zero until TD
                # or component evidence trains it.  t(g(B)) is initialized
                # live, allowing the first update to learn factor directions.
                self.uvfa_asset_factor_projection = nn.Linear(
                    self.uvfa_asset_factor_input_dim,
                    self.uvfa_local_interaction_rank,
                )
                self.uvfa_task_factor_projection = nn.Linear(
                    self.basket_context_dim,
                    self.uvfa_local_interaction_rank,
                )
                self.uvfa_relational_factor_projection = nn.Linear(
                    control_dim,
                    self.uvfa_local_interaction_rank,
                )
                nn.init.zeros_(self.uvfa_asset_factor_projection.weight)
                nn.init.zeros_(self.uvfa_asset_factor_projection.bias)
                nn.init.xavier_uniform_(
                    self.uvfa_task_factor_projection.weight,
                    gain=0.25,
                )
                nn.init.zeros_(self.uvfa_task_factor_projection.bias)
                nn.init.xavier_uniform_(
                    self.uvfa_relational_factor_projection.weight,
                    gain=0.25,
                )
                nn.init.zeros_(self.uvfa_relational_factor_projection.bias)

        with torch.random.fork_rng(devices=[]):
            if self.basket_membership_dim > 0:
                self.basket_membership_proj = nn.Linear(
                    self.basket_membership_dim, set_hidden_dim
                )
                nn.init.xavier_uniform_(
                    self.basket_membership_proj.weight, gain=0.25
                )
                nn.init.zeros_(self.basket_membership_proj.bias)

        # This task path is intentionally separate from episode-relative
        # basket features.  W_m m supplies an exact, additive coordinate for
        # the available global assets; W_c CLS supplies the current learned
        # pool relation.  LayerNorm keeps the combined condition bounded before
        # it is broadcast to local utilities and appended to the global head.
        with torch.random.fork_rng(devices=[]):
            if self.pool_task_conditioning_enabled:
                self.pool_task_mask_projection = nn.Linear(
                    self.max_coins, set_hidden_dim, bias=False
                )
                self.pool_task_cls_projection = nn.Linear(
                    set_hidden_dim, set_hidden_dim, bias=False
                )
                self.pool_task_norm = nn.LayerNorm(set_hidden_dim)
                nn.init.xavier_uniform_(
                    self.pool_task_mask_projection.weight, gain=0.25
                )
                nn.init.xavier_uniform_(
                    self.pool_task_cls_projection.weight, gain=0.25
                )

        # After TD evidence wakes the zero output projection, this gate keeps
        # the learned correction norm bounded independently of hidden width.
        self.allocation_interaction_gate_logit = nn.Parameter(
            torch.tensor(-2.0)
        )
        self.fused_value_max_scale = float(
            getattr(config, "FOUNDATION_FUSED_CONTROL_MAX_SCALE", 0.25)
        )
        self.fused_value_gate_logit = nn.Parameter(torch.tensor(0.0))
        if self.uvfa_local_interaction_enabled:
            self.uvfa_interaction_gate_logit = nn.Parameter(
                torch.tensor(-2.0)
            )

        # The causal one-step per-asset labels are representation supervision,
        # not Bellman values.  Give them a private predictor so their noisy
        # one-step gradients cannot directly rewrite the local Q backbone,
        # asset-value head, allocation adapter, or low-rank Q correction that
        # defines the actor's steering surface.
        #
        # The private head predicts a unit-exposure asset return from the shared
        # market token.  Its teacher is independent of the current allocation,
        # so an unowned asset can still contribute counterfactual evidence.
        # Exposure enters only in the legacy ablation or later through the
        # TD-owned forecast-times-weight-change bridge.  Only the shared token
        # representation is common with Q, where the existing gradient-ratio
        # controller bounds auxiliary pressure.
        #
        # These modules are built for enabled and disabled critics alike inside
        # fork_rng.  Consequently, toggling the auxiliary neither consumes the
        # historical critic RNG stream nor changes any parameter or output on
        # the established Q path under a matched seed.
        self.component_outcome_aux_enabled = bool(
            component_outcome_aux_enabled
        )
        self.component_outcome_dim = int(component_outcome_dim)
        if self.component_outcome_dim <= 0:
            raise ValueError("component_outcome_dim must be positive")
        self.supervise_local_utilities = False
        self.component_outcome_unit_return_targets = bool(
            component_outcome_unit_return_targets
        )
        if self.local_financial_handoff_enabled and (
            not self.component_outcome_aux_enabled
            or not self.component_outcome_unit_return_targets
            or self.component_outcome_dim != 1
        ):
            raise ValueError(
                "Local-financial handoff requires one supervised unit-return "
                "component target per asset"
            )
        self.component_forecast_bridge_enabled = bool(
            component_forecast_bridge_enabled
        )
        if self.local_financial_handoff_enabled and self.component_forecast_bridge_enabled:
            raise ValueError("The old forecast bridge must be off for local-financial handoff")
        self.component_forecast_bridge_local_only = bool(
            component_forecast_bridge_local_only
        )
        if (
            self.component_forecast_bridge_enabled
            and not self.component_outcome_aux_enabled
        ):
            raise ValueError(
                "The component forecast bridge requires component-outcome "
                "auxiliary supervision."
            )
        self.component_forecast_bridge_return_reference = float(
            component_forecast_bridge_return_reference
        )
        if self.component_forecast_bridge_return_reference <= 0.0:
            raise ValueError(
                "component_forecast_bridge_return_reference must be positive"
            )
        self.component_forecast_bridge_max_abs = float(
            component_forecast_bridge_max_abs
        )
        if self.component_forecast_bridge_max_abs <= 0.0:
            raise ValueError("component_forecast_bridge_max_abs must be positive")
        with torch.random.fork_rng(devices=[]):
            self.component_outcome_backbone = _build_mlp_layers(
                set_hidden_dim,
                net_arch,
                set_hidden_dim,
            )
            self.component_outcome_head = nn.Linear(
                set_hidden_dim,
                self.component_outcome_dim,
            )
            self.component_local_backbone = None
            self.component_context_backbone = None
            if self.local_financial_handoff_enabled:
                # FAM's peer-independent code is the financial anchor. AAM
                # receives a separate, bounded correction; both heads are
                # shared across all assets and all basket permutations.
                self.component_local_backbone = nn.Sequential(
                    nn.LayerNorm(
                        self.token_encoder.fam_asset_vector_dim
                        + (self.asset_identity_dim if self.financial_forecast_identity_enabled else 0)
                    ),
                    nn.Linear(
                        self.token_encoder.fam_asset_vector_dim
                        + (self.asset_identity_dim if self.financial_forecast_identity_enabled else 0),
                        set_hidden_dim,
                    ),
                    nn.SiLU(),
                    nn.Linear(set_hidden_dim, 1),
                )
                self.component_context_backbone = nn.Sequential(
                    nn.LayerNorm(control_dim),
                    nn.Linear(control_dim, set_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(set_hidden_dim, 1),
                )
                nn.init.xavier_uniform_(
                    self.component_local_backbone[-1].weight, gain=0.01
                )
                nn.init.zeros_(self.component_local_backbone[-1].bias)
                nn.init.zeros_(self.component_context_backbone[-1].weight)
                nn.init.zeros_(self.component_context_backbone[-1].bias)
            # Persistent private counterfactual head.  Its six outputs are
            # four net-return horizons, turnover, and downside.  It shares the
            # action-conditioned local backbone but never replaces ``head`` or
            # ``asset_value_head`` in scalar Bellman TD.
            self.foundation_action_outcome_head = nn.Linear(
                set_hidden_dim, 6
            )
            # Component labels are deliberately small causal financial
            # increments.  A small live output keeps the first predictions in
            # that numerical neighborhood while preserving a nonzero encoder
            # gradient for the adaptive gradient-ratio calibration.
            nn.init.xavier_uniform_(
                self.component_outcome_head.weight,
                gain=0.01,
            )
            nn.init.zeros_(self.component_outcome_head.bias)
            nn.init.xavier_uniform_(
                self.foundation_action_outcome_head.weight, gain=0.01
            )
            nn.init.zeros_(self.foundation_action_outcome_head.bias)
            self.component_forecast_bridge_projection = nn.Linear(
                self.component_outcome_dim,
                set_hidden_dim,
                bias=False,
            )
            # The bridge has no Q value and no actor gradient at cold start.
            # Scalar TD wakes it after observing whether the forecast-aligned
            # allocation change improved actual portfolio growth.
            nn.init.zeros_(self.component_forecast_bridge_projection.weight)
            self.predictive_local_delta_head = None
            if self.predictive_moe_enabled:
                self.predictive_local_delta_head = nn.Sequential(
                    nn.LayerNorm(set_hidden_dim + 3),
                    nn.Linear(set_hidden_dim + 3, set_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(set_hidden_dim, set_hidden_dim),
                )
                nn.init.xavier_uniform_(
                    self.predictive_local_delta_head[-1].weight, gain=0.01
                )
                nn.init.zeros_(self.predictive_local_delta_head[-1].bias)
        self.component_forecast_bridge_gate_logit = nn.Parameter(
            torch.tensor(-2.0)
        )

    def _local_financial_forecast(
        self, token_components: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict unit returns from local FAM, then a bounded peer correction."""
        if not self.local_financial_handoff_enabled:
            raise RuntimeError("Local-financial forecast is disabled")
        if self.component_local_backbone is None or self.component_context_backbone is None:
            raise RuntimeError("Local-financial predictor was not constructed")
        local_features = token_components["local_temporal"]
        if self.financial_forecast_identity_enabled:
            identity = token_components.get("forecast_asset_identity")
            if identity is None:
                # Synthetic encoded-state probes may omit the identity block.
                identity = local_features.new_zeros(
                    (*local_features.shape[:2], self.asset_identity_dim)
                )
            local_features = torch.cat((local_features, identity), dim=-1)
        local = self.component_local_backbone(local_features)
        context_raw = self.component_context_backbone(
            token_components["relational_control"]
        )
        bound = self.local_financial_context_max_abs
        context_correction = bound * torch.tanh(context_raw / bound)
        return local + context_correction, local, context_correction

    def _component_forecast_bridge_residual(
        self,
        component_return_predictions: torch.Tensor,
        asset_delta: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a bounded TD-owned residual from a detached return forecast.

        The auxiliary predictor supplies a financial coordinate, but detach()
        prevents scalar TD from training that predictor and prevents component
        loss from training this bridge.  Multiplication by asset_delta makes
        the bridge exactly zero when the proposed allocation is unchanged.
        """
        detached_forecast = torch.tanh(
            component_return_predictions.detach()
            / self.component_forecast_bridge_return_reference
        )
        action_centered_forecast = (
            detached_forecast * asset_delta.unsqueeze(-1)
        )
        raw_residual = self.component_forecast_bridge_projection(
            action_centered_forecast
        )
        gate = self.component_forecast_bridge_max_abs * torch.sigmoid(
            self.component_forecast_bridge_gate_logit
        )
        return (
            gate
            * torch.tanh(raw_residual)
            * mask.unsqueeze(-1)
        )

    def _allocation_interaction_residual(
        self,
        token_components: Dict[str, torch.Tensor],
        prev_assets: torch.Tensor,
        current_assets: torch.Tensor,
        asset_delta: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a bounded token-local allocation residual and diagnostics.

        The four products retain assignment information without naming an
        asset: current exposure, previous exposure, signed trade, and turnover
        are each composed with the same learned behavioral basis.  No product
        is pooled or added to Q directly.
        """
        mask_expanded = mask.unsqueeze(-1)
        behavioral_basis = self._allocation_behavioral_basis(
            token_components, mask
        )
        interaction_roles = torch.cat(
            [
                behavioral_basis * current_assets.unsqueeze(-1),
                behavioral_basis * prev_assets.unsqueeze(-1),
                behavioral_basis * asset_delta.unsqueeze(-1),
                behavioral_basis * asset_delta.abs().unsqueeze(-1),
            ],
            dim=-1,
        )
        unit_update = torch.tanh(
            self.allocation_interaction_up(interaction_roles)
        ) * mask_expanded
        interaction_scale = (
            self.allocation_interaction_max_norm
            * torch.sigmoid(self.allocation_interaction_gate_logit)
            / math.sqrt(float(unit_update.shape[-1]))
        ).to(unit_update.dtype)
        residual_update = interaction_scale * unit_update
        return residual_update, behavioral_basis, interaction_scale

    def _allocation_behavioral_basis(
        self,
        token_components: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the shared identity-free basis used to bind state to action.

        The basis is token-local and uses the same projection as the live
        critic allocation adapter. A permutation only moves complete
        asset/basis rows; it cannot change their contents or introduce a slot
        coordinate. Replay can therefore pool weight-bound roles from this
        basis without losing which observable behavior received each weight.
        """
        mask_expanded = mask.unsqueeze(-1)
        basis_components = []
        for component_name in self.allocation_basis_component_names:
            component = token_components[component_name]
            if (
                component_name == "local_control"
                and self.token_encoder.strict_semantic_gradient_ownership
            ):
                component = component.detach()
            basis_components.append(component)
        basis_source = torch.cat(basis_components, dim=-1)
        behavioral_basis = torch.tanh(
            self.allocation_basis_projection(basis_source)
        ) * mask_expanded
        return behavioral_basis

    def _low_rank_uvfa_local_correction(
        self,
        local_asset_action_input: torch.Tensor,
        relational_control: torch.Tensor,
        basket_context: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compose continuous basket task factors with local action factors.

        The returned scalar correction stays inside each shared local utility.
        It is never pooled or added to Q as a standalone basket value.  Exact
        membership is intentionally absent: g(B) must describe the observable
        opportunity set so the interaction has semantics for unseen assets.
        """
        if self.token_encoder.local_causal_inputs_enabled and basket_context is not None:
            basket_context = torch.zeros_like(basket_context)
        if not self.uvfa_local_interaction_enabled:
            zero_correction = mask.new_zeros(mask.shape)
            zero_asset_factors = local_asset_action_input.new_zeros(
                (*mask.shape, 0)
            )
            zero_task_factors = local_asset_action_input.new_zeros(
                (*mask.shape, 0)
            )
            zero_scale = local_asset_action_input.new_zeros(())
            return (
                zero_correction,
                zero_asset_factors,
                zero_task_factors,
                zero_scale,
            )
        if basket_context is None:
            raise ValueError(
                "basket_context is required by the low-rank local UVFA "
                "interaction"
            )
        if (
            self.uvfa_asset_factor_projection is None
            or self.uvfa_task_factor_projection is None
            or self.uvfa_relational_factor_projection is None
            or self.uvfa_interaction_gate_logit is None
        ):
            raise RuntimeError("Low-rank local UVFA modules are not initialized")

        owned_local_asset_action_input = (
            local_asset_action_input.detach()
            if self.token_encoder.strict_semantic_gradient_ownership
            else local_asset_action_input
        )
        asset_factors = torch.tanh(
            self.uvfa_asset_factor_projection(owned_local_asset_action_input)
        ) * mask.unsqueeze(-1)
        # The task side now says both "what kind of selected basket is this?"
        # and "what does this particular asset mean inside that basket?".
        # This keeps the UVFA identity-neutral while preventing one pooled
        # basket vector from assigning the same contextual role to every row.
        global_task_factors = self.uvfa_task_factor_projection(
            basket_context
        )
        asset_task_factors = self.uvfa_relational_factor_projection(
            relational_control
        )
        task_factors = torch.tanh(
            global_task_factors.unsqueeze(1) + asset_task_factors
        ) * mask.unsqueeze(-1)
        normalized_interaction = (
            asset_factors * task_factors
        ).sum(dim=-1) / float(self.uvfa_local_interaction_rank)
        interaction_scale = (
            self.uvfa_local_interaction_max_abs
            * torch.sigmoid(self.uvfa_interaction_gate_logit)
        ).to(local_asset_action_input.dtype)
        correction = interaction_scale * normalized_interaction * mask
        return correction, asset_factors, task_factors, interaction_scale

    def _typed_control_value_features(
        self,
        token_components: Dict[str, torch.Tensor],
        prev_assets: torch.Tensor,
        current_assets: torch.Tensor,
        asset_delta: torch.Tensor,
        mask: torch.Tensor,
        basket_feat: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compose action-conditioned value features without erasing semantics.

        The local lane can learn reusable behavior × allocation rules.  The
        relational lane can learn how the same behavior changes meaning among
        the current alternatives.  Their cross lane represents composition,
        but none of the three must reconstruct the split from a fused token.
        """
        action_roles = (
            prev_assets.unsqueeze(-1),
            current_assets.unsqueeze(-1),
            asset_delta.unsqueeze(-1),
        )
        local_input = torch.cat(
            (token_components["local_control"], *action_roles), dim=-1
        )
        relational_parts = [
            token_components["relational_control"],
            *action_roles,
        ]
        if basket_feat is not None:
            relational_parts.append(
                basket_feat.unsqueeze(1).expand(-1, self.max_coins, -1)
            )
        relational_input = torch.cat(relational_parts, dim=-1)
        contextual_local_control = (
            token_components["local_control"].detach()
            if self.token_encoder.strict_semantic_gradient_ownership
            else token_components["local_control"]
        )
        cross_input = torch.cat(
            (
                contextual_local_control,
                token_components["relational_control"],
                *action_roles,
            ),
            dim=-1,
        )
        expanded_mask = mask.unsqueeze(-1)
        local_features = F.silu(self.local_control_value(local_input))
        relational_features = F.silu(
            self.relational_control_value(relational_input)
        )
        cross_features = F.silu(self.cross_control_value(cross_input))
        typed_features = (
            local_features + relational_features + cross_features
        ) * expanded_mask
        return typed_features, {
            "local": local_features * expanded_mask,
            "relational": relational_features * expanded_mask,
            "cross": cross_features * expanded_mask,
        }

    def encode_state(
        self,
        coin_features: torch.Tensor,
        asset_ids: torch.Tensor,
        mask: torch.Tensor,
        prev_mask: torch.Tensor,
        prev_assets: torch.Tensor,
        basket_context: Optional[torch.Tensor] = None,
        basket_membership: Optional[torch.Tensor] = None,
        feature_history: Optional[torch.Tensor] = None,
        feature_history_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del prev_mask, prev_assets
        asset_identity = (
            self.asset_identity_embedding(asset_ids)
            if self.asset_identity_embedding is not None
            else None
        )
        return self.token_encoder(
            coin_features,
            mask,
            asset_identity=asset_identity,
            basket_context=basket_context if self.basket_context_dim > 0 else None,
            basket_membership=(
                basket_membership if self.basket_membership_dim > 0 else None
            ),
            feature_history=(
                feature_history if self.token_encoder.fam_enabled else None
            ),
            feature_history_mask=(
                feature_history_mask if self.token_encoder.fam_enabled else None
            ),
        )

    def predictive_representation_loss(
        self,
        coin_features: torch.Tensor,
        asset_ids: torch.Tensor,
        mask: torch.Tensor,
        prev_action: torch.Tensor,
        replay_actions: torch.Tensor,
        basket_context: torch.Tensor,
        basket_membership: torch.Tensor,
        next_coin_features: torch.Tensor,
        next_asset_ids: torch.Tensor,
        next_mask: torch.Tensor,
        next_prev_action: torch.Tensor,
        next_actions: torch.Tensor,
        next_basket_context: torch.Tensor,
        next_basket_membership: torch.Tensor,
        feature_history: torch.Tensor,
        feature_history_mask: torch.Tensor,
        next_feature_history: torch.Tensor,
        next_feature_history_mask: torch.Tensor,
        target_q_network: "DeepSetQNetwork",
        dones: Optional[torch.Tensor] = None,
        *,
        future_actions: Optional[torch.Tensor] = None,
        future_valid_steps: Optional[torch.Tensor] = None,
        future_outcome_valid_steps: Optional[torch.Tensor] = None,
        future_rewards: Optional[torch.Tensor] = None,
        future_dones: Optional[torch.Tensor] = None,
        future_continuation_value_targets: Optional[torch.Tensor] = None,
        current_q_targets: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        horizon_discount: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Train critic-facing latents over a recursive SPR trajectory."""
        if future_actions is None:
            future_actions = replay_actions.unsqueeze(1)
        if future_actions.dim() != 3:
            raise ValueError("critic SPR actions must be [batch, horizon, action]")
        batch_size, horizon = future_actions.shape[:2]

        def _lift_horizon(value: torch.Tensor) -> torch.Tensor:
            return value.unsqueeze(1) if value.dim() >= 1 else value

        # Older one-step callers pass tensors without a horizon coordinate.
        if next_coin_features.dim() == coin_features.dim():
            next_coin_features = _lift_horizon(next_coin_features)
            next_asset_ids = _lift_horizon(next_asset_ids)
            next_mask = _lift_horizon(next_mask)
            next_prev_action = _lift_horizon(next_prev_action)
            next_actions = _lift_horizon(next_actions)
            next_basket_context = _lift_horizon(next_basket_context)
            next_basket_membership = _lift_horizon(next_basket_membership)
            next_feature_history = _lift_horizon(next_feature_history)
            next_feature_history_mask = _lift_horizon(
                next_feature_history_mask
            )
        if int(next_coin_features.shape[1]) != horizon:
            raise ValueError("critic SPR target/action horizons do not match")

        def _flatten_horizon(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch_size * horizon, *value.shape[2:])

        def _restore_horizon(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch_size, horizon, *value.shape[1:])

        flat_next_coin_features = _flatten_horizon(next_coin_features)
        flat_next_asset_ids = _flatten_horizon(next_asset_ids)
        flat_next_mask = _flatten_horizon(next_mask)
        flat_next_prev_action = _flatten_horizon(next_prev_action)
        flat_next_actions = _flatten_horizon(next_actions)
        flat_next_context = _flatten_horizon(next_basket_context)
        flat_next_membership = _flatten_horizon(next_basket_membership)
        flat_next_history = _flatten_horizon(next_feature_history)
        flat_next_history_mask = _flatten_horizon(next_feature_history_mask)

        current_identity = (
            self.asset_identity_embedding(asset_ids)
            if self.asset_identity_embedding is not None
            else None
        )
        current_components = self.token_encoder.encode_components(
            coin_features,
            mask,
            asset_identity=current_identity,
            basket_context=(
                basket_context if self.basket_context_dim > 0 else None
            ),
            basket_membership=(
                basket_membership if self.basket_membership_dim > 0 else None
            ),
            feature_history=feature_history,
            feature_history_mask=feature_history_mask,
            detach_predictive_control_adapter=bool(
                getattr(
                    config,
                    "PREDICTIVE_SSL_CONTROL_ADAPTER_FINANCIAL_ONLY",
                    True,
                )
            ),
        )
        with torch.no_grad():
            flat_next_identity = (
                target_q_network.asset_identity_embedding(flat_next_asset_ids)
                if target_q_network.asset_identity_embedding is not None
                else None
            )
            flat_next_components = target_q_network.token_encoder.encode_components(
                flat_next_coin_features,
                flat_next_mask,
                asset_identity=flat_next_identity,
                basket_context=(
                    flat_next_context
                    if target_q_network.basket_context_dim > 0
                    else None
                ),
                basket_membership=(
                    flat_next_membership
                    if target_q_network.basket_membership_dim > 0
                    else None
                ),
                feature_history=flat_next_history,
                feature_history_mask=flat_next_history_mask,
                detach_predictive_control_adapter=bool(
                    getattr(
                        config,
                        "PREDICTIVE_SSL_CONTROL_ADAPTER_FINANCIAL_ONLY",
                        True,
                    )
                ),
            )
            next_components = {
                name: _restore_horizon(value)
                for name, value in flat_next_components.items()
            }
            target_predictive_layer = (
                target_q_network.token_encoder.predictive_expert_layer
            )
            if future_continuation_value_targets is not None:
                continuation_value_targets = (
                    future_continuation_value_targets.detach().float().reshape(
                        batch_size, horizon, 1
                    )
                )
            elif bool(getattr(config, "VALUE_EQUIVALENT_MODEL_ENABLED", True)):
                continuation_value_targets = (
                    _restore_horizon(
                        target_predictive_layer.state_value(flat_next_components)
                    )
                    if target_predictive_layer is not None
                    else coin_features.new_zeros((batch_size, horizon, 1))
                )
            else:
                continuation_value_targets = coin_features.new_zeros(
                    (batch_size, horizon, 1)
                )

        traded_weights = normalize_raw_actions_to_portfolio_weights(
            future_actions.float().reshape(batch_size * horizon, -1),
            mask.unsqueeze(1)
            .expand(-1, horizon, -1)
            .reshape(batch_size * horizon, -1),
        ).reshape(batch_size, horizon, -1)
        current_assets = traded_weights[:, :, 1:1 + self.max_coins]
        previous_assets = torch.cat(
            [
                prev_action[:, 1:1 + self.max_coins].unsqueeze(1),
                current_assets[:, :-1],
            ],
            dim=1,
        )
        action_roles = torch.stack(
            (
                previous_assets,
                current_assets,
                current_assets - previous_assets,
            ),
            dim=-1,
        )
        if future_valid_steps is not None:
            transition_valid = future_valid_steps.bool().reshape(
                batch_size, horizon
            )
        elif dones is not None:
            transition_valid = (dones.view(-1) < 0.5).unsqueeze(1)
        else:
            transition_valid = torch.ones(
                (batch_size, horizon), dtype=torch.bool, device=mask.device
            )
        stats = self.token_encoder.predictive_loss(
            current_features=coin_features,
            current_components=current_components,
            current_asset_ids=asset_ids,
            current_mask=mask,
            next_features=next_coin_features.detach(),
            next_components=next_components,
            next_asset_ids=next_asset_ids,
            next_mask=next_mask,
            action_roles=action_roles,
            transition_valid=transition_valid,
            outcome_valid_steps=future_outcome_valid_steps,
            future_rewards=future_rewards,
            future_dones=future_dones,
            continuation_value_targets=continuation_value_targets,
            current_q_targets=current_q_targets,
            gamma=gamma,
            feature_weight=float(
                getattr(config, "PREDICTIVE_SSL_FEATURE_WEIGHT", 1.0)
            ),
            fam_weight=float(
                getattr(config, "PREDICTIVE_SSL_FAM_WEIGHT", 0.0)
            ),
            market_weight=float(
                getattr(config, "PREDICTIVE_SSL_MARKET_WEIGHT", 0.25)
            ),
            relation_weight=float(
                getattr(config, "PREDICTIVE_SSL_RELATION_WEIGHT", 0.50)
            ),
            fused_weight=float(
                getattr(config, "PREDICTIVE_SSL_FUSED_WEIGHT", 1.0)
            ),
            load_balance_weight=float(
                getattr(config, "PREDICTIVE_MOE_LOAD_BALANCE_WEIGHT", 0.01)
            ),
            diversity_weight=float(
                getattr(config, "PREDICTIVE_MOE_DIVERSITY_WEIGHT", 0.01)
            ),
            routing_entropy_weight=float(
                getattr(
                    config,
                    "PREDICTIVE_MOE_ROUTING_ENTROPY_WEIGHT",
                    0.01,
                )
            ),
            horizon_discount=horizon_discount,
        )
        if self.predictive_local_delta_head is None:
            return stats

        current_local_features = self._predictive_local_features(
            current_components,
            coin_features,
            mask,
            prev_action,
            replay_actions,
            basket_context,
            basket_membership,
        )
        with torch.no_grad():
            flat_next_local_features = target_q_network._predictive_local_features(
                flat_next_components,
                flat_next_coin_features,
                flat_next_mask,
                flat_next_prev_action,
                flat_next_actions,
                flat_next_context,
                flat_next_membership,
            )
            next_local_features = _restore_horizon(
                flat_next_local_features
            )

        predicted_local_features = current_local_features
        downstream_numerator = current_local_features.new_zeros(())
        downstream_denominator = current_local_features.new_zeros(())
        for step in range(horizon):
            predicted_local_features = (
                predicted_local_features
                + self.predictive_local_delta_head(
                    torch.cat(
                        [predicted_local_features, action_roles[:, step]],
                        dim=-1,
                    )
                )
            )
            aligned_target, has_match = self.token_encoder._align_next_tokens(
                next_local_features[:, step],
                asset_ids,
                next_asset_ids[:, step],
            )
            aligned_mask, _ = self.token_encoder._align_next_tokens(
                next_mask[:, step].unsqueeze(-1),
                asset_ids,
                next_asset_ids[:, step],
            )
            valid = (
                mask.bool()
                & has_match
                & aligned_mask.squeeze(-1).bool()
                & transition_valid[:, step].view(-1, 1)
            )
            error = 1.0 - F.cosine_similarity(
                predicted_local_features, aligned_target, dim=-1
            )
            step_loss = self.token_encoder._masked_scalar_mean(
                error, valid.to(error.dtype)
            )
            step_weight = current_local_features.new_tensor(
                float(horizon_discount) ** step
            ) * valid.any().to(current_local_features.dtype)
            downstream_numerator = (
                downstream_numerator + step_weight * step_loss
            )
            downstream_denominator = downstream_denominator + step_weight
        downstream_loss = downstream_numerator / downstream_denominator.clamp(
            min=1.0
        )
        stats["loss"] = stats["loss"] + float(
            getattr(config, "PREDICTIVE_SSL_DOWNSTREAM_WEIGHT", 1.0)
        ) * downstream_loss
        stats["downstream_loss"] = downstream_loss
        return stats

    def _project_task_features(
        self,
        basket_context: Optional[torch.Tensor],
        basket_membership: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.token_encoder.local_causal_inputs_enabled:
            basket_context = None if basket_context is None else torch.zeros_like(basket_context)
            basket_membership = None if basket_membership is None else torch.zeros_like(basket_membership)
        feat = None
        if self.basket_context_proj is not None:
            if basket_context is None:
                raise ValueError(
                    "DeepSetQNetwork requires basket_context when "
                    f"basket_context_dim={self.basket_context_dim}"
                )
            feat = self.basket_context_proj(basket_context)
        if self.basket_membership_proj is not None:
            if basket_membership is None:
                raise ValueError(
                    "DeepSetQNetwork requires basket_membership when "
                    f"basket_membership_dim={self.basket_membership_dim}"
                )
            membership_feat = self.basket_membership_proj(basket_membership)
            feat = membership_feat if feat is None else feat + membership_feat
        return feat

    def _project_pool_task_features(
        self,
        mask: torch.Tensor,
        pool_cls: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Return the causal task coordinate used by financial Q.

        The global availability mask is the exact observed CSP identity.  The
        pool CLS vector is its state-dependent relational interpretation.  This
        route remains live in local-causal mode because availability is an MDP
        condition, not a peer-normalized market feature.
        """
        if not self.pool_task_conditioning_enabled:
            return None
        if pool_cls is None:
            raise RuntimeError(
                "Pool-task conditioning requires pool_cls from the whole-stock "
                "encoder"
            )
        if mask.ndim != 2 or mask.shape[-1] != self.max_coins:
            raise ValueError(
                "Pool-task availability mask must be [batch, max_coins], got "
                f"{tuple(mask.shape)}"
            )
        if pool_cls.ndim != 2 or pool_cls.shape[0] != mask.shape[0]:
            raise ValueError(
                "Pool-task CLS must be [batch, hidden] and align with mask, got "
                f"mask={tuple(mask.shape)} cls={tuple(pool_cls.shape)}"
            )
        return self.pool_task_norm(
            self.pool_task_mask_projection(mask.float())
            + self.pool_task_cls_projection(pool_cls)
        )

    def _combined_task_features(
        self,
        token_components: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        basket_context: Optional[torch.Tensor],
        basket_membership: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        legacy = self._project_task_features(
            basket_context, basket_membership
        )
        pool_task = self._project_pool_task_features(
            mask, token_components.get("pool_cls")
        )
        portfolio_context = token_components.get('portfolio_context')
        if portfolio_context is not None:
            pool_task = portfolio_context if pool_task is None else pool_task + portfolio_context
        if legacy is None:
            return pool_task
        if pool_task is None:
            return legacy
        return legacy + pool_task

    def _predictive_local_features(
        self,
        token_components: Dict[str, torch.Tensor],
        coin_features: torch.Tensor,
        mask: torch.Tensor,
        prev_action: torch.Tensor,
        actions: torch.Tensor,
        basket_context: torch.Tensor,
        basket_membership: torch.Tensor,
        *,
        actions_are_weights: bool = False,
        return_behavioral_basis: bool = False,
    ):
        """Return the downstream local-Q hidden state without scalar heads."""
        previous_assets = prev_action[:, 1:1 + self.max_coins]
        if actions_are_weights:
            traded_weights = actions.float().clamp(min=0.0)
            traded_weights = torch.cat(
                (
                    traded_weights[:, :1],
                    traded_weights[:, 1 : 1 + self.max_coins] * mask,
                ),
                dim=1,
            )
            traded_weights = traded_weights / traded_weights.sum(
                dim=1, keepdim=True
            ).clamp(min=1e-8)
        else:
            traded_weights = normalize_raw_actions_to_portfolio_weights(
                actions, mask
            )
        current_assets = traded_weights[:, 1:1 + self.max_coins]
        asset_delta = current_assets - previous_assets
        task_feature = self._combined_task_features(
            token_components,
            mask,
            basket_context,
            basket_membership,
        )
        fused_parts = [
            token_components["fused"],
            previous_assets.unsqueeze(-1),
            current_assets.unsqueeze(-1),
            asset_delta.unsqueeze(-1),
        ]
        if task_feature is not None:
            fused_parts.append(
                task_feature.unsqueeze(1).expand(-1, self.max_coins, -1)
            )
        fused_features = self.local_value_backbone(
            torch.cat(fused_parts, dim=-1)
        )
        allocation_residual, behavioral_basis, _ = (
            self._allocation_interaction_residual(
                token_components,
                previous_assets,
                current_assets,
                asset_delta,
                mask,
            )
        )
        if self.foundation_typed_control_enabled:
            typed_features, _ = self._typed_control_value_features(
                token_components,
                previous_assets,
                current_assets,
                asset_delta,
                mask,
                task_feature,
            )
            fused_scale = self.fused_value_max_scale * torch.sigmoid(
                self.fused_value_gate_logit
            )
            hidden = (
                typed_features
                + fused_scale * (fused_features + allocation_residual)
            ) * mask.unsqueeze(-1)
        else:
            hidden = (
                fused_features + allocation_residual
            ) * mask.unsqueeze(-1)
        if return_behavioral_basis:
            return hidden, behavioral_basis
        return hidden

    def foundation_action_outcomes_from_components(
        self,
        token_components: Dict[str, torch.Tensor],
        coin_features: torch.Tensor,
        mask: torch.Tensor,
        prev_action: torch.Tensor,
        candidate_actions: torch.Tensor,
        basket_context: torch.Tensor,
        basket_membership: torch.Tensor,
        *,
        return_behavioral_basis: bool = False,
        return_q_values: bool = False,
    ):
        """Evaluate K allocations after one shared state encoding."""
        batch, candidate_count, action_dim = candidate_actions.shape
        if action_dim != self.max_coins + 1:
            raise ValueError("foundation candidate action width mismatch")

        def _repeat(value):
            return value.unsqueeze(1).expand(
                -1, candidate_count, *value.shape[1:]
            ).reshape(batch * candidate_count, *value.shape[1:])

        # The action-conditioned path reads only the final fused state, the
        # behavioral allocation basis, and (for compatible value helpers) the
        # valid count.  Repeating local_temporal, episode_context, and
        # relational used to materialize diagnostic tensors for every
        # candidate even though no operation below consumed them.  With the
        # production 128 x 16 counterfactual batch that needless broadcast was
        # a material part of the CUDA high-water mark.
        required_component_names = {
            "fused",
            "valid_counts",
            *self.allocation_basis_component_names,
        }
        if 'portfolio_context' in token_components:
            required_component_names.add('portfolio_context')
        if self.pool_task_conditioning_enabled:
            required_component_names.add("pool_cls")
        if self.local_financial_handoff_enabled:
            required_component_names.add("local_temporal")
        repeated_components = {
            key: _repeat(token_components[key])
            for key in required_component_names
            if key in token_components
        }
        repeated_mask = _repeat(mask)
        repeated_previous = _repeat(prev_action)
        repeated_context = _repeat(basket_context)
        repeated_membership = _repeat(basket_membership)
        repeated_features = _repeat(coin_features)
        flat_candidates = candidate_actions.reshape(
            batch * candidate_count, action_dim
        )
        flat_candidate_weights = flat_candidates.float().clamp(min=0.0)
        flat_candidate_weights = torch.cat(
            (
                flat_candidate_weights[:, :1],
                flat_candidate_weights[:, 1 : 1 + self.max_coins]
                * repeated_mask,
            ),
            dim=1,
        )
        flat_candidate_weights = flat_candidate_weights / (
            flat_candidate_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        )
        local_result = self._predictive_local_features(
            repeated_components,
            repeated_features,
            repeated_mask,
            repeated_previous,
            flat_candidate_weights,
            repeated_context,
            repeated_membership,
            actions_are_weights=True,
            return_behavioral_basis=(
                return_behavioral_basis or return_q_values
            ),
        )
        if return_behavioral_basis or return_q_values:
            local_hidden, behavioral_basis = local_result
        else:
            local_hidden = local_result
            behavioral_basis = None
        traded_weights = (
            flat_candidate_weights[:, 1 : 1 + self.max_coins]
            * repeated_mask
        )
        weighted_hidden = (
            local_hidden * traded_weights.unsqueeze(-1)
        ).sum(dim=1)
        # Cash has zero market outcome; retain scale when an allocation is
        # partly cash instead of renormalizing the invested assets to one.
        outcomes = self.foundation_action_outcome_head(weighted_hidden).reshape(
            batch, candidate_count, 6
        )
        q_values = None
        if return_q_values:
            # The optional Foundation learner must use the same production
            # action-value composition as Bellman TD so the centered-advantage
            # contract cannot silently diverge in this older optional path.
            q_values = self.action_q_values_from_components(
                token_components,
                coin_features,
                mask,
                prev_action,
                candidate_actions,
                basket_context,
                basket_membership,
            )
        if return_behavioral_basis:
            basis = behavioral_basis.reshape(
                batch,
                candidate_count,
                self.max_coins,
                self.allocation_interaction_dim,
            )
            if return_q_values:
                return outcomes, q_values, basis
            return outcomes, basis
        if return_q_values:
            return outcomes, q_values
        return outcomes

    def action_q_values_from_components(
        self,
        token_components: Dict[str, torch.Tensor],
        coin_features: torch.Tensor,
        mask: torch.Tensor,
        prev_action: torch.Tensor,
        candidate_actions: torch.Tensor,
        basket_context: torch.Tensor,
        basket_membership: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate executable allocations on the live scalar-Q surface.

        Bellman TD, ProExp, and optional Foundation learning interrogate the
        same Q law.  This method expands only action/value composition after
        one state encoding, so no optional consumer gets a private or
        differently centered value surface.
        """
        batch, candidate_count, action_dim = candidate_actions.shape
        if action_dim != self.max_coins + 1:
            raise ValueError("candidate action width does not match critic")

        def _repeat(value: torch.Tensor) -> torch.Tensor:
            return value.unsqueeze(1).expand(
                -1, candidate_count, *value.shape[1:]
            ).reshape(batch * candidate_count, *value.shape[1:])

        required_component_names = {
            "fused",
            "valid_counts",
            *self.allocation_basis_component_names,
        }
        if 'portfolio_context' in token_components:
            required_component_names.add('portfolio_context')
        if self.pool_task_conditioning_enabled:
            required_component_names.add("pool_cls")
        if self.local_financial_handoff_enabled:
            if "local_financial_forecast" in token_components:
                required_component_names.update(
                    (
                        "local_financial_forecast",
                        "local_financial_local",
                        "local_financial_correction",
                    )
                )
            else:
                required_component_names.add("local_temporal")
        repeated_components = {
            name: _repeat(token_components[name])
            for name in required_component_names
        }
        state_value_override = None
        reference_score_override = None
        if self.centered_bellman_enabled:
            base_state_value = self._state_value_from_encoded_state(
                token_components,
                mask,
                prev_action,
                basket_context,
                basket_membership,
            )
            base_reference_score = (
                self._uncentered_action_score_from_encoded_state(
                    token_components,
                    coin_features,
                    mask,
                    prev_action,
                    prev_action,
                    basket_context=basket_context,
                    basket_membership=basket_membership,
                    actions_are_weights=True,
                )
            )
            state_value_override = _repeat(base_state_value)
            reference_score_override = _repeat(base_reference_score)
        flat_q = self._forward_from_encoded_state(
            repeated_components,
            _repeat(coin_features),
            _repeat(mask),
            _repeat(prev_action),
            candidate_actions.reshape(batch * candidate_count, action_dim),
            basket_context=_repeat(basket_context),
            basket_membership=_repeat(basket_membership),
            actions_are_weights=True,
            state_value_override=state_value_override,
            reference_action_score_override=reference_score_override,
        )
        return flat_q.reshape(batch, candidate_count)

    def foundation_q_values_from_components(
        self,
        token_components: Dict[str, torch.Tensor],
        coin_features: torch.Tensor,
        mask: torch.Tensor,
        prev_action: torch.Tensor,
        candidate_actions: torch.Tensor,
        basket_context: torch.Tensor,
        basket_membership: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility alias for the optional Foundation learner."""
        return self.action_q_values_from_components(
            token_components,
            coin_features,
            mask,
            prev_action,
            candidate_actions,
            basket_context,
            basket_membership,
        )

    def _encode_state_components(
        self,
        coin_features: torch.Tensor,
        asset_ids: torch.Tensor,
        mask: torch.Tensor,
        prev_action: torch.Tensor,
        prev_mask: torch.Tensor,
        basket_context: Optional[torch.Tensor] = None,
        basket_membership: Optional[torch.Tensor] = None,
        feature_history: Optional[torch.Tensor] = None,
        feature_history_mask: Optional[torch.Tensor] = None,
        *,
        portfolio_reward_history: Optional[torch.Tensor] = None,
        include_foundation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode the action-independent market state exactly once.

        ProExp evaluates several allocations for the same observation.  The
        FAM/AAM relation is deliberately independent of those allocations, so
        repeating this encoder for every candidate is mathematically redundant.
        Keeping the boundary explicit lets candidate actions share one state
        graph while their gradients still meet in the ordinary value heads.
        """
        del prev_action, prev_mask
        asset_identity = (
            self.asset_identity_embedding(asset_ids)
            if self.asset_identity_embedding is not None
            else None
        )
        components = self.token_encoder.encode_components(
            coin_features,
            mask,
            asset_identity=asset_identity,
            basket_context=basket_context if self.basket_context_dim > 0 else None,
            basket_membership=(
                basket_membership if self.basket_membership_dim > 0 else None
            ),
            feature_history=feature_history,
            feature_history_mask=feature_history_mask,
            control_only=True,
        )
        # Financial Q composition does not consume predictive routing traces,
        # the complete FAM map, gate maps, or intermediate snapshots. Keeping
        # those tensors in the returned dictionary extends their lifetime
        # across every value-head pass (especially costly for no-grad target
        # evaluation). Predictive learning calls token_encoder directly and
        # therefore retains its complete component contract.
        control_component_names = {
            "fused",
            "valid_counts",
            *self.allocation_basis_component_names,
        }
        if self.pool_task_conditioning_enabled:
            control_component_names.add("pool_cls")
        if self.local_financial_handoff_enabled:
            control_component_names.add("local_temporal")
        if include_foundation:
            control_component_names.update(
                {"local_temporal", "episode_context", "relational"}
            )
        selected_components = {
            name: components[name] for name in control_component_names
        }
        selected_components['portfolio_context'] = self.portfolio_history_encoder(
            portfolio_reward_history, mask)
        if self.financial_forecast_identity_enabled:
            selected_components["forecast_asset_identity"] = asset_identity
        if self.local_financial_handoff_enabled:
            forecast, local, correction = self._local_financial_forecast(
                selected_components
            )
            selected_components["local_financial_forecast"] = forecast
            selected_components["local_financial_local"] = local
            selected_components["local_financial_correction"] = correction
        return selected_components

    def _uncentered_action_score_from_encoded_state(
        self,
        token_components: Dict[str, torch.Tensor],
        coin_features: torch.Tensor,
        mask: torch.Tensor,
        prev_action: torch.Tensor,
        actions: torch.Tensor,
        return_component_outcomes: bool = False,
        return_financial_decomposition: bool = False,
        basket_context: Optional[torch.Tensor] = None,
        basket_membership: Optional[torch.Tensor] = None,
        *,
        actions_are_weights: bool = False,
        return_control_decomposition: bool = False,
    ):
        """Return the legacy action-conditioned score F before centering."""
        prev_cash = prev_action[:, 0:1]
        prev_assets = prev_action[:, 1:1 + self.max_coins]

        if actions_are_weights:
            traded_weights = actions.float().clamp(min=0.0)
            traded_weights = torch.cat(
                (
                    traded_weights[:, :1],
                    traded_weights[:, 1 : 1 + self.max_coins] * mask,
                ),
                dim=1,
            )
            traded_weights = traded_weights / traded_weights.sum(
                dim=1, keepdim=True
            ).clamp(min=1e-8)
        else:
            traded_weights = normalize_raw_actions_to_portfolio_weights(
                actions, mask
            )
        current_cash = traded_weights[:, 0:1]
        current_assets = traded_weights[:, 1:1 + self.max_coins]
        asset_delta = current_assets - prev_assets
        state_embeddings = token_components["fused"]
        valid_counts = token_components["valid_counts"]
        active_fraction = valid_counts / float(self.max_coins)

        component_return_predictions = None
        component_local_predictions = None
        component_context_corrections = None
        if self.local_financial_handoff_enabled:
            if "local_financial_forecast" in token_components:
                component_return_predictions = token_components[
                    "local_financial_forecast"
                ]
                component_local_predictions = token_components[
                    "local_financial_local"
                ]
                component_context_corrections = token_components[
                    "local_financial_correction"
                ]
            else:
                (
                    component_return_predictions,
                    component_local_predictions,
                    component_context_corrections,
                ) = self._local_financial_forecast(token_components)
        elif self.component_forecast_bridge_enabled or return_component_outcomes:
            component_return_features = self.component_outcome_backbone(
                state_embeddings
            )
            component_return_predictions = self.component_outcome_head(
                component_return_features
            )

        basket_feat = self._combined_task_features(
            token_components,
            mask,
            basket_context,
            basket_membership,
        )
        if basket_feat is not None:
            basket_feat_expanded = basket_feat.unsqueeze(1).expand(
                -1, self.max_coins, -1
            )
        else:
            basket_feat_expanded = None

        local_asset_action_input = torch.cat(
            [
                token_components["local_control"],
                prev_assets.unsqueeze(-1),
                current_assets.unsqueeze(-1),
                asset_delta.unsqueeze(-1),
            ],
            dim=-1,
        )
        fused_parts = [
            state_embeddings,
            prev_assets.unsqueeze(-1),
            current_assets.unsqueeze(-1),
            asset_delta.unsqueeze(-1),
        ]
        if basket_feat_expanded is not None:
            fused_parts.append(basket_feat_expanded)
        fused_value_input = torch.cat(fused_parts, dim=-1)
        fused_value_features = self.local_value_backbone(fused_value_input)
        allocation_residual, _, _ = self._allocation_interaction_residual(
            token_components,
            prev_assets,
            current_assets,
            asset_delta,
            mask,
        )
        if self.foundation_typed_control_enabled:
            typed_value_features, typed_parts = (
                self._typed_control_value_features(
                    token_components,
                    prev_assets,
                    current_assets,
                    asset_delta,
                    mask,
                    basket_feat,
                )
            )
            fused_scale = self.fused_value_max_scale * torch.sigmoid(
                self.fused_value_gate_logit
            )
            fused_value_residual = fused_scale * (
                fused_value_features + allocation_residual
            )
            base_local_value_features = (
                typed_value_features + fused_value_residual
            ) * mask.unsqueeze(-1)
        else:
            # Restore the established pre-Foundation value law: the complete
            # action-conditioned fused feature is the critic's live base, with
            # the allocation adapter remaining a corrective residual.  The
            # typed heads stay in the state dict for Foundation-compatible
            # checkpoints, but they neither steer Q nor receive TD gradients.
            expanded_mask = mask.unsqueeze(-1)
            inactive = torch.zeros_like(fused_value_features) * expanded_mask
            typed_parts = {
                "local": inactive,
                "relational": inactive,
                "cross": inactive,
            }
            fused_scale = fused_value_features.new_tensor(1.0)
            fused_value_residual = (
                fused_value_features + allocation_residual
            ) * expanded_mask
            base_local_value_features = fused_value_residual
        local_value_features = base_local_value_features
        if self.component_forecast_bridge_enabled:
            if component_return_predictions is None:
                raise RuntimeError(
                    "Forecast bridge enabled without component predictions"
                )
            local_value_features = (
                local_value_features
                + self._component_forecast_bridge_residual(
                    component_return_predictions,
                    asset_delta,
                    mask,
                )
            )
        uvfa_correction, _, _, _ = self._low_rank_uvfa_local_correction(
            local_asset_action_input,
            token_components["relational_control"],
            basket_context,
            mask,
        )
        local_asset_values = (
            self.asset_value_head(local_value_features).squeeze(-1)
            + uvfa_correction
        ) * mask
        pooled_asset_value = local_asset_values.sum(dim=1, keepdim=True) / valid_counts
        global_local_value_features = (
            base_local_value_features
            if self.component_forecast_bridge_local_only
            else local_value_features
        )
        pooled_local_value_features, _ = _masked_mean_pool(
            global_local_value_features,
            mask,
        )

        mean_asset_delta = (asset_delta * mask).sum(dim=1, keepdim=True) / valid_counts
        head_parts = [
            pooled_local_value_features,
            current_cash,
            prev_cash,
            mean_asset_delta,
            active_fraction,
        ]
        if self.global_head_uses_direct_task:
            if basket_feat is None:
                raise RuntimeError(
                    "Direct task-conditioned Q head did not receive a task feature"
                )
            head_parts.append(basket_feat)
        head_input = torch.cat(head_parts, dim=1)
        global_value = self.head(head_input)
        if self.basket_conditioned_allocation_enabled:
            marginal_scores = self.basket_opportunity_head(
                token_components["local_control"],
                token_components["relational_control"],
                state_embeddings, token_components["pool_cls"],
                mask, prev_action,
                portfolio_context=token_components.get('portfolio_context'),
            )
            marginal_value = (asset_delta * marginal_scores).sum(
                dim=1, keepdim=True
            )
        else:
            marginal_scores = torch.zeros_like(asset_delta)
            marginal_value = torch.zeros_like(global_value)
        residual_q_value = global_value + pooled_asset_value + marginal_value
        if self.local_financial_handoff_enabled:
            # The predictor is trained by real per-asset outcomes; scalar TD
            # owns the residual critic.  Normalized executed weights make
            # this term permutation invariant and match environment geometry.
            financial_base = (
                current_assets
                * component_return_predictions.detach().squeeze(-1)
                * mask
            ).sum(dim=1, keepdim=True)
            q_value = financial_base + residual_q_value
        else:
            financial_base = torch.zeros_like(residual_q_value)
            q_value = residual_q_value
        control_decomposition = {
            **typed_parts,
            "fused": fused_value_residual * mask.unsqueeze(-1),
            "asset": pooled_asset_value,
            "global": global_value,
            "basket_marginal": marginal_value,
            "basket_marginal_scores": marginal_scores,
            "task": (
                basket_feat
                if basket_feat is not None
                else global_value.new_zeros((global_value.shape[0], 0))
            ),
            "financial_base": financial_base,
            "bellman_residual": residual_q_value,
            "fused_scale": fused_scale,
        }
        if not return_component_outcomes:
            if return_control_decomposition:
                return q_value, control_decomposition
            return q_value
        if not self.component_outcome_aux_enabled:
            raise RuntimeError(
                "Component-outcome predictions requested from a critic whose "
                "auxiliary supervision is disabled."
            )
        # The unit-return head is supervised for every available asset,
        # including one with zero executed exposure.  Under the local-financial
        # handoff its detached forecast also supplies an explicit Q base; the
        # real Bellman target remains the sole authority over total Q.
        if component_return_predictions is None:
            raise RuntimeError(
                "Component predictions requested but were not constructed"
            )
        if self.component_outcome_unit_return_targets:
            component_predictions = (
                component_return_predictions * mask.unsqueeze(-1)
            )
        else:
            component_predictions = (
                component_return_predictions
                * current_assets.unsqueeze(-1)
                * mask.unsqueeze(-1)
            )
        if return_control_decomposition:
            return q_value, component_predictions, control_decomposition
        if return_financial_decomposition:
            if not self.local_financial_handoff_enabled:
                raise RuntimeError("Financial decomposition requires local-financial handoff")
            return (
                q_value,
                component_predictions,
                component_local_predictions * mask.unsqueeze(-1),
                component_context_corrections * mask.unsqueeze(-1),
                financial_base,
                residual_q_value,
            )
        return q_value, component_predictions

    def _state_value_from_encoded_state(
        self,
        token_components: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        prev_action: torch.Tensor,
        basket_context: Optional[torch.Tensor],
        basket_membership: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the action-independent task baseline V(s, m, w_prev)."""
        if not self.centered_advantage_enabled or self.state_value_head is None:
            raise RuntimeError("Centered critic baseline is not enabled")
        state_embeddings = token_components["fused"]
        valid_counts = token_components["valid_counts"]
        active_fraction = valid_counts / float(self.max_coins)
        prev_cash = prev_action[:, :1]
        prev_assets = prev_action[:, 1 : 1 + self.max_coins] * mask
        pooled_market, _ = _masked_mean_pool(state_embeddings, mask)
        held_market = (
            state_embeddings * prev_assets.unsqueeze(-1)
        ).sum(dim=1)
        task_feature = self._combined_task_features(
            token_components,
            mask,
            basket_context,
            basket_membership,
        )
        baseline_parts = [
            pooled_market,
            held_market,
            prev_cash,
            active_fraction,
        ]
        if self.task_feature_enabled:
            if task_feature is None:
                raise RuntimeError(
                    "Centered critic baseline did not receive its task feature"
                )
            baseline_parts.append(task_feature)
        return self.state_value_head(torch.cat(baseline_parts, dim=1))

    def _forward_from_encoded_state(
        self,
        token_components: Dict[str, torch.Tensor],
        coin_features: torch.Tensor,
        mask: torch.Tensor,
        prev_action: torch.Tensor,
        actions: torch.Tensor,
        return_component_outcomes: bool = False,
        return_financial_decomposition: bool = False,
        basket_context: Optional[torch.Tensor] = None,
        basket_membership: Optional[torch.Tensor] = None,
        *,
        actions_are_weights: bool = False,
        return_control_decomposition: bool = False,
        state_value_override: Optional[torch.Tensor] = None,
        reference_action_score_override: Optional[torch.Tensor] = None,
    ):
        """Train direct Q and derive its reference-centered allocation view.

        In direct_q mode the reference baseline is this same Q evaluated at
        the held book, so V_ref + A_ref is exactly candidate Q. Ordinary TD
        receives the candidate scorer's full parameter gradient, including
        when the recorded allocation equals the held book. Reference views
        are evaluated only for requested control diagnostics. separate_v
        retains the historical independent V + F(candidate) - F(reference).
        """
        candidate_result = self._uncentered_action_score_from_encoded_state(
            token_components,
            coin_features,
            mask,
            prev_action,
            actions,
            return_component_outcomes=return_component_outcomes,
            return_financial_decomposition=return_financial_decomposition,
            basket_context=basket_context,
            basket_membership=basket_membership,
            actions_are_weights=actions_are_weights,
            return_control_decomposition=return_control_decomposition,
        )
        if not self.centered_bellman_enabled:
            if return_control_decomposition and self.centered_advantage_enabled:
                # A derived decomposition, never a second Bellman fit. Its
                # sum equals candidate_result[0] exactly and the ordinary
                # forward/financial paths need no extra reference pass.
                reference_result = self._uncentered_action_score_from_encoded_state(
                    token_components, coin_features, mask, prev_action, prev_action,
                    basket_context=basket_context,
                    basket_membership=basket_membership,
                    actions_are_weights=True,
                    return_control_decomposition=True,
                )
                candidate_score = candidate_result[0]
                reference_score = reference_result[0]
                decomposition = dict(candidate_result[-1])
                decomposition.update(
                    state_value=reference_score,
                    candidate_action_score=candidate_score,
                    reference_action_score=reference_score,
                    centered_advantage=candidate_score - reference_score,
                    reference_global=reference_result[-1]["global"],
                    reference_asset=reference_result[-1]["asset"],
                )
                if return_component_outcomes:
                    return candidate_score, candidate_result[1], decomposition
                return candidate_score, decomposition
            return candidate_result

        candidate_score = (
            candidate_result[0]
            if isinstance(candidate_result, tuple)
            else candidate_result
        )
        if state_value_override is None:
            state_value = self._state_value_from_encoded_state(
                token_components,
                mask,
                prev_action,
                basket_context,
                basket_membership,
            )
        else:
            state_value = state_value_override

        reference_result = None
        if reference_action_score_override is None:
            reference_result = self._uncentered_action_score_from_encoded_state(
                token_components,
                coin_features,
                mask,
                prev_action,
                prev_action,
                return_component_outcomes=return_financial_decomposition,
                return_financial_decomposition=return_financial_decomposition,
                basket_context=basket_context,
                basket_membership=basket_membership,
                actions_are_weights=True,
                return_control_decomposition=return_control_decomposition,
            )
            reference_score = (
                reference_result[0]
                if isinstance(reference_result, tuple)
                else reference_result
            )
        else:
            reference_score = reference_action_score_override

        advantage = candidate_score - reference_score
        q_value = state_value + advantage

        if return_control_decomposition:
            candidate_decomposition = candidate_result[-1]
            reference_decomposition = reference_result[-1]
            decomposition = dict(candidate_decomposition)
            decomposition.update(
                state_value=state_value,
                candidate_action_score=candidate_score,
                reference_action_score=reference_score,
                centered_advantage=advantage,
                reference_global=reference_decomposition["global"],
                reference_asset=reference_decomposition["asset"],
            )
            if return_component_outcomes:
                return q_value, candidate_result[1], decomposition
            return q_value, decomposition

        if return_financial_decomposition:
            if not return_component_outcomes or reference_result is None:
                raise RuntimeError(
                    "Financial decomposition requires component outcomes and "
                    "an explicit reference evaluation"
                )
            centered_financial_base = (
                candidate_result[4] - reference_result[4]
            )
            centered_bellman_residual = (
                state_value + candidate_result[5] - reference_result[5]
            )
            return (
                q_value,
                candidate_result[1],
                candidate_result[2],
                candidate_result[3],
                centered_financial_base,
                centered_bellman_residual,
            )
        if return_component_outcomes:
            return q_value, candidate_result[1]
        return q_value

    def forward_with_control_decomposition(
        self,
        coin_features: torch.Tensor,
        asset_ids: torch.Tensor,
        mask: torch.Tensor,
        prev_action: torch.Tensor,
        prev_mask: torch.Tensor,
        actions: torch.Tensor,
        basket_context: Optional[torch.Tensor] = None,
        basket_membership: Optional[torch.Tensor] = None,
        feature_history: Optional[torch.Tensor] = None,
        feature_history_mask: Optional[torch.Tensor] = None,
        portfolio_reward_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        token_components = self._encode_state_components(
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context=basket_context,
            basket_membership=basket_membership,
            feature_history=feature_history,
            feature_history_mask=feature_history_mask,
            portfolio_reward_history=portfolio_reward_history,
        )
        return self._forward_from_encoded_state(
            token_components,
            coin_features,
            mask,
            prev_action,
            actions,
            basket_context=basket_context,
            basket_membership=basket_membership,
            return_control_decomposition=True,
        )

    def forward(
        self,
        coin_features: torch.Tensor,
        asset_ids: torch.Tensor,
        mask: torch.Tensor,
        prev_action: torch.Tensor,
        prev_mask: torch.Tensor,
        actions: torch.Tensor,
        return_component_outcomes: bool = False,
        return_financial_decomposition: bool = False,
        basket_context: Optional[torch.Tensor] = None,
        basket_membership: Optional[torch.Tensor] = None,
        feature_history: Optional[torch.Tensor] = None,
        feature_history_mask: Optional[torch.Tensor] = None,
        portfolio_reward_history: Optional[torch.Tensor] = None,
    ):
        token_components = self._encode_state_components(
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context=basket_context,
            basket_membership=basket_membership,
            feature_history=feature_history,
            feature_history_mask=feature_history_mask,
            portfolio_reward_history=portfolio_reward_history,
        )
        return self._forward_from_encoded_state(
            token_components,
            coin_features,
            mask,
            prev_action,
            actions,
            return_component_outcomes=return_component_outcomes,
            return_financial_decomposition=return_financial_decomposition,
            basket_context=basket_context,
            basket_membership=basket_membership,
        )


class DeepSetCritic(nn.Module):
    """
    Permutation-invariant critic over (coin_state_i, action_i) pairs.

    The critic consumes raw dict observations so it can use prev_action/prev_mask,
    which the baseline MLP path already had access to via CombinedExtractor.
    """

    expects_precomputed_features = False

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        max_coins: int,
        features_per_coin: int,
        set_hidden_dim: int,
        net_arch: List[int],
        n_heads: int,
        n_layers: int,
        dropout: float,
        use_market_attention: bool,
        use_global_pooling: bool,
        n_critics: int = 2,
        local_feature_dim: Optional[int] = None,
        position_feature_dim: Optional[int] = None,
        context_feature_dim: Optional[int] = None,
        comparative_feature_dim: Optional[int] = None,
        uvfa_local_interaction_enabled: bool = False,
        uvfa_local_interaction_rank: int = 8,
        uvfa_local_interaction_max_abs: float = 0.10,
        component_outcome_aux_enabled: bool = False,
        component_outcome_dim: int = 1,
        component_outcome_unit_return_targets: bool = False,
        component_forecast_bridge_enabled: bool = False,
        component_forecast_bridge_return_reference: float = 0.01,
        component_forecast_bridge_max_abs: float = 0.10,
        component_forecast_bridge_local_only: bool = False,
        num_asset_ids: Optional[int] = None,
        asset_identity_dim: int = 16,
        basket_context_dim: int = 0,
        basket_membership_dim: int = 0,
        predictive_moe_enabled: bool = False,
        predictive_moe_num_experts: int = 4,
        predictive_moe_expert_hidden_dim: int = 64,
        predictive_moe_router_temperature: float = 1.0,
        predictive_moe_max_residual_abs: float = 0.10,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.max_coins = max_coins
        self.features_per_coin = features_per_coin
        self.n_critics = n_critics
        self.require_portfolio_reward_history = 'portfolio_reward_history' in observation_space.spaces
        resolved_num_asset_ids = int(
            max_coins if num_asset_ids is None else num_asset_ids
        )
        self.asset_padding_id = resolved_num_asset_ids
        self.basket_context_dim = int(basket_context_dim)
        if self.basket_context_dim < 0:
            raise ValueError("basket_context_dim must be non-negative")
        self.basket_membership_dim = int(basket_membership_dim)
        if self.basket_membership_dim < 0:
            raise ValueError("basket_membership_dim must be non-negative")

        self.q_networks = nn.ModuleList(
            [
                DeepSetQNetwork(
                    max_coins=max_coins,
                    features_per_coin=features_per_coin,
                    set_hidden_dim=set_hidden_dim,
                    net_arch=net_arch,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    dropout=dropout,
                    use_market_attention=use_market_attention,
                    use_global_pooling=use_global_pooling,
                    local_feature_dim=local_feature_dim,
                    position_feature_dim=position_feature_dim,
                    context_feature_dim=context_feature_dim,
                    comparative_feature_dim=comparative_feature_dim,
                    uvfa_local_interaction_enabled=(
                        uvfa_local_interaction_enabled
                    ),
                    uvfa_local_interaction_rank=uvfa_local_interaction_rank,
                    uvfa_local_interaction_max_abs=(
                        uvfa_local_interaction_max_abs
                    ),
                    component_outcome_aux_enabled=(
                        component_outcome_aux_enabled
                    ),
                    component_outcome_dim=component_outcome_dim,
                    component_outcome_unit_return_targets=(
                        component_outcome_unit_return_targets
                    ),
                    component_forecast_bridge_enabled=(
                        component_forecast_bridge_enabled
                    ),
                    component_forecast_bridge_return_reference=(
                        component_forecast_bridge_return_reference
                    ),
                    component_forecast_bridge_max_abs=(
                        component_forecast_bridge_max_abs
                    ),
                    component_forecast_bridge_local_only=(
                        component_forecast_bridge_local_only
                    ),
                    num_asset_ids=resolved_num_asset_ids,
                    asset_identity_dim=asset_identity_dim,
                    basket_context_dim=self.basket_context_dim,
                    basket_membership_dim=self.basket_membership_dim,
                    predictive_moe_enabled=predictive_moe_enabled,
                    predictive_moe_num_experts=predictive_moe_num_experts,
                    predictive_moe_expert_hidden_dim=(
                        predictive_moe_expert_hidden_dim
                    ),
                    predictive_moe_router_temperature=(
                        predictive_moe_router_temperature
                    ),
                    predictive_moe_max_residual_abs=(
                        predictive_moe_max_residual_abs
                    ),
                )
                for _ in range(n_critics)
            ]
        )

    def _parse_observation(
        self, observations: Dict[str, torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not isinstance(observations, dict):
            raise ValueError(
                "DeepSetCritic expects dict observations with "
                "observations/asset_ids/mask/prev_action/prev_mask "
                "(and basket_context / basket_membership when enabled)."
            )

        if self.require_portfolio_reward_history and 'portfolio_reward_history' not in observations:
            raise ValueError('Critic requires portfolio reward history from the current observation contract')
        obs = observations["observations"].float()
        asset_ids = observations.get("asset_ids")
        mask = observations["mask"].float()
        prev_action = observations.get("prev_action")
        prev_mask = observations.get("prev_mask")
        basket_context = observations.get("basket_context")
        basket_membership = observations.get("basket_membership")
        if "feature_history" in observations:
            raise RuntimeError(
                "Foundation schema v2 rejects legacy feature_history. "
                "Start a fresh run with local_feature_history and "
                "episode_feature_history."
            )
        local_feature_history = observations.get("local_feature_history")
        episode_feature_history = observations.get("episode_feature_history")
        feature_history = None
        feature_history_mask = observations.get("feature_history_mask")

        batch_size = obs.shape[0]
        expected_width = self.max_coins * self.features_per_coin
        if obs.ndim != 2 or obs.shape[1] != expected_width:
            raise ValueError(
                "DeepSetCritic observation width mismatch: expected "
                f"{self.max_coins} × {self.features_per_coin} = "
                f"{expected_width}, received shape {tuple(obs.shape)}."
            )
        coin_features = obs.view(batch_size, self.max_coins, self.features_per_coin)
        encoder = self.q_networks[0].token_encoder
        if encoder.local_causal_inputs_enabled:
            coin_features = local_market_token_view(
                coin_features, encoder.local_feature_dim, encoder.position_feature_dim,
                encoder.comparative_feature_dim,
            )
            basket_context = None if basket_context is None else torch.zeros_like(basket_context)
            basket_membership = None if basket_membership is None else torch.zeros_like(basket_membership)
            if episode_feature_history is not None:
                episode_feature_history = torch.zeros_like(episode_feature_history)

        if asset_ids is None:
            asset_ids = torch.arange(
                self.max_coins, device=obs.device
            ).unsqueeze(0).expand(batch_size, -1)
            padding_fill = torch.full_like(asset_ids, self.asset_padding_id)
            asset_ids = torch.where(mask > 0, asset_ids, padding_fill)
        else:
            asset_ids = asset_ids.long()
            if asset_ids.dim() == 1:
                asset_ids = asset_ids.unsqueeze(0)
            asset_ids = asset_ids.clamp(min=0, max=self.asset_padding_id)

        if prev_action is None:
            prev_action = obs.new_zeros((batch_size, self.max_coins + 1))
        else:
            prev_action = prev_action.float()

        if prev_mask is None:
            prev_mask = mask
        else:
            prev_mask = prev_mask.float()

        if self.basket_context_dim > 0:
            if basket_context is None:
                basket_context = obs.new_zeros((batch_size, self.basket_context_dim))
            else:
                basket_context = basket_context.float()
                if basket_context.dim() == 1:
                    basket_context = basket_context.unsqueeze(0)
                if basket_context.shape[-1] != self.basket_context_dim:
                    raise ValueError(
                        "DeepSetCritic basket_context width mismatch: expected "
                        f"{self.basket_context_dim}, received {basket_context.shape[-1]}."
                    )
        else:
            basket_context = obs.new_zeros((batch_size, 0))

        if self.basket_membership_dim > 0:
            if basket_membership is None:
                basket_membership = obs.new_zeros(
                    (batch_size, self.basket_membership_dim)
                )
            else:
                basket_membership = basket_membership.float()
                if basket_membership.dim() == 1:
                    basket_membership = basket_membership.unsqueeze(0)
                if basket_membership.shape[-1] != self.basket_membership_dim:
                    raise ValueError(
                        "DeepSetCritic basket_membership width mismatch: expected "
                        f"{self.basket_membership_dim}, received "
                        f"{basket_membership.shape[-1]}."
                    )
        else:
            basket_membership = obs.new_zeros((batch_size, 0))

        token_encoder = self.q_networks[0].token_encoder
        if token_encoder.fam_enabled:
            history_length = token_encoder.fam_history_length
            expected_local_shape = (
                batch_size,
                self.max_coins,
                history_length,
                token_encoder.fam_local_feature_dim,
            )
            expected_episode_shape = (
                batch_size,
                self.max_coins,
                history_length,
                token_encoder.fam_episode_feature_dim,
            )
            if local_feature_history is None:
                local_feature_history = obs.new_zeros(expected_local_shape)
            else:
                local_feature_history = local_feature_history.float()
                if tuple(local_feature_history.shape) != expected_local_shape:
                    raise ValueError(
                        "DeepSetCritic local_feature_history shape mismatch: expected "
                        f"{expected_local_shape}, received "
                        f"{tuple(local_feature_history.shape)}."
                    )
            if episode_feature_history is None:
                episode_feature_history = obs.new_zeros(expected_episode_shape)
            else:
                episode_feature_history = episode_feature_history.float()
                if tuple(episode_feature_history.shape) != expected_episode_shape:
                    raise ValueError(
                        "DeepSetCritic episode_feature_history shape mismatch: expected "
                        f"{expected_episode_shape}, received "
                        f"{tuple(episode_feature_history.shape)}."
                    )
            feature_history = torch.cat(
                (local_feature_history, episode_feature_history), dim=-1
            )
            expected_history_mask_shape = (batch_size, history_length)
            if feature_history_mask is None:
                feature_history_mask = obs.new_zeros(expected_history_mask_shape)
            else:
                feature_history_mask = feature_history_mask.float()
                if tuple(feature_history_mask.shape) != expected_history_mask_shape:
                    raise ValueError(
                        "DeepSetCritic feature_history_mask shape mismatch: "
                        f"expected {expected_history_mask_shape}, received "
                        f"{tuple(feature_history_mask.shape)}."
                    )
        else:
            feature_history = obs.new_zeros((batch_size, self.max_coins, 0, 0))
            feature_history_mask = obs.new_zeros((batch_size, 0))

        return (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        )

    def earnmore_masked_reconstruction_loss(
        self,
        observations: Dict[str, torch.Tensor],
        *,
        synthetic_mask: Optional[torch.Tensor] = None,
        reconstruction_mask: Optional[torch.Tensor] = None,
        prepared_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train both twin-critic encoders on the same maskable task."""
        if prepared_inputs is None:
            (
                coin_features,
                asset_ids,
                mask,
                _,
                _,
                basket_context,
                basket_membership,
                feature_history,
                feature_history_mask,
            ) = self._parse_observation(observations)
            pool_history_valid = observations.get("pool_history_valid")
        else:
            coin_features = prepared_inputs["coin_features"]
            asset_ids = prepared_inputs["asset_ids"]
            mask = prepared_inputs["mask"]
            basket_context = prepared_inputs["basket_context"]
            basket_membership = prepared_inputs["basket_membership"]
            feature_history = prepared_inputs["feature_history"]
            feature_history_mask = prepared_inputs["feature_history_mask"]
            pool_history_valid = prepared_inputs.get("pool_history_valid")
        losses = [
            q_network.token_encoder.earnmore_masked_reconstruction_loss(
                coin_features,
                mask,
                asset_identity=(q_network.asset_identity_embedding(asset_ids).detach()
                                if q_network.asset_identity_embedding is not None else None),
                basket_context=(
                    basket_context if self.basket_context_dim > 0 else None
                ),
                basket_membership=(
                    basket_membership
                    if self.basket_membership_dim > 0
                    else None
                ),
                feature_history=feature_history,
                feature_history_mask=feature_history_mask,
                synthetic_mask=synthetic_mask,
                reconstruction_mask=reconstruction_mask,
                pool_history_valid=pool_history_valid,
            )
            for q_network in self.q_networks
        ]
        return {
            "loss": torch.stack([stats["loss"] for stats in losses]).mean(),
            "masked_asset_fraction": torch.stack(
                [stats["masked_asset_fraction"] for stats in losses]
            ).mean(),
            "sampled_mask_ratio": torch.stack(
                [stats["sampled_mask_ratio"] for stats in losses]
            ).mean(),
            "masked_local_coordinate_fraction": torch.stack(
                [stats["masked_local_coordinate_fraction"] for stats in losses]
            ).mean(),
            "masked_assets_per_sample": torch.stack(
                [stats["masked_assets_per_sample"] for stats in losses]
            ).mean(),
            "visible_assets_per_sample": torch.stack(
                [stats["visible_assets_per_sample"] for stats in losses]
            ).mean(),
        }

    def foundation_loss(
        self,
        observations: Dict[str, torch.Tensor],
        future_unit_returns: torch.Tensor,
        future_valid: torch.Tensor,
        candidate_actions: torch.Tensor,
        candidate_targets: torch.Tensor,
        peer_observations: Optional[Dict[str, torch.Tensor]] = None,
        *,
        subsample_candidates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Train both live critics without redefining their scalar Q heads."""
        (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        ) = self._parse_observation(observations)
        peer_parsed = (
            self._parse_observation(peer_observations)
            if peer_observations is not None
            else None
        )
        network_losses = []
        representation_losses = []
        action_losses = []
        q_rank_losses = []
        predictions = []
        evaluated_candidate_counts = []
        for network_index, q_network in enumerate(self.q_networks):
            network_feature_history = feature_history.clone()
            local_width = q_network.token_encoder.fam_local_feature_dim
            local_coordinate = torch.arange(
                network_feature_history[..., :local_width].numel(),
                device=network_feature_history.device,
            ).reshape_as(network_feature_history[..., :local_width])
            private_mask = (
                (local_coordinate + 3 * network_index) % 11 == 0
            )
            network_feature_history[..., :local_width] = (
                network_feature_history[..., :local_width].masked_fill(
                    private_mask, 0.0
                )
            )
            components = q_network._encode_state_components(
                coin_features,
                asset_ids,
                mask,
                prev_action,
                prev_mask,
                basket_context,
                basket_membership,
                network_feature_history,
                feature_history_mask,
                include_foundation=True,
                portfolio_reward_history=observations.get("portfolio_reward_history"),
            )
            peer_local_temporal = None
            shared_peer_mask = None
            if peer_parsed is not None:
                peer_components = q_network._encode_state_components(
                    peer_parsed[0],
                    peer_parsed[1],
                    peer_parsed[2],
                    peer_parsed[3],
                    peer_parsed[4],
                    peer_parsed[5],
                    peer_parsed[6],
                    peer_parsed[7],
                    peer_parsed[8],
                    include_foundation=True,
                    portfolio_reward_history=peer_observations.get("portfolio_reward_history"),
                )
                correspondence = (
                    asset_ids.unsqueeze(2) == peer_parsed[1].unsqueeze(1)
                ) & peer_parsed[2].bool().unsqueeze(1)
                peer_local_temporal = torch.einsum(
                    "bij,bjd->bid",
                    correspondence.to(
                        peer_components["local_temporal"].dtype
                    ),
                    peer_components["local_temporal"],
                )
                shared_peer_mask = correspondence.any(dim=2) & mask.bool()
            representation = q_network.token_encoder.foundation_representation_loss(
                components,
                torch.cat(
                    (
                        observations.get(
                            "foundation_local_history_target",
                            feature_history[
                                ..., : q_network.token_encoder.fam_local_feature_dim
                            ],
                        ).float(),
                        feature_history[
                            ..., q_network.token_encoder.fam_local_feature_dim :
                        ],
                    ),
                    dim=-1,
                ),
                mask,
                future_unit_returns,
                future_valid,
                peer_local_temporal=peer_local_temporal,
                shared_peer_mask=shared_peer_mask,
                local_reconstruction_mask=observations.get(
                    "foundation_local_reconstruction_mask"
                ),
            )
            all_candidate_indices = torch.arange(
                candidate_actions.shape[1], device=candidate_actions.device
            )
            if subsample_candidates:
                # The loss has always assigned alternating candidates to the
                # twins.  Previously each critic evaluated all sixteen and
                # discarded half only after the forward pass.  Slice first so
                # Q1 and Q2 retain the exact same independent supervision while
                # the action-conditioned work is cut in half.
                candidate_indices = all_candidate_indices[
                    (all_candidate_indices + network_index)
                    % len(self.q_networks)
                    == 0
                ]
            else:
                candidate_indices = all_candidate_indices
            network_candidate_actions = candidate_actions.index_select(
                1, candidate_indices
            )
            network_candidate_targets = candidate_targets.index_select(
                1, candidate_indices
            )
            prediction, candidate_q = (
                q_network.foundation_action_outcomes_from_components(
                    components,
                    coin_features,
                    mask,
                    prev_action,
                    network_candidate_actions,
                    basket_context,
                    basket_membership,
                    return_q_values=True,
                )
            )
            # Independent deterministic bootstrap masks preserve twin diversity
            # while keeping tests and resumed runs reproducible.
            row_index = torch.arange(
                prediction.shape[0], device=prediction.device
            )
            bootstrap = ((row_index + network_index) % 5) != 0
            action_error = F.smooth_l1_loss(
                prediction, network_candidate_targets, reduction="none"
            )
            if not subsample_candidates:
                candidate_subset = (
                    (all_candidate_indices + network_index)
                    % len(self.q_networks)
                    == 0
                )
                action_error = action_error[:, candidate_subset]
            action_error = action_error.mean(dim=(1, 2))
            action_loss = (
                action_error[bootstrap].mean()
                if bootstrap.any()
                else action_error.mean()
            )
            utility = network_candidate_targets[..., 3] - float(
                getattr(config, "RISK_ADJUSTMENT_BETA", 2.0)
            ) * torch.clamp(
                network_candidate_targets[..., 5]
                - float(
                    getattr(config, "DOWNSIDE_DEVIATION_TARGET", 0.0)
                )
                * float(getattr(config, "FINANCIAL_REWARD_SCALE", 10.0)),
                min=0.0,
            )
            if subsample_candidates:
                ranking_q = candidate_q
                ranking_utility = utility
            else:
                owned_candidates = (
                    (all_candidate_indices + network_index)
                    % len(self.q_networks)
                    == 0
                )
                ranking_q = candidate_q[:, owned_candidates]
                ranking_utility = utility[:, owned_candidates]
            left, right = torch.triu_indices(
                ranking_q.shape[1],
                ranking_q.shape[1],
                offset=1,
                device=ranking_q.device,
            )
            utility_difference = (
                ranking_utility[:, left] - ranking_utility[:, right]
            )
            informative = utility_difference.abs() > 1e-7
            utility_scale = ranking_utility.std(
                dim=1, keepdim=True, unbiased=False
            ).clamp(min=1e-6)
            standardized_utility = (
                ranking_utility - ranking_utility.mean(dim=1, keepdim=True)
            ) / utility_scale
            q_rank_margin = float(
                getattr(config, "FOUNDATION_Q_RANK_MARGIN", 0.05)
            )
            target_q_difference = q_rank_margin * torch.clamp(
                standardized_utility[:, left]
                - standardized_utility[:, right],
                min=-2.0,
                max=2.0,
            )
            ranking_error = F.smooth_l1_loss(
                ranking_q[:, left] - ranking_q[:, right],
                target_q_difference,
                reduction="none",
                beta=q_rank_margin,
            )
            ranking_valid = informative & bootstrap.unsqueeze(1)
            q_rank_loss = (
                ranking_error[ranking_valid].mean()
                if ranking_valid.any()
                else ranking_q.sum() * 0.0
            )
            q_rank_weight = float(
                getattr(config, "FOUNDATION_Q_RANK_LOSS_WEIGHT", 1.0)
            )
            network_loss = (
                representation["loss"]
                + action_loss
                + q_rank_weight * q_rank_loss
            ) / (2.0 + q_rank_weight)
            network_losses.append(network_loss)
            representation_losses.append(representation["loss"])
            action_losses.append(action_loss)
            q_rank_losses.append(q_rank_loss)
            evaluated_candidate_counts.append(int(prediction.shape[1]))
            if subsample_candidates:
                # Predictions are a diagnostic output, not part of the loss.
                # Preserve the established [twin, batch, 16, outcome] contract
                # and mark candidates not owned by this twin as NaN rather
                # than fabricating a value or keeping their graphs alive.
                diagnostic_prediction = prediction.detach().new_full(
                    (
                        prediction.shape[0],
                        candidate_actions.shape[1],
                        prediction.shape[2],
                    ),
                    torch.nan,
                )
                diagnostic_prediction[:, candidate_indices] = (
                    prediction.detach()
                )
                predictions.append(diagnostic_prediction)
            else:
                predictions.append(prediction)
        prediction_tensor = torch.stack(predictions)
        if subsample_candidates:
            twin_disagreement = prediction_tensor.new_zeros(())
        else:
            twin_disagreement = (
                (predictions[0] - predictions[1]).abs().mean()
                if len(predictions) > 1
                else predictions[0].sum() * 0.0
            )
        return {
            "loss": torch.stack(network_losses).mean(),
            "representation_loss": torch.stack(representation_losses).mean(),
            "action_loss": torch.stack(action_losses).mean(),
            "q_rank_loss": torch.stack(q_rank_losses).mean(),
            "prediction": prediction_tensor,
            "twin_disagreement": twin_disagreement,
            "evaluated_candidates_per_network": tuple(
                evaluated_candidate_counts
            ),
            "candidate_evaluation_fraction": prediction_tensor.new_tensor(
                float(sum(evaluated_candidate_counts))
                / float(
                    max(
                        len(self.q_networks) * candidate_actions.shape[1],
                        1,
                    )
                )
            ),
        }

    def forward(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        ) = self._parse_observation(observations)
        actions = actions.float()
        return tuple(
            q_net(
                coin_features,
                asset_ids,
                mask,
                prev_action,
                prev_mask,
                actions,
                basket_context=basket_context,
                basket_membership=basket_membership,
                feature_history=feature_history,
                feature_history_mask=feature_history_mask,
                portfolio_reward_history=observations.get("portfolio_reward_history"),
            )
            for q_net in self.q_networks
        )

    def forward_with_control_decomposition(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[
        Tuple[torch.Tensor, ...],
        Tuple[Dict[str, torch.Tensor], ...],
    ]:
        """Expose how each independent twin composes typed value evidence."""
        (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        ) = self._parse_observation(observations)
        outputs = tuple(
            q_net.forward_with_control_decomposition(
                coin_features,
                asset_ids,
                mask,
                prev_action,
                prev_mask,
                actions.float(),
                basket_context=basket_context,
                basket_membership=basket_membership,
                feature_history=feature_history,
                feature_history_mask=feature_history_mask,
                portfolio_reward_history=observations.get("portfolio_reward_history"),
            )
            for q_net in self.q_networks
        )
        return (
            tuple(output[0] for output in outputs),
            tuple(output[1] for output in outputs),
        )

    @staticmethod
    def _repeat_encoded_batch(
        tensor: torch.Tensor,
        repeats: int,
    ) -> torch.Tensor:
        """Repeat only an encoded batch axis for candidate value composition."""
        if repeats <= 1:
            return tensor
        batch_size = tensor.shape[0]
        return tensor.unsqueeze(1).expand(
            batch_size,
            repeats,
            *tensor.shape[1:],
        ).reshape(batch_size * repeats, *tensor.shape[1:])

    def evaluate_action_candidates(
        self,
        observations: Dict[str, torch.Tensor],
        candidate_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Evaluate ``[batch, candidates, action]`` with one state encode/Q.

        Every Q network retains its own parameters and therefore performs its
        own state encode.  Within one Q network, however, all candidate actions
        share the exact action-independent FAM/AAM state.  Expanding the
        encoded tensors gives the same value-head outputs and summed encoder
        gradients as repeating the raw observation, without recomputing the
        temporal and cross-asset representation for each candidate.
        """
        if candidate_actions.dim() != 3:
            raise ValueError(
                "critic candidate actions must be [batch, candidates, action]"
            )
        parsed = self._parse_observation(observations)
        (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        ) = parsed
        batch_size, candidate_count, action_dim = candidate_actions.shape
        if batch_size != coin_features.shape[0]:
            raise ValueError(
                "critic candidate batch does not match observation batch"
            )
        flat_actions = candidate_actions.float().reshape(
            batch_size * candidate_count,
            action_dim,
        )

        repeated_coin_features = self._repeat_encoded_batch(
            coin_features, candidate_count
        )
        repeated_mask = self._repeat_encoded_batch(mask, candidate_count)
        repeated_prev_action = self._repeat_encoded_batch(
            prev_action, candidate_count
        )
        repeated_basket_context = self._repeat_encoded_batch(
            basket_context, candidate_count
        )
        repeated_basket_membership = self._repeat_encoded_batch(
            basket_membership, candidate_count
        )

        q_values = []
        for q_net in self.q_networks:
            components = q_net._encode_state_components(
                coin_features,
                asset_ids,
                mask,
                prev_action,
                prev_mask,
                basket_context=basket_context,
                basket_membership=basket_membership,
                feature_history=feature_history,
                feature_history_mask=feature_history_mask,
                portfolio_reward_history=observations.get("portfolio_reward_history"),
            )
            # Candidate value composition consumes only the final fused state,
            # its count, and the behavioral basis fields. Repeating every
            # diagnostic component would materialize the full FAM map
            # [B, assets, history, K*features] once per action even though no
            # value-head operation reads it.
            required_component_names = {
                "fused",
                "valid_counts",
                "portfolio_context",
                *q_net.allocation_basis_component_names,
            }
            if q_net.pool_task_conditioning_enabled:
                required_component_names.add("pool_cls")
            if q_net.local_financial_handoff_enabled:
                required_component_names.update(
                    ("local_financial_forecast", "local_financial_local",
                     "local_financial_correction")
                )
            repeated_components = {
                name: self._repeat_encoded_batch(
                    components[name], candidate_count
                )
                for name in required_component_names
            }
            state_value_override = None
            reference_score_override = None
            if q_net.centered_bellman_enabled:
                state_value_override = self._repeat_encoded_batch(
                    q_net._state_value_from_encoded_state(
                        components,
                        mask,
                        prev_action,
                        basket_context,
                        basket_membership,
                    ),
                    candidate_count,
                )
                reference_score_override = self._repeat_encoded_batch(
                    q_net._uncentered_action_score_from_encoded_state(
                        components,
                        coin_features,
                        mask,
                        prev_action,
                        prev_action,
                        basket_context=basket_context,
                        basket_membership=basket_membership,
                        actions_are_weights=True,
                    ),
                    candidate_count,
                )
            flat_q = q_net._forward_from_encoded_state(
                repeated_components,
                repeated_coin_features,
                repeated_mask,
                repeated_prev_action,
                flat_actions,
                basket_context=repeated_basket_context,
                basket_membership=repeated_basket_membership,
                state_value_override=state_value_override,
                reference_action_score_override=reference_score_override,
            )
            q_values.append(
                flat_q.reshape(batch_size, candidate_count, -1).squeeze(-1)
            )
        return tuple(q_values)

    def q1_forward(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        ) = self._parse_observation(observations)
        actions = actions.float()
        return self.q_networks[0](
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            actions,
            basket_context=basket_context,
            basket_membership=basket_membership,
            feature_history=feature_history,
            feature_history_mask=feature_history_mask,
            portfolio_reward_history=observations.get("portfolio_reward_history"),
        )

    def value_experience_representation(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Identity-neutral FAM/AAM state-action coordinate for replay.

        The replay learner is allowed to recognize observable behavioral
        situations, but it is not allowed to recover an exact basket key.  We
        expose masked moments of the completed local and relational sets, then
        action-conditioned outcome predictions and pooled behavioral allocation
        roles. Raw slot-ordered weights are forbidden: current exposure,
        previous exposure, signed trade, and absolute trade are first bound to
        each asset's identity-free critic basis and only then pooled. This
        summary may be pooled because it is a replay descriptor; it is never
        fed back into an individual asset's local temporal query.
        """
        q_network = self.q_networks[0]
        if q_network.asset_identity_embedding is not None:
            raise RuntimeError(
                "value-experience replay requires Path-2 asset_identity_dim=0"
            )
        if q_network.basket_membership_dim > 0:
            raise RuntimeError(
                "value-experience replay may not consume basket membership"
            )
        (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        ) = self._parse_observation(observations)
        del asset_ids, prev_mask
        actions = actions.float()
        components = q_network.token_encoder.encode_components(
            coin_features,
            mask,
            asset_identity=None,
            basket_context=(
                basket_context if q_network.basket_context_dim > 0 else None
            ),
            basket_membership=None,
            feature_history=(
                feature_history if q_network.token_encoder.fam_enabled else None
            ),
            feature_history_mask=(
                feature_history_mask
                if q_network.token_encoder.fam_enabled
                else None
            ),
            detach_predictive_control_adapter=True,
        )
        action_mask = torch.cat(
            (torch.ones_like(mask[:, :1]), mask), dim=1
        )
        traded_weights = normalize_raw_actions_to_portfolio_weights(
            actions, action_mask
        )
        current_cash = traded_weights[:, :1]
        current_assets = traded_weights[:, 1 : 1 + q_network.max_coins]
        previous_cash = prev_action[:, :1].float().clamp(min=0.0)
        previous_assets = (
            prev_action[:, 1 : 1 + q_network.max_coins].float().clamp(min=0.0)
            * mask
        )
        previous_total = (
            previous_cash + previous_assets.sum(dim=1, keepdim=True)
        ).clamp(min=1e-8)
        previous_cash = previous_cash / previous_total
        previous_assets = previous_assets / previous_total
        asset_delta = current_assets - previous_assets

        def _masked_moments(values: torch.Tensor) -> torch.Tensor:
            expanded_mask = mask.unsqueeze(-1).to(values.dtype)
            count = expanded_mask.sum(dim=1).clamp(min=1.0)
            mean = (values * expanded_mask).sum(dim=1) / count
            variance = (
                (values - mean.unsqueeze(1)).square() * expanded_mask
            ).sum(dim=1) / count
            return torch.cat((mean, torch.sqrt(variance + 1e-6)), dim=-1)

        action_outcomes, candidate_behavioral_basis = (
            q_network.foundation_action_outcomes_from_components(
                components,
                coin_features,
                mask,
                prev_action,
                traded_weights.unsqueeze(1),
                basket_context,
                basket_membership,
                return_behavioral_basis=True,
            )
        )
        action_outcomes = action_outcomes.squeeze(1)
        behavioral_basis = candidate_behavioral_basis.squeeze(1)

        def _pooled_role(role_weights: torch.Tensor) -> torch.Tensor:
            return (
                behavioral_basis * role_weights.unsqueeze(-1)
            ).sum(dim=1)

        # These are the same four state x allocation roles used by the live
        # critic adapter. Pooling happens only after each scalar weight has
        # been attached to its corresponding observable asset behavior.
        allocation_roles = torch.cat(
            (
                _pooled_role(current_assets),
                _pooled_role(previous_assets),
                _pooled_role(asset_delta),
                _pooled_role(asset_delta.abs()),
            ),
            dim=-1,
        )
        previous_weights = torch.cat(
            (previous_cash, previous_assets), dim=-1
        )
        portfolio_statistics = torch.cat(
            (
                current_cash,
                previous_cash,
                current_assets.square().sum(dim=1, keepdim=True),
                previous_assets.square().sum(dim=1, keepdim=True),
                (traded_weights - previous_weights).abs().sum(
                    dim=1, keepdim=True
                ),
                current_assets.max(dim=1, keepdim=True).values,
            ),
            dim=-1,
        )
        return torch.cat(
            (
                _masked_moments(components["local_temporal"]),
                _masked_moments(components["relational"]),
                action_outcomes,
                allocation_roles,
                portfolio_statistics,
            ),
            dim=-1,
        )

    def value_experience_representation_widths(self) -> Tuple[int, int]:
        """Return the state/action boundary of the replay descriptor."""
        q_network = self.q_networks[0]
        state_width = 2 * (
            int(q_network.token_encoder.fam_asset_vector_dim)
            + int(q_network.token_encoder.model_dim)
        )
        outcome_width = int(
            q_network.foundation_action_outcome_head.out_features
        )
        action_width = (
            outcome_width
            + 4 * int(q_network.allocation_interaction_dim)
            + 6
        )
        return state_width, action_width

    def forward_with_component_outcomes(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        *,
        return_financial_decomposition: bool = False,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Return twin Bellman Q values and private causal forecasts."""
        (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        ) = self._parse_observation(observations)
        actions = actions.float()
        q_values = []
        component_predictions = []
        local_predictions = []
        context_corrections = []
        financial_bases = []
        bellman_residuals = []
        for q_net in self.q_networks:
            components = q_net._encode_state_components(
                coin_features,
                asset_ids,
                mask,
                prev_action,
                prev_mask,
                basket_context=basket_context,
                basket_membership=basket_membership,
                feature_history=feature_history,
                feature_history_mask=feature_history_mask,
                portfolio_reward_history=observations.get("portfolio_reward_history"),
            )
            result = q_net._forward_from_encoded_state(
                components,
                coin_features,
                mask,
                prev_action,
                actions,
                return_component_outcomes=True,
                return_financial_decomposition=return_financial_decomposition,
                basket_context=basket_context,
                basket_membership=basket_membership,
            )
            q_value, predictions = result[:2]
            q_values.append(q_value)
            component_predictions.append(predictions)
            if return_financial_decomposition:
                local_predictions.append(result[2])
                context_corrections.append(result[3])
                financial_bases.append(result[4])
                bellman_residuals.append(result[5])
        if return_financial_decomposition:
            result = (
                tuple(q_values),
                tuple(component_predictions),
                tuple(local_predictions),
                tuple(context_corrections),
                tuple(financial_bases),
                tuple(bellman_residuals),
            )
        else:
            result = (tuple(q_values), tuple(component_predictions))
        return result

    def forward_component_forecasts(
        self, observations: Dict[str, torch.Tensor]
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Supervise all original market minutes without another Bellman pass.

        XBR replaces replay rows for TD.  The asset teacher needs the entire
        originally sampled batch, so it separately encodes those real states
        while sharing the same critic parameters and optimizer update.
        """
        (
            coin_features, asset_ids, mask, prev_action, prev_mask,
            basket_context, basket_membership, feature_history,
            feature_history_mask,
        ) = self._parse_observation(observations)
        forecasts, locals_, corrections = [], [], []
        for q_net in self.q_networks:
            if not q_net.local_financial_handoff_enabled:
                raise RuntimeError("Real-batch forecast routing requires local-financial handoff")
            components = q_net._encode_state_components(
                coin_features, asset_ids, mask, prev_action, prev_mask,
                basket_context=basket_context,
                basket_membership=basket_membership,
                feature_history=feature_history,
                feature_history_mask=feature_history_mask,
                portfolio_reward_history=observations.get("portfolio_reward_history"),
            )
            expanded_mask = mask.unsqueeze(-1)
            forecasts.append(components["local_financial_forecast"] * expanded_mask)
            locals_.append(components["local_financial_local"] * expanded_mask)
            corrections.append(components["local_financial_correction"] * expanded_mask)
        return tuple(forecasts), tuple(locals_), tuple(corrections)

    def predictive_representation_loss(
        self,
        observations: Dict[str, torch.Tensor],
        next_observations: Dict[str, torch.Tensor],
        replay_actions: torch.Tensor,
        next_actions: torch.Tensor,
        target_critic: "DeepSetCritic",
        dones: Optional[torch.Tensor] = None,
        *,
        future_observations: Optional[Dict[str, torch.Tensor]] = None,
        future_actions: Optional[torch.Tensor] = None,
        future_next_actions: Optional[torch.Tensor] = None,
        future_valid_steps: Optional[torch.Tensor] = None,
        future_outcome_valid_steps: Optional[torch.Tensor] = None,
        future_rewards: Optional[torch.Tensor] = None,
        future_dones: Optional[torch.Tensor] = None,
        future_continuation_value_targets: Optional[torch.Tensor] = None,
        current_q_targets: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        horizon_discount: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Average recursive multi-step SPR loss across the twin critics."""
        current = self._parse_observation(observations)
        if future_observations is None:
            future_observations = {
                name: value.unsqueeze(1)
                for name, value in next_observations.items()
            }
        if future_actions is None:
            future_actions = replay_actions.unsqueeze(1)
        if future_next_actions is None:
            future_next_actions = next_actions.unsqueeze(1).expand(
                -1, future_actions.shape[1], -1
            )
        if future_actions.dim() != 3:
            raise ValueError("critic SPR actions must include a horizon axis")
        batch_size, horizon = future_actions.shape[:2]
        flat_future_observations = {
            name: value.reshape(batch_size * horizon, *value.shape[2:])
            for name, value in future_observations.items()
        }
        flat_next_state = target_critic._parse_observation(
            flat_future_observations
        )

        def _horizon_view(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch_size, horizon, *value.shape[1:])

        next_state = tuple(_horizon_view(value) for value in flat_next_state)
        (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            _,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        ) = current
        (
            next_coin_features,
            next_asset_ids,
            next_mask,
            next_prev_action,
            _,
            next_basket_context,
            next_basket_membership,
            next_feature_history,
            next_feature_history_mask,
        ) = next_state
        losses = [
            q_network.predictive_representation_loss(
                coin_features=coin_features,
                asset_ids=asset_ids,
                mask=mask,
                prev_action=prev_action,
                replay_actions=replay_actions,
                basket_context=basket_context,
                basket_membership=basket_membership,
                next_coin_features=next_coin_features,
                next_asset_ids=next_asset_ids,
                next_mask=next_mask,
                next_prev_action=next_prev_action,
                next_actions=future_next_actions,
                next_basket_context=next_basket_context,
                next_basket_membership=next_basket_membership,
                feature_history=feature_history,
                feature_history_mask=feature_history_mask,
                next_feature_history=next_feature_history,
                next_feature_history_mask=next_feature_history_mask,
                target_q_network=target_q_network,
                dones=dones,
                future_actions=future_actions,
                future_valid_steps=future_valid_steps,
                future_outcome_valid_steps=future_outcome_valid_steps,
                future_rewards=future_rewards,
                future_dones=future_dones,
                future_continuation_value_targets=(
                    future_continuation_value_targets
                ),
                current_q_targets=current_q_targets,
                gamma=gamma,
                horizon_discount=horizon_discount,
            )
            for q_network, target_q_network in zip(
                self.q_networks, target_critic.q_networks
            )
        ]
        metric_names = set().union(*(loss.keys() for loss in losses))
        return {
            name: torch.stack([loss[name] for loss in losses]).mean()
            for name in metric_names
        }

    def component_encoder_parameters(self):
        """Yield shared encoders whose auxiliary pressure is ratio-controlled."""
        for q_net in self.q_networks:
            yield from q_net.token_encoder.parameters()
            if q_net.asset_identity_embedding is not None:
                yield from q_net.asset_identity_embedding.parameters()
            if q_net.basket_context_proj is not None:
                yield from q_net.basket_context_proj.parameters()
            if q_net.basket_membership_proj is not None:
                yield from q_net.basket_membership_proj.parameters()
            if q_net.pool_task_mask_projection is not None:
                yield from q_net.pool_task_mask_projection.parameters()
            if q_net.pool_task_cls_projection is not None:
                yield from q_net.pool_task_cls_projection.parameters()
            if q_net.pool_task_norm is not None:
                yield from q_net.pool_task_norm.parameters()
            if q_net.uvfa_asset_factor_projection is not None:
                yield from q_net.uvfa_asset_factor_projection.parameters()
            if q_net.uvfa_task_factor_projection is not None:
                yield from q_net.uvfa_task_factor_projection.parameters()
            if q_net.uvfa_relational_factor_projection is not None:
                yield from q_net.uvfa_relational_factor_projection.parameters()
            if q_net.uvfa_interaction_gate_logit is not None:
                yield q_net.uvfa_interaction_gate_logit

    def encode_state(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            coin_features,
            asset_ids,
            mask,
            prev_action,
            prev_mask,
            basket_context,
            basket_membership,
            feature_history,
            feature_history_mask,
        ) = self._parse_observation(observations)
        prev_assets = prev_action[:, 1:1 + self.max_coins]
        return self.q_networks[0].encode_state(
            coin_features,
            asset_ids,
            mask,
            prev_mask,
            prev_assets,
            basket_context=basket_context,
            basket_membership=basket_membership,
            feature_history=feature_history,
            feature_history_mask=feature_history_mask,
        )

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


class MultiHeadAttentionExtractor(BaseFeaturesExtractor):
    """
    Enhanced Multi-Head Attention extractor for variable portfolio sizes.
    Handles both Box and Dict observation spaces for backward compatibility.
    """

    def __init__(
        self,
        observation_space: Union[spaces.Box, spaces.Dict],
        features_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        d_model: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        
        # Determine if we're dealing with Dict (variable) or Box (fixed) observations
        self.is_dict_obs = isinstance(observation_space, spaces.Dict)
        
        if self.is_dict_obs:
            # Dict observation space: extract total dimension
            self.input_dim = observation_space['observations'].shape[0]
            self.mask_dim = observation_space['mask'].shape[0]
        else:
            # Box observation space: use shape directly
            self.input_dim = observation_space.shape[0]
            self.mask_dim = 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Input projection to d_model dimensions
        self.input_projection = nn.Linear(self.input_dim, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))  # Max 100 features
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Feed-forward networks
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        
        # Final projection to desired features_dim
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if self.is_dict_obs:
            # Handle Dict observations
            obs = observations['observations']
            # Note: For this simpler attention, we don't use the mask
            # but it's available if needed
        else:
            # Handle Box observations (legacy)
            obs = observations
            
        batch_size = obs.shape[0]
        
        # Reshape observation to work with attention
        x = obs.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Project to d_model dimensions
        x = self.input_projection(x)  # [batch_size, 1, d_model]
        
        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply attention layers
        for i in range(self.n_layers):
            # Multi-head attention with residual connection
            attn_output, _ = self.attention_layers[i](x, x, x)
            x = self.layer_norms[i](x + attn_output)
            
            # Feed-forward with residual connection
            ff_output = self.feed_forwards[i](x)
            x = self.layer_norms[i](x + ff_output)
        
        # Global average pooling across sequence dimension
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Final projection
        features = self.output_projection(x)  # [batch_size, features_dim]
        
        return features


class CoinAttentionExtractor(BaseFeaturesExtractor):
    """
    Legacy coin attention extractor with variable portfolio support.
    Enhanced to handle both Box and Dict observation spaces.
    """

    def __init__(
        self,
        observation_space: Union[spaces.Box, spaces.Dict],
        features_dim: int = 256,
        max_coins: int = 5,
        features_per_coin: int = 14,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        **kwargs  # Accept additional arguments for backward compatibility
    ):
        super().__init__(observation_space, features_dim)
        
        # Determine observation type and setup
        self.is_dict_obs = isinstance(observation_space, spaces.Dict)
        
        if self.is_dict_obs:
            self.input_dim = observation_space['observations'].shape[0]
            self.max_coins = max_coins
            self.features_per_coin = features_per_coin
        else:
            # Legacy mode: infer from Box observation space
            self.input_dim = observation_space.shape[0]
            self.max_coins = kwargs.get('n_coins', 3)  # Backward compatibility
            self.features_per_coin = self.input_dim // self.max_coins
            
        self.d_model = 128
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Project each coin's features to d_model
        self.coin_projection = nn.Linear(self.features_per_coin, self.d_model)
        
        # Learnable coin embeddings
        self.coin_embeddings = nn.Parameter(torch.randn(self.max_coins, self.d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Market context attention
        self.market_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model * self.max_coins, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim)
        )

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if self.is_dict_obs:
            # Handle Dict observations
            obs = observations['observations']
            mask = observations['mask']
        else:
            # Handle Box observations (legacy)
            obs = observations
            mask = None
        
        batch_size = obs.shape[0]
        
        # Reshape observations to [batch_size, max_coins, features_per_coin]
        coin_features = obs.view(batch_size, self.max_coins, self.features_per_coin)
        
        # Project each coin's features
        coin_embeddings = self.coin_projection(coin_features)  # [batch_size, max_coins, d_model]
        
        # Add coin-specific embeddings
        coin_embeddings = coin_embeddings + self.coin_embeddings.unsqueeze(0)
        
        # Apply transformer with optional masking
        if mask is not None:
            attention_mask = ~mask.bool()  # True = ignore
            transformed = self.transformer(coin_embeddings, src_key_padding_mask=attention_mask)
        else:
            transformed = self.transformer(coin_embeddings)
        
        # Apply market attention
        if mask is not None:
            market_context, _ = self.market_attention(
                transformed, transformed, transformed,
                key_padding_mask=~mask.bool()
            )
        else:
            market_context, _ = self.market_attention(transformed, transformed, transformed)
        
        # Combine original and context
        enhanced_features = transformed + market_context
        
        # Flatten and project to final features
        flattened = enhanced_features.reshape(batch_size, -1)
        features = self.output_projection(flattened)
        
        return features


class AttentionActorCriticPolicy(ActorCriticPolicy):
    """
    Enhanced Actor-Critic policy supporting both fixed and variable portfolio sizes.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        attention_type: str = "variable_coin_attention",
        features_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        dirichlet_beta: float = 0.05,
        dirichlet_alpha: float = 1.0,
        **kwargs
    ):
        # Set the features extractor based on attention type
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_class"] = VariableCoinAttentionExtractor
        elif attention_type == "multihead":
            kwargs["features_extractor_class"] = MultiHeadAttentionExtractor
        elif attention_type == "coin_attention":
            kwargs["features_extractor_class"] = CoinAttentionExtractor
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        kwargs["features_extractor_kwargs"] = {
            "features_dim": features_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
        }
        
        # Add specific parameters for variable coin attention
        if attention_type == "variable_coin_attention":
            max_coins_cfg = kwargs.pop("max_coins", getattr(config, "MAX_COINS", 5))
            fpc_cfg = kwargs.pop("features_per_coin", getattr(config, "FEATURES_PER_COIN", 14))
            kwargs["features_extractor_kwargs"].update({
                "max_coins": max_coins_cfg,
                "features_per_coin": fpc_cfg,
            })
        elif attention_type == "coin_attention":
            kwargs["features_extractor_kwargs"].update({
                "max_coins": kwargs.pop("max_coins", kwargs.pop("n_coins", 3)),
                "features_per_coin": kwargs.pop("features_per_coin", 14),
                "n_coins": kwargs.pop("n_coins", 3)  # Backward compatibility
            })
        
        # Remove any extra parameters that shouldn't go to base policy
        extra_params = ["max_coins", "features_per_coin", "n_coins"]
        for param in extra_params:
            kwargs.pop(param, None)
        
        # Set net_arch to match features_dim
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = dict(pi=[features_dim], vf=[features_dim])
            
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # Dirichlet exploration over active dims (cash + active coins)
        self._dirichlet_beta = float(max(0.0, dirichlet_beta))
        self._dirichlet_alpha = float(max(1e-6, dirichlet_alpha))


class AttentionTD3Policy(TD3Policy):
    """
    TD3 policy with attention-based feature extraction.
    Supports custom actor and critic learning rates for CustomTD3_AC compatibility.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        attention_type: str = "variable_coin_attention",
        features_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        actor_lr: Optional[float] = None,  # NEW: Support custom actor learning rate
        critic_lr: Optional[float] = None,  # NEW: Support custom critic learning rate
        component_outcome_aux_enabled: Optional[bool] = None,
        component_outcome_dim: Optional[int] = None,
        component_outcome_unit_return_targets: Optional[bool] = None,
        component_forecast_bridge_enabled: Optional[bool] = None,
        component_forecast_bridge_return_reference: Optional[float] = None,
        component_forecast_bridge_max_abs: Optional[float] = None,
        component_forecast_bridge_local_only: Optional[bool] = None,
        uvfa_local_interaction_enabled: Optional[bool] = None,
        uvfa_local_interaction_rank: Optional[int] = None,
        uvfa_local_interaction_max_abs: Optional[float] = None,
        predictive_moe_enabled: Optional[bool] = None,
        predictive_moe_num_experts: Optional[int] = None,
        predictive_moe_expert_hidden_dim: Optional[int] = None,
        predictive_moe_router_temperature: Optional[float] = None,
        predictive_moe_max_residual_abs: Optional[float] = None,
        **kwargs
    ):
        # Store configured features_dim for robust actor construction even if the
        # features_extractor instance is not yet available when make_actor is called
        self.configured_features_dim = features_dim
        self._use_deepset_variable_policy = attention_type == "variable_coin_attention"
        self._configured_max_coins = kwargs.get("max_coins", 5)
        self._configured_features_per_coin = kwargs.get("features_per_coin", 14)
        self._configured_num_asset_ids = int(
            kwargs.get("num_asset_ids", self._configured_max_coins)
        )
        self._configured_asset_identity_dim = int(
            kwargs.get(
                "asset_identity_dim",
                0,
            )
        )
        default_basket_dim = int(
            getattr(config, "BASKET_CONTEXT_DIM", 0)
            if bool(getattr(config, "BASKET_CONTEXT_ENABLED", False))
            else 0
        )
        self._configured_basket_context_dim = int(
            kwargs.get("basket_context_dim", default_basket_dim)
        )
        default_membership_dim = int(
            getattr(config, "BASKET_MEMBERSHIP_DIM", 0)
            if bool(getattr(config, "BASKET_MEMBERSHIP_ENABLED", False))
            else 0
        )
        self._configured_basket_membership_dim = int(
            kwargs.get("basket_membership_dim", default_membership_dim)
        )
        (
            self._configured_local_feature_dim,
            self._configured_position_feature_dim,
            self._configured_comparative_feature_dim,
            self._configured_context_feature_dim,
        ) = _resolve_token_scope_dims(
            self._configured_features_per_coin,
            kwargs.get("local_feature_dim"),
            kwargs.get("position_feature_dim"),
            kwargs.get("context_feature_dim"),
            kwargs.get("comparative_feature_dim"),
        )
        self._configured_n_heads = n_heads
        self._configured_n_layers = n_layers
        self._configured_dropout = dropout
        self._configured_use_market_attention = kwargs.get("use_market_attention", True)
        self._configured_use_global_pooling = kwargs.get("use_global_pooling", True)
        self._configured_component_outcome_aux_enabled = bool(
            False
            if component_outcome_aux_enabled is None
            else component_outcome_aux_enabled
        )
        self._configured_component_outcome_dim = int(
            len(getattr(config, "COMPONENT_OUTCOME_HORIZONS", (1,)))
            if component_outcome_dim is None
            else component_outcome_dim
        )
        self._configured_component_outcome_unit_return_targets = bool(
            getattr(config, "COMPONENT_OUTCOME_UNIT_RETURN_TARGETS", False)
            if component_outcome_unit_return_targets is None
            else component_outcome_unit_return_targets
        )
        self._configured_component_forecast_bridge_enabled = bool(
            (
                getattr(config, "COMPONENT_FORECAST_BRIDGE_ENABLED", False)
                and self._configured_component_outcome_aux_enabled
            )
            if component_forecast_bridge_enabled is None
            else component_forecast_bridge_enabled
        )
        self._configured_component_forecast_bridge_return_reference = float(
            getattr(
                config,
                "COMPONENT_FORECAST_BRIDGE_RETURN_REFERENCE",
                0.01,
            )
            if component_forecast_bridge_return_reference is None
            else component_forecast_bridge_return_reference
        )
        self._configured_component_forecast_bridge_max_abs = float(
            getattr(config, "COMPONENT_FORECAST_BRIDGE_MAX_ABS", 0.10)
            if component_forecast_bridge_max_abs is None
            else component_forecast_bridge_max_abs
        )
        self._configured_component_forecast_bridge_local_only = bool(
            getattr(config, "COMPONENT_FORECAST_BRIDGE_LOCAL_ONLY", False)
            if component_forecast_bridge_local_only is None
            else component_forecast_bridge_local_only
        )
        self._configured_uvfa_local_interaction_enabled = bool(
            getattr(config, "UVFA_LOCAL_INTERACTION_ENABLED", False)
            if uvfa_local_interaction_enabled is None
            else uvfa_local_interaction_enabled
        )
        self._configured_uvfa_local_interaction_rank = int(
            getattr(config, "UVFA_LOCAL_INTERACTION_RANK", 8)
            if uvfa_local_interaction_rank is None
            else uvfa_local_interaction_rank
        )
        self._configured_uvfa_local_interaction_max_abs = float(
            getattr(config, "UVFA_LOCAL_INTERACTION_MAX_ABS", 0.10)
            if uvfa_local_interaction_max_abs is None
            else uvfa_local_interaction_max_abs
        )
        self._configured_predictive_moe_enabled = bool(
            False
            if predictive_moe_enabled is None
            else predictive_moe_enabled
        )
        self._configured_predictive_moe_num_experts = int(
            getattr(config, "PREDICTIVE_MOE_NUM_EXPERTS", 4)
            if predictive_moe_num_experts is None
            else predictive_moe_num_experts
        )
        self._configured_predictive_moe_expert_hidden_dim = int(
            getattr(config, "PREDICTIVE_MOE_EXPERT_HIDDEN_DIM", 64)
            if predictive_moe_expert_hidden_dim is None
            else predictive_moe_expert_hidden_dim
        )
        self._configured_predictive_moe_router_temperature = float(
            getattr(config, "PREDICTIVE_MOE_ROUTER_TEMPERATURE", 1.0)
            if predictive_moe_router_temperature is None
            else predictive_moe_router_temperature
        )
        self._configured_predictive_moe_max_residual_abs = float(
            getattr(config, "PREDICTIVE_MOE_MAX_RESIDUAL_ABS", 0.10)
            if predictive_moe_max_residual_abs is None
            else predictive_moe_max_residual_abs
        )
        # Set the features extractor based on attention type
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_class"] = VariableCoinAttentionExtractor
        elif attention_type == "multihead":
            kwargs["features_extractor_class"] = MultiHeadAttentionExtractor
        elif attention_type == "coin_attention":
            kwargs["features_extractor_class"] = CoinAttentionExtractor
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        extractor_kwargs = {
            "features_dim": features_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
        }
        
        # Add specific parameters for variable coin attention
        if attention_type == "variable_coin_attention":
            self._use_global_pooling = kwargs.pop("use_global_pooling", True)
            kwargs.pop("local_feature_dim", None)
            kwargs.pop("position_feature_dim", None)
            kwargs.pop("context_feature_dim", None)
            kwargs.pop("comparative_feature_dim", None)
            kwargs.pop("num_asset_ids", None)
            kwargs.pop("asset_identity_dim", None)
            kwargs.pop("basket_context_dim", None)
            kwargs.pop("basket_membership_dim", None)
            extractor_kwargs.update({
                "max_coins": kwargs.pop("max_coins", 5),
                "features_per_coin": kwargs.pop("features_per_coin", 14),
                "use_market_attention": kwargs.pop("use_market_attention", True),
                "use_global_pooling": self._use_global_pooling,
            })
            self._configured_use_global_pooling = extractor_kwargs["use_global_pooling"]
            self._configured_use_market_attention = extractor_kwargs["use_market_attention"]
        elif attention_type == "coin_attention":
            extractor_kwargs.update({
                "max_coins": kwargs.pop("max_coins", kwargs.pop("n_coins", 3)),
                "features_per_coin": kwargs.pop("features_per_coin", 14),
                "n_coins": kwargs.pop("n_coins", 3)  # Backward compatibility
            })
        # Attach finalized extractor kwargs
        kwargs["features_extractor_kwargs"] = extractor_kwargs

        # Remove any extra parameters that shouldn't go to base policy
        extra_params = [
            "max_coins", "features_per_coin", "n_coins",
            "use_market_attention", "use_global_pooling",
            "local_feature_dim", "position_feature_dim", "context_feature_dim",
            "comparative_feature_dim",
            "num_asset_ids", "asset_identity_dim", "basket_context_dim",
            "basket_membership_dim",
        ]
        for param in extra_params:
            kwargs.pop(param, None)

        # Set net_arch from attention_configs if not already specified
        if "net_arch" not in kwargs and attention_type in ATTENTION_CONFIGS:
            if "net_arch" in ATTENTION_CONFIGS[attention_type]:
                 kwargs["net_arch"] = ATTENTION_CONFIGS[attention_type]["net_arch"]

        # Default net_arch if not specified anywhere
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = [256, 256]

        # Save attention configuration for computing true actor input dim when extractor not yet built
        self._attention_type = attention_type
        self._attention_max_coins = extractor_kwargs.get("max_coins", 5)
        self._attention_d_model = features_dim // 2

        # ✅ CustomTD3_AC COMPATIBILITY: Store custom learning rates before calling parent
        self.actor_lr_override = actor_lr
        self.critic_lr_override = critic_lr
        
        # Remove actor_lr and critic_lr from kwargs before passing to parent
        kwargs.pop("actor_lr", None)
        kwargs.pop("critic_lr", None)
            
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_actor(self, features_extractor: nn.Module = None) -> nn.Module:
        if self._use_deepset_variable_policy:
            net_arch = self.net_arch["pi"] if isinstance(self.net_arch, dict) else self.net_arch
            return DeepSetActor(
                require_portfolio_reward_history=(
                    'portfolio_reward_history' in self.observation_space.spaces),
                max_coins=self._configured_max_coins,
                features_per_coin=self._configured_features_per_coin,
                action_dim=self.action_space.shape[0],
                set_hidden_dim=self.configured_features_dim,
                net_arch=net_arch,
                n_heads=self._configured_n_heads,
                n_layers=self._configured_n_layers,
                dropout=self._configured_dropout,
                use_market_attention=self._configured_use_market_attention,
                use_global_pooling=self._configured_use_global_pooling,
                local_feature_dim=self._configured_local_feature_dim,
                position_feature_dim=self._configured_position_feature_dim,
                context_feature_dim=self._configured_context_feature_dim,
                comparative_feature_dim=self._configured_comparative_feature_dim,
                num_asset_ids=self._configured_num_asset_ids,
                asset_identity_dim=self._configured_asset_identity_dim,
                basket_context_dim=self._configured_basket_context_dim,
                basket_membership_dim=self._configured_basket_membership_dim,
                predictive_moe_enabled=self._configured_predictive_moe_enabled,
                predictive_moe_num_experts=(
                    self._configured_predictive_moe_num_experts
                ),
                predictive_moe_expert_hidden_dim=(
                    self._configured_predictive_moe_expert_hidden_dim
                ),
                predictive_moe_router_temperature=(
                    self._configured_predictive_moe_router_temperature
                ),
                predictive_moe_max_residual_abs=(
                    self._configured_predictive_moe_max_residual_abs
                ),
                # Without temporal conv, per-token representations are
                # homogeneous enough for independent scoring via pooled_mean
                # context. CRL alone enforces smoothness — no mixer needed.
                # This matches the validated base design (preog_base).
                use_action_mixer=False,    
            ).to(self.device)

        """Create a NormalisedActor for attention TD3 to produce stable [0,1] actions.
        TD3Policy._build calls make_actor(None), so the extractor does not exist yet; we must
        create it here to get the correct features_dim (flatten_dim + d_model when use_global_pooling).
        """
        extractor = features_extractor or getattr(self, "features_extractor", None)
        if extractor is None:
            # _build calls make_actor(None); create extractor so features_dim matches forward() output
            extractor = self.make_features_extractor()
            self.features_extractor = extractor
        if extractor is not None and hasattr(extractor, "features_dim"):
            obs_dim = extractor.features_dim
        else:
            # Fallback if features_dim not available
            if getattr(self, "_attention_type", None) == "variable_coin_attention":
                use_gp = getattr(self, "_use_global_pooling", True)
                obs_dim = self._attention_d_model * self._attention_max_coins
                if use_gp:
                    obs_dim += self._attention_d_model
            elif hasattr(self, "configured_features_dim") and isinstance(self.configured_features_dim, int):
                obs_dim = self.configured_features_dim
            else:                
                if isinstance(self.observation_space, spaces.Dict):
                    obs_dim = self.observation_space["observations"].shape[0]
                else:
                    obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]

        # net_arch may be a dict(pi=..., qf=...) or a list
        net_arch = self.net_arch['pi'] if isinstance(self.net_arch, dict) else self.net_arch

        actor = NormalisedActor(
            observation_dim=obs_dim,
            action_dim=action_dim,
            net_arch=net_arch,
        )
        
        # ✅ NEW: Debug the dimensions
        print(f"🔍 Actor Debug: observation_dim={obs_dim}, action_dim={action_dim}")
        if extractor is not None and hasattr(extractor, "features_dim"):
            print(f"   Using extractor.features_dim: {extractor.features_dim}")
        else:
            print(f"   Calculated obs_dim: {obs_dim}")
        
        return actor

    def make_critic(self, features_extractor: nn.Module = None) -> nn.Module:
        if self._use_deepset_variable_policy:
            critic_arch = self.critic_kwargs["net_arch"] if isinstance(self.critic_kwargs["net_arch"], list) else self.net_arch
            return DeepSetCritic(
                observation_space=self.observation_space,
                action_space=self.action_space,
                max_coins=self._configured_max_coins,
                features_per_coin=self._configured_features_per_coin,
                set_hidden_dim=self.configured_features_dim,
                net_arch=critic_arch,
                n_heads=self._configured_n_heads,
                n_layers=self._configured_n_layers,
                dropout=self._configured_dropout,
                use_market_attention=self._configured_use_market_attention,
                use_global_pooling=self._configured_use_global_pooling,
                n_critics=self.critic_kwargs["n_critics"],
                local_feature_dim=self._configured_local_feature_dim,
                position_feature_dim=self._configured_position_feature_dim,
                context_feature_dim=self._configured_context_feature_dim,
                comparative_feature_dim=self._configured_comparative_feature_dim,
                uvfa_local_interaction_enabled=(
                    self._configured_uvfa_local_interaction_enabled
                ),
                uvfa_local_interaction_rank=(
                    self._configured_uvfa_local_interaction_rank
                ),
                uvfa_local_interaction_max_abs=(
                    self._configured_uvfa_local_interaction_max_abs
                ),
                component_outcome_aux_enabled=(
                    self._configured_component_outcome_aux_enabled
                ),
                component_outcome_dim=self._configured_component_outcome_dim,
                component_outcome_unit_return_targets=(
                    self._configured_component_outcome_unit_return_targets
                ),
                component_forecast_bridge_enabled=(
                    self._configured_component_forecast_bridge_enabled
                ),
                component_forecast_bridge_return_reference=(
                    self._configured_component_forecast_bridge_return_reference
                ),
                component_forecast_bridge_max_abs=(
                    self._configured_component_forecast_bridge_max_abs
                ),
                component_forecast_bridge_local_only=(
                    self._configured_component_forecast_bridge_local_only
                ),
                num_asset_ids=self._configured_num_asset_ids,
                asset_identity_dim=self._configured_asset_identity_dim,
                basket_context_dim=self._configured_basket_context_dim,
                basket_membership_dim=self._configured_basket_membership_dim,
                predictive_moe_enabled=self._configured_predictive_moe_enabled,
                predictive_moe_num_experts=(
                    self._configured_predictive_moe_num_experts
                ),
                predictive_moe_expert_hidden_dim=(
                    self._configured_predictive_moe_expert_hidden_dim
                ),
                predictive_moe_router_temperature=(
                    self._configured_predictive_moe_router_temperature
                ),
                predictive_moe_max_residual_abs=(
                    self._configured_predictive_moe_max_residual_abs
                ),
            ).to(self.device)
        return super().make_critic(features_extractor)

    def _compile_deepset_modules(self) -> None:
        if not self._use_deepset_variable_policy:
            return
        strategy = str(
            getattr(config, "TORCH_COMPILE_STRATEGY", "regional")
        ).strip().lower()
        if not bool(getattr(config, "TORCH_COMPILE_ENABLED", False)) or strategy == "off":
            return

        if strategy == "full":
            _maybe_compile_method(self.actor, "forward", "DeepSetActor")
            _maybe_compile_method(self.actor_target, "forward", "DeepSetActorTarget")
            _maybe_compile_method(self.critic, "forward", "DeepSetCritic")
            _maybe_compile_method(self.critic, "q1_forward", "DeepSetCritic")
            if self._configured_component_outcome_aux_enabled:
                _maybe_compile_method(
                    self.critic,
                    "forward_with_component_outcomes",
                    "DeepSetCritic",
                )
            _maybe_compile_method(self.critic_target, "forward", "DeepSetCriticTarget")
            _maybe_compile_method(self.critic_target, "q1_forward", "DeepSetCriticTarget")
            return

        if strategy != "regional":
            raise ValueError(f"Unsupported torch.compile strategy: {strategy}")

        roots = (
            ("actor", self.actor),
            ("actor_target", self.actor_target),
            *(
                (f"critic.q{index + 1}", q_network)
                for index, q_network in enumerate(self.critic.q_networks)
            ),
            *(
                (f"critic_target.q{index + 1}", q_network)
                for index, q_network in enumerate(self.critic_target.q_networks)
            ),
        )
        compiled_ids = set()
        for root_label, root in roots:
            encoder = getattr(root, "token_encoder", None)
            if encoder is None:
                continue
            regions = [
                ("fam", getattr(encoder, "feature_attention", None)),
                ("fam_map_embedding", getattr(encoder, "fam_asset_embedding", None)),
                ("market_encoder", getattr(encoder, "market_encoder", None)),
                ("fam_to_model_projection", getattr(encoder, "fam_to_model_projection", None)),
                ("fusion_encoder", getattr(encoder, "fusion_encoder", None)),
                ("predictive_experts", getattr(encoder, "predictive_expert_layer", None)),
            ]
            regions.extend(
                (f"aam_layer_{index}", layer)
                for index, layer in enumerate(getattr(encoder, "transformer", ()))
            )
            for region_name, region in regions:
                if region is None or id(region) in compiled_ids:
                    continue
                if region_name in {
                    "market_encoder",
                    "fam_to_model_projection",
                    "fusion_encoder",
                } and not isinstance(region, nn.Sequential):
                    continue
                if _maybe_compile_module(region, f"{root_label}.{region_name}"):
                    compiled_ids.add(id(region))

            # Regional compilation of SoftPredictiveExpertLayer.forward only
            # covers control-time expert composition. SSL calls predict_next()
            # directly, where several small affine heads would otherwise be
            # separate CUDA launches at every update. The production SSL
            # minibatch and one-step horizon have fixed shapes, so specialize
            # this training-only method on the live actor and twin critics.
            if "_target" not in root_label and encoder.predictive_expert_layer is not None:
                _maybe_compile_method(
                    encoder.predictive_expert_layer,
                    "predict_next",
                    f"{root_label}.predictive_experts",
                    dynamic_override=False,
                    fullgraph=False,
                )

            # The action and value heads consume the same set representation,
            # but remain outside the compiled encoder so TD3 can keep its
            # actor/critic ownership boundaries. Their repeated MLP stems are
            # fixed-shape tensor regions and are safe to compile independently.
            for region_name in (
                "action_token_backbone",
                "local_value_backbone",
                "component_outcome_backbone",
                "component_local_backbone",
                "component_context_backbone",
            ):
                region = getattr(root, region_name, None)
                if not isinstance(region, nn.Sequential) or id(region) in compiled_ids:
                    continue
                if _maybe_compile_module(region, f"{root_label}.{region_name}"):
                    compiled_ids.add(id(region))

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Build the network architecture with attention-based feature extraction.
        Override optimizers with custom learning rates if provided (for CustomTD3_AC).
        """
        # Create the network architecture using the parent's _build method
        super()._build(lr_schedule)

        # ✅ CustomTD3_AC COMPATIBILITY: Override optimizers with custom LRs if provided
        if self.actor_lr_override is not None and self.critic_lr_override is not None:
            print(f"🔧 AttentionTD3Policy: Using custom learning rates: Actor={self.actor_lr_override}, Critic={self.critic_lr_override}")
            
            # Re-create the optimizers with the specific learning rates
            optimizer_kwargs = self.optimizer_kwargs.copy()
            
            self.actor.optimizer = self.optimizer_class(
                self.actor.parameters(), lr=self.actor_lr_override, **optimizer_kwargs
            )
            self.critic.optimizer = self.optimizer_class(
                self.critic.parameters(), lr=self.critic_lr_override, **optimizer_kwargs
            )

        self._compile_deepset_modules()

    def _ensure_features_extractor(self) -> None:
        """Lazily create features_extractor if it's not yet built (version-safe)."""
        if self._use_deepset_variable_policy:
            return
        if getattr(self, "features_extractor", None) is None:
            self.features_extractor = self.make_features_extractor()

    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Training-time action for either Deep Sets raw obs or feature-based attention."""
        if self._use_deepset_variable_policy:
            return self.actor(observation)
        self._ensure_features_extractor()
        features = self.extract_features(observation, self.features_extractor)
        action = self.actor(features)

        # ✅ FIXED: Remove action masking - let environment handle it
        # This prevents masking from interfering with action diversity
        # The environment will apply masking during action processing
            
        return action
        # """Training-time action. Uses default TD3 Actor; it does extract_features internally and outputs [-1,1] (tanh)."""
        # return self.actor(observation)
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Inference-time action for either Deep Sets raw obs or feature-based attention."""
        if self._use_deepset_variable_policy:
            with torch.no_grad():
                return self.actor(observation)
        self._ensure_features_extractor()
        features = self.extract_features(observation, self.features_extractor)
        # """Inference-time action. Same as forward; env maps [-1,1] to [0,1] via (x+1)*0.5."""

        with torch.no_grad():
            action_01 = self.actor(features)
            # ✅ FIXED: Remove action masking - let environment handle it
            # This prevents masking from interfering with action diversity
        return action_01
            # return self.actor(observation)



class AttentionSACPolicy(SACPolicy):
    """
    SAC policy with attention-based feature extraction.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        attention_type: str = "variable_coin_attention",
        features_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        # Set the features extractor based on attention type
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_class"] = VariableCoinAttentionExtractor
        elif attention_type == "multihead":
            kwargs["features_extractor_class"] = MultiHeadAttentionExtractor
        elif attention_type == "coin_attention":
            kwargs["features_extractor_class"] = CoinAttentionExtractor
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        kwargs["features_extractor_kwargs"] = {
            "features_dim": features_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
        }
        
        # Add specific parameters for variable coin attention
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_kwargs"].update({
                "max_coins": kwargs.pop("max_coins", 5),
                "features_per_coin": kwargs.pop("features_per_coin", 14)
            })
        elif attention_type == "coin_attention":
            kwargs["features_extractor_kwargs"].update({
                "max_coins": kwargs.pop("max_coins", kwargs.pop("n_coins", 3)),
                "features_per_coin": kwargs.pop("features_per_coin", 14),
                "n_coins": kwargs.pop("n_coins", 3)  # Backward compatibility
            })
        
        # Remove any extra parameters that shouldn't go to base policy
        extra_params = ["max_coins", "features_per_coin", "n_coins"]
        for param in extra_params:
            kwargs.pop(param, None)
        
        # Set net_arch to match features_dim
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = [features_dim, features_dim]
            
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)


class AttentionDDPGPolicy(TD3Policy):
    """
    DDPG policy with attention-based feature extraction.
    Inherits from TD3Policy as per SB3 design.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        attention_type: str = "variable_coin_attention",
        features_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        # Set the features extractor based on attention type
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_class"] = VariableCoinAttentionExtractor
        elif attention_type == "multihead":
            kwargs["features_extractor_class"] = MultiHeadAttentionExtractor
        elif attention_type == "coin_attention":
            kwargs["features_extractor_class"] = CoinAttentionExtractor
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        kwargs["features_extractor_kwargs"] = {
            "features_dim": features_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
        }
        
        # Add specific parameters for variable coin attention
        if attention_type == "variable_coin_attention":
            kwargs["features_extractor_kwargs"].update({
                "max_coins": kwargs.pop("max_coins", 5),
                "features_per_coin": kwargs.pop("features_per_coin", 14)
            })
        elif attention_type == "coin_attention":
            kwargs["features_extractor_kwargs"].update({
                "max_coins": kwargs.pop("max_coins", kwargs.pop("n_coins", 3)),
                "features_per_coin": kwargs.pop("features_per_coin", 14),
                "n_coins": kwargs.pop("n_coins", 3)  # Backward compatibility
            })
        
        # Remove any extra parameters that shouldn't go to base policy
        extra_params = ["max_coins", "features_per_coin", "n_coins"]
        for param in extra_params:
            kwargs.pop(param, None)
        
        # Set net_arch to match features_dim
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = [features_dim, features_dim]
            
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)


# =============================================================================
# SIMPLE MLP POLICY CLASSES (NO ATTENTION) - REFACTORED
# =============================================================================

class SimpleMlpActorCriticPolicy(ActorCriticPolicy):
    """
    Simple MLP Actor-Critic policy (for PPO).
    This class is now a standard MlpPolicy that will use the default FlattenExtractor.
    The network architecture is passed via `policy_kwargs` in the model constructor.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Build the network architecture.
        This is overriden to use custom learning rates for actor and critic.
        """
        # Create the network architecture
        super()._build(lr_schedule)

        # Override optimizers if custom learning rates are provided
        if "actor_lr" in self.optimizer_kwargs and "critic_lr" in self.optimizer_kwargs:
            actor_lr = self.optimizer_kwargs.pop("actor_lr")
            critic_lr = self.optimizer_kwargs.pop("critic_lr")

            # Create new optimizers with the specified learning rates
            self.actor.optimizer = self.optimizer_class(
                self.actor.parameters(), lr=actor_lr, **self.optimizer_kwargs
            )
            self.critic.optimizer = self.optimizer_class(
                self.critic.parameters(), lr=critic_lr, **self.optimizer_kwargs
            )
            print(f"✅ Using custom learning rates: Actor={actor_lr}, Critic={critic_lr}")

# =============================================================================
# Custom Actor with Sigmoid Activation for [0, 1] Output
# =============================================================================

class ActorSigmoid(TD3Actor):
    """
    Custom TD3 Actor that uses a sigmoid activation function in the final layer
    to output actions directly in the [0, 1] range. This is a replacement for
    the default actor which uses tanh for [-1, 1] output.
    """
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in the actor network.

        :param obs: Observation
        :return: Action
        """
        # The standard TD3 Actor applies tanh to the output of self.mu
        # We override this to apply sigmoid instead.
        features = self.features_extractor(obs)
        
        # ✅ DATA TYPE FIX: Cast the features to float32 right before the network call.
        # This is the most robust way to handle float64 observations from the environment,
        # ensuring the input to the neural network matches its float32 weights.
        features = features.float()
        
        return torch.sigmoid(self.mu(features))


class ContinuousCriticFloat(ContinuousCritic):
    """
    Custom ContinuousCritic that casts the final input tensor to float32
    to handle environments that use float64 (double) precision.
    """
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Re-implementation of the parent's forward pass to insert a dtype cast.
        with torch.set_grad_enabled(not self.share_features_extractor):
            # The features extractor will run on the original (potentially float64) obs.
            features = self.extract_features(obs, self.features_extractor)

        # The 'actions' tensor can also be float64. Concatenation will produce float64.
        qvalue_input = torch.cat([features, actions], dim=1)

        # ✅ DATA TYPE FIX: Cast the final, combined input to float32 right before the network call.
        qvalue_input = qvalue_input.float()

        return tuple(q_net(qvalue_input) for q_net in self.q_networks)


# =============================================================================
# Simple MLP Policy (TD3)
# =============================================================================

class SimpleMlpTD3Policy(TD3Policy):
    """
    Simple MLP TD3 policy.
    This class is now a standard MlpPolicy that will use the default FlattenExtractor.
    The network architecture is passed via `policy_kwargs` (e.g., dict(pi=[...], qf=[...]))
    in the model constructor.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        actor_lr: Optional[float] = None,
        critic_lr: Optional[float] = None,
        **kwargs,
    ):
        # ✅ FIX: Set the custom LRs on the instance *before* calling the parent constructor.
        # This prevents an AttributeError because the parent's __init__ calls self._build(),
        # which needs these attributes to be available.
        self.actor_lr_override = actor_lr
        self.critic_lr_override = critic_lr
        
        # ✅ FIX: Pass the custom learning rates into the optimizer_kwargs so they are available
        # to the parent class during initialization.
        if "optimizer_kwargs" not in kwargs:
            kwargs["optimizer_kwargs"] = {}
        # if actor_lr is not None:
        #     kwargs["optimizer_kwargs"]["actor_lr"] = actor_lr
        # if critic_lr is not None:
        #     kwargs["optimizer_kwargs"]["critic_lr"] = critic_lr
            
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Build the network architecture.
        This is overridden to use custom learning rates for actor and critic if provided.
        
        ✅ CRITICAL FIX: Only override optimizers when NOT using CustomTD3_AC.
        CustomTD3_AC manages its own optimizers with learning rate schedules.
        """
        # Create the network architecture using the parent's _build method
        super()._build(lr_schedule)

        # ✅ FIX: Check if we're being used with CustomTD3_AC by looking for the learning rate schedules
        # If CustomTD3_AC is managing optimizers, don't interfere
        using_custom_td3_ac = hasattr(self, '_custom_td3_ac_managed') and self._custom_td3_ac_managed
        
        # If custom learning rates were passed AND we're not using CustomTD3_AC, override optimizers
        if (self.actor_lr_override is not None and self.critic_lr_override is not None 
            and not using_custom_td3_ac):
            print(f"🔧 SimpleMlpTD3Policy: Overriding optimizers with custom learning rates: Actor={self.actor_lr_override}, Critic={self.critic_lr_override}")
            
            # Re-create the optimizers with the specific learning rates
            # We pop our custom LRs from optimizer_kwargs so they aren't passed to Adam again.
            optimizer_kwargs = self.optimizer_kwargs.copy()
            optimizer_kwargs.pop("actor_lr", None)
            optimizer_kwargs.pop("critic_lr", None)
            
            self.actor.optimizer = self.optimizer_class(
                self.actor.parameters(), lr=self.actor_lr_override, **optimizer_kwargs
            )
            self.critic.optimizer = self.optimizer_class(
                self.critic.parameters(), lr=self.critic_lr_override, **optimizer_kwargs
            )
        elif using_custom_td3_ac:
            print(f"🔧 SimpleMlpTD3Policy: Skipping optimizer override - CustomTD3_AC will manage optimizers with learning rate schedules")

class SimpleMlpPPOPolicy(ActorCriticPolicy):
    """
    Simple MLP PPO policy with BOUNDED actions via Tanh and action clipping.
    
    CRITICAL: PPO samples actions from DiagGaussian distribution N(mean, std).
    Even with Tanh on the mean, sampled actions can be unbounded due to learned std.
    This policy both bounds the mean AND clips sampled actions to [-1, 1].
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
    
    def _build(self, lr_schedule: Schedule) -> None:
        """Build networks with Tanh activation on mean output."""
        # First build everything normally
        super()._build(lr_schedule)
        
        # ✅ FIX 1: Wrap action_net with Tanh to bound MEAN to [-1, 1]
        original_action_net = self.action_net
        self.action_net = nn.Sequential(
            original_action_net,
            nn.Tanh()
        )
        
        print(f"✅ SimpleMlpPPOPolicy: Added Tanh activation to bound mean to [-1, 1]")
        print(f"✅ SimpleMlpPPOPolicy: Actions will be clipped to [-1, 1] after sampling")
    

class SimpleMlpSACPolicy(SACPolicy):
    """
    Simple MLP SAC policy.
    This class is now a standard MlpPolicy that will use the default FlattenExtractor.
    The network architecture is passed via `policy_kwargs` (e.g., dict(pi=[...], qf=[...]))
    in the model constructor.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class SimpleMlpDDPGPolicy(TD3Policy):
    """
    Simple MLP DDPG policy. Inherits from TD3Policy as per SB3.
    This class is now a standard MlpPolicy that will use the default FlattenExtractor.
    The network architecture is passed via `policy_kwargs` (e.g., dict(pi=[...], qf=[...]))
    in the model constructor.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Enhanced factory function
def create_attention_policy(attention_type: str = "variable_coin_attention", algorithm: str = "PPO", **kwargs):
    """
    Factory function to create policies with attention or simple MLP.
    
    Args:
        attention_type: "variable_coin_attention", "multihead", "coin_attention", or "mlp"
        algorithm: "PPO", "TD3", "SAC", or "DDPG"
        **kwargs: Additional arguments for the attention mechanism or MLP
    
    Returns:
        Policy class that can be used with SB3 algorithms
    """
    
    # Correctly select the policy class based on attention type and algorithm
    if attention_type.lower() == "mlp":
        # ---- MLP Policies (NO ATTENTION) ----
        if algorithm.upper() == "PPO":
            return SimpleMlpActorCriticPolicy
        elif algorithm.upper() == "TD3":
            return SimpleMlpTD3Policy
        elif algorithm.upper() == "SAC":
            return SimpleMlpSACPolicy
        elif algorithm.upper() == "DDPG":
            return SimpleMlpDDPGPolicy
        else:
            raise ValueError(f"Unsupported algorithm for MLP: {algorithm}")

    else:
        # ---- Attention-Based Policies ----
        if algorithm.upper() == "PPO":
            BasePolicyClass = AttentionActorCriticPolicy
        elif algorithm.upper() == "TD3":
            BasePolicyClass = AttentionTD3Policy
        elif algorithm.upper() == "SAC":
            BasePolicyClass = AttentionSACPolicy
        elif algorithm.upper() == "DDPG":
            BasePolicyClass = AttentionDDPGPolicy
        else:
            raise ValueError(f"Unsupported algorithm for Attention: {algorithm}")

        # Create and return the custom attention policy
        class CustomAttentionPolicy(BasePolicyClass):
            def __init__(self, *args, **policy_kwargs):
                super().__init__(*args, attention_type=attention_type, **kwargs, **policy_kwargs)
        return CustomAttentionPolicy


# Enhanced configuration for variable portfolios and MLPs
ATTENTION_CONFIGS = {
    "light": {
        "features_dim": 64,
        "n_heads": 4,
        "n_layers": 1,
        "dropout": 0.1,
        "max_coins": 5,
        "features_per_coin": 14,
        "asset_identity_dim": 0,
        "basket_context_dim": 8,
        "basket_membership_dim": 0,
        "use_market_attention": False,  # Faster: one transformer stack only
        "use_global_pooling": True,
        "net_arch": [400, 300]  # ✅ FIX: Use same deep network as MLP
    },
    "medium": {
        "features_dim": 128,
        "n_heads": 8,
        "n_layers": 2,
        "dropout": 0.1,
        "max_coins": 5,
        "features_per_coin": 14,
        "asset_identity_dim": 0,
        "basket_context_dim": 8,
        "basket_membership_dim": 0,
        "use_market_attention": False,
        "use_global_pooling": True,
        "net_arch": [400, 300]  # ✅ FIX: Use same deep network as MLP
    },
    # 8-of-13 task-conditioned capacity experiment. Runtime observation-schema
    # widths are deliberately absent: train_simple_mlp_og.py derives them from
    # config so this profile also remains valid when the universe changes.
    "multitask": {
        "features_dim": 256,
        "n_heads": 8,
        "n_layers": 2,
        "dropout": 0.0,
        "asset_identity_dim": 0,
        "use_market_attention": False,
        "use_global_pooling": True,
        "net_arch": [512, 384],
    },
    "heavy": {
        "features_dim": 256,
        "n_heads": 16,
        "n_layers": 3,
        "dropout": 0.2,
        "max_coins": 5,
        "features_per_coin": 14,
        "asset_identity_dim": 0,
        "basket_context_dim": 8,
        "basket_membership_dim": 0,
        "use_market_attention": False,
        "use_global_pooling": True,
        "net_arch": [400, 300]  # ✅ FIX: Use same deep network as MLP
    }
}

# Simple MLP configurations are now defined in the training script.
# The `net_arch` is passed directly to the model via `policy_kwargs`.
MLP_CONFIGS = {
    "light": {
        "net_arch": [256, 128],
        "features_dim": 128,
    },
    "medium": {
        "net_arch": [400, 300],
        "features_dim": 300,
    },
    "heavy": {
        "net_arch": [512, 512, 512, 512, 512,  256], 
        "features_dim": 256, 
    }
}


class NormalisedActor(nn.Module):
    """
    ✅ ENHANCED SOLUTION: Custom actor that outputs normalized portfolio weights with better exploration.
    
    This prevents the many-to-one mapping issue while allowing proper gradient flow and exploration.
    
    Key improvements:
    1. L1 normalization instead of softmax (allows negative logits)
    2. Better gradient flow for learning
    3. Injective mapping: each unique network output → unique portfolio allocation
    4. Proper exploration and differentiation based on observations
    """
    
    def __init__(self, observation_dim: int, action_dim: int, net_arch: List[int] = [400, 300]):
        super().__init__()
        self.expects_precomputed_features = True
        
        # Build MLP layers (same architecture as standard TD3)
        layers = []
        prev_dim = observation_dim
        
        for layer_dim in net_arch:
            layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.ReLU(),
            ])
            prev_dim = layer_dim
        
        # Output layer (no activation - we'll apply custom normalization)
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # ✅ FIXED: Use standard SB3 initialization instead of custom biased initialization
        # This allows natural exploration and prevents similar action outputs
        # Standard SB3 initialization provides better diversity
        with torch.no_grad():
            # Use standard bias initialization (no custom biases)
            # Let the agent learn naturally without forced biases
            
            # Add minimal noise for diversity (much smaller than before)
            noise = torch.randn(action_dim) * 0.01  # Very small noise for diversity
            self.mlp[-1].bias.data += noise
        
        print(f"🎯 Enhanced NormalisedActor created:")
        print(f"   📊 Architecture: {observation_dim} → {net_arch} → {action_dim}")
        print(f"   ✅ Uses tanh activation to output [-1,1] (env maps to [0,1] via (x+1)*0.5)")
        print(f"   🔄 Standard SB3 initialization (natural exploration)")
        print(f"   🎲 Agent learns to produce diverse weights through reward signal")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: observations -> portfolio weight candidates.
        
        Output in [-1,1] via tanh so env (x+1)*0.5 gives [0,1]. Keeps CRL stable
        (custom actor on features) without sigmoid cash bias or unbounded logits.
        """
        # ✅ DTYPE FIX: Ensure observations are float32 for PyTorch compatibility
        if observations.dtype != torch.float32:
            observations = observations.float()
        
        # Pass through MLP layers to get raw logits
        logits = self.mlp(observations)
        
        # Output in [-1,1] so env's (x+1)*0.5 maps to [0,1]. Avoids unbounded logits and
        # keeps CRL smooth (custom actor on features); no sigmoid so no 0.75 cash bias.
        portfolio_weights = torch.tanh(logits)
        
        # Debug output to verify diverse allocations
        if torch.rand(1).item() < 0.01:  # Print 1% of the time to avoid spam
            # ✅ BATCH FIX: Handle batch processing by printing the first item
            first_item_actions = portfolio_weights[0]
            
            print(f"🔍 NormalisedActor (Portfolio Weights) Debug:")
            print(f"   Raw logits (first item): {logits[0].detach().cpu().numpy()}")
            print(f"   Tanh weights (first item): {first_item_actions.detach().cpu().numpy()}")
            print(f"   Weight sum: {first_item_actions.sum().item():.6f}")
            print(f"   Weight range: [{first_item_actions.min().item():.3f}, {first_item_actions.max().item():.3f}]")
            print(f"   Weight std: {first_item_actions.std().item():.6f}")
            print(f"   ✅ Tanh ensures [-1,1]; env (x+1)*0.5 -> [0,1]")
            
            # Check for diversity in tanh outputs
            weight_std = first_item_actions.std().item()
            if weight_std < 0.05:
                print(f"⚠️  WARNING: Tanh weights have low diversity (std={weight_std:.6f})")
                print(f"   This may indicate the policy needs better exploration.")
            else:
                print(f"   ✅ Good diversity: std={weight_std:.6f} indicates varied allocations")
            
        return portfolio_weights
    
    def set_training_mode(self, mode: bool) -> None:
        """Set training mode for the actor (required by SB3)."""
        self.train(mode)


class NormalisedTD3Policy(TD3Policy):
    """
    ✅ COMPREHENSIVE FIX: TD3 policy with NormalisedActor to prevent plateau convergence.
    
    This custom policy completely overrides SB3's action handling to ensure our
    normalized portfolio weights are preserved during both training and inference.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: List[int] = [400, 300],
        features_dim: int = 300,
        **kwargs
    ):
        # We'll override the actor creation, so set basic parameters first
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        # Store architecture for custom actor
        self.net_arch = net_arch
        self.features_dim = features_dim
        
        print(f"🎯 NormalisedTD3Policy created:")
        print(f"   🔄 Uses NormalisedActor (portfolio weights)")
        print(f"   🚫 No normalization - natural learning (like MLP)")
        print(f"   ✅ Agent learns valid portfolio weights through reward signal")
    
    def make_actor(self, features_extractor: nn.Module = None) -> nn.Module:
        """Create custom NormalisedActor instead of standard TD3 actor."""
        
        # Get observation dimension from features extractor or directly
        if features_extractor is not None:
            obs_dim = features_extractor.features_dim
        else:
            if isinstance(self.observation_space, spaces.Dict):
                obs_dim = self.observation_space['observations'].shape[0]
            else:
                obs_dim = self.observation_space.shape[0]
        
        action_dim = self.action_space.shape[0]
        
        # Create normalized actor
        actor = NormalisedActor(
            observation_dim=obs_dim,
            action_dim=action_dim,
            net_arch=self.net_arch
        )
        
        print(f"✅ Custom NormalisedActor created: {obs_dim} → {action_dim}")
        return actor
    
    def _build(self, lr_schedule: Schedule) -> None:
        """Build the policy networks with custom actor."""
        
        # Create features extractor
        self.features_extractor = self.make_features_extractor()
        
        # Create custom normalized actor
        self.actor = self.make_actor(self.features_extractor)
        
        # Create standard critic
        self.critic = self.make_critic(self.features_extractor)
        
        # Create target networks
        self.actor_target = self.make_actor(self.features_extractor)
        self.critic_target = self.make_critic(self.features_extractor)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Create optimizers
        self.actor_optimizer = self.optimizer_class(
            self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        self.critic_optimizer = self.optimizer_class(
            self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        
        # ✅ CRITICAL FIX: Assign optimizer to actor so SB3 can find it
        # SB3's TD3 calls self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        # This line prevents AttributeError: 'NormalisedActor' object has no attribute 'optimizer'
        self.actor.optimizer = self.actor_optimizer
        self.critic.optimizer = self.critic_optimizer
        
        print(f"✅ NormalisedTD3Policy networks built successfully")
        print(f"✅ Optimizer assignment fix applied - prevents SB3 AttributeError")
    
    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        ✅ COMPREHENSIVE FIX: Override forward method used during training.
        
        This ensures our normalized actions are used during training data collection,
        not just during inference via _predict().
        """
        # Extract features using the correct SB3 method signature
        features = self.extract_features(observation, self.features_extractor)
        
        # Get normalized weights directly from our actor (no SB3 rescaling)
        action = self.actor(features)
        
        # Debug logging to trace training actions
        if torch.rand(1).item() < 0.005:  # Log 0.5% of training actions
            print(f"🔍 Training Action Debug:")
            print(f"   NormalisedActor output: {action.detach().cpu().numpy()}")
            print(f"   Action sum: {action.sum().item():.6f}")
        
        return action
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        ✅ CRITICAL FIX: Override SB3's action prediction to use our normalized weights directly.
        
        SB3's default predict applies Box(0,1) rescaling which breaks our normalized weights.
        This method bypasses that rescaling and returns the actor's softmax output directly.
        
        Note: Action noise is handled by the TD3 algorithm, not the policy.
        """
        # Extract features using the correct SB3 method signature
        features = self.extract_features(observation, self.features_extractor)
        
        # Get normalized weights directly from our actor (no SB3 rescaling)
        with torch.no_grad():
            action = self.actor(features)
            
            # Action noise is handled by the TD3 algorithm externally
            # Our actor already outputs proper portfolio weights via softmax
        
        # Debug logging to trace inference actions
        if torch.rand(1).item() < 0.01:  # Log 1% of inference actions
            print(f"🔍 Inference Action Debug:")
            print(f"   NormalisedActor output: {action.detach().cpu().numpy()}")
            print(f"   Action sum: {action.sum().item():.6f}")
        
        return action
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        ✅ COMPREHENSIVE FIX: Override the main predict method to ensure no rescaling.
        
        This is the public interface used by SB3 algorithms and external code.
        We need to ensure our normalized actions pass through unchanged.
        """
        # Convert observation to tensor
        if isinstance(observation, dict):
            observation_tensor = {}
            for key, value in observation.items():
                observation_tensor[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        else:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        
        # If single observation, add batch dimension
        if observation_tensor.ndim == 1 or (isinstance(observation_tensor, dict) and 
                                           next(iter(observation_tensor.values())).ndim == 1):
            if isinstance(observation_tensor, dict):
                observation_tensor = {key: value.unsqueeze(0) for key, value in observation_tensor.items()}
            else:
                observation_tensor = observation_tensor.unsqueeze(0)
            single_observation = True
        else:
            single_observation = False
        
        # Get action from our custom _predict method
        with torch.no_grad():
            action_tensor = self._predict(observation_tensor, deterministic=deterministic)
        
        # Convert back to numpy and remove batch dimension if needed
        action = action_tensor.cpu().numpy()
        if single_observation:
            action = action[0]
        
        # Debug logging for the final output
        if np.random.rand() < 0.01:  # Log 1% of final actions
            print(f"🔍 Final Action Debug:")
            print(f"   Predict output: {action}")
            print(f"   Action sum: {np.sum(action):.6f}")
            print(f"   Action type: {type(action)}")
        
        return action, state 


class NormalisedDDPGPolicy(TD3Policy):
    """
    ✅ COMPREHENSIVE FIX: DDPG policy with NormalisedActor to prevent plateau convergence.
    
    This custom policy completely overrides SB3's action handling to ensure our
    normalized portfolio weights are preserved during both training and inference.
    Inherits from TD3Policy as per SB3 design for DDPG.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: List[int] = [400, 300],
        features_dim: int = 300,
        **kwargs
    ):
        # We'll override the actor creation, so set basic parameters first
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        # Store architecture for custom actor
        self.net_arch = net_arch
        self.features_dim = features_dim
        
        print(f"🎯 NormalisedDDPGPolicy created:")
        print(f"   🔄 Uses NormalisedActor (portfolio weights)")
        print(f"   🚫 No normalization - natural learning (like MLP)")
        print(f"   ✅ Agent learns valid portfolio weights through reward signal")
    
    def make_actor(self, features_extractor: nn.Module = None) -> nn.Module:
        """Create custom NormalisedActor instead of standard DDPG actor."""
        
        # Get observation dimension from features extractor or directly
        if features_extractor is not None:
            obs_dim = features_extractor.features_dim
        else:
            if isinstance(self.observation_space, spaces.Dict):
                obs_dim = self.observation_space['observations'].shape[0]
            else:
                obs_dim = self.observation_space.shape[0]
        
        action_dim = self.action_space.shape[0]
        
        # Create normalized actor
        actor = NormalisedActor(
            observation_dim=obs_dim,
            action_dim=action_dim,
            net_arch=self.net_arch
        )
        
        print(f"✅ Custom NormalisedActor created for DDPG: {obs_dim} → {action_dim}")
        return actor
    
    def _build(self, lr_schedule: Schedule) -> None:
        """Build the policy networks with custom actor."""
        
        # Create features extractor
        self.features_extractor = self.make_features_extractor()
        
        # Create custom normalized actor
        self.actor = self.make_actor(self.features_extractor)
        
        # Create standard critic
        self.critic = self.make_critic(self.features_extractor)
        
        # Create target networks
        self.actor_target = self.make_actor(self.features_extractor)
        self.critic_target = self.make_critic(self.features_extractor)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Create optimizers
        self.actor_optimizer = self.optimizer_class(
            self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        self.critic_optimizer = self.optimizer_class(
            self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        
        # ✅ CRITICAL FIX: Assign optimizer to actor so SB3 can find it
        self.actor.optimizer = self.actor_optimizer
        self.critic.optimizer = self.critic_optimizer
        
        print(f"✅ NormalisedDDPGPolicy networks built successfully")
        print(f"✅ Optimizer assignment fix applied - prevents SB3 AttributeError")
    
    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        ✅ COMPREHENSIVE FIX: Override forward method used during training.
        """
        features = self.extract_features(observation, self.features_extractor)
        action = self.actor(features)
        
        if torch.rand(1).item() < 0.005:  # Log 0.5% of training actions
            print(f"🔍 DDPG Training Action Debug: Sum={action.sum().item():.6f}")
        
        return action
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        ✅ CRITICAL FIX: Override SB3's action prediction to use our normalized weights directly.
        """
        features = self.extract_features(observation, self.features_extractor)
        with torch.no_grad():
            action = self.actor(features)
        
        if torch.rand(1).item() < 0.01:  # Log 1% of inference actions
            print(f"🔍 DDPG Inference Action Debug: Sum={action.sum().item():.6f}")
        
        return action
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        ✅ COMPREHENSIVE FIX: Override the main predict method to ensure no rescaling.
        """
        if isinstance(observation, dict):
            observation_tensor = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in observation.items()}
        else:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)

        if observation_tensor.ndim == 1 or (isinstance(observation_tensor, dict) and next(iter(observation_tensor.values())).ndim == 1):
            if isinstance(observation_tensor, dict):
                observation_tensor = {k: v.unsqueeze(0) for k, v in observation_tensor.items()}
            else:
                observation_tensor = observation_tensor.unsqueeze(0)
            single_observation = True
        else:
            single_observation = False

        with torch.no_grad():
            action_tensor = self._predict(observation_tensor, deterministic=deterministic)

        action = action_tensor.cpu().numpy()
        if single_observation:
            action = action[0]

        if np.random.rand() < 0.01:
            print(f"🔍 DDPG Final Action Debug: Sum={np.sum(action):.6f}")
            
        return action, state 
