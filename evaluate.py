# %%
#!/usr/bin/env python3

"""
Evaluation script for RL portfolio allocation using Stable Baselines 3.
Compatible with attention-based policies and gymnasium environments.
"""

import numpy as np
import pandas as pd 
import os
import argparse
import hashlib
import pickle
import re
import random
import sys
import importlib
import matplotlib.pyplot as plt
import torch
from functools import lru_cache
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from typing import List, Dict, Any, Optional, Tuple
from attention_policy import (
    SimpleMlpTD3Policy, 
    SimpleMlpSACPolicy, 
    SimpleMlpDDPGPolicy, 
    SimpleMlpActorCriticPolicy,
    NormalisedTD3Policy,
    NormalisedDDPGPolicy,
    AttentionTD3Policy,
    AttentionSACPolicy,
    AttentionDDPGPolicy,
    AttentionActorCriticPolicy,
    DeepSetActor,
    # FlatStochasticDeepSetPolicy,
    # HierarchicalStochasticDeepSetPolicy,
    create_attention_policy
)

# from attention_policy_preog import (
#     AttentionTD3Policy
# )
# from custom_td3_actor_critic_batch import CustomTD3_AC
# from custom_td3_actor_critic_2crl import CustomTD3_AC
# from custom_td3_actor_critic_2crl_og import CustomTD3_AC
# from custom_td3_actor_critic_2crl_ogmid import CustomTD3_AC

# from custom_td3_actor_critic_2crl_safe import CustomTD3_AC
# from custom_td3_actor_critic_2crl_safe2 import CustomTD3_AC
# from custom_td3_actor_critic_2crl_safe3 import CustomTD3_AC
import attention_policy_preog
# from custom_td3_actor_critic_2crl_safe_pro import CustomTD3_AC
# from custom_td3_actor_critic_2crl_safe_pro_attention import CustomTD3_AC as CustomTD3_AC_Attention
from custom_td3_actor_critic_2crl_safe_pro_exp_attention import CustomTD3_AC as CustomTD3_AC_Attention

# from custom_td3_actor_critic_2crl_safe_phased import CustomTD3_AC 
# from custom_td3_actor_critic_2crl_safe_old_pre_og_past import CustomTD3_AC
# from custom_td3_actor_critic_2crl_safe_og import CustomTD3_AC


import warnings
warnings.filterwarnings('ignore')

import config
from enviorment import PortfolioEnv


def timestamp_as_utc(ts):
    """Normalize feed/env timestamps to UTC (config.data_set stores US/Pacific)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def episode_plot_datetime_index(timestamps):
    """Matplotlib x-axis: UTC wall clock so 13:00 UTC does not display as 06:00 Pacific."""
    return pd.DatetimeIndex([timestamp_as_utc(t) for t in timestamps])


def market_data_dates_for_plots(date_series):
    return pd.DatetimeIndex([timestamp_as_utc(t) for t in date_series])


def plot_calendar_date_label(timestamps):
    """mm/dd/yyyy of the UTC timestamps drawn on a plot. A multi-day span uses first–last."""
    if isinstance(timestamps, pd.DatetimeIndex):
        index = timestamps
    else:
        index = episode_plot_datetime_index(timestamps)
    if len(index) == 0:
        return ""
    unique_days = pd.DatetimeIndex(index).tz_convert("UTC").normalize().unique().sort_values()
    labels = [pd.Timestamp(day).strftime("%m/%d/%Y") for day in unique_days]
    if len(labels) == 1:
        return labels[0]
    return f"{labels[0]}–{labels[-1]}"


def title_with_plot_date(title, date_label):
    if not date_label:
        return title
    return f"{title} {date_label}"

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
RLKIT_ROOT = os.path.join(PROJECT_ROOT, "rlkit")
if RLKIT_ROOT not in sys.path:
    sys.path.insert(0, RLKIT_ROOT)


def get_rlkit_gaussian_policy_class():
    if RLKIT_ROOT not in sys.path:
        sys.path.insert(0, RLKIT_ROOT)
    return importlib.import_module("rlkit.torch.sac.policies").GaussianPolicy


def get_rlkit_attention_deepset_policy_class():
    if RLKIT_ROOT not in sys.path:
        sys.path.insert(0, RLKIT_ROOT)
    return importlib.import_module("rlkit.torch.sac.policies").AttentionDeepSetGaussianPolicy


def get_rlkit_attention_iql_policy_class():
    if RLKIT_ROOT not in sys.path:
        sys.path.insert(0, RLKIT_ROOT)
    return importlib.import_module(
        "rlkit.torch.sac.policies.attention_iql_policy"
    ).AttentionIQLGaussianPolicy


# Default IQL checkpoint: canonical recipe is hierarchical_stochastic_deepset
# + stochastic_log_prob (Beta/Dirichlet actor, IQL Eq.7 advantage-weighted log_prob).
# Override with --model if your run lives under a different output_dir.
# Keep this as None (or a single path string). An empty "(...)" with every path
# commented out becomes an empty tuple and breaks os.path.isdir downstream.
DEFAULT_IQL_CHECKPOINT = None
## "rlkit/data/04-28-iql-mapo-offline-conventional/04-28-iql-mapo-offline-conventional_2026_04_28_12_43_04_0000--s-25580/iql_checkpoint.pt"
# "rlkit/data/05-01-iql-mapo-offline-conventional/05-01-iql-mapo-offline-conventional_2026_05_01_16_24_19_0000--s-25580/iql_checkpoint.pt"
# "rlkit/data/05-01-iql-mapo-offline-conventional/05-01-iql-mapo-offline-conventional_2026_05_01_17_32_59_0000--s-25580/iql_checkpoint.pt"
# "rlkit/data/05-02-iql-mapo-ad-offline-attentiondeepset/05-02-iql-mapo-ad-offline-attentiondeepset_2026_05_02_00_10_10_0000--s-25580/iql_attention_deepset_checkpoint.pt"
# "rlkit/data/05-09-iql-mapo-ad-offline-attentiondeepset/05-09-iql-mapo-ad-offline-attentiondeepset_2026_05_09_23_50_21_0000--s-25580/model_250000_steps.pt"
# "rlkit/data/05-11-iql-mapo-ad-offline-attentiondeepset/05-11-iql-mapo-ad-offline-attentiondeepset_2026_05_11_17_40_34_0000--s-25580/iql_attention_deepset_checkpoint.pt"
# "rlkit/data/05-11-iql-mapo-ad-offline-attentiondeepset/05-11-iql-mapo-ad-offline-attentiondeepset_2026_05_11_17_56_53_0000--s-25580/iql_attention_deepset_checkpoint.pt"
# "rlkit/data/05-11-iql-mapo-ad-offline-attentiondeepset/05-11-iql-mapo-ad-offline-attentiondeepset_2026_05_11_21_56_10_0000--s-25580/iql_attention_deepset_checkpoint.pt"
# "rlkit/data/05-12-iql-mapo-ad-offline-attentiondeepset/05-12-iql-mapo-ad-offline-attentiondeepset_2026_05_12_03_23_21_0000--s-25580/iql_attention_deepset_checkpoint.pt"
# "rlkit/data/05-12-iql-mapo-ad-offline-attentiondeepset/05-12-iql-mapo-ad-offline-attentiondeepset_2026_05_12_17_32_30_0000--s-25580/iql_attention_deepset_checkpoint.pt"
# "rlkit/data/05-17-iql-mapo-ad-canonical-replay-only/05-17-iql-mapo-ad-canonical-replay-only_2026_05_17_16_31_32_0000--s-25580/iql_attention_deepset_canonical_replay_only_checkpoint.pt"
# "rlkit/data/05-12-iql-mapo-ad-offline-attentiondeepset/05-12-iql-mapo-ad-offline-attentiondeepset_2026_05_12_17_32_30_0000--s-25580/iql_attention_deepset_checkpoint.pt"
# "rlkit/data/05-20-iql-mapo-ad-v4-attentiondeepset/05-20-iql-mapo-ad-v4-attentiondeepset_2026_05_20_18_51_15_0000--s-25580/iql_attention_deepset_checkpoint.pt"
# "rlkit/data/05-20-iql-mapo-ad-offline/05-20-iql-mapo-ad-offline_2026_05_20_22_30_49_0000--s-25580/iql_attention_deepset_checkpoint.pt"
# "rlkit/data/05-22-iql-mapo-ad-offline/05-22-iql-mapo-ad-offline_2026_05_22_16_52_56_0000--s-25580/iql_attention_deepset_checkpoint.pt"
# "rlkit/data/05-29-iql-mapo-ad-final/05-29-iql-mapo-ad-final_2026_05_29_08_21_17_0000--s-25580/iql_attention_final_checkpoint.pt"
# To re-enable IQL defaults, set DEFAULT_IQL_CHECKPOINT = "rlkit/data/.../checkpoint.pt"
# above (single string), not an empty parenthesized block.
DEFAULT_TD3_SOURCE_CHECKPOINT = (
    # "models/TD3_tf_agents_style_medium_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_20260427_125830_baseline/checkpoints/model_2600000_steps.zip"
    # "models/TD3_tf_agents_style_medium_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_20260805_170415_baseline/checkpoints/model_50000_steps.zip"
    
    
    # "models/TD3_tf_agents_style_multitask_STRUCTURED_CREDIT_variable_coin_attention_fixed8of13_CRL_adaptive_proexp_disagreement_20260814_222043_baseline/best_model.zip"

    # "models/TD3_tf_agents_style_multitask_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_earnmore_global13_select8_whole_stock_pool_task_conditioned_critic_centered_advantage_bellman_popart_xbr_actor_real_20260928_201154_baseline/best_model.zip"
    # "models/TD3_tf_agents_style_multitask_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_earnmore_global13_select8_whole_stock_pool_task_conditioned_critic_centered_advantage_bellman_popart_xbr_actor_real_20260928_201154_baseline/checkpoints/model_1150000_steps.zip"

    # Each uncommented path is one evaluation. Commas are required: without them
    # Python concatenates the strings into a single path and only one run starts.
    "models/TD3_tf_agents_style_multitask_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_basket_history_g966_norecon_20261013_215231_baseline/checkpoints/model_500000_steps.zip",#-2.62
    "models/TD3_tf_agents_style_multitask_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_basket_history_g966_norecon_20261013_215231_baseline/checkpoints/model_1400000_steps.zip",#-3.19
    "models/TD3_tf_agents_style_multitask_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_basket_history_g966_norecon_20261013_215231_baseline/checkpoints/model_1500000_steps.zip",#-2.64
    "models/TD3_tf_agents_style_multitask_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_basket_history_g966_norecon_20261013_215231_baseline/checkpoints/model_1650000_steps.zip",#-3.28
    "models/TD3_tf_agents_style_multitask_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_basket_history_g966_norecon_20261013_215231_baseline/checkpoints/model_1850000_steps.zip",#-4.20 #9/25/26
)
MAPO_IQL_CHECKPOINT_FORMAT = "rlkit_mapo_iql_checkpoint"
MAPO_AD_IQL_CHECKPOINT_FORMAT = "rlkit_mapo_attention_deepset_iql_checkpoint"
MAPO_AD_FINAL_IQL_CHECKPOINT_FORMAT = "rlkit_mapo_attention_iql_final_checkpoint"
MAPO_DEFAULT_OBSERVATION_KEYS = (
    "observations",
    "asset_ids",
    "mask",
    "prev_action",
    "prev_mask",
)


@lru_cache(maxsize=32)
def load_supported_pt_checkpoint(model_path: str) -> Optional[Dict[str, Any]]:
    normalized_path = os.path.abspath(str(model_path))
    if not normalized_path.lower().endswith(".pt"):
        return None
    if not os.path.exists(normalized_path):
        return None
    try:
        checkpoint = torch.load(normalized_path, map_location="cpu")
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            return None
        try:
            checkpoint = torch.load(
                normalized_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception:
            return None
    except Exception:
        return None
    return checkpoint if isinstance(checkpoint, dict) else None


def deployed_policy_value_estimate(model, observation, action):
    """Return mean/min/max Q at the executed action, without fitting a model.

    For deterministic execution, Q(s, pi(s)) is a policy state-value proxy.
    These TD3 estimates retain their bootstrapped/ProExp training semantics;
    they are not RECAP's separately trained Monte Carlo state-value model.
    For stochastic execution this is a sampled-action Q, not E_a[Q(s, a)].
    """
    critic = getattr(model, "critic", None)
    policy = getattr(model, "policy", None)
    if critic is None or policy is None or not callable(critic):
        return float("nan"), float("nan"), float("nan")
    if getattr(deployed_policy_value_estimate, "_unavailable", False):
        return float("nan"), float("nan"), float("nan")
    try:
        policy.set_training_mode(False)
        obs_tensor, _vectorized = policy.obs_to_tensor(observation)
        if isinstance(obs_tensor, dict):
            obs_tensor = {
                key: value.float() if torch.is_floating_point(value) else value
                for key, value in obs_tensor.items()
            }
        elif torch.is_tensor(obs_tensor):
            obs_tensor = obs_tensor.float()
        gate = getattr(model, "_gate_observations_with_sigma", None)
        if callable(gate):
            obs_tensor = gate(obs_tensor)
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.ndim == 1:
            action_array = action_array.reshape(1, -1)
        scaled_action = policy.scale_action(action_array)
        action_tensor = torch.as_tensor(
            scaled_action, device=model.device, dtype=torch.float32
        )
        with torch.no_grad():
            q_values = critic(obs_tensor, action_tensor)
            if isinstance(q_values, torch.Tensor):
                stacked = q_values.reshape(q_values.shape[0], -1)
            else:
                stacked = torch.cat(
                    [value.reshape(value.shape[0], -1) for value in q_values],
                    dim=1,
                )
            q_min = stacked.min(dim=1).values
            q_max = stacked.max(dim=1).values
            q_mean = stacked.mean(dim=1)
        return float(q_mean[0].item()), float(q_min[0].item()), float(q_max[0].item())
    except Exception as exc:
        deployed_policy_value_estimate._unavailable = True
        print(f"  Critic score unavailable for this model: {exc}")
        return float("nan"), float("nan"), float("nan")


def deployed_action_critic_score(model, observation, action):
    """Compatibility view of the deployed action's lower Q and critic spread."""
    _, q_min, q_max = deployed_policy_value_estimate(model, observation, action)
    return q_min, q_max - q_min


def episode_value_diagnostics(episode_data):
    """Align future-return targets and existing value proxies to decision states.

    rewards[0] is the reset placeholder; rewards[t + 1] follows action[t].
    RECAP's empirical target sums the subsequent rewards without discounting.
    A second target uses the checkpoint's gamma for comparison with TD3.
    Both targets use hindsight and must never become policy observations.
    Only genuine termination establishes a zero-valued absorbing continuation;
    truncated windows have an unknown value beyond their observed boundary.
    """
    rewards = np.asarray(episode_data["rewards"], dtype=np.float64)
    state_count = len(episode_data["timestamps"])
    if rewards.ndim != 1 or rewards.size != state_count or state_count == 0:
        raise ValueError("Value diagnostics need one reward entry per state, including reset")
    if not np.isfinite(rewards).all():
        raise ValueError("Value diagnostics need finite observed rewards")
    gamma = float(episode_data.get("discount_factor", 1.0))
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("The value comparison discount factor must be between 0 and 1")

    steps = state_count - 1
    returns = np.zeros(state_count, dtype=np.float64)
    discounted_returns = np.zeros(state_count, dtype=np.float64)
    for t in range(steps - 1, -1, -1):
        returns[t] = rewards[t + 1] + returns[t + 1]
        discounted_returns[t] = rewards[t + 1] + gamma * discounted_returns[t + 1]

    def aligned_scores(key):
        recorded = np.asarray(episode_data.get(key, []), dtype=np.float64)
        if recorded.ndim != 1 or recorded.size > steps:
            raise ValueError(f"{key} must contain at most one score per executed decision")
        result = np.full(state_count, np.nan, dtype=np.float64)
        result[:recorded.size] = recorded
        result[~np.isfinite(result)] = np.nan
        return result

    value = aligned_scores("critic_q_mean")
    low = aligned_scores("critic_q_min")
    gap = aligned_scores("critic_q_gap")
    # Old episode dictionaries recorded precisely two critics, min and gap.
    if "critic_q_mean" not in episode_data:
        value = low + 0.5 * gap
    terminal_known = bool(episode_data.get("terminated", False))
    has_value = bool(np.isfinite(value[:steps]).any())
    if terminal_known and has_value:
        value[-1] = 0.0
    continuation = np.zeros(steps) if gamma == 0.0 else gamma * value[1:]
    td_residual = rewards[1:] + continuation - value[:-1]
    return {
        "return_to_go": returns,
        "discounted_return_to_go": discounted_returns,
        "policy_value": value,
        "critic_low": low,
        "critic_gap": gap,
        "td_residual": td_residual,
        "discount_factor": gamma,
        "has_value_estimate": has_value,
        "terminal_value_known": terminal_known,
    }


def evaluate_policy_sb3(
    model,
    env,
    num_episodes=1,
    deterministic=True,
    render=False,
    evaluation_seed=42,
    action_mode="auto",
):
    """
    Evaluate a Stable Baselines 3 policy and collect detailed episode data.
    Compatible with both fixed and variable portfolio environments.
    
    Args:
        model: Trained SB3 model
        env: Portfolio environment
        num_episodes: Number of episodes to run
        deterministic: Use deterministic actions
        render: Whether to render the environment
        evaluation_seed: Fixed seed for reproducible data segment selection (default: 42)
    
    Returns:
        List of episode data dictionaries
    """
    all_episode_data = []
    deployed_policy_value_estimate._unavailable = False

    for episode in range(num_episodes):
        print(f"Running evaluation episode {episode + 1}/{num_episodes}...")
        
        # Reset environment with the same numpy seed and Gym seed for every model.
        # The environment still uses np.random internally for the starting portfolio
        # and Python random for coin subset order, so both seeds matter.
        episode_seed = int(evaluation_seed) + episode
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)
        obs, info = env.reset(seed=episode_seed)

        
        # Detect portfolio type and extract episode coins
        is_variable_portfolio = hasattr(env, 'use_variable_portfolio') and env.use_variable_portfolio
        episode_coins = info.get('episode_coins', config.COINS) if is_variable_portfolio else config.COINS
        normalized_action_mode = str(action_mode).strip().lower()
        if hasattr(model, "resolve_deterministic_flag"):
            effective_deterministic = model.resolve_deterministic_flag(action_mode)
        elif normalized_action_mode == "stochastic":
            effective_deterministic = False
        elif normalized_action_mode == "deterministic":
            effective_deterministic = True
        else:
            effective_deterministic = bool(deterministic)
        effective_action_mode = "deterministic" if effective_deterministic else "stochastic"
        
        print(f"  Episode coins: {episode_coins} ({'variable' if is_variable_portfolio else 'fixed'} portfolio)")
        print(f"  Action mode: requested={action_mode}, effective={effective_action_mode}")
        
        episode_data = {
            "timestamps": [],
            "values": [],
            "allocations": [],
            "market_data": [],
            "rewards": [],
            "actions": [],
            "observations": [],
            "episode_coins": episode_coins,  # Store episode-specific coins
            "is_variable_portfolio": is_variable_portfolio,
            "action_mode": effective_action_mode,
            "action_normalization": getattr(env, "action_normalization", "hierarchical"),
            "discount_factor": float(getattr(model, "gamma", 1.0)),
            "critic_q_mean": [],
            "critic_q_min": [],
            "critic_q_gap": [],
        }
        
        # Store initial state
        episode_data["timestamps"].append(timestamp_as_utc(env.current_time))
        episode_data["values"].append(env.current_value)
        episode_data["allocations"].append(env.money_split_ratio.copy())
        episode_data["market_data"].append(env.dfslice.copy())
        episode_data["rewards"].append(0.0)  # Initial reward is 0
        
        # Handle initial action based on portfolio type
        if is_variable_portfolio:
            initial_action = [0.0] * (config.MAX_COINS + 1)
        else:
            initial_action = [0.0] * (len(config.COINS) + 1)
        episode_data["actions"].append(initial_action)
        episode_data["observations"].append(obs.copy())
        
        terminated = truncated = False
        step = 0
        
        while not (terminated or truncated):
            # Get action from the model, then ask the twin critics what future
            # that same action is worth before the environment advances.
            action, _ = model.predict(obs, deterministic=effective_deterministic)
            q_mean, q_min, q_max = deployed_policy_value_estimate(model, obs, action)
            episode_data["critic_q_mean"].append(q_mean)
            episode_data["critic_q_min"].append(q_min)
            episode_data["critic_q_gap"].append(q_max - q_min)

            # Apply action to environment
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # Store step data
            episode_data["timestamps"].append(timestamp_as_utc(env.current_time))
            episode_data["values"].append(env.current_value)
            episode_data["allocations"].append(env.money_split_ratio.copy())
            episode_data["market_data"].append(env.dfslice.copy())
            episode_data["rewards"].append(reward)
            episode_data["actions"].append(action.copy())
            episode_data["observations"].append(obs.copy())
            
            if render:
                env.render()
                
            # Progress indicator
            if step % 100 == 0:
                print(f"  Step {step}, Portfolio Value: ${env.current_value:.2f}, Reward: {reward:.4f}")
        
        episode_data["terminated"] = bool(terminated)
        episode_data["truncated"] = bool(truncated)

        # Consolidate market data into a single DataFrame
        market_data_list = episode_data["market_data"]
        if market_data_list:
            episode_data["market_data"] = pd.concat(market_data_list, ignore_index=True)
        else:
            episode_data["market_data"] = pd.DataFrame()
        
        # Calculate episode statistics
        total_return = episode_data["values"][-1] - episode_data["values"][0]
        percent_return = (total_return / episode_data["values"][0]) * 100
        total_reward = sum(episode_data["rewards"])
        
        print(f"  Episode {episode + 1} completed:")
        print(f"    Steps: {step}")
        print(f"    Total Return: ${total_return:.2f} ({percent_return:.2f}%)")
        print(f"    Total Reward: {total_reward:.4f}")
        print(f"    Final Portfolio Value: ${episode_data['values'][-1]:.2f}")
        finite_q = [
            value for value in episode_data["critic_q_min"] if np.isfinite(value)
        ]
        if finite_q:
            print(
                f"    Conservative Q: mean={np.mean(finite_q):.4f} "
                f"min={np.min(finite_q):.4f} max={np.max(finite_q):.4f}"
            )
        
        all_episode_data.append(episode_data)

    return all_episode_data


def extract_meaningful_allocations(episode_data):
    """
    Extract meaningful allocation data from episode data, handling both fixed and variable portfolios.
    
    Args:
        episode_data: Episode data dictionary
        
    Returns:
        tuple: (allocations_df, episode_coins)
    """
    is_variable_portfolio = episode_data.get("is_variable_portfolio", False)
    episode_coins = episode_data.get("episode_coins", config.COINS)
    allocations_raw = episode_data["allocations"]
    timestamps = episode_data["timestamps"]
    
    datetime_index = episode_plot_datetime_index(timestamps)
    
    if is_variable_portfolio:
        # For variable portfolio: extract only meaningful allocations
        meaningful_allocations = []
        
        for allocation_array in allocations_raw:
            # allocation_array has shape (MAX_COINS + 1,) = (6,)
            # [cash, coin1, coin2, coin3, coin4, coin5]
            # But only cash + len(episode_coins) are meaningful
            
            meaningful_allocation = [allocation_array[0]]  # Cash
            
            # Add allocations for active coins
            for i in range(len(episode_coins)):
                meaningful_allocation.append(allocation_array[i + 1])
            
            meaningful_allocations.append(meaningful_allocation)
        
        # Create column names
        column_names = ["Cash"] + episode_coins
        
        # Create DataFrame
        allocations_df = pd.DataFrame(
            meaningful_allocations,
            columns=column_names,
            index=datetime_index
        )
        
        print(f"  📊 Extracted {len(episode_coins)} coin allocations: {episode_coins}")
        
    else:
        # For fixed portfolio: use traditional approach
        allocations_df = pd.DataFrame(
            allocations_raw,
            columns=["Cash"] + config.COINS,
            index=datetime_index
        )
        
        print(f"  📊 Using fixed allocations: {config.COINS}")
    
    return allocations_df, episode_coins


def extract_meaningful_actions(episode_data):
    """
    Extract meaningful action data from episode data, handling both fixed and variable portfolios.
    
    Args:
        episode_data: Episode data dictionary
        
    Returns:
        DataFrame: Actions dataframe with appropriate columns
    """
    is_variable_portfolio = episode_data.get("is_variable_portfolio", False)
    episode_coins = episode_data.get("episode_coins", config.COINS)
    actions_raw = episode_data["actions"]
    timestamps = episode_data["timestamps"]
    
    datetime_index = episode_plot_datetime_index(timestamps)
    
    if is_variable_portfolio:
        # For variable portfolio: extract only meaningful actions
        meaningful_actions = []
        
        for action_array in actions_raw:
            action_array = _episode_order_policy_action(action_array, episode_coins)
            # action_array has shape (MAX_COINS + 1,) = (6,)
            # But only cash + len(episode_coins) are meaningful
            
            meaningful_action = [action_array[0]]  # Cash action
            
            # Add actions for active coins
            for i in range(len(episode_coins)):
                if i + 1 < len(action_array):
                    meaningful_action.append(action_array[i + 1])
                else:
                    meaningful_action.append(0.0)  # Padding if needed
            
            meaningful_actions.append(meaningful_action)
        
        # Create column names
        column_names = ["Cash"] + episode_coins
        
        # Create DataFrame
        actions_df = pd.DataFrame(
            meaningful_actions,
            columns=column_names,
            index=datetime_index
        )
        
    else:
        # For fixed portfolio: use traditional approach
        actions_df = pd.DataFrame(
            actions_raw,
            columns=["Cash"] + config.COINS,
            index=datetime_index
        )
    
    return actions_df


def _episode_order_policy_action(action_array, episode_coins):
    """Global policy coordinates -> compact episode order used by accounting plots."""
    raw = np.asarray(action_array, dtype=np.float32).reshape(-1)
    if config.EARNMORE_GLOBAL_POOL_ENABLED and raw.size == len(config.COIN_POOL) + 1:
        indices = [0] + [config.TICKER_TO_ID[coin] + 1 for coin in episode_coins]
        return raw[indices]
    return raw


def _normalize_raw_action_to_target_allocation(
    action_array,
    active_asset_count,
    action_normalization="hierarchical",
):
    """
    Apply the same hierarchical cash-gated normalization used by the environment.

    Fig. 1 variable meanings:
    raw action = policy output in [-1, 1]
    cash gate = first raw coordinate mapped to [0, 1]
    asset sleeve = remaining active asset coordinates normalized relative to one another
    target allocation = executed portfolio weights implied by the raw action before price drift
    """
    active_asset_count = int(active_asset_count)
    active_width = active_asset_count + 1
    raw = np.asarray(action_array, dtype=np.float32).reshape(-1)
    active_raw = raw[:active_width]
    if action_normalization == "flowrl_joint_simplex_v1":
        shifted = (np.clip(active_raw, -1.0, 1.0) + 1.0) * 0.5
        total = shifted.sum()
        return shifted / total if total > 1e-8 else np.full(active_width, 1.0 / active_width, dtype=np.float32)
    if action_normalization != "hierarchical":
        raise ValueError(f"Unknown action normalization: {action_normalization}")

    cash_logit = float(active_raw[0]) if active_raw.size > 0 else -1.0
    asset_logits = active_raw[1:]

    cash_weight = np.clip((cash_logit + 1.0) * 0.5, 0.0, 1.0)
    if active_asset_count <= 0:
        return np.asarray([cash_weight], dtype=np.float32)

    asset_shifted = np.clip((asset_logits + 1.0) * 0.5, a_min=0.0, a_max=None)
    asset_sum = float(np.sum(asset_shifted))
    if asset_sum > 1e-8:
        asset_weights_relative = asset_shifted / asset_sum
    else:
        asset_weights_relative = np.ones(active_asset_count, dtype=np.float32) / float(active_asset_count)

    equity_allocation = 1.0 - cash_weight
    final_asset_weights = asset_weights_relative * equity_allocation
    return np.concatenate(([cash_weight], final_asset_weights)).astype(np.float32, copy=False)


def extract_meaningful_target_allocations(episode_data):
    """
    Convert the stored raw actions into the target allocations implied at execution time.

    This is the clean bridge between the raw action plot and the environment's
    hierarchical execution contract. It lets us distinguish:

    1. the target allocation implied by the policy's raw action at time t
    2. the actual allocation recorded after the environment advances and prices move
    """
    actions_raw = episode_data["actions"]
    timestamps = episode_data["timestamps"]
    is_variable_portfolio = episode_data.get("is_variable_portfolio", False)
    episode_coins = episode_data.get("episode_coins", config.COINS)
    datetime_index = episode_plot_datetime_index(timestamps)

    if is_variable_portfolio:
        active_asset_count = len(episode_coins)
        column_names = ["Cash"] + episode_coins
        target_allocations = [
            _normalize_raw_action_to_target_allocation(
                _episode_order_policy_action(action_array, episode_coins), active_asset_count,
                episode_data.get("action_normalization", "hierarchical"),
            )
            for action_array in actions_raw
        ]
        return pd.DataFrame(target_allocations, columns=column_names, index=datetime_index)

    active_asset_count = len(config.COINS)
    column_names = ["Cash"] + config.COINS
    target_allocations = [
        _normalize_raw_action_to_target_allocation(action_array, active_asset_count,
                                                 episode_data.get("action_normalization", "hierarchical"))
        for action_array in actions_raw
    ]
    return pd.DataFrame(target_allocations, columns=column_names, index=datetime_index)


# A single macOS path component cannot exceed 255 bytes (Errno 63).
# The longest suffix appended after this prefix is "_target_allocation_strategy.png".
_LONGEST_PLOT_SUFFIX_LEN = len("_target_allocation_strategy.png")
_MAX_PLOT_PREFIX_LEN = 255 - _LONGEST_PLOT_SUFFIX_LEN - 1


def safe_plot_prefix(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_")
    cleaned = cleaned or "evaluation"
    if len(cleaned.encode("utf-8")) <= _MAX_PLOT_PREFIX_LEN:
        return cleaned

    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:10]
    tail_len = 72
    head_len = _MAX_PLOT_PREFIX_LEN - len(digest) - tail_len - 2
    head = cleaned[:head_len].rstrip("_.-")
    tail = cleaned[-tail_len:].lstrip("_.-")
    shortened = f"{head}_{digest}_{tail}"
    if len(shortened.encode("utf-8")) > _MAX_PLOT_PREFIX_LEN:
        shortened = shortened.encode("utf-8")[:_MAX_PLOT_PREFIX_LEN].decode("utf-8", errors="ignore").rstrip("_.-")
    return shortened or "evaluation"


LIVE_EVAL_PNG_NAME = "live_eval_latest.png"


def create_live_eval_env_for_model(model_path: str):
    """PortfolioEnv on enviorment_live so config.data_set appends are visible to the policy."""
    from enviorment_live import create_portfolio_env_for_live_model

    reward_type, use_variable_portfolio, use_temporal_attention = infer_environment_settings(model_path)
    eval_env = create_portfolio_env_for_live_model(
        model_path,
        reward_type=reward_type,
        use_variable_portfolio=use_variable_portfolio,
        use_temporal_attention=use_temporal_attention,
    )
    print(
        f"✅ Live eval env (enviorment_live): reward_type={reward_type}, "
        f"variable_portfolio={use_variable_portfolio}, temporal={use_temporal_attention}"
    )
    return eval_env


def new_live_evaluation_history(env, obs, info):
    is_variable_portfolio = hasattr(env, "use_variable_portfolio") and env.use_variable_portfolio
    episode_coins = info.get("episode_coins", config.COINS) if is_variable_portfolio else config.COINS
    if is_variable_portfolio:
        initial_action = [0.0] * (config.MAX_COINS + 1)
    else:
        initial_action = [0.0] * (len(config.COINS) + 1)
    obs_copy = (
        {k: np.asarray(v).copy() for k, v in obs.items()}
        if isinstance(obs, dict)
        else np.asarray(obs).copy()
    )
    return {
        "timestamps": [timestamp_as_utc(env.current_time)],
        "values": [float(env.current_value)],
        "allocations": [env.money_split_ratio.copy()],
        "market_data": [env.dfslice.copy()],
        "rewards": [0.0],
        "actions": [initial_action],
        "observations": [obs_copy],
        "episode_coins": list(episode_coins),
        "is_variable_portfolio": is_variable_portfolio,
        "action_mode": "deterministic",
    }


def append_live_evaluation_step(history, env, obs, action, reward, info):
    obs_copy = (
        {k: np.asarray(v).copy() for k, v in obs.items()}
        if isinstance(obs, dict)
        else np.asarray(obs).copy()
    )
    action_copy = np.asarray(action, dtype=np.float32).copy()
    history["timestamps"].append(timestamp_as_utc(env.current_time))
    history["values"].append(float(env.current_value))
    history["allocations"].append(env.money_split_ratio.copy())
    history["market_data"].append(env.dfslice.copy())
    history["rewards"].append(float(reward))
    history["actions"].append(action_copy)
    history["observations"].append(obs_copy)


def run_live_policy_steps(env, model, history, n_steps, deterministic=True):
    """
    predict + step like evaluate_policy_sb3; append each step to live history.
    Returns number of steps executed.
    """
    if n_steps <= 0:
        return 0
    obs = env.get_policy_observation()
    executed = 0
    for _ in range(int(n_steps)):
        if env.time_index >= len(env.unique_times) - 1:
            break
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        append_live_evaluation_step(history, env, obs, action, reward, info)
        executed += 1
        if terminated or truncated:
            break
    return executed


def update_live_evaluation_dashboard(
    history,
    output_path=None,
    plot_prefix="live_eval",
):
    """
    Single PNG dashboard (overwritten each refresh): portfolio, allocations, per-asset OHLC.
    Mirrors evaluate.py panels in one figure for live monitoring.
    """
    if not history or len(history.get("timestamps", [])) < 1:
        return None

    output_path = output_path or os.path.join(config.LOGDIR, LIVE_EVAL_PNG_NAME)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    timestamps = history["timestamps"]
    portfolio_values = history["values"]
    rewards = history["rewards"]
    episode_coins = list(history.get("episode_coins", config.COINS))
    n_coins = max(1, len(episode_coins))

    datetime_index = episode_plot_datetime_index(timestamps)
    date_label = plot_calendar_date_label(datetime_index)
    portfolio_value_series = pd.Series(portfolio_values, index=datetime_index)
    rewards_series = pd.Series(rewards, index=datetime_index)

    episode_data = dict(history)
    if isinstance(episode_data.get("latest_market_data"), pd.DataFrame):
        episode_data["market_data"] = episode_data["latest_market_data"].copy()
    elif episode_data.get("market_data") and isinstance(episode_data["market_data"][0], pd.DataFrame):
        episode_data["market_data"] = pd.concat(episode_data["market_data"], ignore_index=True)
    else:
        episode_data["market_data"] = pd.DataFrame()

    target_allocations_df = extract_meaningful_target_allocations(episode_data)
    allocations_df, _episode_coins = extract_meaningful_allocations(episode_data)
    market_data_df = episode_data.get("market_data", pd.DataFrame())
    if isinstance(market_data_df, pd.DataFrame) and not market_data_df.empty:
        market_data_df = market_data_df.copy()
        market_data_df["date"] = market_data_dates_for_plots(market_data_df["date"])

    n_rows = 2 + n_coins
    fig_h = 2.8 + 2.2 * n_coins
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, fig_h), sharex=False)
    if n_rows == 1:
        axes = [axes]

    ax_pv, ax_alloc = axes[0], axes[1]
    portfolio_value_series.plot(ax=ax_pv, color="blue", linewidth=2, label="Portfolio value")
    ax_pv.set_ylabel("Portfolio value ($)")
    ax_pv.set_title(title_with_plot_date("Live session — portfolio value (UTC, enviorment_live + feed)", date_label))
    ax_pv.grid(True, linestyle="--", alpha=0.5)
    ax_pv.legend(loc="upper left")
    if len(portfolio_values) >= 2 and portfolio_values[0] > 0:
        pct = ((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]) * 100.0
        ax_pv.text(
            0.02,
            0.95,
            f"Points: {len(portfolio_values)} | Return: {pct:.2f}%",
            transform=ax_pv.transAxes,
            bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.7),
        )

    cumulative_rewards = pd.Series(np.cumsum(rewards), index=datetime_index)
    ax_reward = ax_pv.twinx()
    cumulative_rewards.plot(
        ax=ax_reward, color="red", linewidth=1.5, alpha=0.75, label="Cum. reward"
    )
    ax_reward.set_ylabel("Cumulative reward", color="red")
    ax_reward.tick_params(axis="y", labelcolor="red")

    target_allocations_df.plot.area(ax=ax_alloc, stacked=True, linewidth=0, alpha=0.85)
    ax_alloc.set_ylim(0, 1)
    ax_alloc.set_ylabel("Target allocation")
    ax_alloc.set_title(title_with_plot_date("Live session — TD3 target allocation (evaluate.py normalization)", date_label))
    ax_alloc.grid(True, linestyle="--", alpha=0.5)
    ax_alloc.legend(title="Assets", loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)

    for ax, coin in zip(axes[2:], episode_coins):
        if market_data_df.empty or "coin" not in market_data_df.columns:
            ax.set_title(title_with_plot_date(f"{coin} (no market rows)", date_label))
            ax.axis("off")
            continue
        coin_data = market_data_df[market_data_df["coin"] == coin].set_index("date")
        if coin_data.empty:
            ax.set_title(title_with_plot_date(f"{coin} (empty)", date_label))
            continue
        plot_cols = [c for c in ["open", "high", "low", "close"] if c in coin_data.columns]
        if plot_cols:
            coin_data[plot_cols].plot(ax=ax, linewidth=1.2, alpha=0.85)
        ax.set_title(title_with_plot_date(f"{coin} — OHLC + allocation + cum. return", date_label))
        ax.set_ylabel("Price ($)")
        ax.grid(True, linestyle="--", alpha=0.4)

        if coin in allocations_df.columns:
            ax2 = ax.twinx()
            allocations_df[coin].plot(ax=ax2, color="red", linewidth=2, alpha=0.7)
            ax2.set_ylabel(f"{coin} alloc", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.set_ylim(0, 1)

        if "close" in coin_data.columns and len(coin_data) >= 2:
            initial_price = float(coin_data["close"].iloc[0])
            if initial_price > 0:
                cum_ret = (coin_data["close"] / initial_price - 1.0) * 100.0
                ax3 = ax.twinx()
                ax3.spines["left"].set_position(("outward", 52))
                ax3.spines["left"].set_visible(True)
                ax3.spines["right"].set_visible(False)
                ax3.yaxis.set_ticks_position("left")
                ax3.yaxis.set_label_position("left")
                cum_ret.plot(ax=ax3, color="purple", linestyle=":", linewidth=2.0)
                ax3.set_ylabel("Cum. return %", color="purple")
                ax3.tick_params(axis="y", labelcolor="purple")
                ax3.axhline(0, color="purple", linestyle="--", linewidth=0.8, alpha=0.6)

    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(
        title_with_plot_date(
            f"Live TD3 dashboard (UTC) — {len(portfolio_values)} point(s), "
            f"{len(datetime_index.unique())} unique minute(s)",
            date_label,
        ),
        fontsize=13,
        y=0.995,
    )
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(
        f"📈 Live eval dashboard saved: {output_path} "
        f"({len(portfolio_values)} steps, {n_coins} asset panels)"
    )
    return output_path


def plot_episode_value_function(
    episode_data,
    save_plots=True,
    plot_prefix="evaluation",
    show_plots=True,
):
    """Show RECAP's empirical target and distinguish the available TD3 proxy.

    RECAP IV-A trains a separate V on undiscounted empirical reward-to-go.
    Existing checkpoints do not contain that model. Their deployed-action Q
    estimates are compared with gamma-matched realized returns, not relabeled
    as a RECAP model. No success reward, normalization, or smoothing is added.
    """
    diagnostics = episode_value_diagnostics(episode_data)
    index = episode_plot_datetime_index(episode_data["timestamps"])
    date_label = plot_calendar_date_label(index)
    has_value = diagnostics["has_value_estimate"]
    rows = 4 if has_value else 1
    fig, axes = plt.subplots(rows, 1, figsize=(15, 3.8 * rows), sharex=True, squeeze=False)
    axes = axes[:, 0]

    pd.Series(diagnostics["return_to_go"], index=index).plot(
        ax=axes[0], color="darkgreen", linewidth=2,
        label="Realized reward-to-go (hindsight; gamma=1)",
    )
    axes[0].set_title(title_with_plot_date(
        "Remaining Reward: Monte Carlo Target Used by RECAP (Not a Learned Prediction)",
        date_label,
    ))
    axes[0].set_ylabel("Undiscounted reward")

    if has_value:
        gamma = diagnostics["discount_factor"]
        stochastic = episode_data.get("action_mode") == "stochastic"
        proxy_label = (
            "Sampled-action Q proxy (critic mean)" if stochastic
            else "V_pi proxy: mean Q(s, pi(s))"
        )
        # The terminal zero is a boundary condition used in the residual, not
        # an extra network prediction. Plot only the states actually scored.
        pd.Series(diagnostics["policy_value"][:-1], index=index[:-1]).plot(
            ax=axes[1], color="darkgreen", linewidth=1.8, label=proxy_label,
        )
        pd.Series(diagnostics["discounted_return_to_go"], index=index).plot(
            ax=axes[1], color="darkorange", linestyle="--", linewidth=1.8,
            label=f"Realized discounted reward-to-go (gamma={gamma:g})",
        )
        axes[1].set_title("Existing TD3 Value Proxy vs. Realized Continuation")
        axes[1].set_ylabel("Discounted reward")
        axes[1].text(
            0.01, 0.03,
            "Existing critic estimates; a separately trained RECAP V is not in this checkpoint.",
            transform=axes[1].transAxes, fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )

        pd.Series(diagnostics["td_residual"], index=index[:-1]).plot(
            ax=axes[2], color="purple", linewidth=1.3,
            label="r + gamma * V_proxy(next) - V_proxy(now)",
        )
        axes[2].set_title("Reward-Adjusted Value Change (On-Policy Diagnostic, Not ProExp's Backup)")
        axes[2].set_ylabel("TD residual")

        pd.Series(diagnostics["critic_gap"][:-1], index=index[:-1]).plot(
            ax=axes[3], color="gray", linewidth=1.3, label="Max Q - min Q",
        )
        axes[3].set_title("Critic Disagreement (Not Value or a Calibrated Confidence Interval)")
        axes[3].set_ylabel("Critic spread")

    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend(loc="upper right", fontsize=9)
    if len(index) > 1:
        # Later decision-only series have one fewer point. Pandas otherwise
        # shrinks every shared axis and hides the final observed return.
        full_state_axis = axes[0].lines[0].get_xdata(orig=False)
        axes[-1].set_xlim(full_state_axis[0], full_state_axis[-1])
    axes[-1].set_xlabel("Time (UTC)")
    boundary_note = (
        "Terminal continuation is zero; this does not label the episode a success."
        if diagnostics["terminal_value_known"] else
        "Observed-window targets only: the continuation beyond the boundary is unknown."
    )
    fig.text(
        0.01, 0.005,
        "Future rewards are hindsight targets, never policy inputs. " + boundary_note,
        fontsize=9,
    )
    plt.xticks(rotation=45)
    fig.tight_layout(rect=[0, 0.025, 1, 1])
    if save_plots:
        output_path = os.path.join(config.LOGDIR, f"{safe_plot_prefix(plot_prefix)}_value_function.png")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Value-function diagnostics saved to {output_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_evaluation_sb3(
    episode_data,
    save_plots=True,
    plot_prefix="evaluation",
    show_plots=True,
):
    """
    Plot comprehensive evaluation results with SB3 data.
    Compatible with both fixed and variable portfolio environments.

    Args:
        episode_data: Episode data dictionary from evaluate_policy_sb3
        save_plots: Whether to save plots to files
        show_plots: If False, save PNGs and close figures (for live incremental refresh)
    """
    
    timestamps = episode_data["timestamps"]
    portfolio_values = episode_data["values"]
    rewards = episode_data["rewards"]
    
    # Extract meaningful allocation data
    allocations_df, episode_coins = extract_meaningful_allocations(episode_data)
    target_allocations_df = extract_meaningful_target_allocations(episode_data)
    
    datetime_index = episode_plot_datetime_index(timestamps)
    date_label = plot_calendar_date_label(datetime_index)

    portfolio_value_series = pd.Series(portfolio_values, index=datetime_index)
    value_diagnostics = episode_value_diagnostics(episode_data)

    # --- Plot 1: Wealth, Past Rewards, and Estimated/Realized Continuation ---
    n_summary_rows = 3
    fig, summary_axes = plt.subplots(
        n_summary_rows, 1, figsize=(15, 5 * n_summary_rows), sharex=True
    )
    ax1, ax2 = summary_axes[0], summary_axes[1]
    
    # Portfolio value
    portfolio_value_series.plot(ax=ax1, label='Portfolio Value', color='blue', linewidth=2)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title(title_with_plot_date('Portfolio Performance Over Episode (UTC)', date_label))
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()
    
    # Calculate and display performance metrics
    start_val = portfolio_values[0]
    end_val = portfolio_values[-1]
    percent_return = ((end_val - start_val) / start_val) * 100
    max_val = max(portfolio_values)
    min_val = min(portfolio_values)
    max_drawdown = ((max_val - min_val) / max_val) * 100
    
    performance_text = f'Return: {percent_return:.2f}%\nMax Drawdown: {max_drawdown:.2f}%'
    ax1.text(0.02, 0.95, performance_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7))
    
    # Cumulative rewards share the portfolio series' datetime axis. A raw
    # matplotlib plot uses a different date unit and lands outside the window.
    cumulative_rewards = pd.Series(np.cumsum(rewards), index=datetime_index)
    cumulative_rewards.plot(ax=ax2, label='Cumulative Reward', color='red', linewidth=2)
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title(title_with_plot_date('Cumulative Rewards Over Episode', date_label))
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend()

    if value_diagnostics["has_value_estimate"]:
        proxy_label = (
            "Sampled-action Q proxy (critic mean)"
            if episode_data.get("action_mode") == "stochastic"
            else "V_pi proxy: mean Q(s, pi(s))"
        )
        pd.Series(value_diagnostics["policy_value"][:-1], index=datetime_index[:-1]).plot(
            ax=summary_axes[2], color="darkgreen", linewidth=1.8, label=proxy_label,
        )
        gamma = value_diagnostics["discount_factor"]
        pd.Series(value_diagnostics["discounted_return_to_go"], index=datetime_index).plot(
            ax=summary_axes[2], color="darkorange", linewidth=1.8, linestyle="--",
            label=f"Realized discounted reward-to-go (gamma={gamma:g}; hindsight)",
        )
        summary_axes[2].set_ylabel("Discounted reward")
        value_title = "Existing Policy Value Proxy vs. Realized Remaining Reward"
    else:
        pd.Series(value_diagnostics["return_to_go"], index=datetime_index).plot(
            ax=summary_axes[2], color="darkgreen", linewidth=2,
            label="Realized reward-to-go (hindsight; gamma=1)",
        )
        summary_axes[2].set_ylabel("Remaining reward")
        value_title = "RECAP-Style Monte Carlo Target (No Learned Value Available)"
    summary_axes[2].set_title(title_with_plot_date(value_title, date_label))
    summary_axes[2].axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    summary_axes[2].grid(True, linestyle="--", linewidth=0.5)
    summary_axes[2].legend(loc="upper right")

    summary_axes[-1].set_xlabel('Time (UTC)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_plots:
        output_path = os.path.join(config.LOGDIR, f"{safe_plot_prefix(plot_prefix)}_portfolio_performance.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Portfolio performance plot saved to {output_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    plot_episode_value_function(
        episode_data, save_plots=save_plots, plot_prefix=plot_prefix, show_plots=show_plots,
    )

    # --- Plot 2: Target Allocation Implied By Raw Actions ---
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    target_allocations_df.plot.area(ax=ax, stacked=True, linewidth=0, alpha=0.8)
    ax.set_ylabel('Allocation (%)')
    ax.set_xlabel('Time (UTC)')
    ax.set_title(title_with_plot_date('Target Allocation Implied By Raw Actions (UTC)', date_label))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(title='Assets', loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_ylim(0, 1)

    target_avg_allocations = target_allocations_df.mean()
    target_allocation_text = "Average Target Allocations:\n" + "\n".join(
        [f"{asset}: {pct:.1%}" for asset, pct in target_avg_allocations.items()]
    )
    ax.text(
        0.02,
        0.98,
        target_allocation_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', fc='lightcyan', alpha=0.7),
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_plots:
        output_path = os.path.join(
            config.LOGDIR,
            f"{safe_plot_prefix(plot_prefix)}_target_allocation_strategy.png",
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Target allocation plot saved to {output_path}")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # --- Plot 3: Actual Portfolio Allocation Strategy After Step Advance ---
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    allocations_df.plot.area(ax=ax, stacked=True, linewidth=0, alpha=0.8)
    ax.set_ylabel('Allocation (%)')
    ax.set_xlabel('Time (UTC)')
    ax.set_title(title_with_plot_date('Actual Portfolio Allocation After Step Advance (UTC)', date_label))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(title='Assets', loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_ylim(0, 1)
    
    # Add allocation statistics
    avg_allocations = allocations_df.mean()
    allocation_text = "Average Allocations:\n" + "\n".join([f"{asset}: {pct:.1%}" for asset, pct in avg_allocations.items()])
    ax.text(0.02, 0.98, allocation_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_plots:
        output_path = os.path.join(config.LOGDIR, f"{safe_plot_prefix(plot_prefix)}_allocation_strategy.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Allocation strategy plot saved to {output_path}")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # --- Plot 4: Asset Prices with Allocation Overlay ---
    market_data_df = episode_data.get("market_data", pd.DataFrame())
    if isinstance(market_data_df, list):
        market_data_df = (
            pd.concat(market_data_df, ignore_index=True)
            if market_data_df
            else pd.DataFrame()
        )
    if isinstance(market_data_df, pd.DataFrame) and not market_data_df.empty:
        market_data_df['date'] = market_data_dates_for_plots(market_data_df['date'])

        fig, axes = plt.subplots(len(episode_coins), 1, figsize=(15, 4 * len(episode_coins)), sharex=True)
        if len(episode_coins) == 1:
            axes = [axes]
        
        fig.suptitle(title_with_plot_date('Asset Prices and Allocation Decisions', date_label), fontsize=16, y=0.98)

        for ax, coin in zip(axes, episode_coins):
            coin_data = market_data_df[market_data_df['coin'] == coin].set_index('date')
            
            if not coin_data.empty:
                # Plot OHLC prices
                plot_cols = ['open', 'high', 'low', 'close']
                available_cols = [col for col in plot_cols if col in coin_data.columns]
                coin_data[available_cols].plot(ax=ax, linewidth=1.5, alpha=0.8)
                
                ax.set_title(title_with_plot_date(f'{coin} Price Movement', date_label))
                ax.set_ylabel('Price ($)')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.legend(loc='upper left')

                # Overlay allocation on secondary y-axis
                ax2 = ax.twinx()
                allocations_df[coin].plot(ax=ax2, color='red', linewidth=2, alpha=0.7, label=f'{coin} Allocation')
                ax2.set_ylabel(f'{coin} Allocation', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.set_ylim(0, 1)
                ax2.legend(loc='upper right')

                # Create a third y-axis for cumulative return, offset to the left
                ax3 = ax.twinx()

                # Calculate the cumulative percentage return from the start of the episode
                # We use the 'close' price for this calculation
                initial_price = coin_data['close'].iloc[0]
                cumulative_return = (coin_data['close'] / initial_price - 1) * 100

                # Configure the new axis's spine (the axis line)
                # The 'outward' position moves it to the left of the original price axis
                ax3.spines['left'].set_position(('outward', 60))
                ax3.spines['left'].set_visible(True)
                ax3.spines['right'].set_visible(False)  # Hide the default spine on the right

                # Position the ticks and labels on our new left spine
                ax3.yaxis.set_ticks_position('left')
                ax3.yaxis.set_label_position('left')

                # Plot the cumulative return data on the new axis
                cumulative_return.plot(ax=ax3, color='purple', linestyle=':', linewidth=2.5, label=f'{coin} Cum. Return')
                ax3.set_ylabel(f'{coin} Cumulative Return (%)', color='purple')
                ax3.tick_params(axis='y', labelcolor='purple')

                # Add a horizontal line at 0% for a clear breakeven reference
                ax3.axhline(0, color='purple', linestyle='--', linewidth=1, alpha=0.7)

                # Place the legend for this new plot, e.g., in the lower left to avoid overlap
                ax3.legend(loc='upper right')

        axes[-1].set_xlabel('Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plots:
            output_path = os.path.join(config.LOGDIR, f"{safe_plot_prefix(plot_prefix)}_prices_and_allocations.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Asset prices and allocations plot saved to {output_path}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # --- Plot 5: Action Distribution ---
    actions_df = extract_meaningful_actions(episode_data)
    
    plt.figure(figsize=(15, 6))
    ax = plt.gca()
    actions_df.plot(ax=ax, linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Action Values')
    ax.set_xlabel('Time')
    ax.set_title(title_with_plot_date('Raw Action Values Over Time', date_label))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(title='Actions', loc='upper right')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_plots:
        output_path = os.path.join(config.LOGDIR, f"{safe_plot_prefix(plot_prefix)}_actions.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Actions plot saved to {output_path}")
    if show_plots:
        plt.show()
    else:
        plt.close()

    if not show_plots:
        plt.close("all")


def load_sb3_model(model_path, env):
    """
    Load a Stable Baselines 3 model from file.
    
    Args:
        model_path: Path to the saved model
        env: The environment to associate with the model
        
    Returns:
        Loaded SB3 model
    """
    print(f"Loading model from: {model_path}")
    if flowrl_checkpoint_metadata(model_path) is not None:
        from FlowRL_algo import FlowRL
        return FlowRL.load(model_path, env=env)
    
    # --- FIX: Use custom_objects to specify the custom policy class ---
    # This is crucial for SB3 to know how to reconstruct the model,
    # including its network architecture and optimizers.
    
    # Detect policy type from model path
    model_path_lower = model_path.lower()
    custom_objects = {}
    if "_attention_" in model_path_lower:
        # Variable portfolio or attention models use AttentionTD3Policy
        custom_objects = {"policy_class": AttentionTD3Policy}
        print("🧠 Detected variable/attention model - using AttentionTD3Policy")
    elif "_variable_" in model_path_lower or "_mlp_" in model_path_lower:
        # Fixed portfolio MLP models use SimpleMlpTD3Policy  
        custom_objects = {"policy_class": SimpleMlpTD3Policy}
        print("🔧 Detected fixed/MLP model - using SimpleMlpTD3Policy")
    
    # --- FIX: Make algorithm detection more robust ---
    # Check the parent directory name for the algorithm, and check for TD3 first
    # to avoid the "SAC" in "TRANSACTION_COST" issue.
    run_name_parts = model_path.split(os.sep)
    run_name = ""
    if len(run_name_parts) > 2:
        # e.g., .../models/TD3_tf_agents_style_heavy_.../checkpoints/model.zip -> TD3_TF_AGENTS_STYLE_HEAVY_...
        run_name = run_name_parts[-3].upper()

    model = None
    path_has_td3 = "TD3" in run_name or "td3" in model_path_lower
    if path_has_td3:
        if "MLP" in run_name:
            print("Detected TD3 model type from path.")
            # model = CustomTD3_AC.load(model_path, env=env, custom_objects=custom_objects)
            print("✅ Loaded CustomTD3 model")
        elif (
            "ATTENTION" in run_name
            or "variable_coin_attention" in model_path_lower
            or custom_objects.get("policy_class") is AttentionTD3Policy
        ):
            print("Detected TD3 attention model from path.")
            model = CustomTD3_AC_Attention.load(model_path, env=env, custom_objects=custom_objects)
            print("✅ Loaded CustomTD3 attention model")
        else:
            print(f"⚠️ TD3 path but no loader branch matched run_name={run_name!r}")
    elif "SAC" in run_name:
        print("Detected SAC model type from path.")
        model = SAC.load(model_path, env=env, custom_objects=custom_objects)
        print("✅ Loaded SAC model")
    elif "PPO" in run_name:
        print("Detected PPO model type from path.")
        model = PPO.load(model_path, env=env, custom_objects=custom_objects)
        print("✅ Loaded PPO model")
    else:
        print("⚠️ Could not detect algorithm from path folder; using policy_class / path hints...")
        if custom_objects.get("policy_class") is AttentionTD3Policy or "variable_coin_attention" in model_path_lower:
            model = CustomTD3_AC_Attention.load(model_path, env=env, custom_objects=custom_objects)
            print("✅ Loaded CustomTD3 attention model (path fallback)")
        else:
            raise ValueError(
                f"Could not infer SB3 loader for {model_path}. "
                f"Use checkpoints/ under a TD3_* run folder or set AttentionTD3Policy path hints."
            )

    # Safety check to ensure model was loaded successfully
    if model is None:
        raise ValueError("Model loading returned None - this should not happen")
    
    return model


def is_iql_checkpoint(model_path: str) -> bool:
    basename = os.path.basename(model_path).lower()
    if not model_path.lower().endswith(".pt"):
        return False
    if (
        basename.startswith("iql_checkpoint")
        or basename.startswith("iql_policy")
        or basename.startswith("iql_attention_deepset")
        or "iql_offline" in model_path.lower()
        or "iql-mapo" in model_path.lower()
    ):
        return True

    checkpoint = load_supported_pt_checkpoint(model_path)
    if checkpoint is None:
        return False

    if (
        is_mapo_attention_iql_final_checkpoint(checkpoint)
        or is_mapo_attention_deepset_iql_checkpoint(checkpoint)
        or is_mapo_iql_checkpoint(checkpoint)
    ):
        return True

    metadata = checkpoint.get("network_metadata", {})
    return bool(
        checkpoint.get("policy_state_dict")
        or checkpoint.get("policy")
        or checkpoint.get("actor")
    ) and bool(metadata)


def is_mapo_attention_deepset_iql_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    metadata = checkpoint.get("network_metadata", {})
    actor_type = str(metadata.get("actor_type", "")).lower()
    policy_class = str(metadata.get("policy_class", ""))
    return (
        checkpoint.get("format") == MAPO_AD_IQL_CHECKPOINT_FORMAT
        or "attention_deepset" in actor_type
        or policy_class == "AttentionDeepSetGaussianPolicy"
    )


def is_mapo_iql_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    metadata = checkpoint.get("network_metadata", {})
    return (
        checkpoint.get("format") == MAPO_IQL_CHECKPOINT_FORMAT
        or metadata.get("actor_type") == "rlkit_gaussian_policy"
        or metadata.get("policy_class") == "GaussianPolicy"
    )


def is_mapo_attention_iql_final_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    metadata = checkpoint.get("network_metadata", {})
    actor_type = str(metadata.get("actor_type", "")).lower()
    policy_class = str(metadata.get("policy_class", ""))
    policy_module = str(metadata.get("policy_module", "")).lower()
    return (
        checkpoint.get("format") == MAPO_AD_FINAL_IQL_CHECKPOINT_FORMAT
        or actor_type == "rlkit_attention_iql_gaussian_policy"
        or policy_class == "AttentionIQLGaussianPolicy"
        or policy_module.endswith("attention_iql_policy")
    )


def infer_attention_deepset_shape_kwargs(metadata: Dict[str, Any], state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    observation_metadata = metadata.get("observation_metadata", {})
    observations_metadata = observation_metadata.get("observations", {})
    mask_metadata = observation_metadata.get("mask", {})
    asset_metadata = observation_metadata.get("asset_ids", {})

    max_coins = int(metadata.get("max_coins") or mask_metadata.get("flat_dim") or asset_metadata.get("flat_dim"))
    observations_flat_dim = int(observations_metadata.get("flat_dim") or metadata.get("observation_dim"))
    if max_coins <= 0:
        raise ValueError("MAPO AD checkpoint metadata does not contain a valid max_coins/mask dimension.")
    if observations_flat_dim % max_coins != 0:
        raise ValueError(
            f"MAPO AD observations flat_dim={observations_flat_dim} is not divisible by max_coins={max_coins}."
        )

    features_per_coin = int(metadata.get("features_per_coin") or (observations_flat_dim // max_coins))
    action_dim = int(metadata["action_dim"])
    expected_action_dim = max_coins + 1
    if action_dim != expected_action_dim:
        raise ValueError(
            f"MAPO AD action_dim={action_dim} does not match max_coins + cash={expected_action_dim}."
        )

    embedding_weight = state_dict.get("asset_identity_embedding.weight")
    if embedding_weight is not None:
        num_asset_ids = int(embedding_weight.shape[0] - 1)
    else:
        asset_id_max = float(asset_metadata.get("max_id", max_coins))
        num_asset_ids = int(max(asset_id_max, max_coins))

    return dict(
        action_dim=action_dim,
        max_coins=max_coins,
        features_per_coin=features_per_coin,
        num_asset_ids=num_asset_ids,
    )


def resolve_eval_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def flatten_mapo_iql_observation(
    obs,
    observation_keys=MAPO_DEFAULT_OBSERVATION_KEYS,
    normalize_asset_ids=True,
    asset_id_scale=1.0,
):
    if not isinstance(obs, dict):
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    pieces = []
    missing_keys = []
    for key in observation_keys:
        if key not in obs:
            missing_keys.append(key)
            continue
        values = np.asarray(obs[key])
        if key == "asset_ids" and normalize_asset_ids:
            values = values.astype(np.float32) / max(float(asset_id_scale), 1.0)
        else:
            values = values.astype(np.float32)
        pieces.append(values.reshape(-1))

    if missing_keys:
        raise ValueError(f"MAPO IQL observation is missing keys: {missing_keys}")
    if not pieces:
        raise ValueError("MAPO IQL observation flattening produced no features.")
    return np.concatenate(pieces, axis=0).astype(np.float32)


class IQLPolicyAdapter:
    """
    Thin evaluation adapter so the offline IQL actor can be used by the same
    evaluation loop as SB3 models. It intentionally exposes predict().
    """

    def __init__(
        self,
        actor,
        device,
        observation_space=None,
        action_space=None,
        checkpoint_path=None,
        observation_mode="dict",
        actor_output_mode="tensor",
        observation_keys=None,
        normalize_asset_ids=True,
        asset_id_scale=1.0,
        default_action_mode="deterministic",
    ):
        self.actor = actor
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        self.checkpoint_path = checkpoint_path
        self.observation_mode = observation_mode
        self.actor_output_mode = actor_output_mode
        self.observation_keys = tuple(observation_keys or MAPO_DEFAULT_OBSERVATION_KEYS)
        self.normalize_asset_ids = bool(normalize_asset_ids)
        self.asset_id_scale = float(asset_id_scale)
        normalized_action_mode = str(default_action_mode).strip().lower()
        if normalized_action_mode not in ("deterministic", "stochastic"):
            raise ValueError(
                f"default_action_mode must be 'deterministic' or 'stochastic', got {default_action_mode!r}"
            )
        self.default_action_mode = normalized_action_mode
        self.actor.eval()

    def _obs_to_tensor(self, obs):
        if self.observation_mode == "flat":
            flat_obs = flatten_mapo_iql_observation(
                obs,
                observation_keys=self.observation_keys,
                normalize_asset_ids=self.normalize_asset_ids,
                asset_id_scale=self.asset_id_scale,
            )
            return torch.as_tensor(flat_obs[None, :], dtype=torch.float32, device=self.device)

        if not isinstance(obs, dict):
            raise ValueError("IQLPolicyAdapter expects dict observations from the variable portfolio environment.")

        tensors = {}
        for key, value in obs.items():
            array = np.asarray(value)
            if array.ndim == 1:
                array = array[None, ...]
            if key == "asset_ids":
                tensors[key] = torch.as_tensor(array, dtype=torch.long, device=self.device)
            else:
                tensors[key] = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        return tensors

    def predict(self, obs, deterministic=True):
        with torch.no_grad():
            tensor_obs = self._obs_to_tensor(obs)
            actor_output = self.actor(tensor_obs)
            if hasattr(actor_output, "mle_estimate"):
                action_tensor = actor_output.mle_estimate() if deterministic else actor_output.sample()
            else:
                action_tensor = actor_output
            action = action_tensor.detach().cpu().numpy()[0].astype(np.float32)
        if self.action_space is not None:
            action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
        return action, None

    def resolve_deterministic_flag(self, requested_mode: str) -> bool:
        normalized_mode = str(requested_mode).strip().lower()
        if normalized_mode == "auto":
            normalized_mode = self.default_action_mode
        if normalized_mode == "deterministic":
            return True
        if normalized_mode == "stochastic":
            return False
        raise ValueError(f"Unsupported action mode: {requested_mode!r}")


def infer_iql_default_action_mode(metadata: Dict[str, Any]) -> str:
    trainer_kwargs = metadata.get("trainer_kwargs", {}) or {}
    policy_kwargs = metadata.get("policy_kwargs", {}) or {}

    simplex_uses_samples = bool(trainer_kwargs.get("simplex_use_policy_samples", False))
    awr_loss_space = str(trainer_kwargs.get("awr_loss_space", "")).strip().lower()
    action_distribution = str(policy_kwargs.get("action_distribution", "")).strip().lower()

    if simplex_uses_samples and awr_loss_space == "simplex" and "portfolio_aware" in action_distribution:
        return "stochastic"
    return "deterministic"


def load_iql_model(model_path: str, env, device_arg: str = "auto") -> IQLPolicyAdapter:
    device = resolve_eval_device(device_arg)
    print(f"Loading offline IQL checkpoint from: {model_path}")
    checkpoint = load_supported_pt_checkpoint(model_path)
    if checkpoint is None:
        raise ValueError(f"Unable to load supported IQL checkpoint: {model_path}")
    metadata = checkpoint["network_metadata"]

    if is_mapo_attention_iql_final_checkpoint(checkpoint):
        state_dict = (
            checkpoint.get("policy_state_dict")
            or checkpoint.get("policy")
            or checkpoint.get("actor")
        )
        if state_dict is None:
            raise ValueError("MAPO AD final IQL checkpoint has no policy state_dict.")

        policy_kwargs = dict(metadata.get("policy_kwargs", {}))
        shape_kwargs = infer_attention_deepset_shape_kwargs(metadata, state_dict)
        AttentionIQLGaussianPolicy = get_rlkit_attention_iql_policy_class()
        actor = AttentionIQLGaussianPolicy(
            **shape_kwargs,
            **policy_kwargs,
        ).to(device)
        actor.load_state_dict(state_dict)
        actor.eval()

        observation_keys = tuple(
            metadata.get("observation_keys", MAPO_DEFAULT_OBSERVATION_KEYS)
        )
        print(f"✅ Loaded MAPO AD final RLKit IQL actor at update {checkpoint.get('update', 'unknown')}")
        print(f"   Device: {device}")
        print(f"   Actor type: {metadata.get('actor_type', 'rlkit_attention_iql_gaussian_policy')}")
        print(
            "   Actor max_coins/features/action_dim: "
            f"{shape_kwargs['max_coins']}/{shape_kwargs['features_per_coin']}/{shape_kwargs['action_dim']}"
        )
        print(f"   Observation keys: {observation_keys}")
        if policy_kwargs.get("fixed_log_std") is not None:
            print(f"   Fixed log std: {policy_kwargs.get('fixed_log_std')}")
        # The final attention IQL actor is a plain diagonal Gaussian extracted by
        # advantage-weighted regression; deterministic evaluation uses its mean.
        return IQLPolicyAdapter(
            actor=actor,
            device=device,
            observation_space=env.observation_space,
            action_space=env.action_space,
            checkpoint_path=model_path,
            observation_mode="dict",
            actor_output_mode="distribution",
            observation_keys=observation_keys,
            normalize_asset_ids=bool(metadata.get("normalize_asset_ids", False)),
            asset_id_scale=float(metadata.get("asset_id_scale", 1.0)),
            default_action_mode="deterministic",
        )

    if is_mapo_attention_deepset_iql_checkpoint(checkpoint):
        state_dict = (
            checkpoint.get("policy_state_dict")
            or checkpoint.get("policy")
            or checkpoint.get("actor")
        )
        if state_dict is None:
            raise ValueError("MAPO AD IQL checkpoint has no policy state_dict.")

        policy_kwargs = dict(metadata.get("policy_kwargs", {}))
        shape_kwargs = infer_attention_deepset_shape_kwargs(metadata, state_dict)
        RlkitAttentionDeepSetPolicy = get_rlkit_attention_deepset_policy_class()
        actor = RlkitAttentionDeepSetPolicy(
            **shape_kwargs,
            **policy_kwargs,
        ).to(device)
        actor.load_state_dict(state_dict)
        actor.eval()

        observation_keys = tuple(
            metadata.get("observation_keys", MAPO_DEFAULT_OBSERVATION_KEYS)
        )
        print(f"✅ Loaded MAPO AD RLKit IQL actor at update {checkpoint.get('update', 'unknown')}")
        print(f"   Device: {device}")
        print(f"   Actor type: {metadata.get('actor_type', 'rlkit_attention_deepset_gaussian_policy')}")
        print(
            "   Actor max_coins/features/action_dim: "
            f"{shape_kwargs['max_coins']}/{shape_kwargs['features_per_coin']}/{shape_kwargs['action_dim']}"
        )
        print(f"   Observation keys: {observation_keys}")
        print(
            "   Branch contract: "
            f"{metadata.get('iql_branch_name', 'unknown')} / "
            f"{metadata.get('actor_target_source', 'unknown')} / "
            f"{metadata.get('policy_shell_contract', 'unknown')} / "
            f"{metadata.get('reward_contract', 'unknown')}"
        )
        if metadata.get("fixed_log_std") is not None:
            print(f"   Fixed log std: {metadata.get('fixed_log_std')}")
        default_action_mode = infer_iql_default_action_mode(metadata)
        print(f"   Evaluation action mode (auto): {default_action_mode}")
        return IQLPolicyAdapter(
            actor=actor,
            device=device,
            observation_space=env.observation_space,
            action_space=env.action_space,
            checkpoint_path=model_path,
            observation_mode="dict",
            actor_output_mode="distribution",
            observation_keys=observation_keys,
            normalize_asset_ids=bool(metadata.get("normalize_asset_ids", False)),
            asset_id_scale=float(metadata.get("asset_id_scale", 1.0)),
            default_action_mode=default_action_mode,
        )

    if is_mapo_iql_checkpoint(checkpoint):
        policy_kwargs = dict(metadata.get("policy_kwargs", {}))
        obs_dim = int(metadata.get("obs_dim", metadata.get("observation_dim")))
        action_dim = int(metadata["action_dim"])
        RlkitGaussianPolicy = get_rlkit_gaussian_policy_class()
        actor = RlkitGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **policy_kwargs,
        ).to(device)

        state_dict = (
            checkpoint.get("policy_state_dict")
            or checkpoint.get("policy")
            or checkpoint.get("actor")
        )
        if state_dict is None:
            raise ValueError("MAPO IQL checkpoint has no policy state_dict.")
        actor.load_state_dict(state_dict)
        actor.eval()

        observation_metadata = metadata.get("observation_metadata", {})
        asset_metadata = observation_metadata.get("asset_ids", {})
        asset_id_scale = float(
            metadata.get("asset_id_scale", asset_metadata.get("scale", 1.0))
        )
        observation_keys = tuple(
            metadata.get("observation_keys", MAPO_DEFAULT_OBSERVATION_KEYS)
        )
        print(f"✅ Loaded MAPO RLKit IQL actor at update {checkpoint.get('update', 'unknown')}")
        print(f"   Device: {device}")
        print(f"   Actor type: {metadata.get('actor_type', 'rlkit_gaussian_policy')}")
        print(f"   Observation/action dims: {obs_dim}/{action_dim}")
        print(f"   Observation keys: {observation_keys}")
        default_action_mode = infer_iql_default_action_mode(metadata)
        print(f"   Evaluation action mode (auto): {default_action_mode}")
        return IQLPolicyAdapter(
            actor=actor,
            device=device,
            observation_space=env.observation_space,
            action_space=env.action_space,
            checkpoint_path=model_path,
            observation_mode="flat",
            actor_output_mode="distribution",
            observation_keys=observation_keys,
            normalize_asset_ids=bool(metadata.get("normalize_asset_ids", True)),
            asset_id_scale=asset_id_scale,
            default_action_mode=default_action_mode,
        )

    # Fallback mirrors the canonical training default: hierarchical stochastic actor.
    actor_type = metadata.get("actor_type", "hierarchical_stochastic_deepset")
    # if actor_type == "hierarchical_stochastic_deepset":
    #     actor = HierarchicalStochasticDeepSetPolicy(
    #         max_coins=int(metadata["max_coins"]),
    #         features_per_coin=int(metadata["features_per_coin"]),
    #         action_dim=int(metadata["action_dim"]),
    #         set_hidden_dim=int(metadata["features_dim"]),
    #         net_arch=list(metadata["net_arch"]),
    #         num_asset_ids=int(metadata["num_asset_ids"]),
    #         asset_identity_dim=int(metadata["asset_identity_dim"]),
    #         n_heads=int(metadata["n_heads"]),
    #         n_layers=int(metadata["n_layers"]),
    #         dropout=float(metadata["dropout"]),
    #         use_market_attention=bool(metadata.get("use_market_attention", False)),
    #         use_global_pooling=bool(metadata.get("use_global_pooling", True)),
    #         min_log_std=float(metadata.get("policy_log_std_min", -5.0)),
    #         max_log_std=float(metadata.get("policy_log_std_max", 2.0)),
    #         use_prev_action_input=bool(metadata.get("actor_use_prev_action", True)),
    #         prev_action_residual_gain=float(metadata.get("actor_prev_action_gain", 0.05)),
    #         prev_action_dropout_p=float(metadata.get("actor_prev_action_dropout", 0.0)),
    #     ).to(device)
    # elif actor_type == "flat_stochastic_deepset":
    #     actor = FlatStochasticDeepSetPolicy(
    #         max_coins=int(metadata["max_coins"]),
    #         features_per_coin=int(metadata["features_per_coin"]),
    #         action_dim=int(metadata["action_dim"]),
    #         set_hidden_dim=int(metadata["features_dim"]),
    #         net_arch=list(metadata["net_arch"]),
    #         num_asset_ids=int(metadata["num_asset_ids"]),
    #         asset_identity_dim=int(metadata["asset_identity_dim"]),
    #         n_heads=int(metadata["n_heads"]),
    #         n_layers=int(metadata["n_layers"]),
    #         dropout=float(metadata["dropout"]),
    #         use_market_attention=bool(metadata.get("use_market_attention", False)),
    #         use_global_pooling=bool(metadata.get("use_global_pooling", True)),
    #         use_action_mixer=False,
    #         use_prev_action_input=bool(metadata.get("actor_use_prev_action", True)),
    #         prev_action_mode=str(metadata.get("actor_prev_action_mode", "residual")),
    #         prev_action_residual_gain=float(metadata.get("actor_prev_action_gain", 0.05)),
    #         prev_action_dropout_p=float(metadata.get("actor_prev_action_dropout", 0.0)),
    #         min_log_std=float(metadata.get("policy_log_std_min", -5.0)),
    #         max_log_std=float(metadata.get("policy_log_std_max", 2.0)),
    #     ).to(device)
    if True :
        actor = DeepSetActor(
            max_coins=int(metadata["max_coins"]),
            features_per_coin=int(metadata["features_per_coin"]),
            action_dim=int(metadata["action_dim"]),
            set_hidden_dim=int(metadata["features_dim"]),
            net_arch=list(metadata["net_arch"]),
            num_asset_ids=int(metadata["num_asset_ids"]),
            asset_identity_dim=int(metadata["asset_identity_dim"]),
            n_heads=int(metadata["n_heads"]),
            n_layers=int(metadata["n_layers"]),
            dropout=float(metadata["dropout"]),
            use_market_attention=bool(metadata.get("use_market_attention", False)),
            use_global_pooling=bool(metadata.get("use_global_pooling", True)),
            use_action_mixer=False,
            use_prev_action_input=bool(metadata.get("actor_use_prev_action", True)),
            prev_action_mode=str(metadata.get("actor_prev_action_mode", "direct")),
            prev_action_residual_gain=float(metadata.get("actor_prev_action_gain", 0.05)),
            prev_action_dropout_p=float(metadata.get("actor_prev_action_dropout", 0.0)),
        ).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    actor_loss_mode = str(metadata.get("actor_loss_mode", "stochastic_log_prob"))
    print(f"✅ Loaded offline IQL actor at update {checkpoint.get('update', 'unknown')}")
    print(f"   Device: {device}")
    print(f"   Actor type: {actor_type}")
    print(f"   Actor loss (training): {actor_loss_mode}")
    print(f"   Actor max_coins/features/action_dim: {metadata['max_coins']}/{metadata['features_per_coin']}/{metadata['action_dim']}")
    return IQLPolicyAdapter(
        actor=actor,
        device=device,
        observation_space=env.observation_space,
        action_space=env.action_space,
        checkpoint_path=model_path,
    )


def flowrl_checkpoint_metadata(model_path):
    """Read plain SB3 metadata, independent of checkpoint file/folder names."""
    import json
    import zipfile
    if not str(model_path).endswith(".zip") or not zipfile.is_zipfile(model_path):
        return None
    with zipfile.ZipFile(model_path) as archive:
        if "data" not in archive.namelist():
            return None
        metadata = json.loads(archive.read("data"))
    return metadata if metadata.get("algorithm_name") == "FlowRL" else None


def load_model_for_evaluation(model_path: str, env, device_arg: str = "auto"):
    if flowrl_checkpoint_metadata(model_path) is not None:
        from FlowRL_algo import FlowRL
        return FlowRL.load(model_path, env=env, device=device_arg)
    checkpoint = load_supported_pt_checkpoint(model_path)
    if checkpoint is not None and checkpoint.get("format") in {"portfolio_qwm_expo_v1", "portfolio_qwm_expo_v2"}:
        from qwm_expo import QWMEXPO
        return QWMEXPO.load(model_path, device=device_arg)
    if is_iql_checkpoint(model_path):
        return load_iql_model(model_path, env=env, device_arg=device_arg)
    return load_sb3_model(model_path, env=env)


def iql_checkpoint_context_text(model_path: str) -> str:
    if not is_iql_checkpoint(model_path) or not os.path.exists(model_path):
        return model_path
    try:
        checkpoint = load_supported_pt_checkpoint(model_path)
        if checkpoint is None:
            return model_path
        args = checkpoint.get("args", {})
        replay_buffer = args.get("replay_buffer", "")
        metadata = checkpoint.get("network_metadata", {})
        return " ".join([model_path, replay_buffer, str(metadata)])
    except Exception:
        return model_path


def infer_environment_settings(model_path: str) -> Tuple[str, bool, bool]:
    flowrl = flowrl_checkpoint_metadata(model_path)
    if flowrl is not None:
        settings = flowrl["environment_settings"]
        return settings["reward_type"], settings["use_variable_portfolio"], False
    checkpoint = load_supported_pt_checkpoint(model_path)
    if checkpoint is not None:
        if checkpoint.get("format") in {"portfolio_qwm_expo_v1", "portfolio_qwm_expo_v2"}:
            return "STRUCTURED_CREDIT", True, False
        metadata = checkpoint.get("network_metadata", {}) or {}
        reward_type = str(metadata.get("eval_reward_type", "")).strip()
        if reward_type:
            use_temporal_attention = "_tempconv" in model_path.lower()
            use_variable_portfolio = bool(
                metadata.get(
                    "eval_use_variable_portfolio",
                    is_mapo_attention_deepset_iql_checkpoint(checkpoint),
                )
            )
            return reward_type, use_variable_portfolio, use_temporal_attention

    context = iql_checkpoint_context_text(model_path)
    context_upper = context.upper()
    context_lower = context.lower()

    if "TRANSACTION_COST" in context_upper:
        reward_type = "TRANSACTION_COST"
    elif "STRUCTURED_CREDIT" in context_upper:
        reward_type = "STRUCTURED_CREDIT"
    elif "SHAPLEY" in context_upper:
        reward_type = "SHAPLEY"
    elif "POMDP" in context_upper:
        reward_type = "POMDP"
    else:
        reward_type = "simple"

    use_variable_portfolio = (
        is_iql_checkpoint(model_path)
        or "VARIABLE" in context_upper
        or "_variable_" in context_lower
        or "variable_coin_attention" in context_lower
    )
    use_temporal_attention = "_tempconv" in context_lower
    return reward_type, use_variable_portfolio, use_temporal_attention


def create_eval_env_for_model(model_path: str) -> PortfolioEnv:
    reward_type, use_variable_portfolio, use_temporal_attention = infer_environment_settings(model_path)
    flowrl = flowrl_checkpoint_metadata(model_path)
    if flowrl is not None:
        config.set_portfolio_mode(use_variable_portfolio)
        return PortfolioEnv(use_variable_portfolio=use_variable_portfolio, reward_type=reward_type,
                            action_normalization=flowrl["action_normalization"])

    if use_variable_portfolio and use_temporal_attention:
        print(f"🔄 Detected VARIABLE + TEMPORAL CONV model from path")
        eval_env = PortfolioEnv(use_variable_portfolio=True, reward_type=reward_type, use_temporal_attention=True)
    elif use_variable_portfolio:
        print(f"🔄 Detected VARIABLE portfolio model from path")
        eval_env = PortfolioEnv(use_variable_portfolio=True, reward_type=reward_type)
    else:
        print(f"🔄 Detected FIXED portfolio model from path")
        eval_env = PortfolioEnv(use_variable_portfolio=False, reward_type=reward_type)

    checkpoint = load_supported_pt_checkpoint(model_path)
    if checkpoint is not None and checkpoint.get("format") in {"portfolio_qwm_expo_v1", "portfolio_qwm_expo_v2"}:
        from enviorment import QWMPortfolioEnv
        eval_env = QWMPortfolioEnv(eval_env)
    print(f"✅ Environment created with reward_type='{reward_type}', variable_portfolio={use_variable_portfolio}, temporal={use_temporal_attention}")
    print(f"   Observation space: {eval_env.observation_space}")
    print(f"   Action space: {eval_env.action_space}")
    return eval_env


def model_label_from_path(model_path: str) -> str:
    if is_iql_checkpoint(model_path):
        parent = os.path.basename(os.path.dirname(model_path))
        checkpoint_name = os.path.splitext(os.path.basename(model_path))[0]
        return safe_plot_prefix(f"IQL_{parent}_{checkpoint_name}")
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    checkpoint_name = os.path.splitext(os.path.basename(model_path))[0]
    return safe_plot_prefix(f"{run_dir}_{checkpoint_name}")


def shared_episode_signature(episode_data: Dict[str, Any]) -> Tuple[Any, ...]:
    """Identity of the market episode, independent of which policy acted in it."""
    coins = tuple(episode_data.get("episode_coins", ()))
    start = str(episode_data["timestamps"][0]) if episode_data.get("timestamps") else ""
    initial_allocation = np.round(np.asarray(episode_data["allocations"][0], dtype=np.float64), 6)
    return coins, start, tuple(initial_allocation.tolist())


def episode_metrics(model_label: str, model_path: str, episode_data: Dict[str, Any]) -> Dict[str, Any]:
    start_value = float(episode_data["values"][0])
    end_value = float(episode_data["values"][-1])
    total_return = end_value - start_value
    percent_return = (total_return / start_value) * 100.0
    total_reward = float(sum(episode_data["rewards"]))
    steps = len(episode_data["timestamps"]) - 1
    allocations_df, _ = extract_meaningful_allocations(episode_data)
    avg_cash = float(allocations_df["Cash"].mean()) if "Cash" in allocations_df else np.nan
    return {
        "model_label": model_label,
        "model_path": model_path,
        "steps": steps,
        "initial_value": start_value,
        "final_value": end_value,
        "total_return": total_return,
        "percent_return": percent_return,
        "total_reward": total_reward,
        "avg_reward_per_step": total_reward / max(1, steps),
        "avg_cash": avg_cash,
    }


def print_comparison_summary(metrics_rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    ranked = sorted(metrics_rows, key=lambda row: row["percent_return"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        print(f"\nRank {rank}: {row['model_label']}")
        print(f"  Final Value: ${row['final_value']:.2f}")
        print(f"  Total Return: ${row['total_return']:.2f} ({row['percent_return']:.2f}%)")
        print(f"  Total Reward: {row['total_reward']:.6f}")
        print(f"  Avg Reward per Step: {row['avg_reward_per_step']:.8f}")
        print(f"  Avg Cash: {row['avg_cash']:.1%}")


def save_comparison_csv(metrics_rows: List[Dict[str, Any]], output_path: str) -> None:
    if not metrics_rows:
        return
    pd.DataFrame(metrics_rows).to_csv(output_path, index=False)
    print(f"📄 Comparison CSV saved to: {output_path}")


def is_jupyter_kernel_arg(arg: str) -> bool:
    basename = os.path.basename(str(arg))
    return (
        str(arg).startswith("--f=")
        or str(arg) == "-f"
        or str(arg).endswith(".json") and basename.startswith("kernel-")
        or str(arg).endswith(".json") and "jupyter" in str(arg).lower()
    )


def parse_evaluation_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        if all(is_jupyter_kernel_arg(arg) for arg in unknown_args):
            print(f"📓 Ignoring Jupyter kernel arguments: {' '.join(unknown_args)}")
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
    return args


def resolve_evaluation_seed(seed: Optional[int]) -> int:
    if seed is not None:
        return int(seed)
    generated_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    print(f"🎲 No --evaluation-seed provided; using random comparison seed: {generated_seed}")
    return generated_seed


def normalize_model_paths(model_paths: Optional[Any]) -> List[str]:
    """Flatten nested path containers into a clean list of filesystem path strings.

    DEFAULT_* checkpoints are sometimes written as parenthesized path menus. When every
    entry is commented out, Python still builds an empty tuple ``()``. When several
    paths are left active with commas, Python builds a multi-path tuple. Either way,
    ``os.path.isdir`` / ``os.path.isfile`` reject non-path objects, so this helper
    expands nested lists/tuples and drops empty / None entries before resolution.
    """
    if model_paths is None:
        return []

    normalized: List[str] = []

    def _walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                _walk(nested)
            return
        path = str(item).strip()
        if path:
            normalized.append(path)

    _walk(model_paths)
    return normalized


def print_evaluation_summary(all_episode_data):
    """Print a summary of evaluation results for both fixed and variable portfolios."""
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for i, episode_data in enumerate(all_episode_data):
        start_value = episode_data["values"][0]
        end_value = episode_data["values"][-1]
        total_return = end_value - start_value
        percent_return = (total_return / start_value) * 100
        total_reward = sum(episode_data["rewards"])
        steps = len(episode_data["timestamps"]) - 1
        
        # Get episode-specific information
        episode_coins = episode_data.get("episode_coins", config.COINS)
        is_variable = episode_data.get("is_variable_portfolio", False)
        
        print(f"\nEpisode {i+1} ({'Variable' if is_variable else 'Fixed'} Portfolio):")
        print(f"  Active Assets: {episode_coins}")
        print(f"  Duration: {steps} steps")
        print(f"  Initial Value: ${start_value:.2f}")
        print(f"  Final Value: ${end_value:.2f}")
        print(f"  Total Return: ${total_return:.2f} ({percent_return:.2f}%)")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Avg Reward per Step: {total_reward/steps:.6f}")
        
        # Portfolio allocation summary
        allocations_df, episode_coins = extract_meaningful_allocations(episode_data)
        avg_allocations = allocations_df.mean()
        print(f"  Average Allocations:")
        for asset, allocation in avg_allocations.items():
            print(f"    {asset}: {allocation:.1%}")
    
    # Overall summary
    if len(all_episode_data) > 1:
        all_returns = [((ep["values"][-1] - ep["values"][0]) / ep["values"][0]) * 100 
                      for ep in all_episode_data]
        print(f"\nOverall Performance:")
        print(f"  Mean Return: {np.mean(all_returns):.2f}%")
        print(f"  Std Return: {np.std(all_returns):.2f}%")
        print(f"  Min/Max Return: {np.min(all_returns):.2f}% / {np.max(all_returns):.2f}%")


def test_variable_portfolio_evaluation():
    """Test the evaluation system with variable portfolios."""
    print("🧪 Testing variable portfolio evaluation...")
    
    try:
        # Create test environment
        test_env = PortfolioEnv(use_variable_portfolio=True)
        print(f"✅ Variable portfolio environment created")
        print(f"   Observation space: {test_env.observation_space}")
        print(f"   Action space: {test_env.action_space}")
        
        # Test reset and data collection
        obs, info = test_env.reset()
        episode_coins = info.get('episode_coins', [])
        print(f"✅ Environment reset successful")
        print(f"   Episode coins: {episode_coins}")
        print(f" ?  test_env.money_split_ratio: {test_env.money_split_ratio}")
        print(f"   Money split ratio shape: {test_env.money_split_ratio.shape}")
        
        # Create mock episode data for testing
        mock_episode_data = {
            "timestamps": [test_env.current_time],
            "values": [test_env.current_value],
            "allocations": [test_env.money_split_ratio.copy()],
            "market_data": [test_env.dfslice.copy()],
            "rewards": [0.0],
            "actions": [np.zeros(config.MAX_COINS + 1)],
            "observations": [obs],
            "episode_coins": episode_coins,
            "is_variable_portfolio": True
        }
        
        # Test helper functions
        print("🔧 Testing allocation extraction...")
        allocations_df, extracted_coins = extract_meaningful_allocations(mock_episode_data)
        print(f"✅ Allocation extraction successful")
        print(f"   DataFrame shape: {allocations_df.shape}")
        print(f"   Columns: {list(allocations_df.columns)}")
        print(f"   Extracted coins: {extracted_coins}")
        
        print("🔧 Testing action extraction...")
        actions_df = extract_meaningful_actions(mock_episode_data)
        print(f"✅ Action extraction successful")
        print(f"   DataFrame shape: {actions_df.shape}")
        print(f"   Columns: {list(actions_df.columns)}")
        
        test_env.close()
        print("🎉 Variable portfolio evaluation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Variable portfolio evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Configuration
    print("🚀 SB3 Portfolio Evaluation Script")
    print("="*50)

    parser = argparse.ArgumentParser(description="Evaluate SB3, offline IQL and QWM-EXPO portfolio models.")
    parser.add_argument(
        "--model",
        dest="model_paths",
        action="append",
        help="Model checkpoint to evaluate. Can be passed multiple times. Supports SB3 .zip and IQL/QWM-EXPO .pt.",
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=None,
        help="Shared seed for every compared model. If omitted, one random seed is generated and reused.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--action-mode",
        choices=("auto", "deterministic", "stochastic"),
        default="auto",
        help="How to evaluate policy actions. 'auto' uses checkpoint-aware defaults for RLKit IQL.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--run-env-smoke-test", action="store_true")
    args = parse_evaluation_args(parser)
    args.evaluation_seed = resolve_evaluation_seed(args.evaluation_seed)
    
    # Create log directory if it doesn't exist
    os.makedirs(config.LOGDIR, exist_ok=True)

    model_paths = normalize_model_paths(
        args.model_paths
        or [
            DEFAULT_IQL_CHECKPOINT,
            DEFAULT_TD3_SOURCE_CHECKPOINT,
        ]
    )
    
    # Find latest model file in each given directory or file in model_paths
    import glob

    available_models = []
    for model_path in model_paths:
        if os.path.isdir(model_path):
            model_files = glob.glob(os.path.join(model_path, "*.zip"))
            model_files.extend(glob.glob(os.path.join(model_path, "*.pt")))
            model_files.extend(glob.glob(os.path.join(model_path, "**", "*.zip"), recursive=True))
            model_files.extend(glob.glob(os.path.join(model_path, "**", "*.pt"), recursive=True))
            model_files = sorted(set(model_files))
            if not model_files:
                print(f"❌ No model checkpoint files found in directory: {model_path}")
                continue
            latest_model_file = max(model_files, key=os.path.getmtime)
            print(f"✅ Found model file: {latest_model_file}")
            available_models.append(latest_model_file)
        elif os.path.isfile(model_path):
            print(f"✅ Found model file: {model_path}")
            available_models.append(model_path)
        else:
            print(f"❌ Model not found: {model_path}")
            print(f"✅ Found model: {model_path}")
        # else:
        #     print(f"❌ Model not found: {model_path}")
    
    if not available_models:
        print("❌ No models found! Please train a model first.")
        exit(1)

    print(
        f"📊 {len(available_models)} active checkpoint(s) will each be evaluated "
        f"on the same episode (seed {args.evaluation_seed})"
    )
    all_metrics = []
    shared_episode_keys: Dict[int, Tuple[Any, ...]] = {}
    for model_path in available_models:
        model_label = model_label_from_path(model_path)
        print(f"\n📊 Evaluating model: {model_label}")
        print(f"   Path: {model_path}")

        print("🏗️  Creating evaluation environment...")
        eval_env = create_eval_env_for_model(model_path)

        try:
            model = load_model_for_evaluation(model_path, env=eval_env, device_arg=args.device)
            if model is None:
                raise ValueError("Model loading returned None")
            print(f"✅ Model loaded successfully")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Model observation space: {model.observation_space}")
            print(f"   Model action space: {model.action_space}")

            print(f"\n🧪 Running evaluation...")
            print(f"⚙️🔒 Using fixed evaluation seed ({args.evaluation_seed}) for every model")
            evaluation_data = evaluate_policy_sb3(
                model,
                eval_env,
                num_episodes=args.num_episodes,
                deterministic=True,
                render=False,
                evaluation_seed=args.evaluation_seed,
                action_mode=args.action_mode,
            )

            if not args.no_plots:
                print(f"\n📈 Generating plots for {model_label}...")
                for episode_index, episode_data in enumerate(evaluation_data):
                    plot_prefix = model_label
                    if len(evaluation_data) > 1:
                        plot_prefix = f"{model_label}_episode_{episode_index + 1}"
                    plot_evaluation_sb3(episode_data, save_plots=True, plot_prefix=plot_prefix)

            print_evaluation_summary(evaluation_data)
            for episode_index, episode_data in enumerate(evaluation_data):
                signature = shared_episode_signature(episode_data)
                previous = shared_episode_keys.get(episode_index)
                if previous is None:
                    shared_episode_keys[episode_index] = signature
                    print(
                        f"📌 Shared episode {episode_index + 1}: "
                        f"coins={list(signature[0])} start={signature[1]}"
                    )
                elif previous != signature:
                    raise RuntimeError(
                        "Models were not evaluated on the same episode. "
                        f"Episode {episode_index + 1} for {model_label} is "
                        f"coins={list(signature[0])} start={signature[1]}, "
                        f"but the first model saw coins={list(previous[0])} start={previous[1]}."
                    )
                metrics = episode_metrics(model_label, model_path, episode_data)
                metrics["episode"] = episode_index + 1
                metrics["episode_coins"] = list(signature[0])
                metrics["episode_start"] = signature[1]
                all_metrics.append(metrics)
        except Exception as e:
            print(f"❌ Error evaluating model: {e}")
            print(f"   Model path: {model_path}")
            print(f"   Environment observation space: {eval_env.observation_space}")
            print(f"   Environment action space: {eval_env.action_space}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Model evaluation failed: {e}") from e
        finally: 
            eval_env.close()

    print_comparison_summary(all_metrics)
    comparison_csv = os.path.join(config.LOGDIR, "evaluation_model_comparison.csv")
    save_comparison_csv(all_metrics, comparison_csv)

    print(f"\n🎉 Evaluation completed!")
    if args.no_plots:
        print(f"📁 Comparison CSV saved to: {config.LOGDIR}")
    else:
        print(f"📁 Plots and comparison CSV saved to: {config.LOGDIR}")

    if args.run_env_smoke_test:
        test_variable_portfolio_evaluation()
 
 
 # %%

# %% 
