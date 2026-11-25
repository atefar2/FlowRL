# Episode-relative 8-of-13: episode-relative positions, no identity, one-step log growth.
# Run: bash training.sh td3-8of13 [additional training options]
# Medium attention + DeepSets; basket opportunity heads in actor and twin Qs.
# Start a fresh run because the observation and network contracts have changed.
if [ "${1:-}" = "flowrl-8of13" ]; then
    shift
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
    source .venv/bin/activate || exit 1
    exec python train_simple_mlp_og.py \
        --algorithm FlowRL --mlp-size multitask \
        --timesteps 2600000 --reward-type STRUCTURED_CREDIT \
        --crl-profile adaptive --variable-portfolio --encoder-variant baseline \
        --use-attention --attention-type variable_coin_attention "$@"
fi
if [ "${1:-}" = "td3-8of13" ]; then
    shift
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
    source .venv/bin/activate || exit 1
    exec python train_simple_mlp_og.py \
        --algorithm TD3 --mlp-size medium \
        --timesteps 2600000 --reward-type LOG_RETURN \
        --crl-profile adaptive --variable-portfolio --encoder-variant baseline \
        --use-attention --attention-type variable_coin_attention \
        --proportional-exp-mode boltzmann_disagreement "$@"
fi

echo "Usage: bash training.sh {td3-8of13|flowrl-8of13} [training options]" >&2
exit 2

# The historical command catalogue contains prose and is reference material.
: <<'LEGACY_TRAINING_COMMAND_CATALOGUE'
# Recommended First Training Run
python train_with_attention.py --algorithm PPO --attention coin_attention --config medium --timesteps 50000

# Activate virtual environment
source .venv/bin/activate
python train_variable_attention.py --algorithm PPO --attention variable_coin_attention --config heavy --timesteps 200000 --envs 4 --variable

# --- Recommended First Training Run ---
# A good starting point to verify the setup works.
echo "Running initial PPO training run..."
python train_with_attention.py --algorithm PPO --attention coin_attention --config medium --timesteps 50000 --variable

# Move data to RUNPOD 
ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91  "mkdir -p /workspace/data" && rsync -avz --progress -e "ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" /Users/amen/Documents/RL_MultiHeadAttention/data/causal_cleaned_preprocessed_restructured.csv root@213.192.2.91 :/workspace/data/ && ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91  "ls -lh /workspace/data/" | cat

# Perfect! I've updated your training.sh script with the new RunPod connection details from your interface. Here's what I changed: 
# Updated in line 14:
# IP Address: 213.192.2.91 8 → 213.192.2.917

# Show the last 10 lines of crl_history.csv:
tail crl_history.csv

# To grep lines matching a pattern (e.g., containing "SUCCESS") from the file and see last 10 matches:
grep SUCCESS crl_history.csv | tail

# To view the last 20 values from column 13 (assuming comma is the separator):
tail -n 50 output_ar_gamma.csv | cut -d',' -f3

# To grep, then see just column 5 from those matches:
grep "SUCCESS" crl_history.csv | tail | cut -d',' -f5

python3 evaluate_run.py
python3 evaluate_run_data.py
# Port: 22173 → 40144
# move file from checkpoints to parent directory
mv model_500000_steps.zip ..
#!
# new runpod - install rsync first, then transfer data folder
# ssh root@213.192.2.91 -p 40144 -i ~/.ssh/id_ed25519
/Users/amen/.ssh/config
# Cursor Remote SSH config for the RunPod above:
# Host runpod-td3
#     HostName 213.192.2.91
#     User root
#     IdentityFile ~/.ssh/id_ed25519
#     Port 40144
#     ServerAliveInterval 60
#     ServerAliveCountMax 3
#     TCPKeepAlive yes
#     Compression yes
#     StrictHostKeyChecking no
#     UserKnownHostsFile ~/.ssh/known_hosts
# Open remote workspace in Cursor:
# cursor --new-window --folder-uri "vscode-remote://ssh-remote+runpod-td3/workspace"

# download models folder stated from runpod:
rsync -av --partial --progress --stats \
  -e 'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p 40144 -i /Users/amen/.ssh/id_ed25519' \
  'root@213.192.2.91:/workspace/models/FlowRL_tf_agents_style_multitask_STRUCTURED_CREDIT_variable_coin_attention_variable_20261029_020717_241717' \
  '/Users/amen/Documents/RL_MultiHeadAttention/models/'

# data/causal_...* .csv
ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "apt update && apt install -y rsync && mkdir -p /workspace/data" && rsync -avz --progress -e "ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" /Users/amen/Documents/RL_MultiHeadAttention/data/causal_cleaned_preprocessed_restructured_og.csv root@213.192.2.91:/workspace/data/ && ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "ls -lh /workspace/data/" | cat

# models/TD3_tf_agents_style_multitask_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_basket_history_g966_norecon_20261013_215231_baseline
ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "apt update && apt install -y rsync && mkdir -p /workspace/models" && rsync -av --partial --progress -e "ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" /Users/amen/Documents/RL_MultiHeadAttention/models/TD3_tf_agents_style_multitask_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_basket_history_g966_norecon_20261013_215231_baseline root@213.192.2.91:/workspace/models/ && ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "ls -lh /workspace/models/TD3_tf_agents_style_multitask_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_basket_history_g966_norecon_20261013_215231_baseline/"

#RLKIT — sync whole rlkit tree to RunPod but omit rlkit/data (top-level data dir only)
ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "apt update && apt install -y rsync && mkdir -p /workspace/rlkit" && rsync -avz --progress --exclude='/data/' -e "ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" /Users/amen/Documents/RL_MultiHeadAttention/rlkit/ root@213.192.2.91:/workspace/rlkit/ && ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "ls -lah /workspace/rlkit/ | head -50" | cat

# Send this specific TD3 model folder and all of its contents to RunPod
ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "apt update && apt install -y rsync && mkdir -p /workspace/models/TD3_tf_agents_style_medium_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_20260427_125830_baseline" && rsync -avz --progress -e "ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" /Users/amen/Documents/RL_MultiHeadAttention/models/TD3_tf_agents_style_medium_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_20260427_125830_baseline/ root@213.192.2.91:/workspace/models/TD3_tf_agents_style_medium_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_20260427_125830_baseline/ && ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "ls -lah /workspace/models/TD3_tf_agents_style_medium_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_20260427_125830_baseline/" | cat

# Alternative: Use scp instead of rsync (no installation needed)
# ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91  "mkdir -p /workspace/data" && scp -P 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no /Users/amen/Documents/RL_MultiHeadAttention/data/causal_cleaned_preprocessed_restructured.csv root@213.192.2.91 :/workspace/data/ && ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91  "ls -lh /workspace/data/" | cat

# I'll explain the issue before we make any changes to your code.
# with subtraction not division 
RL_MODEL_PATH = "models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260211_225942_baseline/model_2072400_steps.zip"
# RL_MODEL_PATH = "models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260211_152526_baseline/model_2304000_steps.zip"

# with division not subtranction e
# RL_MODEL_PATH = "models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260214_185338_encoder_res_aux0.05.05/model_1268400_steps.zip"
# Why Your GPU Isn't Being U sed on RunPod 

# Upload the 3 model files to RunPod models folder using rsync
# Create subfolders for each TD3 model directory and upload each zip to its corresponding subfolder
# Model 1: TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260211_225942_baseline
ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "mkdir -p /workspace/models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260211_225942_baseline" && rsync -avz --progress -e "ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" /Users/amen/Documents/RL_MultiHeadAttention/models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260211_225942_baseline/model_2072400_steps.zip root@213.192.2.91:/workspace/models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260211_225942_baseline/ && ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "ls -lh /workspace/models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260211_225942_baseline/" | cat

# Model 2: TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260211_152526_baseline
ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "mkdir -p /workspace/models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260211_152526_baseline" && rsync -avz --progress -e "ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" /Users/amen/Documents/RL_MultiHeadAttention/models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260211_152526_baseline/model_2304000_steps.zip root@213.192.2.91:/workspace/models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260211_152526_baseline/ && ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "ls -lh /workspace/models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260211_152526_baseline/" | cat

# Model 3: TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260214_185338_encoder_res_aux0.05/model_1268400_steps.zip
ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "mkdir -p /workspace/models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260214_185338_encoder_res_aux0.05" && rsync -avz --progress -e "ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" /Users/amen/Documents/RL_MultiHeadAttention/models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260214_185338_encoder_res_aux0.05/model_1268400_steps.zip root@213.192.2.91:/workspace/models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260214_185338_encoder_res_aux0.05/ && ssh -p 40144 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.192.2.91 "ls -lh /workspace/models/TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_mlp_variable_CRL_adaptive_20260214_185338_encoder_res_aux0.05/" | cat

Looking at your code, I can see you're using `device="auto"` in your model initialization (lines 626 and 1065 in `train_simple_mlp.py`). While this **should** automatically detect and use the GPU, there are several common reasons why it's not working:

### Most Likely Causes:

1. **PyTorch installed without CUDA support (MOST COMMON)**
   - When you install PyTorch via `pip install torch`, it often installs the **CPU-only version** by default
   - On RunPod with an A40 GPU, you need PyTorch compiled with CUDA support
   - You can check this by running: `python -c "import torch; print(torch.cuda.is_available())"`
   - If it prints `False`, that's your problem

2. **CUDA/Driver mismatch**
   - RunPod's NVIDIA drivers might be a different version than what your PyTorch expects
   - The A40 requires CUDA 11.x or 12.x

3. **Wrong virtual environment**
   - Your `.venv` might have been created on your Mac (CPU-only PyTorch)
   - When you transferred it to RunPod, it still has the Mac's CPU-only version

### What's Happening Now:

When `device="auto"` doesn't detect a GPU, Stable Baselines3 falls back to CPU. Your code runs fine, but slowly, because:
- `device="auto"` → checks `torch.cuda.is_available()` → returns `False` → uses CPU
- No error is raised, so you don't see any warning

### How to Diagnose (before fixing):

**On your RunPod instance, run these commands:**

```bash
# Check if NVIDIA GPU is visible
nvidia-smi

# Check if PyTorch can see CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device count:', torch.cuda.device_count())"

# Check installed PyTorch version
pip show torch
```

**Expected results if GPU is working:**
- `nvidia-smi` should show your A40 GPU
- `torch.cuda.is_available()` should print `True`
- `torch.cuda.device_count()` should print `1` or more

**If it prints `False`**, you need to reinstall PyTorch with CUDA support.

---

Would you like me to:
1. **First show you the diagnostic commands** to confirm the issue?
2. **Then provide the fix** (reinstalling PyTorch with CUDA + adding GPU logging to your code)?

# Let me know what you find from the diagnostics, and I'll provide the exact solution!

python train_simple_mlp.py \
    --algorithm PPO \
    --mlp-size heavy \
    --timesteps 10000 \
    --reward-type STRUCTURED_CREDIT \
    --variable-portfolio \
    --crl-profile adaptive

# === SIMPLE MLP TRAINING (NO ATTENTION) ===
python train_simple_mlp.py \                                         
    --algorithm TD3 \
    --mlp-size heavy \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --disable-crl \
    --variable-portfolio

# ✅ RECOMMENDED: Use train_simple_mlp.py with clear --mlp-size parameter
# ORIGINAL (PROBLEMATIC) - 96 coins is too many for stable learning
# python train_simple_mlp.py \
#     --algorithm TD3 \
#     --mlp-size heavy \
#     --timesteps 10000 \
#     --
reward-type STRUCTURED_CREDIT \
#     --crl-profile adaptive \
#     --variable-portfolio

# ✅  SCALABILITY FIX: Limit to manageable portfolio size for learning
python train_simple_mlp.py \
    --algorithm TD3 \
    --mlp-size heavy \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --aux-coef 0.05 \
    --aux-mask-prob 0.15

python train_simple_mlp_og.py \
    --algorithm TD3 \
    --mlp-size heavy \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \

ps -eo pid,ppid,rss,%mem,%cpu,cmd --sort=-rss | head -30
kill -9 xxxx xxx xxx   
ps -eo pid,ppid,rss,%mem,%cpu,cmd --sort=-rss | grep 'train_simple_mlp_og.py'
ps -eo pid,ppid,rss,%mem,%cpu,lstart,cmd --sort=lstart | grep 'train_simple_mlp_og.py' | tac

python train_simple_mlp_og.py \
    --algorithm TD3 \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention

# SAC nondeterministic counterpart to the attention TD3 dual-CRL run
python train_simple_mlp_og.py \
    --algorithm SAC \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention

# Proportional-exp ablation: local Boltzmann only
python train_simple_mlp_og.py \
    --algorithm TD3 \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --proportional-exp-mode local_boltzmann

#⭐️ Proportional-exp ablation: Boltzmann + disagreement-aware 
python train_simple_mlp_og.py \
    --algorithm TD3 \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --proportional-exp-mode boltzmann_disagreement 

python train_simple_mlp_og.py \
    --algorithm TD3 \
    --mlp-size multitask \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --proportional-exp-mode boltzmann_disagreement 

python train_simple_mlp_og.py \
    --algorithm TD3 \
    --mlp-size multitask \
    --timesteps 41600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --proportional-exp-mode boltzmann_disagreement \
    --gpu-sim-envs 32 \
    --gpu-sim-updates-per-step 2

python train_simple_mlp_og.py \
    --algorithm TD3 \
    --mlp-size multitask \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --proportional-exp-mode boltzmann_disagreement \
    --resume-checkpoint models/TD3_tf_agents_style_multitask_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_basket_history_g966_norecon_20261013_215231_baseline/checkpoints/model_1150000_steps.zip

python train_simple_mlp_og.py \
    --algorithm TD3 \
    --mlp-size multitask \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --fixed-eight-of-thirteen \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --proportional-exp-mode boltzmann_disagreement 

# NO Proportional-exp ablation: Boltzmann + disagreement-aware 
python train_simple_mlp_og.py \
    --algorithm TD3 \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention

# partial data loading for ram efficency
python train_simple_mlp_og.py \
    --train-data-load-mode window \
    --train-window-days 90 \
    --train-window-rotation auto \
    --train-window-direction forward \
    --algorithm TD3 \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention

#⭐️ DATA LOADER Proportional-exp ablation: Boltzmann + disagreement-aware 
python train_simple_mlp_og.py \
    --train-data-load-mode window \
    --train-window-days 90 \
    --train-window-rotation auto \
    --train-window-direction forward \
    --algorithm TD3 \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --proportional-exp-mode boltzmann_disagreement

# ssl — financial-primary reward, global scaler, fixed-basket eval via config.py defaults.
python train_simple_mlp_og.py \
    --train-data-load-mode window \
    --train-window-days 90 \
    --train-window-rotation auto \
    --train-window-direction forward \
    --algorithm TD3 \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --mask-ssl-coef 1.0 \
    --aux-mask-prob 0.15 \
    --mask-ssl-lr 1e-5 \
    --mask-ssl-weight-decay 1e-4 \
    --mask-ssl-update-interval 1 \
    --mask-ssl-warmup-steps 0 \
    --mask-ssl-actor-warmup-steps 250000 \
    --mask-ssl-actor-ramp-steps 200000 \
    --proportional-exp-mode boltzmann_disagreement

# 8-of-13 multitask capacity experiment. The profile itself selects the
# 256-wide actor/critic, batch_size=200 original rows, learning_starts=1,000, and the
# gamma 0.966, whose reciprocal effective horizon is about 29.4 market steps.
python train_simple_mlp_og.py \
    --train-data-load-mode window \
    --train-window-days 90 \
    --train-window-rotation auto \
    --train-window-direction forward \
    --algorithm TD3 \
    --mlp-size multitask \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --proportional-exp-mode boltzmann_disagreement

# Fixed-task ablation of the experiment above. The CSV, asset-id vocabulary,
# membership vector, policy, critic, reward, replay, and hyperparameters remain
# the current system; only episode basket sampling is collapsed to the exact
# April-27 eight-name basket inside the full 13-asset universe.
python train_simple_mlp_og.py \
    --train-data-load-mode window \
    --train-window-days 90 \
    --train-window-rotation auto \
    --train-window-direction forward \
    --algorithm TD3 \
    --mlp-size multitask \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --fixed-eight-of-thirteen \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --proportional-exp-mode boltzmann_disagreement

# ⭐️
python train_simple_mlp_og.py \
    --train-data-load-mode window \
    --train-window-days 90 \
    --train-window-rotation auto \
    --train-window-direction forward \
    --algorithm TD3 \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --mask-ssl-coef 0.05 \
    --aux-mask-prob 0.15 \
    --mask-ssl-lr 1e-5 \
    --mask-ssl-weight-decay 1e-4 \
    --mask-ssl-update-interval 1 \
    --mask-ssl-critic-warmup-steps 20000 \
    --mask-ssl-critic-ramp-steps 80000 \
    --mask-ssl-disable-actor \
    --proportional-exp-mode boltzmann_disagreement


python train_simple_mlp_og.py \
    --train-data-load-mode window \
    --train-window-days 90 \
    --train-window-rotation auto \
    --train-window-direction forward \
    --algorithm TD3 \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --mask-ssl-coef 1.0 \
    --aux-mask-prob 0.15 \
    --mask-ssl-lr 1e-5 \
    --mask-ssl-weight-decay 1e-4 \
    --mask-ssl-update-interval 1 \
    --mask-ssl-warmup-steps 0 \
    --mask-ssl-actor-warmup-steps 250000 \
    --mask-ssl-actor-ramp-steps 200000 \
    --proportional-exp-mode boltzmann_disagreement \
    --rank-aux-critic-lambda 0.01 \
    --rank-aux-actor-lambda 0.005 \
    --rank-aux-horizons 5,10 \
    --rank-aux-batch-size 64


# Cross sectional ListMLE intergrated to TD3

python train_simple_mlp_og.py \
    --train-data-load-mode window \
    --train-window-days 90 \
    --train-window-rotation auto \
    --train-window-direction forward \
    --algorithm TD3 \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --proportional-exp-mode boltzmann_disagreement \
    --rank-aux-critic-score-mode factorized \
    --rank-aux-critic-lambda 0.01 \
    --rank-aux-actor-lambda 0.005 \
    --rank-aux-horizons 5,10 \
    --rank-aux-batch-size 64


python train_simple_mlp_og.py \
    --train-data-load-mode window \
    --train-window-days 90 \
    --train-window-rotation auto \
    --train-window-direction forward \
    --algorithm TD3 \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --proportional-exp-mode boltzmann_disagreement \
    --rank-aux-critic-score-mode factorized \
    --rank-aux-critic-lambda 0.01 \
    --rank-aux-actor-lambda 0.005 \
    --rank-aux-horizons 5,10 \
    --rank-aux-batch-size 64 \
    --rank-aux-update-interval 4 \
    --plot-every-evals 5


# IQL A practical first RunPod command would be:
python train_iql_offline.py \
  --replay-buffer models/TD3_tf_agents_style_medium_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_proexp_disagreement_20260427_125830_baseline/checkpoints/replaybuffer_150000_steps.zip \
  --updates 1000 \
  --batch-size 200 \
  --mlp-size medium \
  --device cuda

# Resume training from checkpoint (RunPod interrupted at step 1132000)
python train_simple_mlp_og.py \
    --algorithm TD3 \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline \
    --use-attention \
    --attention-type variable_coin_attention \
    --resume-checkpoint models/TD3_tf_agents_style_medium_STRUCTURED_CREDIT_variable_coin_attention_variable_CRL_adaptive_20260325_220250_baseline/checkpoints/model_1132000_steps.zip

python train_simple_mlp_og.py \
    --algorithm TD3 \
    --mlp-size medium \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --use-attention \
    --attention-type variable_coin_attention \
    --use-temporal-conv


# Extractor.py    
python train_simple_mlp.py \
    --algorithm TD3 \
    --mlp-size heavy \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant encoder \
    --aux-coef 0.05 \
    --aux-mask-prob 0.15
# Extractor_res.py
python train_simple_mlp_og.py \
    --algorithm TD3 \
    --mlp-size heavy \
    --timesteps 2600000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant encoder_res \
    --aux-coef 0.05 \
    --aux-mask-prob 0.15


#--------------------    
    # --crl-profile aggressive
    # --use-fixed-portfolio
python train_simple_mlp.py \
    --algorithm TD3 \
    --mlp-size heavy \
    --timesteps 2600000 \
    --reward-type TRANSACTION_COST \
    --crl-profile adaptive \
    --variable-portfolio
python train_simple_mlp.py \
    --algorithm TD3 \
    --mlp-size heavy \
    --timesteps 2600000 \
    --reward-type TRANSACTION_COST \
    --crl-profile adaptive \
    --variable-portfolio \
    --encoder-variant baseline 
    
# Polychromic objective (Set RL)
# Run with Set RL enabled:
python train_simple_mlp.py \
    --algorithm TD3 \
    --variable-portfolio \
    --crl-profile adaptive \
    --use-setrl \
    --setrl-profile balanced \
    --reward-type STRUCTURED_CREDIT \
    --timesteps 800000

# === ATTENTION-BASED TRAINING (VARIABLE PORTFOLIO) ===
# Same setup as the MLP above, but enables the attention mechanism.
python train_simple_mlp.py \
    --algorithm TD3 \
    --mlp-size heavy \
    --timesteps 10000 \
    --reward-type TRANSACTION_COST \
    --crl-profile adaptive \
    --variable-portfolio \
    --use-attention \
    --attention-type variable_coin_attention


    # --crl-profile aggressive
    # --use-fixed-portfolio


# === VARIABLE PORTFOLIO MLP TRAINING ===
# FOR VARIABLE MLP, USE USE_VARIABLE_PORTFOLIO
# 🎲 Episode coins: ['DASH', 'STR', 'LTC'] (3 coins)
python train_simple_mlp.py \
    --algorithm TD3 \
    --mlp-size heavy \
    --timesteps 100000 \
    --reward-type TRANSACTION_COST \
    --variable-portfolio \
    --crl-profile adaptive

# Alternative configurations (commented out):
python train_simple_mlp.py \
    --algorithm TD3 \
    --mlp-size heavy \
    --timesteps 100000 \
    --reward-type TRANSACTION_COST \
    --variable-portfolio \
    --disable-crl

# Second training command:
python train_simple_mlp.py \
    --algorithm TD3 \
    --mlp-size heavy \
    --timesteps 100000 \
    --reward-type TRANSACTION_COST \
    --variable-portfolio \
    --crl-profile adaptive


# ------------------------------------------------------------

# Train with CRL (adaptive penalty)
python train_simple_mlp.py --algorithm TD3 --crl-profile balanced

# Train with different profiles
python train_simple_mlp.py --algorithm TD3 --crl-profile conservative
python train_simple_mlp.py --algorithm TD3 --crl-profile aggressive

# Train with CRL (adaptive)
python train_simple_mlp.py --algorithm TD3 --crl-profile adaptive

# Disable CRL (use fixed penalty)
python train_simple_mlp.py --algorithm TD3 --disable-crl

# Test the integration
python test_crl_integration.py

# Profile |	Constraint Threshold |	Description
# conservative | 0.01 | Minimal trading, very stable
# balanced | 0.05 | Balance stability and responsiveness
# aggressive | 0.15 | Allow quick market responses
# adaptive | 0.03 | Start conservative, adapt over time

# ------------------------------------------------------------

# Test MLP functionality (recommended first step)
python train_simple_mlp.py --test

# Main MLP training with clear parameter naming # light is 256, 128
python train_simple_mlp.py --algorithm DDPG --mlp-size light --timesteps 10000

# heavy is 1024, 512, 256
python train_simple_mlp.py --algorithm DDPG --mlp-size heavy --timesteps 100000

# Different MLP sizes (architectures)
python train_simple_mlp.py --algorithm PPO --mlp-size light --timesteps 50000    # Small: [256, 128] → 128 features
python train_simple_mlp.py --algorithm PPO --mlp-size heavy --timesteps 800000   # Large: [1024, 512, 256] → 512 features

# MLP with different algorithms (NO ATTENTION)
python train_simple_mlp.py --algorithm DDPG --mlp-size medium --timesteps 100000
python train_simple_mlp.py --algorithm SAC --mlp-size medium --timesteps 100000



# Compare all algorithms with MLP
python train_simple_mlp.py --compare
# Quick test (600 steps) - recommended for verification
python train_simple_mlp.py --test-issue3
# Full training with model saving
python train_simple_mlp.py --tf-agents-style --algorithm DDPG --mlp-size heavy

# Alternative approach (confusing naming - "config" refers to MLP architecture when using --attention mlp)
# python train_with_attention.py --algorithm PPO --attention mlp --config medium --timesteps 100000

# fixed coin attention no variable
python train_variable_attention.py \
    --algorithm DDPG \
    --attention coin_attention \
    --config heavy \
    --timesteps 200000 \
    --envs 1 \
    --fixed

# --- Production Model Training (PPO) ---
# Example of training a more robust model with more steps and parallel environments.
echo "Starting production PPO training..."
python train_variable_attention.py \
    --algorithm PPO \
    --attention variable_coin_attention \
    --config heavy \
    --timesteps 200000 \
    --envs 4 \
    --variable


# --- TD3 Training Example ---
# TD3 is an off-policy algorithm suitable for continuous action spaces.
echo "Starting TD3 training..."
python train_variable_attention.py \
    --algorithm TD3 \
    --attention variable_coin_attention \
    --config heavy \
    --timesteps 10000 \
    --envs 1 \
    --variable


# --- DDPG Training Example ---
# DDPG is another off-policy algorithm, a precursor to TD3.
echo "Starting DDPG training..."
python train_variable_attention.py \
    --algorithm DDPG \
    --attention variable_coin_attention \
    --config heavy \
    --timesteps 200000 \
    --envs 1 \
    --variable


# --- Other Alternative Configurations ---
echo "Running other configurations..."
python train_variable_attention.py --algorithm SAC --config heavy
python train_variable_attention.py --algorithm DDPG --config medium
python train_variable_attention.py --attention coin_attention --config light


# # Quick test training (5K steps)
# python train_with_attention.py --algorithm PPO --timesteps 5000

# # Full training with coin attention (recommended)
# python train_with_attention.py --algorithm PPO --attention coin_attention --config medium --timesteps 100000

# # Train with SAC (continuous control algorithm)
# python train_with_attention.py --algorithm SAC --attention coin_attention --config heavy --timesteps 200000

# # Compare different algorithms
# python train_with_attention.py --compare
LEGACY_TRAINING_COMMAND_CATALOGUE
