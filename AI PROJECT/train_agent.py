#!/usr/bin/env python3
import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from torch.nn import ReLU
import policy
from multiprocessing import freeze_support

# ================== Network Architecture ==================
def flatten_actor_params(model):
    """Extract and flatten parameters from deep network"""
    params = []
    policy_net = model.policy.mlp_extractor.policy_net
    
    # Layer 1 (8->256)
    params.append(policy_net[0].weight.data.cpu().numpy().flatten())
    params.append(policy_net[0].bias.data.cpu().numpy().flatten())
    
    # Layer 2 (256->256)
    params.append(policy_net[2].weight.data.cpu().numpy().flatten())
    params.append(policy_net[2].bias.data.cpu().numpy().flatten())
    
    # Layer 3 (256->128)
    params.append(policy_net[4].weight.data.cpu().numpy().flatten())
    params.append(policy_net[4].bias.data.cpu().numpy().flatten())
    
    # Action layer (128->4)
    params.append(model.policy.action_net.weight.data.cpu().numpy().flatten())
    params.append(model.policy.action_net.bias.data.cpu().numpy().flatten())
    
    flat_params = np.concatenate(params)
    assert flat_params.size == 101508, f"Parameter size mismatch: {flat_params.size}"
    return flat_params

def unflatten_actor_params(model, flat_params):
    """Load parameters into model"""
    idx = 0
    policy_net = model.policy.mlp_extractor.policy_net
    
    # Layer 1 (8->256)
    W1 = flat_params[idx:idx+2048].reshape(256, 8); idx += 2048
    b1 = flat_params[idx:idx+256]; idx += 256
    # Layer 2 (256->256)
    W2 = flat_params[idx:idx+65536].reshape(256, 256); idx += 65536
    b2 = flat_params[idx:idx+256]; idx += 256
    # Layer 3 (256->128)
    W3 = flat_params[idx:idx+32768].reshape(128, 256); idx += 32768
    b3 = flat_params[idx:idx+128]; idx += 128
    # Action layer (128->4)
    W4 = flat_params[idx:idx+512].reshape(4, 128); idx += 512
    b4 = flat_params[idx:idx+4]; idx += 4

    with torch.no_grad():
        policy_net[0].weight.copy_(torch.tensor(W1, dtype=torch.float32))
        policy_net[0].bias.copy_(torch.tensor(b1, dtype=torch.float32))
        policy_net[2].weight.copy_(torch.tensor(W2, dtype=torch.float32))
        policy_net[2].bias.copy_(torch.tensor(b2, dtype=torch.float32))
        policy_net[4].weight.copy_(torch.tensor(W3, dtype=torch.float32))
        policy_net[4].bias.copy_(torch.tensor(b3, dtype=torch.float32))
        model.policy.action_net.weight.copy_(torch.tensor(W4, dtype=torch.float32))
        model.policy.action_net.bias.copy_(torch.tensor(b4, dtype=torch.float32))

# ================== Training Configuration ==================
policy_kwargs = dict(
    activation_fn=ReLU,
    net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
    ortho_init=True
)

ppo_config = {
    "learning_rate": 2.5e-4,
    "n_steps": 8192,
    "batch_size": 2048,
    "n_epochs": 20,
    "gamma": 0.999,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "ent_coef": 0.002,
    "vf_coef": 0.6,
    "max_grad_norm": 0.7,
    "target_kl": 0.02,
    "policy_kwargs": policy_kwargs,
    "device": "cpu"  # Force CPU usage
}

# ================== Environment Setup ==================
def make_env(seed=None):
    def _init():
        env = gym.make("LunarLander-v3")
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init

# ================== Resumable Training Logic ==================
class HighScoreCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_avg_reward = -np.inf
        self.rewards_buffer = []
        self.last_log = 0
        self.log_interval = 100000
        
        if os.path.exists("best_score.npy"):
            try:
                score_array = np.load("best_score.npy")
                self.best_avg_reward = float(score_array.item())
                print(f"Resumed best score: {self.best_avg_reward:.2f}")
            except Exception as e:
                print(f"Error loading best score: {str(e)}")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards_buffer.append(info["episode"]["r"])
        
        if self.num_timesteps - self.last_log >= self.log_interval:
            if len(self.rewards_buffer) >= 100:
                avg_reward = np.mean(self.rewards_buffer[-100:])
                print(f"\nTimestep {self.num_timesteps:,} | Average 100-episode reward: {avg_reward:.2f}")
            self.last_log = self.num_timesteps
        
        if len(self.rewards_buffer) >= 100:
            current_avg = np.mean(self.rewards_buffer[-100:])
            if current_avg > self.best_avg_reward:
                self.best_avg_reward = current_avg
                flat_params = flatten_actor_params(self.model)
                np.save("best_policy.npy", flat_params)
                np.save("best_score.npy", np.array(self.best_avg_reward))
                print(f"🔥 New Best: {current_avg:.2f} | Policy Saved!")
        
        return True

def load_or_initialize_model(env):
    """Handle resumable training logic"""
    model = PPO(
        "MlpPolicy",
        env,
        verbose=2,
        **ppo_config
    )
    
    if os.path.exists("best_policy.npy"):
        try:
            flat_params = np.load("best_policy.npy")
            unflatten_actor_params(model, flat_params)
            print("✅ Resumed training from best_policy.npy")
        except Exception as e:
            print(f"⚠️ Failed to load policy: {str(e)}")
            print("Starting fresh training session")
    
    return model

# ================== Main Execution ==================
if __name__ == "__main__":
    freeze_support()
    
    num_envs = 8
    env = SubprocVecEnv([make_env(seed=i) for i in range(num_envs)])

    model = load_or_initialize_model(env)

    checkpoint_cb = CheckpointCallback(
        save_freq=250_000,
        save_path="./checkpoints",
        name_prefix="deep_llander"
    )
    highscore_cb = HighScoreCallback()

    print("Starting deep network training...")
    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_cb, highscore_cb],
        progress_bar=True
    )

    try:
        from evaluate_agent import evaluate_policy
        best_params = np.load("best_policy.npy")
        avg_score = evaluate_policy(best_params, policy.policy_action)
        print(f"\n🏆 Final Average Score: {avg_score:.2f}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
    finally:
        env.close()