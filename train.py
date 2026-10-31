"""Train a new Lunar Lander PPO agent for a selected reward experiment."""
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from custom_reward import CenteringReward, HoverReward, CorrectedReward, PrecisionReward

EXPERIMENT = "precision"  # baseline, center, hover, corrected, precision
TOTAL_TIMESTEPS = 500_000
SEED = 42


def make_env(experiment):
    env = gym.make("LunarLander-v3")
    if experiment == "center":
        env = CenteringReward(env, strength=5.0, gamma=0.99)
    elif experiment == "hover":
        env = HoverReward(env, bonus=1.0)
    elif experiment == "corrected":
        env = CorrectedReward(env, step_penalty=0.05)
    elif experiment == "precision":
        env = PrecisionReward(env, target_x=0.15, landing_bonus=50.0,
                              step_penalty=0.05)
    elif experiment != "baseline":
        raise ValueError(f"Unknown experiment: {experiment}")
    return Monitor(env)


def main():
    os.makedirs("models", exist_ok=True)
    env = make_env(EXPERIMENT)
    try:
        model = PPO("MlpPolicy", env, device="cpu", verbose=1, seed=SEED)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        path = f"models/lunar_ppo_{EXPERIMENT}_500k"
        model.save(path)
        print(f"Saved: {path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
