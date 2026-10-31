"""Compare shaped and original rewards on the same hover-model episodes."""
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from custom_reward import HoverReward

MODEL_PATH = "models/lunar_ppo_hover_500k"
SEEDS = range(100, 120)
BONUS = 1.0


def main():
    model = PPO.load(MODEL_PATH, device="cpu")
    results = []
    for seed in SEEDS:
        env = HoverReward(gym.make("LunarLander-v3"), bonus=BONUS)
        try:
            obs, info = env.reset(seed=seed)
            shaped = original = 0.0
            bonus_steps = steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                paid = HoverReward.qualifies(obs, terminated)
                bonus_steps += int(paid)
                shaped += reward
                original += reward - (BONUS if paid else 0.0)
                steps += 1
                if terminated or truncated:
                    break
            print(f"Seed {seed}: original={original:.1f}, modified={shaped:.1f}, "
                  f"bonus_steps={bonus_steps}, steps={steps}, timeout={truncated}")
            results.append((original, shaped, bonus_steps, int(truncated)))
        finally:
            env.close()
    data = np.asarray(results, dtype=float)
    print("\n========== SUMMARY ==========")
    print(f"Mean original reward: {data[:, 0].mean():.2f}")
    print(f"Mean modified reward: {data[:, 1].mean():.2f}")
    print(f"Mean bonus steps: {data[:, 2].mean():.1f}")
    print(f"Timeouts: {int(data[:, 3].sum())}/{len(SEEDS)}")
    print("=============================")


if __name__ == "__main__":
    main()
