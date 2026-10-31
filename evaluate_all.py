"""Evaluate original, hover and corrected policies under ORIGINAL reward."""
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

MODELS = {
    "Original PPO": "models/lunar_ppo_500k",
    "Hover PPO": "models/lunar_ppo_hover_500k",
    "Corrected PPO": "models/lunar_ppo_corrected_500k",
}
SEEDS = range(100, 120)


def evaluate(path):
    model = PPO.load(path, device="cpu")
    rewards, lengths, successful_x = [], [], []
    success = crashes = timeouts = 0
    for seed in SEEDS:
        env = gym.make("LunarLander-v3")
        try:
            obs, info = env.reset(seed=seed)
            total = 0.0
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total += reward
                steps += 1
                if terminated or truncated:
                    break
            rewards.append(total)
            lengths.append(steps)
            if terminated and reward == 100.0:
                success += 1
                successful_x.append(abs(float(obs[0])))
            elif truncated:
                timeouts += 1
            else:
                crashes += 1
        finally:
            env.close()
    return {
        "mean_reward": np.mean(rewards),
        "mean_length": np.mean(lengths),
        "successes": success,
        "crashes": crashes,
        "timeouts": timeouts,
        "score_200": sum(r >= 200 for r in rewards),
        "mean_success_x": np.mean(successful_x) if successful_x else None,
    }


def main():
    for name, path in MODELS.items():
        r = evaluate(path)
        print(f"\n========== {name} ==========")
        print(f"Mean reward: {r['mean_reward']:.2f}")
        print(f"Mean episode length: {r['mean_length']:.1f}")
        print(f"Successful rests: {r['successes']}/20")
        print(f"Crashes: {r['crashes']}/20")
        print(f"Timeouts: {r['timeouts']}/20")
        print(f"Episodes with reward >= 200: {r['score_200']}/20")
        if r['mean_success_x'] is not None:
            print(f"Mean successful landing |x|: {r['mean_success_x']:.4f}")
        else:
            print("No successful landings to measure")
        print("================================")


if __name__ == "__main__":
    main()
