"""Check successful centred rests under the ORIGINAL environment reward."""
import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = "models/lunar_ppo_precision_500k"  # switch to corrected for baseline
TARGET_X = 0.15
SEEDS = range(100, 120)


def main():
    model = PPO.load(MODEL_PATH, device="cpu")
    successes = centered = crashes = timeouts = 0
    for seed in SEEDS:
        env = gym.make("LunarLander-v3")
        try:
            obs, info = env.reset(seed=seed)
            total = 0.0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total += reward
                if terminated or truncated:
                    break
            final_x = float(obs[0])
            landed = terminated and reward == 100.0
            inside = landed and abs(final_x) <= TARGET_X
            successes += int(landed)
            centered += int(inside)
            timeouts += int(truncated)
            crashes += int(terminated and not landed)
            print(f"Seed {seed}: Reward={total:.1f}, Final x={final_x:.3f}, "
                  f"Landed={landed}, Centered={inside}")
        finally:
            env.close()
    print("\n========== PRECISION SUMMARY ==========")
    print(f"Successful rests: {successes}/{len(SEEDS)}")
    print(f"Centered landings: {centered}/{len(SEEDS)}")
    print(f"Crashes: {crashes}/{len(SEEDS)}")
    print(f"Timeouts: {timeouts}/{len(SEEDS)}")
    if successes:
        print(f"Precision among successful rests: {100 * centered / successes:.1f}%")
    print("=======================================")


if __name__ == "__main__":
    main()
