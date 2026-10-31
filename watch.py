"""Watch a saved PPO model fly five Lunar Lander episodes."""
import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = "models/lunar_ppo_precision_500k"
EPISODES = 5


def main():
    model = PPO.load(MODEL_PATH, device="cpu")
    env = gym.make("LunarLander-v3", render_mode="human")
    try:
        for episode in range(EPISODES):
            obs, info = env.reset()
            total_reward = 0.0
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    print(f"Episode {episode + 1}: reward={total_reward:.2f}, "
                          f"steps={steps}, terminated={terminated}, "
                          f"truncated={truncated}, final_x={float(obs[0]):.3f}")
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
