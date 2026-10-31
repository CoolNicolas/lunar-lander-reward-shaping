"""Record trained policy episodes as video; optional dependency: moviepy."""
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

MODEL_PATH = "models/lunar_ppo_precision_500k"
EPISODES = 3


def main():
    model = PPO.load(MODEL_PATH, device="cpu")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda episode: True,
                      name_prefix="precision-ppo")
    try:
        for _ in range(EPISODES):
            obs, info = env.reset()
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    break
    finally:
        env.close()
    print("Videos saved in videos/")


if __name__ == "__main__":
    main()
