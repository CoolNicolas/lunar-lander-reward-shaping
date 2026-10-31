"""Custom reward wrappers for Gymnasium LunarLander-v3."""
import gymnasium as gym


class CenteringReward(gym.Wrapper):
    """Potential shaping based on horizontal alignment."""

    def __init__(self, env, strength=5.0, gamma=0.99):
        super().__init__(env)
        self.strength = strength
        self.gamma = gamma
        self.previous_potential = 0.0

    def potential(self, observation):
        return -self.strength * abs(float(observation[0]))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_potential = self.potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_potential = 0.0 if terminated else self.potential(obs)
        shaping = self.gamma * current_potential - self.previous_potential
        self.previous_potential = current_potential
        return obs, reward + shaping, terminated, truncated, info


class HoverReward(gym.Wrapper):
    """Intentionally flawed per-step reward for remaining slow/upright/airborne."""

    def __init__(self, env, bonus=1.0):
        super().__init__(env)
        self.bonus = bonus

    @staticmethod
    def qualifies(obs, terminated):
        upright = abs(float(obs[4])) < 0.15
        slow = (abs(float(obs[2])) < 0.25 and
                abs(float(obs[3])) < 0.25 and
                abs(float(obs[5])) < 0.25)
        airborne = float(obs[6]) == 0.0 and float(obs[7]) == 0.0
        return upright and slow and airborne and not terminated

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.qualifies(obs, terminated):
            reward += self.bonus
        return obs, reward, terminated, truncated, info


class CorrectedReward(gym.Wrapper):
    """Original reward with a small timestep cost; no hover bonus."""

    def __init__(self, env, step_penalty=0.05):
        super().__init__(env)
        self.step_penalty = step_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - self.step_penalty, terminated, truncated, info


class PrecisionReward(gym.Wrapper):
    """Original reward, timestep cost, and one-time centred-rest bonus."""

    def __init__(self, env, target_x=0.15, landing_bonus=50.0,
                 step_penalty=0.05):
        super().__init__(env)
        self.target_x = target_x
        self.landing_bonus = landing_bonus
        self.step_penalty = step_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        successful_rest = terminated and reward == 100.0
        centered = abs(float(obs[0])) <= self.target_x
        modified_reward = reward - self.step_penalty
        if successful_rest and centered:
            modified_reward += self.landing_bonus
        return obs, modified_reward, terminated, truncated, info
