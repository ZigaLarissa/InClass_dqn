import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ClassEnvironment(gym.Env):
    """
    A classroom simulation where the agent moves through obstacles and rewards.
    """
    def __init__(self):
        super(ClassEnvironment, self).__init__()
        
        # Define the action space (0 = No action, 1 = Jump)
        self.action_space = spaces.Discrete(2)
        
        # Define the observation space (agent position, vertical position, and upcoming obstacles/rewards)
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(4,), dtype=np.float32  # [x, y, next_obstacle_x, next_obstacle_type]
        )
        
        # Initialize environment
        self.agent_pos = [0, 0]  # [x, y]
        self.velocity_y = 0
        self.jumping = False
        self.gravity = 1
        
        # Level layout (x-coordinates and types: 1 = obstacle, 2 = reward, 3 = goal)
        self.level_layout = [
            {"x": 3, "type": 1},  # Cheating
            {"x": 5, "type": 2},  # Studying
            {"x": 7, "type": 1},  # Skipping Class
            {"x": 9, "type": 2},  # Studying
            {"x": 11, "type": 3},  # Passing (Goal)
        ]
        self.current_step = 0
        self.max_steps = len(self.level_layout)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        self.agent_pos = [0, 0]
        self.velocity_y = 0
        self.jumping = False
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Construct the observation.
        """
        if self.current_step < self.max_steps:
            next_obstacle = self.level_layout[self.current_step]
            return np.array([self.agent_pos[0], self.agent_pos[1], next_obstacle["x"], next_obstacle["type"]])
        else:
            return np.array([self.agent_pos[0], self.agent_pos[1], 10, 3])  # Goal state

    def step(self, action):
        """
        Apply the action and update the environment.
        """
        # Update agent position
        self.agent_pos[0] += 0.1  # Move forward automatically

        # Apply jump logic
        if action == 1 and not self.jumping:
            self.velocity_y = -5
            self.jumping = True

        # Apply gravity
        self.velocity_y += self.gravity
        self.agent_pos[1] += self.velocity_y

        # Stop the agent at ground level
        if self.agent_pos[1] < 0:
            self.agent_pos[1] = 0
            self.jumping = False

        # Check collision
        done = False
        reward = 0
        if self.current_step < self.max_steps:
            next_obstacle = self.level_layout[self.current_step]
            if abs(self.agent_pos[0] - next_obstacle["x"]) < 0.2:  # Collision zone
                if next_obstacle["type"] == 1:  # Obstacle
                    reward -= 5  # Penalty
                    done = True
                elif next_obstacle["type"] == 2:  # Reward
                    reward += 10  # Reward for studying
                    self.current_step += 1  # Move to the next obstacle/reward
                elif next_obstacle["type"] == 3:  # Goal
                    reward += 50  # Big reward for passing
                    done = True

        # Check if the agent has finished the level
        if self.agent_pos[0] >= 10:
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self):
        """
        Render the environment (print the state for debugging).
        """
        print(f"Agent Position: {self.agent_pos}, Next Obstacle: {self.level_layout[self.current_step]}")
