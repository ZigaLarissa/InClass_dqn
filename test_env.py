import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ClassEnvironment(gym.Env):
    """
    A simple classroom environment where actions affect student knowledge.
    """
    def __init__(self):
        super(ClassEnvironment, self).__init__()
        
        # Define the action space (e.g., 0 = teach, 1 = test, 2 = assign homework)
        self.action_space = spaces.Discrete(3)
        
        # Define the state space (student knowledge level, boredom, energy)
        self.observation_space = spaces.Box(low=0, high=10, shape=(3,), dtype=np.float32)
        
        # Initialize state
        self.state = np.array([5.0, 5.0, 5.0])  # [knowledge, boredom, energy]
        
        # Other parameters
        self.max_steps = 100
        self.current_step = 0

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        """
        self.state = np.array([5.0, 5.0, 5.0])
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        """
        Apply the action and update the environment state.
        """
        knowledge, boredom, energy = self.state

        # Define the effect of each action
        if action == 0:  # Teach
            knowledge += 2
            boredom += 1
            energy -= 1
        elif action == 1:  # Test
            knowledge += 1
            boredom += 1
            energy -= 2
        elif action == 2:  # Assign Homework
            knowledge += 1
            boredom -= 1
            energy -= 1

        # Rewards based on state
        reward = knowledge - (boredom * 0.5) - (max(0, 10 - energy) * 0.5)

        # Update state
        self.state = np.clip([knowledge, boredom, energy], 0, 10)
        self.current_step += 1

        # Check if the episode is done
        done = self.current_step >= self.max_steps

        return self.state, reward, done, False, {}

    def render(self):
        """
        Render the environment (can be expanded for a visual UI).
        """
        print(f"State: {self.state}")
