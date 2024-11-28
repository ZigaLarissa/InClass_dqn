import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from test_env import ClassEnvironment

# Create a vectorized environment
env = make_vec_env(lambda: ClassEnvironment(), n_envs=1)

# Define the agent with enhanced PPO parameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,              # Default is 3e-4, adjust to 1e-4 for finer learning
    n_steps=2048,                   # Number of steps per update (higher means more stable gradients)
    batch_size=64,                  # Mini-batch size for updates
    n_epochs=10,                    # Number of epochs when optimizing the surrogate loss
    gamma=0.99,                     # Discount factor for future rewards
    gae_lambda=0.95,                # GAE parameter for advantage estimation
    clip_range=0.2,                 # PPO clipping parameter
    ent_coef=0.01,                  # Entropy coefficient for exploration
    vf_coef=0.5,                    # Value function coefficient
    max_grad_norm=0.5,              # Gradient clipping
    target_kl=0.03                  # Stop training if KL divergence exceeds this value
)

# Train the model
print("Training the PPO agent...")
model.learn(total_timesteps=200000)  # Increased training steps for better performance

# Save the model
model.save("classroom_policy")
print("Training complete and model saved.")
