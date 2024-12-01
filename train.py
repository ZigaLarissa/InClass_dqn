import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from test_env import ClassEnvironment  # Import your custom environment

# 1. Create the environment
# Wrap the environment in a DummyVecEnv for compatibility with Stable-Baselines3
env = DummyVecEnv([lambda: ClassEnvironment()])

# 2. Define the DQN Agent
model = DQN(
    "MlpPolicy",  # Multi-Layer Perceptron (MLP) policy
    env,
    learning_rate=1e-3,
    buffer_size=50000,  # Replay buffer size
    learning_starts=1000,  # Number of steps before learning starts
    batch_size=32,  # Mini-batch size
    tau=1e-3,  # Soft update coefficient for the target network
    gamma=0.99,  # Discount factor
    train_freq=4,  # Train every 4 steps
    target_update_interval=1000,  # Update the target network every 1000 steps
    verbose=1,  # Print training progress
)

# 3. Train the model
print("Training the DQN agent...")
model.learn(total_timesteps=50000)  # Number of training steps

# 4. Save the model
model.save("classroom_dqn_model")
print("Model saved to 'classroom_dqn_model.zip'.")

# 5. Test the trained agent
print("Testing the trained agent...")
env = ClassEnvironment()
obs, _ = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()

print("Testing complete.")
