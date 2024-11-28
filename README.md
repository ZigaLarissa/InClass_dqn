# Classroom Environment Simulation with Reinforcement Learning

This project demonstrates a reinforcement learning (RL) agent optimizing its actions in a simulated classroom environment. The agent is trained to maximize student knowledge while balancing boredom and energy levels.

## Features
- A custom **Classroom Environment** implemented using the `gymnasium` framework.
- Reinforcement learning agent powered by **Stable-Baselines3** and trained with the **PPO** algorithm.
- Dynamic 2D visualization using **pygame**, showing real-time updates of the agent's actions and their impact on the environment.

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Environment Details](#environment-details)
- [Visualization](#visualization)
- [Future Work](#future-work)

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install dependencies**:
   ```bash
   pip install gymnasium stable-baselines3 pygame tensorflow tabulate
   ```

3. **Verify installation**:
   Ensure all dependencies are installed without errors.

---

## Usage

### Training the RL Agent
1. Train the agent using the `train.py` script:
   ```bash
   python train.py
   ```
   The trained model will be saved as `classroom_policy.zip`.

### Simulating the Trained Agent
2. Visualize the agent's decisions in a 2D simulation:
   ```bash
   python play.py
   ```
   This will open a window showing:
   - **Bars** representing student knowledge, boredom, and energy levels.
   - The **agent's actions** (e.g., *Teach*, *Test*, *Assign Homework*).
   - The **reward** received after each action.

---

## Files

### `test_env.py`
- Implements the custom **Classroom Environment** for the RL agent.
- State variables:
  - `Knowledge`: Represents student understanding of the subject.
  - `Boredom`: Increases with repetitive or unengaging actions.
  - `Energy`: Decreases with intensive tasks or actions.

### `train.py`
- Trains the RL agent using the PPO algorithm from `Stable-Baselines3`.
- Saves the trained policy as `classroom_policy.zip`.

### `play.py`
- Loads the trained agent and simulates its behavior in the environment.
- Uses **pygame** to provide a real-time 2D visualization of state changes and actions.

---

## Environment Details

### State Space
- `Knowledge`, `Boredom`, and `Energy` values range from 0 to 10.

### Action Space
- **Teach**: Increases knowledge but raises boredom and decreases energy.
- **Test**: Moderately increases knowledge but significantly reduces energy.
- **Assign Homework**: Slightly increases knowledge, reduces boredom, and moderately decreases energy.

### Rewards
- The agent is rewarded based on a combination of high knowledge, low boredom, and sufficient energy levels:
  ```python
  reward = knowledge - (boredom * 0.5) - (max(0, 10 - energy) * 0.5)
  ```

---

## Visualization

The **pygame-based simulation** provides a dynamic visualization:
- **Bars**: Represent knowledge, boredom, and energy levels.
- **Action Feedback**: Displays the agent's current action (e.g., *Teach*, *Test*).
- **Reward Feedback**: Shows the reward received for each action.
- **Simulation Over**: Indicates when the episode ends.

---

## Future Work

- Add **interactive controls** (e.g., pause, restart, or manual action selection).
- Improve the environment with more complex states and actions.
- Integrate a **cumulative reward tracker** in the visualization.
- Expand visualization to include **student performance metrics**.

---

## License
This project is open-source and available under the MIT License.
