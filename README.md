# Classroom DQN Agent

A reinforcement learning project where a Deep Q-Network (DQN) agent is trained to navigate a classroom-like simulation. The agent learns to avoid obstacles, collect rewards, and reach the goal while improving its strategy over time.

---

## Video Demo

- 🎥 **Watch the simulation video here**: [Classroom DQN Agent Simulation](https://www.loom.com/share/d91eb70716894a10aa42d7d601c29b15?sid=cad75f30-24b4-44ab-85f5-e789370e0366)
- 🎥 **Watch the demo video here**: [Classroom DQN Agent Demo](https://youtu.be/KBZWZwZCfq0)

---

## Brief Description

This project uses reinforcement learning to create an intelligent agent capable of navigating a classroom environment. The agent:
- Learns from interactions with the environment using the DQN algorithm.
- Adapts to avoid obstacles, collect rewards, and reach the designated goal.
- Operates in a visually rendered simulation built with Pygame.

---

## How to Set Up and Run the Project

### Prerequisites

- Python 3.8 or higher
- pip for managing dependencies

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ZigaLarissa/InClass_dqn.git
   cd classroom-dqn-agent
   ```

2. **Install Dependencies**:
   Use the `requirements.txt` file to install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Agent

To train the agent from scratch:
1. Ensure your environment is defined in `test_env.py`.
2. Run the training script:
   ```bash
   python train.py
   ```
   - The model will be saved as `classroom_dqn_model.zip` upon completion.

### Running the Simulation

1. Place the necessary images (`agent.png`, `obstacle.png`, `reward.png`, `goal.png`) in the `images/` folder.
2. Run the simulation with the trained model:
   ```bash
   python play.py
   ```
3. Watch as the agent navigates the environment autonomously!

---

## Project Structure

```plaintext
├── README.md               # Project description and instructions
├── classroom_dqn_model.zip # Pre-trained DQN model
├── play.py                 # Gameplay and rendering script
├── test_env.py             # Custom classroom environment
├── train.py                # Training script for the DQN agent
├── images/                 # Assets for the game (agent, obstacles, rewards, etc.)
├── requirements.txt        # Python dependencies
└── __pycache__/            # Cached Python files
```

