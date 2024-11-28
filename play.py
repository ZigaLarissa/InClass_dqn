import pygame
from stable_baselines3 import PPO
from test_env import ClassEnvironment

# Constants for visualization
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
BAR_WIDTH = 100
BAR_HEIGHT = 300
BAR_GAP = 50
BAR_COLORS = {"Knowledge": (0, 255, 0), "Boredom": (255, 0, 0), "Energy": (0, 0, 255)}
ACTIONS = ["Teach", "Test", "Assign Homework"]

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Classroom Environment Simulation")

# Load the trained model
model = PPO.load("classroom_policy")

# Create the environment
env = ClassEnvironment()
state, _ = env.reset()

# Variables for timing decisions
frames_per_decision = 30  # How many frames before the agent decides
frame_counter = 0  # Counts frames between decisions
next_state = state.copy()  # To interpolate smoothly

# Main simulation loop
done = False
clock = pygame.time.Clock()
action_text = "Starting Simulation"
reward = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    if not done and frame_counter % frames_per_decision == 0:
        # Predict the action using the trained model
        action, _states = model.predict(state, deterministic=True)
        action_text = f"Action: {ACTIONS[action]}"
        next_state, reward, done, _, _ = env.step(action)
        frame_counter = 1  # Reset the frame counter for the next decision

    # Gradually update the bar values for smooth transitions
    interpolated_state = [
        state[i] + (next_state[i] - state[i]) * (frame_counter / frames_per_decision)
        for i in range(len(state))
    ]
    state = next_state if frame_counter == frames_per_decision else state

    # Clear the screen
    screen.fill((255, 255, 255))  # White background

    # Draw the bars for each state variable
    labels = ["Knowledge", "Boredom", "Energy"]
    for i, (label, value) in enumerate(zip(labels, interpolated_state)):
        x = BAR_GAP + i * (BAR_WIDTH + BAR_GAP)
        y = WINDOW_HEIGHT - BAR_HEIGHT
        bar_height = int(value / 10 * BAR_HEIGHT)  # Scale value to bar height
        pygame.draw.rect(screen, BAR_COLORS[label], (x, y + BAR_HEIGHT - bar_height, BAR_WIDTH, bar_height))
        
        # Add label and value
        font = pygame.font.Font(None, 36)
        text = font.render(f"{label}: {value:.1f}", True, (0, 0, 0))
        screen.blit(text, (x, y + BAR_HEIGHT + 10))

    # Display action and reward
    font = pygame.font.Font(None, 48)
    action_display = font.render(action_text, True, (0, 0, 0))
    reward_display = font.render(f"Reward: {reward:.2f}", True, (0, 0, 0))
    screen.blit(action_display, (WINDOW_WIDTH // 2 - 150, 20))
    screen.blit(reward_display, (WINDOW_WIDTH // 2 - 150, 60))

    # Show "Simulation Over" when done
    if done:
        end_text = font.render("Simulation Over", True, (255, 0, 0))
        screen.blit(end_text, (WINDOW_WIDTH // 2 - 120, 100))

    # Update the display
    pygame.display.flip()

    # Increment frame counter
    frame_counter += 1

    # Limit frame rate for better visualization
    clock.tick(30)
