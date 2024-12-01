import pygame
from stable_baselines3 import DQN
from test_env import ClassEnvironment

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 400
GROUND_HEIGHT = WINDOW_HEIGHT - 50
SCROLL_SPEED = 3
FPS = 30
FONT_SIZE = 24

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Classroom Simulation")
clock = pygame.time.Clock()

# Load images
agent_image = pygame.image.load("images/agent.png")
obstacle_image = pygame.image.load("images/obstacle.png")
reward_image = pygame.image.load("images/reward.png")
goal_image = pygame.image.load("images/goal.png")

# Resize images for consistency
agent_image = pygame.transform.scale(agent_image, (50, 50))
obstacle_image = pygame.transform.scale(obstacle_image, (50, 50))
reward_image = pygame.transform.scale(reward_image, (50, 50))
goal_image = pygame.transform.scale(goal_image, (50, 50))

# Load font
font = pygame.font.Font(None, FONT_SIZE)

# Predefined level layout
LEVEL_LAYOUT = [
    {"x": 300, "type": "obstacle"},  # Fixed obstacle position
    {"x": 600, "type": "reward"},    # Fixed reward position
    {"x": 900, "type": "obstacle"},  # Fixed obstacle position
    {"x": 1200, "type": "reward"},   # Fixed reward position
    {"x": 1500, "type": "obstacle"},  # Fixed obstacle position
    {"x": 1800, "type": "reward"},    # Fixed reward position
    {"x": 2100, "type": "obstacle"},  # Fixed obstacle position
    {"x": 2400, "type": "reward"},   # Fixed reward position
    {"x": 2700, "type": "obstacle"},  # Fixed obstacle position
    {"x": 3000, "type": "reward"},    # Fixed reward position
    {"x": 3300, "type": "obstacle"},  # Fixed obstacle position
    {"x": 3600, "type": "obstacle"},  # Fixed obstacle position
    {"x": 3900, "type": "obstacle"},  # Fixed obstacle position
    {"x": 4200, "type": "reward"},   # Fixed reward position
    {"x": 4500, "type": "reward"},   # Fixed reward position
    {"x": 4800, "type": "goal"},     # Fixed goal position
]

# Agent class
class Agent:
    def __init__(self):
        self.image = agent_image
        self.x = 100
        self.y = GROUND_HEIGHT
        self.velocity_y = 0
        self.jumping = False

    def jump(self):
        if not self.jumping:  # Only jump if on the ground
            self.velocity_y = -15  # Jump strength
            self.jumping = True

    def apply_gravity(self):
        # Apply gravity and update vertical position
        if self.jumping:
            self.velocity_y += 0.5  # Adjusted gravity
            self.y += self.velocity_y

        # Stop at the ground
        if self.y >= GROUND_HEIGHT:
            self.y = GROUND_HEIGHT
            self.jumping = False

    def draw(self, surface):
        surface.blit(self.image, (self.x, self.y - 50))  # Adjust to center


def draw_score(surface, score):
    """
    Display the current score on the screen.
    """
    score_text = font.render(f"Score: {score}", True, BLACK)
    surface.blit(score_text, (10, 10))


def end_game(message, final_score):
    """
    Display a Game Over or Victory message and final score.
    """
    screen.fill(WHITE)
    message_text = font.render(message, True, RED)
    score_text = font.render(f"Final Score: {final_score}", True, BLACK)

    screen.blit(message_text, (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 - 30))
    screen.blit(score_text, (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 + 10))

    pygame.display.flip()
    pygame.time.wait(3000)  # Wait for 3 seconds before closing


# Main game loop
def main():
    # Load the trained model
    model = DQN.load("classroom_dqn_model")

    # Create the environment
    env = ClassEnvironment()
    state, _ = env.reset()

    agent = Agent()
    objects = [{"x": obj["x"], "type": obj["type"], "image": obstacle_image if obj["type"] == "obstacle" else
               reward_image if obj["type"] == "reward" else goal_image} for obj in LEVEL_LAYOUT]
    score = 0
    running = True
    scroll_offset = 0

    while running:
        screen.fill(WHITE)  # Clear the screen

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Scroll the level
        scroll_offset += SCROLL_SPEED

        # Check for nearby obstacles and jump
        for obj in objects:
            obj_x = obj["x"] - scroll_offset
            # Improved jump logic with increased range
            if obj["type"] == "obstacle" and 100 < obj_x - agent.x < 130 and not agent.jumping:
                agent.jump()

        # Apply gravity and update the agent
        agent.apply_gravity()
        agent.draw(screen)

        # Draw ground
        pygame.draw.line(screen, BLACK, (0, GROUND_HEIGHT + 10), (WINDOW_WIDTH, GROUND_HEIGHT + 10), 2)

        # Draw obstacles, rewards, and goals
        for obj in list(objects):
            obj_x = obj["x"] - scroll_offset

            if 0 < obj_x < WINDOW_WIDTH:  # Only draw if visible
                screen.blit(obj["image"], (obj_x, GROUND_HEIGHT - 50 if obj["type"] != "reward" else GROUND_HEIGHT - 100))

                # Collision detection
                if obj["type"] == "obstacle" and abs(agent.x - obj_x) < 30 and agent.y >= GROUND_HEIGHT - 50:
                    end_game("Game Over!", score)
                    running = False
                elif obj["type"] == "reward" and abs(agent.x - obj_x) < 30 and agent.y >= GROUND_HEIGHT - 100:
                    score += 1
                    objects.remove(obj)
                elif obj["type"] == "goal" and abs(agent.x - obj_x) < 30:
                    end_game("You Win!", score)
                    running = False

            if obj_x < -50:  # Remove objects off-screen
                objects.remove(obj)

        # Update state
        action = 0
        state, _, _, _, _ = env.step(action)

        # Display the score
        draw_score(screen, score)

        # Update the display
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
