import random
from abc import ABC

import pygame
import torch

from reforcing import Environment


class SnakeGame(Environment, ABC):
    def __init__(self, width, height, window_size=(800, 800)):
        super().__init__()
        self.width = width
        self.height = height
        self.snake = []
        self.food = (0, 0)
        self.direction = (1, 0)
        self.score = 0
        self.reward = 0
        self.game_over = False
        self.reset()
        self.window_size = window_size
        self.screen_rate = (self.window_size[0] // self.width, self.window_size[1] // self.height)
        self.background_color = (0, 0, 0)
        self.head_color = (0, 255, 255)
        self.body_color = (0, 255, 0)
        self.food_color = (255, 0, 0)
        self.text_color = (255, 255, 255)

    def generate_snake(self):
        self.snake = [(random.randint(3, self.width - 4), random.randint(3, self.height - 4))]
        self.snake.append((self.snake[0][0] - self.direction[0], self.snake[0][1] - self.direction[1]))

    def generate_food(self):
        self.food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def is_eat_self(self, new_head):
        return new_head in self.snake

    def is_strike_wall(self, new_head):
        return new_head[0] < 0 or new_head[0] >= self.width or new_head[1] < 0 or new_head[1] >= self.height

    def update(self):
        if self.game_over:
            return
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        if self.is_eat_self(new_head) or self.is_strike_wall(new_head):
            self.game_over = True
            return
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.reward = 2
            self.score += 1
            self.generate_food()
        else:
            self.reward = 0
            self.snake.pop()
        self.state = self.get_state()

    def draw(self, screen):
        screen.fill((0, 0, 0))
        for segment in self.snake:
            if segment == self.snake[0]:
                top_left = (segment[0] * self.screen_rate[0], segment[1] * self.screen_rate[1])
                rect = (top_left[0], top_left[1], self.screen_rate[0], self.screen_rate[1])
                pygame.draw.rect(screen, self.head_color, rect)
            else:
                top_left = (segment[0] * self.screen_rate[0], segment[1] * self.screen_rate[1])
                rect = (top_left[0], top_left[1], self.screen_rate[0], self.screen_rate[1])
                pygame.draw.rect(screen, self.body_color, rect)

        top_left = (self.food[0] * self.screen_rate[0], self.food[1] * self.screen_rate[1])
        rect = (top_left[0], top_left[1], self.screen_rate[0], self.screen_rate[1])
        pygame.draw.rect(screen, self.food_color, rect)
        if self.game_over:
            font = pygame.font.Font(None, 36)
            text = font.render("Game Over", 1, self.text_color)
            screen.blit(text, (self.width * 5, self.height * 5))
            text = font.render(f"Score: {self.score}", 1, self.text_color)
            screen.blit(text, (self.width * 5, self.height * 5 + 40))

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and self.direction != (0, 1):
                self.direction = (0, -1)
            elif event.key == pygame.K_DOWN and self.direction != (0, -1):
                self.direction = (0, 1)
            elif event.key == pygame.K_LEFT and self.direction != (1, 0):
                self.direction = (-1, 0)
            elif event.key == pygame.K_RIGHT and self.direction != (-1, 0):
                self.direction = (1, 0)

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode(self.window_size)
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                self.handle_event(event)
            self.update()
            self.draw(screen)
            pygame.display.flip()
            clock.tick(10)

    def get_state(self):
        head = self.snake[0]
        dist = (head[0] - self.food[0], head[1] - self.food[1])
        dist = (dist[0] / self.width, dist[1] / self.height)
        barriers = [0, 0, 0, 0]
        checkpoints = [(head[0] + 1, head[1]), (head[0], head[1] + 1), (head[0] - 1, head[1]), (head[0], head[1] - 1)]
        for segment in self.snake[1:]:
            for i, checkpoint in enumerate(checkpoints):
                if checkpoint == segment:
                    barriers[i] = 1
        for i, checkpoint in enumerate(checkpoints):
            if checkpoint[0] < 0 or checkpoint[0] >= self.width or checkpoint[1] < 0 or checkpoint[1] >= self.height:
                barriers[i] = 1
        state = torch.tensor([dist[0], dist[1]] + barriers, dtype=torch.float32)
        return state

    def reset(self):
        self.direction = (1, 0)
        self.generate_snake()
        self.generate_food()
        self.score = 0
        self.game_over = False
        return self.get_state()

    def step(self, action):
        if action == 0 and self.direction != (0, 1):
            self.direction = (0, -1)
        elif action == 1 and self.direction != (0, -1):
            self.direction = (0, 1)
        elif action == 2 and self.direction != (1, 0):
            self.direction = (-1, 0)
        elif action == 3 and self.direction != (-1, 0):
            self.direction = (1, 0)
        self.update()
        return self.get_state(), self.game_over, (self.reward - 0.5) if self.game_over else self.score

    def close(self):
        pass


if __name__ == "__main__":
    game = SnakeGame(40, 40)
    game.run()
