import math
import random
from abc import ABC

import pygame
import torch

from reforcing import Environment


class SnakeGame(Environment, ABC):
    def __init__(self, width, height, window_size=(800, 800), hungry_mode=True, vision_mode=True, hungry_max=150):
        super().__init__()
        self.width = width
        self.height = height
        self.snake = []
        self.food = (0, 0)
        self.direction = (1, 0)
        self.score = 0
        self.reward = 0
        self.game_over = False
        self.window_size = window_size
        self.screen_rate = (self.window_size[0] // self.width, self.window_size[1] // self.height)
        self.background_color = (245, 245, 220)  # 米白色背景
        self.head_color = (0, 255, 255)  # 蛇头颜色
        self.body_color = (0, 255, 0)  # 蛇身颜色
        self.food_color = (255, 0, 0)  # 食物颜色
        self.text_color = (255, 255, 255)  # 文本颜色
        self.hungry_mode = hungry_mode
        self.hungry_max = hungry_max
        self.hungry = hungry_max
        self.vision_mode = vision_mode
        self.reset()
        pygame.font.init()
        self.font = pygame.font.Font(None, 36)

    def generate_snake(self):
        self.snake = [(random.randint(3, self.width - 4), random.randint(3, self.height - 4))]
        self.snake.append((self.snake[0][0] - self.direction[0], self.snake[0][1] - self.direction[1]))

    def generate_food(self):
        self.food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def is_eat_self(self, new_head):
        return new_head in self.snake

    def is_strike_wall(self, new_head):
        return new_head[0] < 0 or new_head[0] >= self.width or new_head[1] < 0 or new_head[1] >= self.height

    def is_strive(self):
        return self.hungry <= 0

    def update(self):
        if self.game_over:
            return
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        if self.is_eat_self(new_head) or self.is_strike_wall(new_head) or self.is_strive():
            self.game_over = True
            return
        self.snake.insert(0, new_head)
        if new_head == self.food:
            fresh_reward = -math.log(1 - self.hungry / self.hungry_max + 1e-9) * 10
            self.reward = fresh_reward if self.hungry_mode else 2
            self.score += 1
            self.hungry = self.hungry_max
            self.generate_food()
        else:
            if self.hungry_mode:
                self.hungry -= 1
                strive_punish = -(self.hungry_max * 0.2 - self.hungry) / self.hungry_max
                self.reward = 0 if self.hungry > self.hungry_max * 0.2 else strive_punish
            else:
                self.reward = 0
            self.snake.pop()
        self.state = self.get_state()

    def draw_grid(self, screen):
        # 计算每个格子的宽度和高度
        grid_size = self.window_size[0] // self.width  # 每个格子的宽度
        grid_height = self.window_size[1] // self.height  # 每个格子的高度

        # 使用细线绘制网格
        for x in range(0, self.window_size[0], grid_size):
            pygame.draw.line(screen, (255, 255, 255), (x, 0), (x, self.window_size[1]), 1)  # 垂直线

        for y in range(0, self.window_size[1], grid_height):
            pygame.draw.line(screen, (255, 255, 255), (0, y), (self.window_size[0], y), 1)  # 水平线

    def draw(self, screen):
        screen.fill(self.background_color)  # 设置米白色背景

        # Draw grid first
        self.draw_grid(screen)

        # 绘制蛇身及阴影
        for segment in self.snake:
            shadow_offset = (5, 5)  # 阴影偏移
            if segment == self.snake[0]:
                # 蛇头：圆角矩形加阴影
                top_left = (segment[0] * self.screen_rate[0], segment[1] * self.screen_rate[1])
                rect = (top_left[0], top_left[1], self.screen_rate[0], self.screen_rate[1])
                pygame.draw.rect(screen, (0, 0, 0),
                                 (rect[0] + shadow_offset[0], rect[1] + shadow_offset[1], rect[2], rect[3]),
                                 border_radius=10)  # 绘制阴影
                pygame.draw.rect(screen, self.head_color, rect, border_radius=10)  # 蛇头是圆角矩形
            else:
                # 蛇身：圆角矩形加阴影
                top_left = (segment[0] * self.screen_rate[0], segment[1] * self.screen_rate[1])
                rect = (top_left[0], top_left[1], self.screen_rate[0], self.screen_rate[1])
                pygame.draw.rect(screen, (0, 0, 0),
                                 (rect[0] + shadow_offset[0], rect[1] + shadow_offset[1], rect[2], rect[3]),
                                 border_radius=5)  # 绘制阴影
                pygame.draw.rect(screen, self.body_color, rect, border_radius=5)  # 蛇身是圆角矩形

        # 食物：圆形加阴影
        food_radius = self.screen_rate[0] // 2
        food_center = (
            self.food[0] * self.screen_rate[0] + food_radius, self.food[1] * self.screen_rate[1] + food_radius)
        food_shadow_offset = (2, 2)  # 食物的阴影偏移
        pygame.draw.circle(screen, (0, 0, 0),
                           (food_center[0] + food_shadow_offset[0], food_center[1] + food_shadow_offset[1]),
                           food_radius + 2)  # 绘制阴影
        pygame.draw.circle(screen, self.food_color, food_center, food_radius)  # 食物是圆形

        # Display score in a stylish manner
        score_text = self.font.render(f"Score: {self.score}", True, self.text_color)
        screen.blit(score_text, (10, 10))

        if self.game_over:
            game_over_text = self.font.render("Game Over", True, self.text_color)
            screen.blit(game_over_text,
                        (self.window_size[0] // 2 - game_over_text.get_width() // 2, self.window_size[1] // 2 - 50))
            score_text = self.font.render(f"Score: {self.score}", True, self.text_color)
            screen.blit(score_text, (self.window_size[0] // 2 - score_text.get_width() // 2, self.window_size[1] // 2))

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

    def get_abstract_state(self):
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

    def get_vision_state(self):
        state = torch.zeros((self.width, self.height, 3), dtype=torch.float32)
        for segment in self.snake[1:]:
            state[segment[0], segment[1], 0] = self.hungry / 100
        state[self.food[0], self.food[1], 1] = 1
        state[self.snake[0][0], self.snake[0][1], 2] = self.hungry / 100
        return state.flatten()

    def get_state(self):
        return self.get_vision_state() if self.vision_mode else self.get_abstract_state()

    def reset(self):
        self.direction = (1, 0)
        self.generate_snake()
        self.generate_food()
        self.hungry = self.hungry_max
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
