import os
import threading
from time import sleep

import pygame
import torch

from games import SnakeGame
from models.assist import ActorNetwork


def run_game(update_mode=50):
    screen = pygame.display.set_mode((800, 800))
    width = 9
    height = 9
    actor_output_dim = 4
    env = SnakeGame(width, height)
    state_dim = 6
    hidden_dim = 6 * 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor = ActorNetwork(state_dim, hidden_dim, actor_output_dim).to(device)
    state = env.reset().to(device)
    if os.path.exists('actor.pth'):
        print('Load model from actor.pth')
        actor.load_state_dict(torch.load('actor.pth'))
    i = 1
    while True:
        actor.eval()
        if os.path.exists('actor.pth') and i % update_mode == 0:
            print('Load model from actor.pth')
            actor.load_state_dict(torch.load('actor.pth'))
            i += 1
        action, _ = actor.get_action(state)
        state, done, _ = env.step(action.item())
        state = state.to(device)
        if done:
            i += 1
            env.reset()
        sleep(0.1)
        env.draw(screen)


def auto_run():
    pygame.init()
    clock = pygame.time.Clock()
    thread = threading.Thread(target=run_game)
    thread.start()
    while True:
        pygame.display.update()
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return


if __name__ == "__main__":
    auto_run()
