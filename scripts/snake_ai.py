import os
import threading
from time import sleep

import pygame
import torch

from games import SnakeGame
from models.assist import CriticNetwork, ActorNetwork
from reforcing import PPO


class Trainer:
    def __init__(self, **config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.width = config.get('width', 16)
        self.height = config.get('height', 16)
        self.actor_output_dim = config.get('actor_output_dim', 4)
        self.critic_output_dim = config.get('critic_output_dim', 1)
        self.state_dim = config.get('state_dim', 6)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.checkpoints_dir = config.get('checkpoints_dir', 'checkpoints/snake')
        self.critic = CriticNetwork(self.state_dim, self.hidden_dim, self.critic_output_dim).to(self.device)
        self.actor = ActorNetwork(self.state_dim, self.hidden_dim, self.actor_output_dim).to(self.device)

    def train(self, **config):
        max_iter = config.get('max_iter', 100000)
        batch_size = config.get('batch_size', 256)
        target_step = config.get('target_step', 4096)
        lr = config.get('lr', 1e-4)
        repeat_times = config.get('repeat_times', 1)
        gamma = config.get('gamma', 0.99)
        reward_scale = config.get('reward_scale', 1)
        vision_mode = config.get('vision_mode', False)
        hungry_mode = config.get('hungry_mode', True)
        hungry_max = config.get('hungry_max', 150)
        env = SnakeGame(self.width, self.height, vision_mode=vision_mode, hungry_mode=hungry_mode,
                        hungry_max=hungry_max)
        ppo = PPO(env, self.critic, self.actor, device=self.device, gamma=gamma, reward_scale=reward_scale)
        ppo.train(max_iter=max_iter, batch_size=batch_size, target_step=target_step, lr=lr, repeat_times=repeat_times,
                  save_dir=self.checkpoints_dir)

    def auto_run(self, config):
        update_mode = config.get('update_mode', 5)
        window_size = config.get('window_size', (800, 800))
        checkpoint_name = config.get('checkpoint_name', 'best_actor.pth')
        vision_mode = config.get('vision_mode', False)
        hungry_mode = config.get('hungry_mode', True)
        hungry_max = config.get('hungry_max', 150)
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)
        screen = pygame.display.set_mode(window_size)
        env = SnakeGame(self.width, self.height, vision_mode=vision_mode, hungry_mode=hungry_mode,
                        hungry_max=hungry_max)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if os.path.exists(checkpoint_path):
            print(f'Load model from {checkpoint_path}')
            self.actor.load_state_dict(torch.load(checkpoint_path))

        state = env.reset().to(device)
        i = 1
        while True:
            self.actor.eval()
            if os.path.exists(checkpoint_path) and i % update_mode == 0:
                print(f'Load model from {checkpoint_path}')
                self.actor.load_state_dict(torch.load(checkpoint_path))
                i += 1
            action, _ = self.actor.get_action(state)
            state, done, _ = env.step(action.item())
            state = state.to(device)
            if done:
                i += 1
                env.reset()
            sleep(0.1)
            env.draw(screen)

    def test(self, **config):
        pygame.init()
        clock = pygame.time.Clock()
        thread = threading.Thread(target=self.auto_run, args=(config,))
        thread.start()
        while True:
            pygame.display.update()
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
