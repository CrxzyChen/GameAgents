import torch

from games import SnakeGame
from models.assist import CriticNetwork, ActorNetwork
from reforcing import PPO


def main():
    width = 16
    height = 16
    actor_output_dim = 4
    critic_output_dim = 1
    env = SnakeGame(width, height)
    state_dim = width * height * 3
    hidden_dim = state_dim * 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    critic = CriticNetwork(state_dim, hidden_dim, critic_output_dim).to(device)
    actor = ActorNetwork(state_dim, hidden_dim, actor_output_dim).to(device)
    ppo = PPO(env, critic, actor, device=device, gamma=0.99, reward_scale=1)
    ppo.train(max_iter=100000, batch_size=256, target_step=4096, lr=1e-4, repeat_times=1)
    print(ppo.test())


if __name__ == "__main__":
    main()
