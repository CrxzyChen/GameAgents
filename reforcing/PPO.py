import os

import torch

from .Environment import Environment


class PPO:
    def __init__(self, env: Environment, critic, actor, if_gae: bool = True, max_history_len=65536, gamma=0.99,
                 lam=0.95, reward_scale=0.99, ratio_clip=0.2, entropy_coeff=0.01, device='cpu'):
        self.env = env
        self.critic = critic
        self.actor = actor
        self.if_gae = if_gae
        self.history = list()
        self.max_history_len = max_history_len
        self.gamma = gamma
        self.lam = lam
        self.reward_scale = reward_scale
        self.ratio_clip = ratio_clip
        self.entropy_coeff = entropy_coeff  # New parameter for entropy regularization
        self.device = device
        self.state = None

    def explore_env(self, target_step, reward_scale):
        trajectory_list = list()
        state = self.env.reset().to(self.device)
        for _ in range(target_step):
            action, prob = self.actor.get_action(state)
            next_state, done, reward = self.env.step(action)
            trajectory_list.append((state, (reward * reward_scale, done, action, prob)))
            state = self.env.reset().to(self.device) if done else next_state.to(self.device)
        self.state = state
        return trajectory_list

    def compute_advantages(self, trajectory_list):
        states = [item[0] for item in trajectory_list]
        values = self.critic(torch.stack(states)).squeeze().tolist()
        advantages = []
        state_rewards = []
        last_gae_lam = 0
        last_reward = 0
        next_value = self.critic.predict(self.state)

        for t in reversed(range(len(trajectory_list))):
            reward, done = trajectory_list[t][1][0], trajectory_list[t][1][1]
            if done:

                delta = reward - values[t]
                last_gae_lam = delta
                last_reward = reward
            else:
                delta = reward + self.gamma * next_value - values[t]
                last_gae_lam = delta + self.gamma * self.lam * last_gae_lam
                last_reward = reward + self.gamma * last_reward
                next_value = values[t]

            advantages.insert(0, last_gae_lam)
            state_rewards.insert(0, last_reward)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.tolist(), state_rewards

    def train(self, max_iter=1000, target_step=1000, batch_size=64, repeat_times=10, lr=1e-4, save_iter=100):
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        critic_criterion = torch.nn.SmoothL1Loss()
        device = self.device
        if os.path.exists('actor.pth'):
            print('Load model from actor.pth')
            self.actor.load_state_dict(torch.load('actor.pth'))
        if os.path.exists('critic.pth'):
            print('Load model from critic.pth')
            self.critic.load_state_dict(torch.load('critic.pth'))
        best_score = -0.5
        for i in range(max_iter):

            # 探索阶段：
            with torch.no_grad():
                trajectory_list = self.explore_env(target_step, self.reward_scale)
                # 计算优势
                advantages, last_reward = self.compute_advantages(trajectory_list)

            buffer = [(item[0], last_reward[i], advantages[i], item[1][2], item[1][3]) for i, item in
                      enumerate(trajectory_list)]

            self.history.extend(buffer)
            if len(self.history) > self.max_history_len:
                self.history = self.history[-self.max_history_len:]

            # 训练阶段：更新网络
            self.actor.train()
            self.critic.train()
            for _ in range(len(self.history) // batch_size * repeat_times):
                indices = torch.randint(len(self.history), size=(batch_size,))
                batch = [self.history[idx] for idx in indices]
                states, rewards, advantages, actions, old_probs = zip(*batch)
                states_tensor = torch.stack(states, dim=0).to(device)
                rewards_tenser = torch.tensor(rewards, dtype=torch.float32, device=device)
                advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
                actions_tensor = torch.tensor(actions, dtype=torch.long, device=device).view(-1, 1)
                old_probs_tenser = torch.tensor(old_probs, dtype=torch.float32, device=device)

                # 更新actor网络
                new_probs = self.actor.get_prob(states_tensor, actions_tensor)
                ratio = (new_probs - old_probs_tenser).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.ratio_clip, 1 + self.ratio_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 添加熵正则化项
                entropy = self.actor.compute_entropy(states_tensor)
                actor_loss += self.entropy_coeff * entropy  # Add entropy regularization term
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # 更新critic网络
                values = self.critic(states_tensor)
                critic_loss = critic_criterion(values, rewards_tenser.view(-1, 1)) / (rewards_tenser.std() + 1e-6)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

            scores = self.test()
            print(f'Iteration {i}, Average Reward: {scores}')

            if scores > best_score:
                best_score = scores
                torch.save(self.actor.state_dict(), 'best_actor.pth')
                torch.save(self.critic.state_dict(), 'best_critic.pth')

            if i % save_iter == 0:
                torch.save(self.actor.state_dict(), 'actor.pth')
                torch.save(self.critic.state_dict(), 'critic.pth')

    def test(self, game_rounds=20):
        self.actor.eval()
        with torch.no_grad():
            scores = []
            for _ in range(game_rounds):
                state = self.env.reset().to(self.device)
                done = False
                total_reward = 0
                while not done:
                    action, _ = self.actor.get_action(state)
                    state, done, reward = self.env.step(action)
                    total_reward += reward
                    state = state.to(self.device)
                scores.append(total_reward)
            return sum(scores) / len(scores)
