import torch
from torch.distributions import Categorical


class CriticNetwork(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = torch.nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc5 = torch.nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc6 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        out = x
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        out = x + out
        x = torch.relu(self.fc4(out))
        x = torch.relu(self.fc5(x))
        x = out + x
        return self.fc6(x)

    def predict(self, state):
        with torch.no_grad():
            value = self.forward(state)
        return value.item()


class ActorNetwork(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = torch.nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc5 = torch.nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc6 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        out = x
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        out = out + x
        x = torch.relu(self.fc4(out))
        x = torch.relu(self.fc5(x))
        x = out + x
        x = torch.softmax(self.fc6(x), dim=-1)  # softmax to get action probabilities
        return x

    def get_action(self, state):
        with torch.no_grad():
            action_prob = self.forward(state)
            dist = Categorical(action_prob)
            action = torch.multinomial(action_prob, 1)  # Sample an action
            return action, dist.log_prob(action)

    def get_prob(self, state, action):
        action_prob = self.forward(state)
        dist = Categorical(action_prob)
        return dist.log_prob(action.view(-1))

    def compute_entropy(self, state):
        action_prob = self.forward(state)
        entropy = -torch.sum(action_prob * torch.log(action_prob + 1e-8),
                             dim=-1).mean()  # Add small value to avoid log(0)
        return entropy
