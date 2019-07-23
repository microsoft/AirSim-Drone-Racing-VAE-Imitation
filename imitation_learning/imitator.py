import torch
import torch.nn as nn

from imitation_learning.demonstrations import Demonstrations

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """ Actor model used by the imitator"""
    def __init__(self, state_dim, action_dim):
        """

        Args:
            state_dim: H x W x C
            action_dim: N
        """
        super(Actor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(state_dim[2], 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(64 * 34 * 62, action_dim)

    def forward(self, obs):
        out = self.layer1(obs)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        # TODO @praveen-palanisamy: Clamp/scale appropriately
        return out


class Discriminator(nn.Module):
    """ Discriminator used by the imitator"""
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(state_dim[2], 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(64 * 34 * 62 + action_dim, 1)

    def forward(self, obs, action):
        out = self.layer1(obs)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = torch.cat([out, action], 1)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out


class Imitator(object):
    """ Imitator implementation using adversarial CE loss"""
    def __init__(self, task_name, state_dim, action_dim, lr=1e-4):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.demonstrator = Demonstrations(task_name)
        self.loss = nn.BCELoss()
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters())

    def get_action(self, obs):
        """Get action based on imitator's current policy
        
        Arguments:
            obs {np.array} -- Image observation/state
        
        Returns:
            np.array -- Np.array of size `action_dim`
        """
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        return self.actor(obs).cpu().data.numpy().flatten()

    def learn(self, num_iter, batch_size=256):
        """Main routine to update the imitator model (actor, discriminator)
        
        Arguments:
            num_iter {int} -- Number of iterations to train
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {256})
        """
        for i in range(num_iter):
            # Sample from the demonstrator
            demo_state, demo_action = self.demonstrator.sample(batch_size)
            demo_state = demo_state.to(device)
            demo_action = demo_action.to(device)

            # Sample from the actor
            state, gt_action = self.demonstrator.sample(batch_size)
            state = state.to(device)
            actor_action = self.actor(state)

            # Prepare targets for learning
            target_demonstrator = torch.full((batch_size, 1), 1, device=device)
            target_actor = torch.full((batch_size, 1), 0, device=device)

            # Update Discriminator
            prob_demonstrator = self.discriminator(demo_state, demo_action)
            prob_actor = self.discriminator(state, actor_action.detach())

            loss_demonstrator = self.loss(prob_demonstrator,
                                          target_demonstrator)
            loss_policy = self.loss(prob_actor, target_actor)
            loss = loss_demonstrator + loss_policy
            self.discriminator_optim.zero_grad()
            loss.backward()
            self.discriminator_optim.step()

            # Update Actor
            loss_actor = - self.discriminator(state, actor_action)
            self.actor_optim.zero_grad()
            loss_actor.mean().backward()
            self.actor_optim.step()
