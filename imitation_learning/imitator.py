import os

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
    def __init__(self, algo, task_name, state_dim, action_dim, lr=1e-4):
        self.algo = algo
        if self.algo == "adv":
            self.actor = Actor(state_dim, action_dim).to(device)
            self.discriminator = Discriminator(state_dim, action_dim).to(device)
            self.demonstrator = Demonstrations(task_name)
            self.loss = nn.BCELoss()
            self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
            self.discriminator_optim = torch.optim.Adam(
                self.discriminator.parameters())
            self.learn = self._learn_adversarial
        elif algo == "bc":
            from imitation_learning.models import MobileNetV2
            self.demonstrator = Demonstrations(task_name)
            self.actor = MobileNetV2()
            _state_dict = torch.load("imitation_learning/trained_models/mobilenet_v2.ptm")
            self.actor.load_state_dict(_state_dict)
            self.actor.classifier = nn.Linear(in_features=1280, out_features=4)
            self.actor = self.actor.to(device)
            for params in self.actor.features.parameters():
                params.requires_grad = False
            self.loss = nn.MSELoss()
            self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
            self.learn = self._learn_bc


    def get_action(self, obs):
        """Get action based on imitator's current policy
        
        Arguments:
            obs {np.array} -- Image observation/state
        
        Returns:
            np.array -- Np.array of size `action_dim`
        """
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        return self.actor(obs).cpu().data.numpy().flatten()

    def _learn_bc(self, num_iter: int, batch_size: int=64) -> dict:
        """ Learn an imitator model using behavior cloning

        Args:
            num_iter: -- Number of iterations
            batch_size:

        Returns: statistics

        """
        stats = {"loss_a": []}
        for i in range(num_iter):
            # Sample from the demonstrator
            demo_states, demo_actions = self.demonstrator.sample(batch_size)
            demo_states = demo_states.to(device)
            demo_actions = demo_actions.to(device)

            # Get predictions from actor
            actor_actions = self.actor(demo_states)

            # Update actor
            loss = self.loss(demo_actions, actor_actions)
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            stats["loss_a"].append(loss.item())
        return stats

    def _learn_adversarial(self, num_iter: int, batch_size: int=256) -> dict:
        """Learn an imitator model (actor, discriminator) using adversarial loss
        
        Args:
            num_iter {int} -- Number of iterations to train
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {256})
        Returns: statistics
        """
        stats = {"loss_d": [], "loss_a": []}
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
            stats["loss_d"].append(loss.item())

            # Update Actor
            loss_actor = - self.discriminator(state, actor_action)
            self.actor_optim.zero_grad()
            loss_actor.mean().backward()
            self.actor_optim.step()
            stats["loss_a"].append(loss.item())
        return stats

    def save(self, save_dir="imitation_learning/trained_models"):
        """Save current model states to save_dir"""
        os.makedirs(save_dir, exist_ok=True)
        if self.algo == "adv":
            torch.save(self.actor.state_dict(), f"{save_dir}/actor_adv.ptm")
            torch.save(self.discriminator.state_dict(), f"{save_dir}/discriminator_adv.ptm")
            print(f"Saved actor_adv.ptm & discriminator_adv.ptm")
        elif self.algo == "bc":
            torch.save(self.actor.state_dict(), f"{save_dir}/actor_bc.ptm")
            print(f"Saved {save_dir}/actor_bc.ptm")

    def load(self, load_dir="imitation_learning/trained_models"):
        """Load model states from load_dir"""
        if self.algo == "adv":
            actor_model_file = f"{load_dir}/actor_adv.ptm"
            discriminator_model_file = f"{load_dir}/discriminator_adv.ptm"
            if os.path.isfile(actor_model_file) and os.path.isfile(
                discriminator_model_file
            ):
                self.actor.load_state_dict(torch.load(actor_model_file))
                self.discriminator.load_state_dict(torch.load(discriminator_model_file))
                print("Loaded actor & discriminator model.")
            else:
                raise ValueError("Model files not found. Please check trained-models-dir")
        if self.algo == "bc":
            actor_model_file = f"{load_dir}/actor_bc.ptm"
            if os.path.isfile(actor_model_file):
                self.actor.load_state_dict(torch.load(actor_model_file))
                print("Loaded actor_bc model.")
            else:
                raise ValueError("Model files not found. Please check trained-models-dir")

