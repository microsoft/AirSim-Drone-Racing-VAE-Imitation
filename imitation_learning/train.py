#!/usr/bin/env python3
from argparse import ArgumentParser

import numpy as np

from imitation_learning.imitator import Imitator


def train(args):
    """Training routine"""
    algo = args.algo
    task_name = args.task_name
    num_iter = args.num_iter
    lr = args.lr
    batch_size = args.batch_size
    state_dim = (144, 256, 3)  # Dim of input images
    action_dim = 4  # V_x, V_y, V_z, V_{\psi}
    num_ep = 1000  # Number of training episodes/epochs
    imitator = Imitator(algo, task_name, state_dim, action_dim, lr=lr)
    if args.load_model:
        imitator.load(args.trained_model_dir)
    for ep in range(num_ep):
        # Train the Imitator
        stats = imitator.learn(num_iter, batch_size)
        print(f"Ep #: {ep}; loss_a={np.mean(stats['loss_a'])}")
        if ep and ep % 10 == 0:
            imitator.save(args.trained_model_dir)
        # TODO @praveen-palanisamy: Add an eval loop to track progress


if __name__ == "__main__":
    parser = ArgumentParser("Imitation Learning")
    parser.add_argument("--task-name", default="soccer-field-course",
                        help="Name of the task")
    parser.add_argument("--num-iter", default=1000, type=int,
                        help="Number of iterations to train")
    parser.add_argument("--lr", default=1e-5, type=float,
                        help="Learning rate")
    parser.add_argument("--batch-size", default=256, type=int,
                        help="Batch size for training iter")
    parser.add_argument("--trained-model-dir",
                        default="imitation_learning/trained_models",
                        help="Pre-trained model directory")
    parser.add_argument("--load-model", action="store_true",
                        help="Load pre-trained model")
    parser.add_argument("--algo", default="bc",
                        help="Algorithm to use (adv:Adversarial or bc: Behavior Cloning)")
    args = parser.parse_args()

    train(args)
