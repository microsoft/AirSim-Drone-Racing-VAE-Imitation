#!/usr/bin/env python3
from argparse import ArgumentParser

import torch

from imitation_learning.imitator import Imitator

def train(task_name :str="soccer-field-course", num_iter:int=1000, lr: float=1e-5, batch_size: int=32):
    """Training routine
    
    Keyword Arguments:
        task_name {str} -- Name of the task (default: {"soccer-field-course})
        num_iter {int} -- Number of iterations to train (default: {1000})
        lr {[type]} --  Learning rate (default: {1e-5})
        batch_size {int} -- Batch size (default: {256})
    """
    state_dim = (144, 256, 3)  # Dim of input images
    action_dim = 4  # V_x, V_y, V_z, V_{\psi}
    num_ep = 1000  # Number of training episodes/epochs
    imitator = Imitator(task_name, state_dim, action_dim)
    for ep in range(num_ep):
        # Train the Imitator
        imitator.learn(num_iter, batch_size)
        print(f"Ep #: {ep}")
        # TODO @praveen-palanisamy: Add an eval loop to track progress


if __name__ == "__main__":
    # parser = ArgumentParser("Imitation Learning")
    # parser.add_argument("--task-name", default="soccer-field-course",
    #                    help="Name of the task")
    # parser.add_argument("--num-iter", default=1000, type=int,
    #                    help="Number of iterations to train")
    # parser.add_argument("--lr", default=1e-5, type=float,
    #                    help="Learning rate")
    # parser.add_argument("--batch-size", default=256,
    #                    help="Batch size for training iter")
    # args = parser.parse_args()

    train()
