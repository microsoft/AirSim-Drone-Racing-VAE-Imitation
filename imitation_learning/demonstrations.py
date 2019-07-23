import numpy as np
import torch
import cv2


class Demonstrations(object):
    """ Expert demonstrations or trajectories"""
    def __init__(self, task_name):
        self.states = np.genfromtxt("imitation_learning/demonstrations/{}/states.txt".format(
            task_name
        ), dtype=str)
        self.actions = np.genfromtxt("imitation_learning/demonstrations/{}/actions.txt".format(
            task_name
        ), dtype=float, delimiter=',')
        self.num_demos = len(self.actions)

    def _process_state(self, state):
        """ Process the state input and return state image data as np array"""
        if type(state) == np.str_:
            state = cv2.imread(state, flags=cv2.IMREAD_COLOR)
        elif type(state) == np.ndarray:
            pass
        return np.transpose(state, (2, 0, 1))

    def sample(self, batch_size):
        """Sample `batch_size` number of samples and return states & actions"""
        sample_indices = np.random.randint(0, self.num_demos, size=batch_size)
        states = []
        actions = []
        for idx in sample_indices:
            states.append(self._process_state(self.states[idx]))
            actions.append(self.actions[idx])
        return torch.FloatTensor(np.array(states)), torch.FloatTensor(np.array(actions))
