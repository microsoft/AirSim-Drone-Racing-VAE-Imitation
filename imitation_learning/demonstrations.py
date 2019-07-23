import numpy as np
import cv2


class Demonstrations(object):
    """ Expert demonstrations or trajectories"""
    def __init__(self, task_name):
        self.states = np.loadtxt("demonstrations/{}/states.txt".format(
            task_name
        ))
        self.actions = np.loadtxt("demonstrations/{}/actions.txt".format(
            task_name
        ))
        self.num_demos = len(self.actions)

    def _process_state(self, state):
        """ Process the state input and return state image data as np array"""
        if type(state) == str:
            state_im = cv2.imread(state, mode='RGB')
            return state_im
        elif type(state) == np.array:
            return state

    def sample(self, batch_size):
        """Sample `batch_size` number of samples and return states & actions"""
        sample_indices = np.random.randint(0, self.num_demos, size=batch_size)
        states = []
        actions = []
        for idx in sample_indices:
            states.append(self._process_state(self.states[idx]))
            actions.append(self._process_state(self.actions[idx]))
        return np.array(states), np.array(actions)
