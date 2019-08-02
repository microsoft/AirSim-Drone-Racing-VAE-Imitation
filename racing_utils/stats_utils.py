import numpy as np
import matplotlib.pyplot as plt

def calculate_gate_stats(predictions, poses):
    # display averages
    mean_pred = np.mean(predictions, axis=0)
    mean_pose = np.mean(poses, axis=0)
    print('Means (prediction, GT) : R({} , {}) Theta({} , {}) Psi({} , {}) Phi_rel({} , {})'.format(
        mean_pred[0], mean_pose[0], mean_pred[1], mean_pose[1], mean_pred[2], mean_pose[2], mean_pred[3], mean_pose[3]))
    # display mean absolute error
    abs_diff = np.abs(predictions-poses)
    mae = np.mean(abs_diff, axis=0)
    print('MAE : R({}) Theta({}) Psi({}) Phi_rel({})'.format(mae[0], mae[1], mae[2], mae[3]))
    # display max errors
    max_diff = np.max(abs_diff, axis=0)
    print('Max error : R({}) Theta({}) Psi({}) Phi_rel({})'.format(max_diff[0], max_diff[1], max_diff[2], max_diff[3]))
    plt.title("R MAE histogram")
    _ = plt.hist(abs_diff[:, 0], np.linspace(0.0, 10.0, num=1000))
    plt.show()
    plt.title("Theta MAE histogram")
    _ = plt.hist(abs_diff[:, 1], np.linspace(0.0, np.pi, num=1000))
    plt.show()
    plt.title("Phi MAE histogram")
    _ = plt.hist(abs_diff[:, 2], np.linspace(0.0, np.pi, num=1000))
    plt.show()
    plt.title("Phi_rel MAE histogram")
    _ = plt.hist(abs_diff[:, 3], np.linspace(0.0, np.pi, num=100))
    plt.show()


