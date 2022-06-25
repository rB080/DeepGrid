import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_ROC(false_positive_rate, true_positive_rate, auroc=None, display=False, save_path=None):
    plt.title('Receiver Operating Characteristic')
    if auroc is not None:
        plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.5f' % auroc)
        plt.legend(loc='lower right')
    else: plt.plot(false_positive_rate, true_positive_rate, 'b')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if display: plt.show()
    if save_path is not None: plt.savefig(save_path)

def plot_training_curves(parameter_list, parameter_name, display=False, save_path=None):
    E = []
    for i in range(len(parameter_list)): E.append(i+1)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(E, parameter_list, 'r')
    plt.ylabel(parameter_name)
    plt.xlabel('Epochs')
    if display: plt.show()
    if save_path is not None: plt.savefig(save_path)

def store_map(T, type='bin_map', save_path=None):
    assert type in ['bgr_map', 'bin_map'], 'Tensor type not found'
    if type == 'bin_map':
        T = np.array(T[0, :, :].detach().cpu().numpy() * 256, dtype=np.uint8)
        if save_path is not None: cv2.imwrite(save_path, T)
    else:
        T = np.array(T.detach().cpu().numpy().transpose(1, 2, 0) * 256, dtype=np.uint8)
        if save_path is not None: cv2.imwrite(save_path, T)
