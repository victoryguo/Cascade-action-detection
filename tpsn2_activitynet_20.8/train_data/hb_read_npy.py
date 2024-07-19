import numpy as np


def _read(file):
    data = np.load(file)
    return data


if __name__=='__main__':
    # file = './train_labels.npy'    # type : numpy.ndarray, shape: (200, 20)

    # file = './rgb_features/1.npy'  # type : numpy.ndarry, shape: (256, 1024)
    # file = './rgb_features/2.npy'  # shape: (259, 1024)
    # file = './rgb_features/3.npy'  # shape: (308, 1024)

    # file = './flow_features/1.npy' # shape: (265, 1024)
    # file = './flow_features/2.npy' # shape: (259, 1024)
    file = './flow_features/3.npy'   # shape: (308, 1024)
    data = _read(file)

    print('data.type(): ', type(data))
    print('data:', data.shape)
    print('d: ', data)

