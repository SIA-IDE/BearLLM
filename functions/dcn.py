import numpy as np
from scipy.fft import dct


def pad_or_cut(data: np.ndarray, length=24000):
    if len(data) < length:
        data = np.pad(data, (0, length - len(data)))
    else:
        data = data[:length]
    return data


def normalize(data: np.ndarray):
    power = np.sum(data ** 2)
    data = data * np.sqrt(len(data) / power) * 0.01
    return data.astype(np.float32)


def dcn(data: np.ndarray, length=24000):
    data = dct(data)
    data = pad_or_cut(data, length)
    data = normalize(data)
    return data


if __name__ == '__main__':
    def test():
        data = np.random.randn(24000)
        data = dcn(data)
        print(data.size)

    test()