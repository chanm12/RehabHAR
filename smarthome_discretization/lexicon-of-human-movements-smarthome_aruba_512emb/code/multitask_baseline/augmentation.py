import numpy as np
from scipy.interpolate import CubicSpline  # for warping
from transforms3d.axangles import axangle2mat


# The augmentation transforms are taken from:
# https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob
# /master/Example_DataAugmentation_TimeseriesData.ipynb

# The rest are taken from this paper: https://arxiv.org/pdf/1907.11879.pdf


class Jitter(object):
    """
    Adds random noise to the data
    """

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, data):
        # We need to deep copy data! We destroy the dataset otherwise
        data = np.copy(data)

        label = 0
        if np.random.rand() > 0.5:
            noise = np.random.normal(loc=0, scale=self.sigma, size=data.shape)
            data += noise
            label = 1

        return data, label


class Scaling(object):
    """
    Randomly scales the data by a factor
    """

    def __init__(self, sigma=0.25):
        self.sigma = sigma

    def __call__(self, data):
        data = np.copy(data)

        label = 0
        if np.random.rand() > 0.5:
            scaling_factor = np.random.normal(
                loc=1.0, scale=self.sigma, size=(1, data.shape[1])
            )
            noise = np.matmul(np.ones((data.shape[0], 1)), scaling_factor)
            data *= noise
            label = 1

        return data, label


class Rotation(object):
    """
    Performing rotations
    """

    def __init__(self):
        pass

    def DA_Rotation(self, X):
        axis = np.random.uniform(low=-1, high=1, size=3)
        angle = np.random.uniform(low=-np.pi, high=np.pi)

        if X.shape[-1] == 3:
            return np.matmul(X, axangle2mat(axis, angle))
        elif X.shape[-1] == 6:
            acc = np.matmul(X[:, :3], axangle2mat(axis, angle))
            gyro = np.matmul(X[:, 3:], axangle2mat(axis, angle))
            data = np.hstack((acc, gyro))
        elif X.shape[-1] == 60:
            # This is only for Skoda right now
            for i in range(20):
                sensor = np.matmul(X[:, i * 3 : (i + 1) * 3], axangle2mat(axis, angle))

                if i == 0:
                    data = sensor
                else:
                    data = np.hstack((data, sensor))

        return data

    def __call__(self, data):
        data = np.copy(data)

        label = 0
        if np.random.rand() > 0.5:
            data = self.DA_Rotation(data)
            label = 1

        return data, label


class Negation(object):
    """
    Negating all sensory values. Simulates situations where the sensor is
    upside-down.
    """

    def __int__(self):
        pass

    def __call__(self, data):
        data = np.copy(data)

        label = 0
        if np.random.rand() > 0.5:
            data = data * -1
            label = 1
        return data, label


class HorizontalFlipping(object):
    """
    Reversing the temporal order
    """

    def __init__(self):
        pass

    def __call__(self, data):
        data = np.copy(data)

        label = 0
        if np.random.rand() > 0.5:
            data = np.fliplr(data).copy()  # yells bloody murder here if we
            # dont copy
            label = 1
        return data, label


class Permutation(object):
    """
    Performing rotations
    """

    def __init__(self):
        pass

    # 2 and 5 are a combination that produces results instantly
    def DA_Permutation(self, X, nPerm=2, minSegLength=5):
        X_new = np.zeros(X.shape)
        idx = np.random.permutation(nPerm)
        bWhile = True
        while bWhile == True:
            segs = np.zeros(nPerm + 1, dtype=int)
            segs[1:-1] = np.sort(
                np.random.randint(minSegLength, X.shape[0] - minSegLength, nPerm - 1)
            )
            segs[-1] = X.shape[0]
            if np.min(segs[1:] - segs[0:-1]) > minSegLength:
                bWhile = False
        pp = 0
        for ii in range(nPerm):
            x_temp = X[segs[idx[ii]] : segs[idx[ii] + 1], :]
            X_new[pp : pp + len(x_temp), :] = x_temp
            pp += len(x_temp)
        return X_new

    def __call__(self, data):
        data = np.copy(data)

        label = 0
        if np.random.rand() > 0.5:
            data = self.DA_Permutation(data)
            label = 1

        return data, label


class TimeWarping(object):
    """
    Performing rotations
    """

    def __init__(self):
        pass

    def GenerateRandomCurves(self, X, sigma=0.2, knot=4):
        xx = (
            np.ones((X.shape[1], 1))
            * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))
        ).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
        x_range = np.arange(X.shape[0])

        random_curves = np.zeros((X.shape[0], X.shape[1]))
        for i in range(X.shape[1]):
            spline = CubicSpline(xx[:, i], yy[:, i])
            random_curves[:, i] = np.array(spline(x_range))

        return random_curves

    def DistortTimesteps(self, X, sigma=0.2):
        # Regard these samples around 1 as time intervals
        tt = self.GenerateRandomCurves(X, sigma)

        # Add intervals to make a cumulative graph
        # Make the last value to have X.shape[0]
        tt_cum = np.cumsum(tt, axis=0)

        t_scale = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            t_scale[i] = (X.shape[0] - 1) / (tt_cum[-1, i])
            tt_cum[:, i] = tt_cum[:, i] * t_scale[i]

        return tt_cum

    def DA_TimeWarp(self, X, sigma=0.2):
        tt_new = self.DistortTimesteps(X, sigma)
        X_new = np.zeros(X.shape)
        x_range = np.arange(X.shape[0])

        for i in range(X.shape[1]):
            X_new[:, i] = np.interp(x_range, tt_new[:, i], X[:, i])

        return X_new

    def __call__(self, data):
        data = np.copy(data)

        label = 0
        # TODO: vectorize the operations
        if np.random.rand() > 0.5:
            data = self.DA_TimeWarp(data)
            label = 1

        return data, label


class ChannelShuffling(object):
    """
    Randomly shuffling across channels.
    """

    def __init__(self):
        pass

    def __call__(self, data):
        data = np.copy(data)

        label = 0
        if np.random.rand() > 0.5:
            length = np.arange(0, data.shape[1])
            length = np.random.permutation(length)
            data = data[:, length]
            label = 1

        return data, label
