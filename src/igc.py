from typing import List

import numpy as np
from geopy.distance import geodesic
from igc_reader import igc_reader
from scipy.interpolate import make_smoothing_spline


def interpolate_besier(data, t=0.5):
    data2 = np.zeros(len(data))
    for i in range(0, len(data)):
        try:
            data2[i] = (bezier_interpolate(data[i - 2], data[i - 1], data[i + 1], data[i + 2], t) + data[i]) / 2
        except:
            data2[i] = data[i]
    return data2


def bezier_interpolate(p0, p1, p2, p3, t=0.5):
    """Interpolates the middle point using a cubic Bézier curve from 5 evenly spaced points."""

    # Define Bézier control points
    B0, B1, B2, B3 = p0, p1, p2, p3

    # Compute cubic Bézier interpolation
    middle_point = ((1 - t) ** 3 * B0 +
                    3 * (1 - t) ** 2 * t * B1 +
                    3 * (1 - t) * t ** 2 * B2 +
                    t ** 3 * B3)

    return middle_point


class Interpolator():
    def __init__(self, data):
        self.data = data

    def get(self, time: float):
        index = int(time)
        if index < 2:
            return self.data[index]
        try:
            t = time - index
            m = 1.0 / 3.0

            tt = t / 3 + m
            tt1 = t / 3 + 2 * m
            tt2 = t / 3

            b = bezier_interpolate(self.data[index - 1], self.data[index], self.data[index + 1], self.data[index + 2], tt)
            b1 = bezier_interpolate(self.data[index - 2], self.data[index - 1], self.data[index], self.data[index + 1], tt1)
            b2 = bezier_interpolate(self.data[index], self.data[index + 1], self.data[index + 2], self.data[index + 3], tt2)

            return (b * m * 2) + (b1 * (1.0 - t) * m) + (b2 * t * m)
        except:
            return self.data[index]

    def __getitem__(self, key):
        return self.get(key)

    def __len__(self):
        return len(self.data)


class Interpolator2():
    def __init__(self, data: List, l: float = 5):
        self.data = data
        x = np.arange(len(self.data))
        y = self.data
        self.spl = make_smoothing_spline(x, y, lam=l)

    def get(self, time: float):
        return float(self.spl(time))

    def __getitem__(self, key):
        return self.get(key)

    def __len__(self):
        return len(self.data)


class IgcReader():
    def __init__(self):
        self._igc_reader = igc_reader.read_igc
        self._raw_data = None
        self._track = None
        self.altitude = None

    def load(self, path):
        self._raw_data = self._igc_reader.from_file(path)
        self._track = self._raw_data.track

        self.altitude = Interpolator2(self._track.gps_alt)
        self.latitude = Interpolator2(self._track.latitude, 0.5)
        self.longitude = Interpolator2(self._track.longitude, 0.5)

    def getLength(self):
        return len(self._track.latitude)

    def getSpeed(self, index):
        i = index
        distance = geodesic((self.latitude[i], self.longitude[i]), (self.latitude[i - 1], self.longitude[i - 1])).meters
        return distance * 3.6  # km/h

    def getAltitude(self, index):
        return self.altitude[index]

    def getVerticalSpeed(self, index):
        return self.altitude[index + 1] - self.altitude[index]

    def getTime(self, index):
        pass
