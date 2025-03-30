import math

import attrs
import numpy as np
from igc_reader import igc_reader
from scipy.interpolate import UnivariateSpline
from geopy.distance import geodesic

SMOOTHING = 0.07


# def interpolate(data, smoothing=0.00000005):
#     # return data
#     x = np.arange(len(data))
#     spline = UnivariateSpline(x, data, s=smoothing)
#     return spline(x)


def interpolate_besier(data, t=0.5):
    data2 = np.zeros(len(data))
    for i in range(0, len(data)):
        try:
            data2[i] = (bezier_interpolate(data[i - 2], data[i - 1], data[i + 1], data[i + 2], t) + data[i]) / 2
        except:
            data2[i] = data[i]
    return data2


# def distance(point1, point2):
#     R = 6371000
#     degToRad = math.pi / 180.0
#     return R * degToRad * math.sqrt(math.pow(math.cos(point1.lat * degToRad) * (point1.lng - point2.lng), 2) + math.pow(point1.lat - point2.lat, 2))

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

            # return (bezier_interpolate(self.data[index - 2], self.data[index - 1], self.data[index + 1], self.data[index + 2], t) + self.data[index]) / 2
        except:
            return self.data[index]

    def __getitem__(self, key):
        return self.get(key)

    def __len__(self):
        return len(self.data)


class Interpolator2():
    def __init__(self, data):
        self.data = data

    def calc(self,time):
        index = int(time)
        if index < 2:
            return self.data[index]
        try:
            t = time - index
            smoothing = 0.3

            data = self.data[index - 4:index + 5]
            x = np.arange(len(data))
            spline = UnivariateSpline(x, data, s=smoothing)

            s1 = spline(4)
            # return s1

            data = self.data[index - 3:index + 6]
            x = np.arange(len(data))
            spline = UnivariateSpline(x, data, s=smoothing)

            s2 = spline(4)
            # return (s1+s2)/2

            x = [0, 1]
            y = [s1, s2]

            return np.interp(t, x, y)
        except:
            return self.data[index]


    def get(self, time: float):

        return (self.calc(time)+self.calc(time+1))/2

        index = int(time)
        if index < 2:
            return self.data[index]
        try:
            t = time - index

            # tt = t / 3 + 0.333
            # tt1 = t / 3 + 0.666
            # tt2 = t / 3
            #
            # m = 1.0 / 3.0
            #
            # b = bezier_interpolate(self.data[index - 1], self.data[index], self.data[index + 1], self.data[index + 2], tt)
            # b1 = bezier_interpolate(self.data[index - 2], self.data[index - 1], self.data[index], self.data[index + 1], tt1)
            # b2 = bezier_interpolate(self.data[index], self.data[index + 1], self.data[index + 2], self.data[index + 3], tt2)
            #
            # # return (b * m * 2) + (b1 * (1.0 - t) * m) + (b2 * t * m)
            #
            # tt3 = t/3/2+0.5
            # tt4 = t/3/2+m
            # b3 = bezier_interpolate(self.data[index - 2], self.data[index - 1], self.data[index + 1], self.data[index + 2], tt3)
            # b4 = bezier_interpolate(self.data[index - 1], self.data[index], self.data[index + 2], self.data[index + 3], tt4)
            #
            # #return ((b3+b4)/2+self.data[index])/2
            #
            # a1 = (self.data[index - 1] + self.data[index] + self.data[index + 1])/3
            # a2 = (self.data[index] + self.data[index + 1] + self.data[index + 2])/3
            #
            # # return (a2*t+(1-t)*a1 + self.data[index]*(1-t) + self.data[index+1]*t)/2

            # b3 = bezier_interpolate(self.data[index - 2], self.data[index - 1], self.data[index + 1], self.data[index + 2], 0.5)
            # b4 = bezier_interpolate(self.data[index - 1], self.data[index], self.data[index + 2], self.data[index + 3], 0.5)
            # x = [0, 1]
            # y = [b3, b4]
            #
            # yy = [self.data[index], self.data[index + 1]]
            #
            # return (np.interp(t, x, y) + np.interp(t, x, yy)) / 2

            smoothing = 100

            data = self.data[index - 4:index + 5]
            x = np.arange(len(data))
            spline = UnivariateSpline(x, data,s=smoothing)

            s1 = spline(4 )
            # return s1

            data = self.data[index - 3:index + 6]
            x = np.arange(len(data))
            spline = UnivariateSpline(x, data, s=smoothing)

            s2 = spline( 4)
            # return (s1+s2)/2

            x = [0, 1]
            y = [s1, s2]

            # return np.interp(t, x, y)

            #control points

            data = self.data[index - 5:index + 4]
            x = np.arange(len(data))
            spline = UnivariateSpline(x, data, s=smoothing)

            sc1 = spline(4)
            # return s1

            data = self.data[index - 3:index + 7]
            x = np.arange(len(data))
            spline = UnivariateSpline(x, data, s=smoothing)

            sc2 = spline(4)

            return bezier_interpolate(sc1, s1, s2, sc2, t/3+(1.0/3.0))


        except:
            return self.data[index]

    def __getitem__(self, key):
        return self.get(key)

    def __len__(self):
        return len(self.data)


class IgcReader():
    def __init__(self):
        self._igc_reader = igc_reader.read_igc
        self._raw_data = None
        self._track = None
        self._altitude = None

    def load(self, path):
        self._raw_data = self._igc_reader.from_file(path)
        self._track = self._raw_data.track

        self._altitude = Interpolator2(self._track.gps_alt)
        self.latitude = Interpolator(self._track.latitude)
        self.longitude = Interpolator(self._track.longitude)

        self.distance = np.zeros(len(self._track.latitude))

        for i in range(1, len(self._track.latitude)):
            distance = geodesic((self.latitude[i], self.longitude[i]), (self.latitude[i - 1], self.longitude[i - 1])).meters
            self.distance[i] = distance

        self.speed = np.zeros(len(self._altitude))
        for i in range(0, len(self._altitude)):
            # try:
            #     self.speed[i] = (bezier_interpolate_middle(self.distance[i - 2], self.distance[i - 1], self.distance[i + 1], self.distance[i + 2]) + self.distance[i]) / 2
            # except:
            self.speed[i] = self.distance[i]

    def getLength(self):
        return len(self._track.latitude)

    def getSpeed(self, index):
        # p = Point(self.latitude[index], self.longitude[index])
        # p2 = Point(self.latitude[index + 1], self.longitude[index + 1])
        # speed = p.distance(p2)  # m/s (each sample is 1 second)

        i = index
        distance = geodesic((self.latitude[i], self.longitude[i]), (self.latitude[i - 1], self.longitude[i - 1])).meters

        return distance * 3.6  # km/h

    def getAltitude(self, index):
        return self._altitude[index]

    def getVerticalSpeed(self, index):
        return self._altitude[index + 1] - self._altitude[index]

    def getTime(self, index):
        pass

# path = r"O:\GoogleDrive\paragliding\flight_logs\2025\25-03-22-stanwell_park\250322013012.igc"
# igc_data = igc_reader.read_igc.from_file(path)
#
# igc_data.track.latitude
#
# from datetime import datetime, time
#
# dt1 = datetime.combine(datetime.now(), igc_data.track.timestamp[10])
# dt2 = datetime.combine(datetime.now(), igc_data.track.timestamp[9])
# delta = dt1 - dt2
# delta.total_seconds()
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import UnivariateSpline
#
# # Given rounded integer values
# samples = igc_data.track.p_alt
#
# y_rounded = samples[1:2200]  # np.array([146, 147, 147, 148, 149, 150, 150, 151, 151, 152,
# # 153, 153, 154, 154, 155, 155, 154, 154, 154, 154,
# # 155, 156, 156, 157, 157, 158, 159, 160, 160, 161,
# # 161, 162, 163, 164, 165, 166, 167, 168, 169, 169])
#
# x = np.arange(len(y_rounded))  # Create x values (indices)
#
# # Fit a cubic spline with smoothing factor s
# spline = UnivariateSpline(x, y_rounded, s=len(y_rounded) * 0.07)  # Adjust 's' for more/less smoothing
#
# y_smooth = spline(x)
#
# # Plot results
# plt.figure(figsize=(18, 5))
# plt.plot(x, y_rounded, 'o', label='Rounded Data', markersize=5, alpha=0.6)
# plt.plot(x, y_smooth, '-', label='Smoothed Curve', linewidth=2)
# plt.legend()
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.title("Restoring a Smooth Curve from Rounded Data")
# plt.grid()
# plt.show()
#
# print(y_rounded[115])
# print(y_smooth[115])
#
# print(y_rounded[115] - y_rounded[114])
# print(y_smooth[115] - y_smooth[114])
