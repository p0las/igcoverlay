from pathlib import Path

import aerofiles
import pytest

from igc import IgcReader, bezier_interpolate, Interpolator
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import csv


@pytest.fixture()
def csv_data():
    csv_path = Path(__file__).parent.parent / "test_data" / "processed_data.csv"

    d = []
    with open(csv_path, mode='r') as file:
        data = csv.reader(file)

        for i, row in enumerate(data):
            if i == 0:
                continue
            r = {}
            r['latitude'] = float(row[2][3:10])
            r['longitude'] = float(row[3])
            r['altitude'] = float(row[4])
            r['speed'] = float(row[7])
            r['vertical_speed'] = float(row[8])
            d.append(r)
    return d


@pytest.fixture()
def igc_reader():
    path = Path(__file__).parent.parent / "test_data" / "source.igc"
    igc = IgcReader()
    igc.load(path.as_posix())
    return igc


class Test_interpolator():
    def test_test(self):
        input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        i = Interpolator(input)
        print(i.get(0))
        assert i.get(0) == 1
        assert i.get(1) == 2

        for n in [4, 4.25, 4.5, 4.75, 5]:
            assert i.get(n) == pytest.approx(n + 1, rel=1e-2)

        plt.figure(figsize=(18, 5))
        # plt.plot(long, lat, '-', label='Rounded Data', markersize=5, alpha=0.6)
        # plt.plot(x, y_rounded, '-', label='curve', linewidth=2)
        # plt.plot(long2, lat2, '-', label='curve', linewidth=2)
        # plt.plot(long, lat, 'o', label='curve', linewidth=2,markersize=5)

        x = np.arange(len(input))
        xx = np.arange(0, 10, 0.1)
        ii = [i.get(n) for n in xx]
        plt.plot(x, input, 'o', label='reference', markersize=5)
        plt.plot(xx, ii, '-', label='curve', linewidth=2)

        plt.legend()
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Restoring a Smooth Curve from Rounded Data")
        plt.grid()
        plt.show()

    def test_array(self):
        input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        i = Interpolator(input)

        for n in [4, 4.25, 4.5, 4.75, 5]:
            print(i[n])


class Test_test():

    def test_lat_long(self, igc_reader, csv_data):
        igc = igc_reader

        s = 15
        e = 400
        ll = e - s

        lat = np.zeros(ll)
        long = np.zeros(ll)

        for i in range(0, ll):
            assert float(csv_data[i + s]['longitude']) == pytest.approx(igc.longitude[i + s], rel=1e-6)
            assert float(csv_data[i + s]['latitude']) == pytest.approx(igc.latitude[i + s], rel=1e-5)  # 34.2236

            lat[i] = csv_data[i + s]['latitude']
            long[i] = csv_data[i + s]['longitude']

            # print(f"{csv_data[i + s]['longitude']} == {igc.longitude[i + s]}")
            # print(f"{csv_data[i + s]['latitude']} == {igc.latitude[i + s]}")

        # x = np.arange(ll)

        plt.figure(figsize=(18, 18))
        # plt.plot(long, lat, '-', label='Rounded Data', markersize=5, alpha=0.6)
        # plt.plot(x, y_rounded, '-', label='curve', linewidth=2)

        lat = [igc.latitude[i] for i in np.arange(s, e, 0.1)]
        long = [igc.longitude[i] for i in np.arange(s, e,0.1)]

        plt.plot(lat, long, '-', label='curve', linewidth=2)

        plt.plot(igc._track.latitude[s:e], igc._track.longitude[s:e], 'o', label='raw', markersize=5)
        #
        # plt.plot(lat, long, 'o', label='csv reference', markersize=3)

        # lat1 = [igc.latitude[i] for i in range(s, e)]
        # lon1 = [igc.longitude[i] for i in range(s, e)]
        #
        # xx = np.arange(s, e, 0.1)

        # lat1 = [igc.latitude[i] for i in xx]
        # lon1 = [igc.longitude[i] for i in xx]
        #
        # lat1 = np.zeros(ll * 5)
        # lon1 = np.zeros(ll * 5)
        # for i in range(s, e):
        #     slat = np.zeros(5)
        #     slon = np.zeros(5)
        #     slat2 = np.zeros(5)
        #     slon2 = np.zeros(5)
        #     slat3 = np.zeros(5)
        #     slon3 = np.zeros(5)
        #
        #     for ii in np.arange(i, i + 1, 0.2):
        #         t = ii - i
        #         tt = t
        #
        #         tt = t / 3 + 0.333
        #         tt1 = t / 3 + 0.666
        #         tt2 = t / 3
        #
        #         m=1.0/3.0
        #
        #         b = bezier_interpolate(igc._track.latitude[i - 1], igc._track.latitude[i], igc._track.latitude[i + 1], igc._track.latitude[i + 2], tt)
        #         b1 = bezier_interpolate(igc._track.latitude[i - 2], igc._track.latitude[i - 1], igc._track.latitude[i], igc._track.latitude[i + 1], tt1)
        #         b2 = bezier_interpolate(igc._track.latitude[i], igc._track.latitude[i + 1], igc._track.latitude[i + 2], igc._track.latitude[i + 3], tt2)
        #         slat[int((t + 0.01) * 5)] = b
        #         slat2[int((t + 0.01) * 5)] = b1
        #         slat3[int((t + 0.01) * 5)] = b2
        #
        #         lat1[int((ii - s + 0.01) * 5)] = (b * m*2) + (b1 * (1.0 - t) * m) + (b2 * t * m)
        #
        #         c = bezier_interpolate(igc._track.longitude[i - 1], igc._track.longitude[i], igc._track.longitude[i + 1], igc._track.longitude[i + 2], tt)
        #         c1 = bezier_interpolate(igc._track.longitude[i - 2], igc._track.longitude[i - 1], igc._track.longitude[i], igc._track.longitude[i + 1], tt1)
        #         c2 = bezier_interpolate(igc._track.longitude[i], igc._track.longitude[i + 1], igc._track.longitude[i + 2], igc._track.longitude[i + 3], tt2)
        #         slon[int((t + 0.01) * 5)] = c
        #         slon2[int((t + 0.01) * 5)] = c1
        #         slon3[int((t + 0.01) * 5)] = c2
        #
        #         lon1[int((ii - s + 0.01) * 5)] = (c * m*2) + (c1 * (1.0 - t) * m) + (c2 * t * m)
        #
        #         # print(round(ii,2))
        #         # if (round(ii,2) == s + 2+0.4):
        #         #     plt.scatter(b,c, s=100, marker='+', label='point')
        #         #     plt.scatter(b1, c1, s=100, marker='+', label='point-1')
        #         #     plt.scatter(b2, c2, s=100, marker='+', label='point+1')
        #     #
        #     # if (i==s+2):
        #     #     plt.plot(slat, slon, '-', label='besier', linewidth=2, alpha=0.6)
        #     #     plt.plot(slat2, slon2, '--', label='besier-1', linewidth=2, alpha=0.6)
        #     #     plt.plot(slat3, slon3, '-.', label='besier+1', linewidth=2, alpha=0.6)
        #     #     plt.scatter(igc._track.latitude[i], igc._track.longitude[i], s=100, c='red', marker='x', label='point')
        #     #     plt.scatter(igc._track.latitude[i+1], igc._track.longitude[i+1], s=100, c='green', marker='x', label='point2')
        #
        # # t = (time - index) / 2 + 0.5
        # # b1 = bezier_interpolate(self.data[index - 2], self.data[index - 1], self.data[index], self.data[index + 1], t)
        # # t = (time - index) / 2 + 0.5
        # # b2 = bezier_interpolate(self.data[index - 1], self.data[index], self.data[index + 1], self.data[index + 2], t)
        # print(igc._track.latitude[s:e], igc._track.longitude[s:e])
        # print(lat1, lon1)
        # plt.plot(lat1, lon1, '-', label='interpolated', linewidth=1)

        plt.legend()
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Interpolating imprecise data")
        plt.grid()
        plt.show()

    def test_altitude(self, igc_reader, csv_data):

        s = 100
        e = 200
        ll = e - s

        x = np.arange(ll)

        ref = np.zeros(ll)
        data = np.zeros(ll)
        for i in range(0, ll):
            ref[i] = csv_data[i + s]['altitude']
            data[i] = igc_reader.getAltitude(i + s)

        plt.figure(figsize=(18, 5))
        # plt.plot(long, lat, '-', label='Rounded Data', markersize=5, alpha=0.6)
        # plt.plot(x, y_rounded, '-', label='curve', linewidth=2)
        # plt.plot(long2, lat2, '-', label='curve', linewidth=2)
        # plt.plot(long, lat, 'o', label='curve', linewidth=2,markersize=5)

        plt.plot(x, ref, 'o', label='reference', markersize=5)

        data = [igc_reader.getAltitude(a) for a in np.arange(s, e,0.1)]
        x= np.arange(0, e-s,0.1)

        plt.plot(x, data, '-', label='curve', linewidth=2)

        plt.legend()
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Restoring a Smooth Curve from Rounded Data")
        plt.grid()
        plt.show()

        for i in range(e, e):
            print(csv_data[i]['altitude'], igc_reader.getAltitude(i))
            assert float(csv_data[i]['altitude']) == pytest.approx(igc_reader.getAltitude(i), rel=1e-4)
        # assert round(igc_reader.getAltitude(0), 3) == 200.173

    def test_vario(self, igc_reader, csv_data):

        s = 100
        e = 140
        ll = e - s

        x = np.arange(ll)  # Create x values (indices)

        speed = [d['vertical_speed'] for d in csv_data[s:e]]

        data = np.zeros(ll)
        for i in range(0, ll):
            data[i] = igc_reader.getVerticalSpeed(i + s)

        data_avg = np.zeros(ll)
        for i in range(0, ll):
            try:
                # average
                data_avg[i] = (data[i] + data[i + 1]) / 2
            except:
                data_avg[i] = data[i]

        def bezier_interpolate_middle(p0, p1, p2, p3):
            """Interpolates the middle point using a cubic Bézier curve from 5 evenly spaced points."""
            t = 0.5  # Middle of the curve

            # Define Bézier control points (using the middle 4 points)
            B0, B1, B2, B3 = p0, p1, p2, p3  # p4 is ignored since we only use cubic Bézier

            # Compute cubic Bézier interpolation
            middle_point = ((1 - t) ** 3 * B0 +
                            3 * (1 - t) ** 2 * t * B1 +
                            3 * (1 - t) * t ** 2 * B2 +
                            t ** 3 * B3)

            return middle_point

        data_b = np.zeros(ll)
        for i in range(0, ll):
            try:
                # average
                data_b[i] = (bezier_interpolate_middle(data[i - 2], data[i - 1], data[i + 1], data[i + 2]) + data[i]) / 2
            except:
                data_avg[i] = data[i]

        # Fit a cubic spline with smoothing factor s
        # spline = UnivariateSpline(x, y_rounded, s=0.000000001)  # Adjust 's' for more/less smoothing
        #
        # y_smooth = spline(x)
        # y_smooth = igc.latitude
        plt.figure(figsize=(18, 5))
        plt.plot(x, speed, 'o', label='reference', markersize=5, alpha=0.6)
        plt.plot(x, data, '-', label='calculated', linewidth=1)
        plt.plot(x, data_avg, '-', label='avg', linewidth=2)
        plt.plot(x, data_b, '-', label='bezier', linewidth=2)

        data = [igc_reader.getVerticalSpeed(a) for a in np.arange(s, e,0.1)]
        x= np.arange(0, e-s,0.1)

        plt.plot(x, data, '-', label='curve', linewidth=2)


        # plt.plot(x, y_rounded, '-', label='curve', linewidth=2)
        # plt.plot(x, igc.distance_smooth[1700:2200], '-', label='curve', linewidth=2)

        # Plot results
        # plt.plot(x, y_rounded, 'o', label='Rounded Data', markersize=5, alpha=0.6)
        # plt.plot(x, y_smooth, '-', label='Smoothed Curve', linewidth=2)

        # plt.plot(igc.latitude[1700:2000], igc.longitude[1700:2000], '-', label='Smoothed Curve', linewidth=2)
        # plt.plot(igc._track.latitude[1700:2000], igc._track.longitude[1700:2000], 'o', label='Smoothed Curve', markersize=2)

        plt.legend()
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Restoring a Smooth Curve from Rounded Data")
        plt.grid()
        plt.show()

        for i in range(0, 100):
            assert float(csv_data[i]['vertical_speed']) == pytest.approx(igc_reader.getVerticalSpeed(i), rel=1e-6)
        # assert round(igc_reader.getVerticalSpeed(0), 3) == 0.0

    def test_speed(self, igc_reader, csv_data):

        s = 100
        e = 140
        ll = e - s

        speed = [d['speed'] for d in csv_data[s:e]]

        data = [igc_reader.getSpeed(a) for a in range(s, e)]
        data_avg = np.zeros(ll)
        data_b = np.zeros(ll)
        for i, d in enumerate(data):
            print(i)
            print(d)
            try:
                # average
                data_avg[i] = (data[i - 1] + data[i] + data[i + 1] + data[i + 2]) / 4
            except:
                data_avg[i] = data[i]

            try:
                # average
                data_b[i] = (bezier_interpolate(data[i - 2], data[i - 1], data[i + 1], data[i + 2]) + data[i]) / 2
            except:
                data_b[i] = data[i]

        x = np.arange(ll)  # Create x values (indices)

        # Fit a cubic spline with smoothing factor s
        # spline = UnivariateSpline(x, y_rounded, s=0.000000001)  # Adjust 's' for more/less smoothing
        #
        # y_smooth = spline(x)
        # y_smooth = igc.latitude
        plt.figure(figsize=(18, 5))
        plt.plot(x, speed, 'o', label='reference', markersize=5, alpha=0.6)
        plt.plot(x, data, '-', label='calculated', linewidth=1)
        # plt.plot(x, data_avg, '-', label='avg', linewidth=2)
        # plt.plot(x, data_b, '-', label='bezier', linewidth=2)
        # # plt.plot(x, y_rounded, '-', label='curve', linewidth=2)
        # plt.plot(x, igc.distance_smooth[1700:2200], '-', label='curve', linewidth=2)

        # Plot results
        # plt.plot(x, y_rounded, 'o', label='Rounded Data', markersize=5, alpha=0.6)
        # plt.plot(x, y_smooth, '-', label='Smoothed Curve', linewidth=2)

        # plt.plot(igc.latitude[1700:2000], igc.longitude[1700:2000], '-', label='Smoothed Curve', linewidth=2)
        # plt.plot(igc._track.latitude[1700:2000], igc._track.longitude[1700:2000], 'o', label='Smoothed Curve', markersize=2)

        plt.legend()
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Restoring a Smooth Curve from Rounded Data")
        plt.grid()
        plt.show()


class Test_chat():
    def test_test(self, igc_reader):
        import numpy as np
        from scipy.special import comb

        def bernstein_poly(i, n, t):
            """
             The Bernstein polynomial of n, i as a function of t
            """

            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

        def bezier_curve(points, nTimes=1000):
            """
               Given a set of control points, return the
               bezier curve defined by the control points.

               points should be a list of lists, or list of tuples
               such as [ [1,1],
                         [2,3],
                         [4,5], ..[Xn, Yn] ]
                nTimes is the number of time steps, defaults to 1000

                See http://processingjs.nihongoresources.com/bezierinfo/
            """

            nPoints = len(points)
            xPoints = np.array([p[0] for p in points])
            yPoints = np.array([p[1] for p in points])

            t = np.linspace(0.0, 1.0, nTimes)

            polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

            xvals = np.dot(xPoints, polynomial_array)
            yvals = np.dot(yPoints, polynomial_array)

            return xvals, yvals

        from matplotlib import pyplot as plt

        s = 15
        e = 20

        nPoints = 9
        points = np.random.rand(nPoints, 2) * 200
        xpoints = [p[0] for p in points]
        ypoints = [p[1] for p in points]

        xpoints = igc_reader._track.latitude[s:e]
        ypoints = igc_reader._track.longitude[s:e]

        xvals, yvals = bezier_curve(points, nTimes=1000)
        plt.plot(xvals, yvals)
        plt.plot(xpoints, ypoints, "ro")
        for nr in range(len(points)):
            plt.text(points[nr][0], points[nr][1], nr)

        plt.show()
