from pathlib import Path

import aerofiles
import pytest

from igc import IgcReader, bezier_interpolate, Interpolator
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import csv_reader


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

        for n in [4,4.25,4.5,4.75,5]:
            assert i.get(n) == pytest.approx(n+1, rel=1e-2)

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

        for n in [4,4.25,4.5,4.75,5]:
            print(i[n])




class Test_test():

    def test_lat_long(self, igc_reader, csv_data):
        igc = igc_reader

        s = 100
        e = 200
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

        x = np.arange(ll)

        plt.figure(figsize=(18, 5))
        # plt.plot(long, lat, '-', label='Rounded Data', markersize=5, alpha=0.6)
        # plt.plot(x, y_rounded, '-', label='curve', linewidth=2)
        # plt.plot(long2, lat2, '-', label='curve', linewidth=2)
        # plt.plot(long, lat, 'o', label='curve', linewidth=2,markersize=5)

        plt.plot(lat, long, '-', label='reference', linewidth=2)
        plt.plot(igc.latitude[s:e], igc.longitude[s:e], '-', label='curve', linewidth=2)

        plt.legend()
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Restoring a Smooth Curve from Rounded Data")
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
