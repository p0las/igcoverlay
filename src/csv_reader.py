import csv
from pathlib import Path


class CsvReader():
    def __init__(self):

        self._raw_data = None
        self._track = None
        self._altitude = None

    def load(self, path):
        csv_path = Path(path)

        self.d = []
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
                self.d.append(r)

    def getLength(self):
        return len(self.d)

    def getSpeed(self, index):
        # p = Point(self.latitude[index], self.longitude[index])
        # p2 = Point(self.latitude[index + 1], self.longitude[index + 1])
        # speed = p.distance(p2)  # m/s (each sample is 1 second)

        return self.d[index]['speed']

    def getAltitude(self, index):
        return self.d[index]['altitude']

    def getVerticalSpeed(self, index):
        return self.d[index]['vertical_speed']

    def getTime(self, index):
        pass