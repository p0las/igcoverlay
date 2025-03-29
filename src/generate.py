import argparse

from csv_reader import CsvReader
from igc import IgcReader
from overlay import OverlayImage, makeOverlayPath


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--igc', type=str, help='Input IGC file')
    parser.add_argument('--csv', type=str, help='Input CSV file')

    parser.add_argument('--out', type=str, help='Output folder')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.igc:
        igc = IgcReader()
        igc.load(args.igc)
    elif args.csv:
        igc = CsvReader()
        igc.load(args.csv)
    else:
        raise ValueError("No input file provided")

    for i in range(igc.getLength()-1):
        o = OverlayImage(1000, 1000)
        p = makeOverlayPath(args.out, "overlay", i)
        data = {
            "vs": igc.getVerticalSpeed(i),
            "speed": igc.getSpeed(i),
            "alt": igc.getAltitude(i)
        }
        print(data)
        o.generate(p, data)
