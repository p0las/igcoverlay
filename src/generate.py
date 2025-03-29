import argparse
from pathlib import Path

from igc import IgcReader
from overlay import OverlayImage, makeOverlayPath
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--igc', type=str, required=True, help='Input IGC file')

    parser.add_argument('--out', type=str, required=True, help='Output folder')
    parser.add_argument('--fps', type=int, default=1, help='overlay FPS, default 1')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    igc = IgcReader()
    igc.load(args.igc)

    fps = args.fps
    for i in tqdm(range((igc.getLength() - 1) * fps)):
        o = OverlayImage(1000, 1000)
        p = makeOverlayPath(args.out, "overlay_" + Path(args.igc).stem, i)
        data = {
            "vs": igc.getVerticalSpeed(float(i) / fps),
            "speed": igc.getSpeed(float(i) / fps),
            "alt": igc.getAltitude(float(i) / fps)
        }
        o.generate(p, data)
