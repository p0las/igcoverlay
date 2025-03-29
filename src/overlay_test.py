from overlay import OverlayImage, makeOverlayPath


class Test_test():
    def test_overlay(self):
        # makeOverlay("c:/temp", 1, 2.3)

        o = OverlayImage(1000, 1000)
        p = makeOverlayPath("c:/temp", "test", 1)
        o.generate(p, {"vs": 2.3, "speed": 45.0, "alt": 297.0})
