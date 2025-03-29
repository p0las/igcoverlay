from pathlib import Path
from typing import Any, Dict

from PIL import Image, ImageDraw, ImageFont

FONT = "trebucbd.ttf"


def textWithBorder(draw, x, y, text, font, fillcolor, shadowcolor):
    # thicker border
    draw.text((x - 1, y - 1), text, font=font, fill=shadowcolor)
    draw.text((x + 1, y - 1), text, font=font, fill=shadowcolor)
    draw.text((x - 1, y + 1), text, font=font, fill=shadowcolor)
    draw.text((x + 1, y + 1), text, font=font, fill=shadowcolor)

    # now draw the text over it
    draw.text((x, y), text, font=font, fill=fillcolor)


def makeOverlay(path, index, vs):
    # Create an image
    img = Image.new('RGBA', (1000, 1000), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Use a built-in font or load a custom one
    font = ImageFont.truetype(FONT, size=80)
    font_small = ImageFont.truetype(FONT, size=20)

    # Draw text on the image
    draw.text((10, 15), "VS (m/s)", fill="white", font=font_small)
    # draw.text((10, 45), str(round(vs, 2)), fill="white", font=font)

    textWithBorder(draw, 10, 45, str(round(vs, 2)), font, "white", "black")

    # Save or display the image

    img.save(Path(path) / ("overlay." + str(index).zfill(4) + ".png"))


def makeOverlayPath(folder, image_name, index):
    return Path(folder) / (image_name + "." + str(index).zfill(4) + ".png")


class OverlayImage():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.img = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
        self.scale = height / 1000
        self.draw = ImageDraw.Draw(self.img)
        self.font = ImageFont.truetype(FONT, size=160 * self.scale)
        self.font_small = ImageFont.truetype(FONT, size=40 * self.scale)

        self.colour = 'white'
        self.outline_colour = 'black'
        self.border_thickness = 4
        self.section_offset = 0.08
        self.label_offset = 0.003

        self.font_height = float(self.getFontHeight(self.font) + self.border_thickness * 2) / self.height
        self.font_small_height = float(self.getFontHeight(self.font_small)) / self.height

        self.current_top = 0

    def textWithBorder(self, x, y, text, border_thickness, font, colour, outline_colour):
        """x,y are in % of the image size"""

        x, y = self.toPixels(x, y)

        thickness = int(border_thickness * self.scale)
        self.draw.text((x - thickness, y - thickness), text, font=font, fill=outline_colour)
        self.draw.text((x + thickness, y - thickness), text, font=font, fill=outline_colour)
        self.draw.text((x - thickness, y + thickness), text, font=font, fill=outline_colour)
        self.draw.text((x + thickness, y + thickness), text, font=font, fill=outline_colour)

        # now draw the text over it
        self.draw.text((x, y), text, font=font, fill=colour)

    def toPixels(self, x, y):
        x = int(x * self.width)
        y = int(y * self.width)
        return x, y

    def getFontHeight(self, font: ImageFont):
        return font.getbbox("2")[3] - font.getbbox("2")[1]

    def text(self, x, y, text):
        """x,y are in % of the image size"""

        x, y = self.toPixels(x, y)
        self.draw.text((x, y), text, font=self.font_small, fill=self.colour)

    def generate(self, path, data: Dict[str, Any]):
        left_margin = 0.01
        self.current_top = 0.01

        if 'speed' in data and data['speed'] is not None:
            self.addSection(left_margin, "Speed (km/h)", data['speed'])
        if 'alt' in data and data['alt'] is not None:
            self.addSection(left_margin, "Altitude (m)", data['alt'])
        if 'vs' in data and data['vs'] is not None:
            self.addSection(left_margin, "VS (m/s)", data['vs'])

        self.img.save(path)

    def addSection(self, left_margin, label, value):

        self.textWithBorder(left_margin, self.current_top, label, self.border_thickness/2, self.font_small, self.colour, self.outline_colour)
        self.moveTopLabel()
        self.textWithBorder(left_margin, self.current_top, str(round(value, 2)), self.border_thickness, self.font, self.colour, self.outline_colour)
        self.moveTopSection()

    def moveTopSection(self):
        self.current_top += self.section_offset + self.font_height

    def moveTopLabel(self):
        self.current_top += self.label_offset + self.font_small_height
