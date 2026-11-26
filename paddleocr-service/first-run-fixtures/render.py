"""Render the original PII-free documents shipped for first-run OCR."""

import argparse
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

INK = "#212728"
Fonts = Callable[[str, int], ImageFont.FreeTypeFont]


def upright(font: Fonts) -> Image.Image:
    image = Image.new("RGB", (680, 900), "#edece6")
    draw = ImageDraw.Draw(image)
    draw.rectangle((39, 24, 640, 875), fill="#fffffb")
    draw.text((340, 76), "SAMPLE GROCER", font=font("Sans-Bold", 41), fill=INK, anchor="mt")
    draw.text((340, 138), "MARKET RECEIPT", font=font("SansMono", 25), fill=INK, anchor="mt")
    draw.line((84, 194, 596, 194), fill=INK, width=2)
    draw.multiline_text(
        (84, 221),
        "18 NOV 2026           10:25\n"
        "\n"
        "Sourdough loaf        4.50\n"
        "Apples                3.25\n"
        "Oat milk              2.75\n"
        "Tomato soup           5.40\n"
        "\n"
        "SUBTOTAL             15.90\n"
        "TAX                   0.95",
        font=font("SansMono", 29),
        fill=INK,
        spacing=14,
    )
    draw.line((84, 602, 596, 602), fill=INK, width=2)
    draw.text((84, 632), "TOTAL", font=font("SansMono-Bold", 33), fill=INK)
    draw.text((596, 632), "$16.85", font=font("SansMono-Bold", 33), fill=INK, anchor="ra")
    draw.text((84, 704), "Paid in cash", font=font("SansMono", 27), fill=INK)
    draw.text((340, 794), "Thank you for shopping", font=font("Sans", 25), fill=INK, anchor="mt")
    return image


def rotated(font: Fonts) -> Image.Image:
    image = Image.new("RGB", (690, 930), "#d4dedd")
    draw = ImageDraw.Draw(image)
    draw.rectangle((38, 33, 650, 894), fill="#fffdf7")
    draw.text((345, 78), "SAMPLE EVENING CAFE", font=font("Serif", 37), fill=INK, anchor="mt")
    draw.text((345, 139), "Dine-in order", font=font("Serif-Italic", 28), fill=INK, anchor="mt")
    draw.line((87, 196, 604, 196), fill="#40332c", width=3)
    draw.text((88, 217), "Table 8", font=font("Sans", 27), fill=INK)
    draw.text((604, 217), "17 Nov 2026  18:40", font=font("Sans", 27), fill=INK, anchor="ra")
    for index, (item, amount) in enumerate(
        [
            ("2   Roast vegetable wraps", "16.00"),
            ("1   Potato wedges", "4.50"),
            ("2   Lemon iced teas", "6.00"),
        ]
    ):
        y = 300 + 58 * index
        draw.text((88, y), item, font=font("Sans", 29), fill=INK)
        draw.text((604, y), amount, font=font("Sans", 29), fill=INK, anchor="ra")
    draw.line((87, 477, 604, 477), fill="#40332c", width=3)
    draw.multiline_text(
        (88, 509), "Food and drinks\nTax\nGratuity", font=font("Sans", 28), fill=INK, spacing=15
    )
    draw.multiline_text(
        (604, 509),
        "26.50\n1.59\n4.00",
        font=font("Sans", 28),
        fill=INK,
        spacing=15,
        align="right",
        anchor="ra",
    )
    draw.rectangle((88, 663, 604, 728), fill="#f5eadd")
    draw.text((106, 695), "Amount paid", font=font("Sans-Bold", 32), fill=INK, anchor="lm")
    draw.text((585, 695), "$32.09", font=font("Sans-Bold", 32), fill=INK, anchor="rm")
    draw.text(
        (345, 802), "Thank you for visiting", font=font("Serif-Italic", 27), fill=INK, anchor="mt"
    )
    return image.transpose(Image.Transpose.ROTATE_90)


def warped(font: Fonts) -> Image.Image:
    width, height = 810, 1170
    background = "#d6d3ca"
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((58, 46, 752, 1124), fill="#fffef4")
    draw.rectangle((105, 98, 704, 158), fill="#e9e4d4")
    draw.text((405, 128), "SAMPLE HARDWARE", font=font("SansMono-Bold", 37), fill=INK, anchor="mm")
    draw.text((110, 196), "SALES RECEIPT  H-031", font=font("SansMono-Bold", 29), fill=INK)
    draw.text((110, 241), "16 Nov 2026     14:18", font=font("SansMono", 29), fill=INK)
    draw.line((108, 302, 700, 302), fill=INK, width=2)
    draw.multiline_text(
        (110, 334),
        "2  Steel hinges          7.80\n"
        "1  Wood glue             6.20\n"
        "3  Sandpaper sheets      4.50\n"
        "1  Paint brush           5.90\n"
        "2  Cabinet knobs        11.00\n"
        "1  Masking tape          3.60\n"
        "1  Safety glasses        8.00\n"
        "1  Drop cloth            6.00",
        font=font("SansMono", 29),
        fill=INK,
        spacing=17,
    )
    draw.line((108, 737, 700, 737), fill=INK, width=2)
    draw.multiline_text(
        (110, 761),
        "Subtotal               53.00\nTax                     3.18\nTotal                 $56.18",
        font=font("SansMono-Bold", 29),
        fill=INK,
        spacing=18,
    )
    draw.text(
        (405, 987),
        "Keep this receipt for returns",
        font=font("SansMono", 26),
        fill=INK,
        anchor="mt",
    )

    def source(x: int, y: int) -> tuple[float, float]:
        center = width / 2
        edge = (x - center) / center
        scale = 0.92 + 0.16 * y / height
        return center + (x - center) / scale + (
            y - height / 2
        ) * 0.055, y + 84 * edge**2 - 27 * edge

    mesh = []
    for x in range(0, width, 30):
        for y in range(0, height, 90):
            right, bottom = x + 30, y + 90
            corners = (source(x, y), source(x, bottom), source(right, bottom), source(right, y))
            quad = tuple(coordinate for point in corners for coordinate in point)
            mesh.append(((x, y, right, bottom), quad))
    return image.transform(
        image.size, Image.Transform.MESH, mesh, Image.Resampling.BICUBIC, fillcolor=background
    )


def table(font: Fonts) -> Image.Image:
    image = Image.new("RGB", (1000, 900), "#ffffff")
    draw = ImageDraw.Draw(image)
    draw.text((77, 65), "SAMPLE SUPPLY WORKSHOP", font=font("Sans-Bold", 36), fill="#153936")
    draw.text((77, 142), "Invoice for art supplies", font=font("Sans", 31), fill=INK)
    draw.text((923, 147), "16 November 2026", font=font("Sans", 25), fill=INK, anchor="ra")
    draw.text((77, 204), "Invoice SW-104", font=font("Sans", 25), fill=INK)
    rows = [
        ("Item description", "Qty", "Unit price", "Amount"),
        ("Sketchbook, A4", "3", "$8.00", "$24.00"),
        ("Graphite pencil set", "2", "$5.50", "$11.00"),
        ("Watercolor paper pad", "1", "$12.00", "$12.00"),
        ("Brush set", "2", "$7.25", "$14.50"),
        ("Metal ruler", "4", "$2.25", "$9.00"),
        ("Subtotal", "", "", "$70.50"),
        ("Sales tax", "", "", "$4.23"),
        ("Total due", "", "", "$74.73"),
    ]
    columns = (76, 459, 557, 738, 924)
    top, row_height = 274, 54
    for index, row in enumerate(rows):
        y = top + index * row_height
        if index == 0:
            draw.rectangle((columns[0], y, columns[-1], y + row_height), fill="#dcebe7")
        draw.line((columns[0], y, columns[-1], y), fill="#516161", width=2)
        row_font = font("Sans-Bold" if index in {0, len(rows) - 1} else "Sans", 24)
        for column, cell in enumerate(row):
            if column == 0:
                draw.text(
                    (columns[column] + 16, y + 27), cell, font=row_font, fill=INK, anchor="lm"
                )
            else:
                draw.text(
                    (columns[column + 1] - 16, y + 27), cell, font=row_font, fill=INK, anchor="rm"
                )
    bottom = top + len(rows) * row_height
    draw.line((columns[0], bottom, columns[-1], bottom), fill="#516161", width=2)
    for x in columns:
        draw.line((x, top, x, bottom), fill="#516161", width=2)
    draw.text((77, 800), "All amounts are in US dollars.", font=font("Sans", 22), fill=INK)
    return image


def formula(font: Fonts) -> Image.Image:
    image = Image.new("RGB", (1000, 900), "#fdfcf9")
    draw = ImageDraw.Draw(image)
    draw.text((90, 90), "ALGEBRA NOTES", font=font("Sans-Bold", 25), fill="#44525b")
    draw.text((90, 150), "Solving a quadratic equation", font=font("Serif", 44), fill=INK)
    draw.line((90, 222, 910, 222), fill="#66747a", width=2)
    draw.multiline_text(
        (90, 255),
        "A quadratic equation has the following form,\nwhere a is a nonzero coefficient.",
        font=font("Serif", 29),
        fill=INK,
        spacing=11,
    )
    draw.text((500, 410), "ax² + bx + c = 0", font=font("Serif-Italic", 66), fill=INK, anchor="mm")
    draw.text(
        (90, 521),
        "Its solutions are given by the quadratic formula.",
        font=font("Serif", 29),
        fill=INK,
    )
    math = font("Serif-Italic", 57)
    draw.text((217, 664), "x =", font=math, fill=INK, anchor="lm")
    draw.text((545, 627), "−b ± √(b² − 4ac)", font=math, fill=INK, anchor="mm")
    draw.line((326, 665, 765, 665), fill=INK, width=3)
    draw.text((545, 704), "2a", font=math, fill=INK, anchor="mm")
    draw.text(
        (90, 808),
        "The expression under the square root is the discriminant.",
        font=font("Serif", 25),
        fill=INK,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="directory for the five rendered PNGs")
    parser.add_argument("--font-dir", type=Path, default=Path("/usr/share/fonts/truetype/dejavu"))
    args = parser.parse_args()

    def font(style: str, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(args.font_dir / f"DejaVu{style}.ttf"), size)

    args.output.mkdir(parents=True, exist_ok=True)
    for render in (upright, rotated, warped, table, formula):
        render(font).save(args.output / f"{render.__name__}.png", optimize=True)


if __name__ == "__main__":
    main()
