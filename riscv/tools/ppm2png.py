"""Convert the emulator's binary PPM frame dumps to PNG, with an optional integer upscale."""

import struct
import sys
import zlib


def read_ppm(path):
    data = open(path, "rb").read()
    parts = data.split(maxsplit=4)
    assert parts[0] == b"P6" and parts[3] == b"255"
    width, height = int(parts[1]), int(parts[2])
    return width, height, parts[4][: width * height * 3]


def write_png(path, width, height, rgb, scale=1):
    rows = []
    for y in range(height):
        row = rgb[y * width * 3:(y + 1) * width * 3]
        if scale > 1:
            row = b"".join(row[x * 3:x * 3 + 3] * scale for x in range(width))
        rows += [b"\0" + row] * scale
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    header = struct.pack(">IIBBBBB", width * scale, height * scale, 8, 2, 0, 0, 0)
    with open(path, "wb") as out:
        out.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                  + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    scale = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    for path in sys.argv[3:] or [sys.argv[1]]:
        width, height, rgb = read_ppm(path)
        write_png(path.rsplit(".", 1)[0] + ".png", width, height, rgb, scale)
