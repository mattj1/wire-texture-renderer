import math

# import numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops

image_dim = 1024
brush_dim = 16
brush_radius = 2


class NumPyImage:

    def __init__(self, img) -> None:
        super().__init__()
        self.orig = img

        arr = np.array(img)

        self.rgb = arr[..., :3].astype(np.float32) / 255.0
        self.a = arr[..., 3].astype(np.float32) / 255.0

    def to_image(self):
        self.a = np.clip(self.a, 0, 1)
        self.rgb = np.clip(self.rgb, 0, 1)
        # TODO: don't store "orig"
        out = np.zeros_like(self.orig)
        out[..., :3] = self.rgb * 255
        out[..., 3] = self.a * 255

        return Image.fromarray(np.uint8(out))


def blit_add2(src: NumPyImage, dst: NumPyImage, pos):
    x = int(pos[0])
    y = int(pos[1])

    src_size = src.rgb.shape[1::-1]
    dst_size = dst.rgb.shape[1::-1]

    dst_left = x
    dst_right = x + src_size[0]

    dst_top = y
    dst_bottom = y + src_size[1]

    src_left = 0
    src_right = src_size[0]
    src_top = 0
    src_bottom = src_size[1]

    if dst_left < 0:
        src_left += dst_left
        dst_left = 0

    if dst_top < 0:
        src_top += dst_top
        dst_top = 0

    if dst_right >= dst_size[0]:
        src_right -= dst_right - dst_size[0]
        dst_right = dst_size[0]

    if dst_bottom >= dst_size[1]:
        src_bottom -= dst_bottom - dst_size[1]
        dst_bottom = dst_size[1]

    if src_right <= src_left:
        return

    if src_bottom <= src_top:
        return



    # dst_a[x:x + 16, 0:16] = src_a + dst_a[x:x + 16, 0:16] * (1.0 - src_a)
    # dst.a[x:x + 16, 0:16] = src.a + dst.a[x:x + 16, 0:16]
    dst.a[dst_top:dst_bottom, dst_left:dst_right] += src.a[src_top:src_bottom, src_left:src_right]  # + dst.a[x:x + 16, 0:16]

    # dst_rgb[x:x+16, 0:16] = (src_rgb * src_a[..., None]
    #                          + dst_rgb[x:x+16, 0:16] * dst_a[x:x+16, 0:16, None] * (1.0 - src_a[..., None])) \
    #                         / dst_a[x:x+16, 0:16, None]

    dst.rgb[dst_top:dst_bottom, dst_left:dst_right] += src.rgb[src_top:src_bottom, src_left:src_right]  # (src.rgb + dst.rgb[x:x + 16, 0:16])


def gen_poly_mask(pts):
    im = Image.new("L", (image_dim, image_dim))
    draw = ImageDraw.Draw(im)
    draw.polygon(pts, fill=(255))
    return im


def blit_add(im0: Image, im1, pos):
    # blit im1 onto im0 at pos
    pixels0 = im0.load()
    # pixels1 = im1.load()

    # a = im0.getdata()
    # print(a[0])

    for y in range(0, im1.size[1]):
        for x in range(0, im1.size[0]):
            x0 = int(x + pos[0])
            if x0 < 0 or x0 >= im0.size[0]:
                continue

            y0 = int(y + pos[1])
            if y0 < 0 or y0 >= im0.size[1]:
                continue

            src = im1.getpixel((x, y))
            dst = im0.getpixel((x0, y0))

            im0.putpixel((x0, y0), (dst[0] + src[0], dst[1] + src[1], dst[2] + src[2], dst[3] + src[3]))
            # pixels0[x, y] = (255, 255, 255, 255)

            # a[y * im0.size[0] + x] = (255, 255,255,255)


def brush_line(im, im_brush, p0, p1):
    # draw = ImageDraw.Draw(im)
    print("line...")

    sx = p0[0] - im_brush.size[0] / 2
    sy = p0[1] - im_brush.size[1] / 2

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]

    dist = math.sqrt(dx * dx + dy * dy)
    delta = 1.0 / dist

    t = 0
    while t <= 1:
        x = sx + t * dx
        y = sy + t * dy

        # new_im = Image.new("RGBA", (image_dim, image_dim))
        # new_im_draw = ImageDraw.Draw(im)

        # draw.bitmap((x, y), bitmap=im_brush)
        # new_im_draw.bitmap((x, y), bitmap=im_brush)

        t += delta

        # im = ImageChops.add(im, new_im)

        blit_add(im, im_brush, (x, y))

    return im


def brush_line2(im: NumPyImage, im_brush: NumPyImage, p0, p1):
    # draw = ImageDraw.Draw(im)
    print("line...")

    brush_size = im_brush.rgb.shape[1::-1]

    sx = p0[0] - brush_size[0] / 2
    sy = p0[1] - brush_size[1] / 2

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]

    dist = math.sqrt(dx * dx + dy * dy)
    delta = 1.0 / dist

    t = 0
    while t <= 1:
        x = sx + t * dx
        y = sy + t * dy
        t += delta

        blit_add2(im_brush, im, (x, y))

    return im



print("RUN")

im_brush = Image.new("RGBA", (brush_dim, brush_dim), 0)
im_brush_draw = ImageDraw.Draw(im_brush)

im_brush_draw.ellipse((brush_dim / 2 - brush_radius, brush_dim / 2 - brush_radius, brush_dim / 2 + brush_radius,
                       brush_dim / 2 + brush_radius),
                      fill=(255, 255, 255, 127))
im_brush = im_brush.filter(ImageFilter.GaussianBlur(brush_radius / 2))

im = Image.new('RGBA', (image_dim, image_dim))

dest = np.array(im)
src2 = np.array(im_brush)

print(dest.shape[1::-1])
print(src2.shape[1::-1])


np_image_dst = NumPyImage(dest)
np_image_brush = NumPyImage(src2)

for x in range(0, 32):
    blit_add2(np_image_brush, np_image_dst, (x, 0))

# im2 = np_image_dst.to_image()
# im2.show()
# draw = ImageDraw.Draw(im)

# im_mask = gen_poly_mask((20, 20) + (140, 20) + (140, 130) + (20, 130))
# np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
#
# brush_line2(np_im_lines, np_image_brush, p0=(20, 20), p1=(140, 20))
# brush_line2(np_im_lines, np_image_brush, p0=(20, 20), p1=(140, 20))
# brush_line2(np_im_lines, np_image_brush, p0=(140, 20), p1=(140, 130))
# brush_line2(np_im_lines, np_image_brush, p0=(140, 130), p1=(20, 130))
# brush_line2(np_im_lines, np_image_brush, p0=(20, 130), p1=(20, 20))
# im = Image.composite(np_im_lines.to_image(), im, im_mask)

im_mask = gen_poly_mask([(160.42666625976562, 887.9542846679688), (10.239999771118164, 887.9542846679688), (10.239999771118164, 1013.760009765625), (160.42666625976562, 1013.760009765625)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(160.42666625976562, 887.9542846679688), p1=(10.239999771118164, 887.9542846679688))
brush_line2(np_im_lines, np_image_brush, p0=(10.239999771118164, 887.9542846679688), p1=(10.239999771118164, 1013.760009765625))
brush_line2(np_im_lines, np_image_brush, p0=(10.239999771118164, 1013.760009765625), p1=(160.42666625976562, 1013.760009765625))
brush_line2(np_im_lines, np_image_brush, p0=(160.42666625976562, 1013.760009765625), p1=(160.42666625976562, 887.9542846679688))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(1013.760009765625, 10.239999771118164), (692.9066772460938, 10.239999771118164), (692.9066772460938, 282.3314208984375), (1013.760009765625, 282.3314208984375)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(1013.760009765625, 10.239999771118164), p1=(692.9066772460938, 10.239999771118164))
brush_line2(np_im_lines, np_image_brush, p0=(692.9066772460938, 10.239999771118164), p1=(692.9066772460938, 282.3314208984375))
brush_line2(np_im_lines, np_image_brush, p0=(692.9066772460938, 282.3314208984375), p1=(1013.760009765625, 282.3314208984375))
brush_line2(np_im_lines, np_image_brush, p0=(1013.760009765625, 282.3314208984375), p1=(1013.760009765625, 10.239999771118164))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(331.09332275390625, 10.239999771118164), (10.239999771118164, 10.239999771118164), (10.239999771118164, 282.3314208984375), (331.09332275390625, 282.3314208984375)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(331.09332275390625, 10.239999771118164), p1=(10.239999771118164, 10.239999771118164))
brush_line2(np_im_lines, np_image_brush, p0=(10.239999771118164, 10.239999771118164), p1=(10.239999771118164, 282.3314208984375))
brush_line2(np_im_lines, np_image_brush, p0=(10.239999771118164, 282.3314208984375), p1=(331.09332275390625, 282.3314208984375))
brush_line2(np_im_lines, np_image_brush, p0=(331.09332275390625, 282.3314208984375), p1=(331.09332275390625, 10.239999771118164))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(672.4266967773438, 10.239999771118164), (351.5733337402344, 10.239999771118164), (351.5733337402344, 282.3314208984375), (672.4266967773438, 282.3314208984375)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(672.4266967773438, 10.239999771118164), p1=(351.5733337402344, 10.239999771118164))
brush_line2(np_im_lines, np_image_brush, p0=(351.5733337402344, 10.239999771118164), p1=(351.5733337402344, 282.3314208984375))
brush_line2(np_im_lines, np_image_brush, p0=(351.5733337402344, 282.3314208984375), p1=(672.4266967773438, 282.3314208984375))
brush_line2(np_im_lines, np_image_brush, p0=(672.4266967773438, 282.3314208984375), p1=(672.4266967773438, 10.239999771118164))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(331.09332275390625, 595.3828735351562), (10.239999771118164, 595.3828735351562), (10.239999771118164, 867.4743041992188), (331.09332275390625, 867.4743041992188)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(331.09332275390625, 595.3828735351562), p1=(10.239999771118164, 595.3828735351562))
brush_line2(np_im_lines, np_image_brush, p0=(10.239999771118164, 595.3828735351562), p1=(10.239999771118164, 867.4743041992188))
brush_line2(np_im_lines, np_image_brush, p0=(10.239999771118164, 867.4743041992188), p1=(331.09332275390625, 867.4743041992188))
brush_line2(np_im_lines, np_image_brush, p0=(331.09332275390625, 867.4743041992188), p1=(331.09332275390625, 595.3828735351562))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(501.760009765625, 867.4743041992188), (501.760009765625, 595.3828735351562), (351.5733337402344, 595.3828735351562), (351.5733337402344, 867.4743041992188)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(501.760009765625, 867.4743041992188), p1=(501.760009765625, 595.3828735351562))
brush_line2(np_im_lines, np_image_brush, p0=(351.5733337402344, 595.3828735351562), p1=(351.5733337402344, 867.4743041992188))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(522.239990234375, 721.1885986328125), (843.0933227539062, 721.1885986328125), (843.0933227539062, 595.3828735351562), (522.239990234375, 595.3828735351562)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(522.239990234375, 721.1885986328125), p1=(843.0933227539062, 721.1885986328125))
brush_line2(np_im_lines, np_image_brush, p0=(843.0933227539062, 595.3828735351562), p1=(522.239990234375, 595.3828735351562))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(843.0933227539062, 574.90283203125), (843.0933227539062, 302.8114318847656), (692.9066772460938, 302.8114318847656), (692.9066772460938, 574.90283203125)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(843.0933227539062, 574.90283203125), p1=(843.0933227539062, 302.8114318847656))
brush_line2(np_im_lines, np_image_brush, p0=(692.9066772460938, 302.8114318847656), p1=(692.9066772460938, 574.90283203125))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(522.239990234375, 867.4743041992188), (843.0933227539062, 867.4743041992188), (843.0933227539062, 741.6685791015625), (522.239990234375, 741.6685791015625)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(522.239990234375, 867.4743041992188), p1=(843.0933227539062, 867.4743041992188))
brush_line2(np_im_lines, np_image_brush, p0=(843.0933227539062, 741.6685791015625), p1=(522.239990234375, 741.6685791015625))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(1013.760009765625, 741.6685791015625), (863.5733032226562, 741.6685791015625), (863.5733032226562, 867.4743041992188), (1013.760009765625, 867.4743041992188)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(1013.760009765625, 741.6685791015625), p1=(863.5733032226562, 741.6685791015625))
brush_line2(np_im_lines, np_image_brush, p0=(863.5733032226562, 741.6685791015625), p1=(863.5733032226562, 867.4743041992188))
brush_line2(np_im_lines, np_image_brush, p0=(863.5733032226562, 867.4743041992188), p1=(1013.760009765625, 867.4743041992188))
brush_line2(np_im_lines, np_image_brush, p0=(1013.760009765625, 867.4743041992188), p1=(1013.760009765625, 741.6685791015625))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(1013.760009765625, 302.8114318847656), (863.5733032226562, 302.8114318847656), (863.5733032226562, 428.6171569824219), (1013.760009765625, 428.6171569824219)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(1013.760009765625, 302.8114318847656), p1=(863.5733032226562, 302.8114318847656))
brush_line2(np_im_lines, np_image_brush, p0=(863.5733032226562, 302.8114318847656), p1=(863.5733032226562, 428.6171569824219))
brush_line2(np_im_lines, np_image_brush, p0=(863.5733032226562, 428.6171569824219), p1=(1013.760009765625, 428.6171569824219))
brush_line2(np_im_lines, np_image_brush, p0=(1013.760009765625, 428.6171569824219), p1=(1013.760009765625, 302.8114318847656))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(1013.760009765625, 449.0971374511719), (863.5733032226562, 449.0971374511719), (863.5733032226562, 574.90283203125), (1013.760009765625, 574.90283203125)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(1013.760009765625, 449.0971374511719), p1=(863.5733032226562, 449.0971374511719))
brush_line2(np_im_lines, np_image_brush, p0=(863.5733032226562, 449.0971374511719), p1=(863.5733032226562, 574.90283203125))
brush_line2(np_im_lines, np_image_brush, p0=(863.5733032226562, 574.90283203125), p1=(1013.760009765625, 574.90283203125))
brush_line2(np_im_lines, np_image_brush, p0=(1013.760009765625, 574.90283203125), p1=(1013.760009765625, 449.0971374511719))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(1013.760009765625, 595.3828735351562), (863.5733032226562, 595.3828735351562), (863.5733032226562, 721.1885986328125), (1013.760009765625, 721.1885986328125)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(1013.760009765625, 595.3828735351562), p1=(863.5733032226562, 595.3828735351562))
brush_line2(np_im_lines, np_image_brush, p0=(863.5733032226562, 595.3828735351562), p1=(863.5733032226562, 721.1885986328125))
brush_line2(np_im_lines, np_image_brush, p0=(863.5733032226562, 721.1885986328125), p1=(1013.760009765625, 721.1885986328125))
brush_line2(np_im_lines, np_image_brush, p0=(1013.760009765625, 721.1885986328125), p1=(1013.760009765625, 595.3828735351562))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(672.4266967773438, 574.90283203125), (361.8133239746094, 574.90283203125), (672.4266967773438, 313.0514221191406)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(672.4266967773438, 574.90283203125), p1=(361.8133239746094, 574.90283203125))
brush_line2(np_im_lines, np_image_brush, p0=(361.8133239746094, 574.90283203125), p1=(672.4266967773438, 313.0514221191406))
brush_line2(np_im_lines, np_image_brush, p0=(672.4266967773438, 313.0514221191406), p1=(672.4266967773438, 574.90283203125))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(331.09332275390625, 574.90283203125), (20.479999542236328, 574.90283203125), (331.09332275390625, 313.0514221191406)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(331.09332275390625, 574.90283203125), p1=(20.479999542236328, 574.90283203125))
brush_line2(np_im_lines, np_image_brush, p0=(20.479999542236328, 574.90283203125), p1=(331.09332275390625, 313.0514221191406))
brush_line2(np_im_lines, np_image_brush, p0=(331.09332275390625, 313.0514221191406), p1=(331.09332275390625, 574.90283203125))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(10.239999771118164, 302.8114318847656), (320.85333251953125, 302.8114318847656), (10.239999771118164, 564.662841796875)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(10.239999771118164, 302.8114318847656), p1=(320.85333251953125, 302.8114318847656))
brush_line2(np_im_lines, np_image_brush, p0=(320.85333251953125, 302.8114318847656), p1=(10.239999771118164, 564.662841796875))
brush_line2(np_im_lines, np_image_brush, p0=(10.239999771118164, 564.662841796875), p1=(10.239999771118164, 302.8114318847656))
im = Image.composite(np_im_lines.to_image(), im, im_mask)
im_mask = gen_poly_mask([(351.5733337402344, 302.8114318847656), (662.1866455078125, 302.8114318847656), (351.5733337402344, 564.662841796875)])
np_im_lines = NumPyImage(Image.new('RGBA', (image_dim, image_dim)))
brush_line2(np_im_lines, np_image_brush, p0=(351.5733337402344, 302.8114318847656), p1=(662.1866455078125, 302.8114318847656))
brush_line2(np_im_lines, np_image_brush, p0=(662.1866455078125, 302.8114318847656), p1=(351.5733337402344, 564.662841796875))
brush_line2(np_im_lines, np_image_brush, p0=(351.5733337402344, 564.662841796875), p1=(351.5733337402344, 302.8114318847656))
im = Image.composite(np_im_lines.to_image(), im, im_mask)

im = ImageOps.flip(im)
im.show()
