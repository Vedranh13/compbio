"""Credit: https://gist.github.com/glombard/7cd166e311992a828675"""
from PIL import Image
from random import randint
from noise import add_uniform
import matplotlib.pyplot as plt

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


def combine_two_prots_diag(im1, im2, a=0, b=0):
    w1, h1 = im1.size
    w2, h2 = im2.size
    w, h = max(w1, w2), max(h1, h2)
    result = Image.new("L", (w * 2, h * 2))
    im1.thumbnail((w, h), Image.ANTIALIAS)
    im2.thumbnail((w, h), Image.ANTIALIAS)
    result.paste(im1, (0, 0))
    result.paste(im2, (w - a, h - b))
    return result


def combine_two_prots_top(im1, im2, a=0, b=0):
    w1, h1 = im1.size
    w2, h2 = im2.size
    w, h = max(w1, w2), max(h1, h2)
    result = Image.new("L", (w * 2, h * 2))
    im1.thumbnail((w, h), Image.ANTIALIAS)
    im2.thumbnail((w, h), Image.ANTIALIAS)
    result.paste(im1, (0, 0))
    result.paste(im2, (w - a, 0 + b))
    return result


def combine_two_prots_left(im1, im2, a=0, b=0):
    w1, h1 = im1.size
    w2, h2 = im2.size
    w, h = max(w1, w2), max(h1, h2)
    result = Image.new("L", (w * 2, h * 2))
    plt.imsave("im.png", im1)
    im1.thumbnail((w, h), Image.ANTIALIAS)
    im2.thumbnail((w, h), Image.ANTIALIAS)
    result.paste(im1, (0, 0))
    result.paste(im2, (0 + a, h - b))
    return result


def combine_two_prots_any(im1, im2, a=0, b=0):
    """Combines them randomly diag, top, or left"""
    r = randint(0, 2)
    if r == 0:
        return combine_two_prots_diag(im1, im2, a=a, b=b)
    if r == 1:
        return combine_two_prots_top(im1, im2, a=a, b=b)
    if r == 2:
        return combine_two_prots_left(im1, im2, a=a, b=b)


def combine_two_overlap_dw_dh(im1, im2, dw=0, dh=0):
    return combine_two_prots_any(im1, im2, a=int(dw), b=int(dh))


def combine_two_noise(im1, im2, p_overlap, nl=1):
    w1, h1 = im1.size
    w2, h2 = im2.size
    w, h = max(w1, w2), max(h1, h2)
    dw, dh = p_overlap * w, p_overlap * h
    return add_uniform(l=-nl/2, r=nl/2)(combine_two_overlap_dw_dh(im1, im2, dw=dw, dh=dh))

def combine_two(im1, im2, p_overlap):
    w1, h1 = im1.size
    w2, h2 = im2.size
    w, h = max(w1, w2), max(h1, h2)
    dw, dh = p_overlap * w, p_overlap * h
    return combine_two_overlap_dw_dh(im1, im2, dw=dw, dh=dh)


def test():
    import matplotlib.pyplot as plt
    a = Image.open("dataset/zika_0.png")
    b = Image.open("dataset/zika_1.png")
    c = combine_two_noise(a, b, p, nl=nl)
    w1, h1 = a.size
    w2, h2 = b.size
    w, h = max(w1, w2), max(h1, h2)
    dw, dh = p_overlap * w, p_overlap * h
    return c, combine_two_overlap_dw_dh(a, b, dw=dw, dh=dh)
