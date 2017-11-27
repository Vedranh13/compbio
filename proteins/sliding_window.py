def decomp_im(im, w, h):
    W, H = im.size
    grid = zip(range(W - w), range(H - h))
    for x, y in grid:
        box = (x, y, x + w, x + h)
        region = im.crop(box)
        yield region


def test():
    from sim_data import test as tst
    import time
    c = tst()
    for im in decomp_im(c, 100, 100):
        time.sleep(1)
        im.show()
