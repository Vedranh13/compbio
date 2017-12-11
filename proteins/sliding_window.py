import torch
from torch.autograd import Variable


def decomp_im_PIL(im, w, h):
    W, H = im.size
    grid = zip(range(W - w), range(H - h))
    for x, y in grid:
        box = (x, y, x + w, x + h)
        region = im.crop(box)
        yield region


def decomp_im(im, w, h):
    W, H = im.size()[1:]
    # grid = zip(range(W - w), range(H - h))
    for x in range(W - w):
        for y in range(H - h):
             region = torch.cuda.FloatTensor(4, w, h)
             box = (x, y, x + w, y + h)
             rows = torch.cuda.LongTensor(list(range(x, x + w)))
             cols = torch.cuda.LongTensor(list(range(y, y + h)))
             region = torch.index_select(im, 1, rows).index_select(2, cols)
             # import pdb; pdb.set_trace()
             # region = torch.index_select(region, 2, cols)
             # region = im[:, x:x+w, y+h]
             # region = im.crop(box)
             yield region


def test():
    from sim_data import test as tst
    import time
    c = tst()
    for im in decomp_im(c, 100, 100):
        time.sleep(1)
        im.show()
