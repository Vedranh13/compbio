"""Various util functions"""
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from project_FST import fourier_lin, project_fst
import torch
import torch.utils.data
from os import listdir
from PIL import Image

def generate_protein_gauss(n):
    """This simulates a protein's 3D energy potential as just a gaussian blob"""
    return 5 * np.rand.randn(n, n, n)


def load_protein(filename):
    zika_file = mrcfile.open(filename)
    rho = zika_file.data
    return rho

# Mystery is a protein I used to know, og name was EMD-2830.map

def gen_dataset(proteins=["zika", "mystery"], n_each=1000, write=True, dir='dataset'):
    ims = {}
    for protein in proteins:
        rho = load_protein(protein + ".mrc")
        rhoh = fourier_lin(rho)
        for i in range(n_each):
            ims[protein + "_" + str(i)] = gen_randim(rhoh, rho.shape[0])
    if write:
        save_dict(ims, dir=dir)
    return ims

def gen_randim(rhoh, n):
    a_comp = [np.random.randint(0, 3)]
    b_comp = [i for i in range(3) if i not in a_comp]
    a = np.random.rand(3)
    b = np.random.rand(3)
    a[a_comp] = 0
    b[b_comp] = 0
    return project_fst(a, b, n, rhoh=rhoh)


def save_dict(dct, dir='dataset'):
    for k,v in dct.items():
        plt.imsave(dir + '/' + k + ".png", v, format="png", cmap=plt.cm.gray)


def read_dict(dir='dataset'):
    dct = {}
    all_files = listdir(dir)
    for prot in all_files:
        if ".png" in prot:
            dct[prot] = torch.from_numpy(plt.imread(dir + '/' + prot))
    return dct


class ImageLoader(torch.utils.data.Dataset):
    """Credit: https://stackoverflow.com/questions/45099554/how-to-simplify-dataloader-for-autoencoder-in-pytorch"""
    def __init__(self, prots={"zika": 0, "mystery": 1}, dir='dataset', test=False, tform=None, imgloader=Image.open):
        super(ImageLoader, self).__init__()

        self.dir = dir
        if test:
            self.dir += "/test"
        self.prots = prots
        # self.prot = prot
        self.filenames = listdir(dir)
        for fn in self.filenames:
            if '.png' not in fn:
                self.filenames.remove(fn)
        self.tform = tform
        self.imgloader = imgloader

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        out = self.imgloader(self.dir + "/" + self.filenames[i])
        # HOW TO RESIZE AND COMBINE SHITE http://scipy-cookbook.readthedocs.io/items/Matplotlib_AdjustingImageSize.html
        # out.resize((124,100,4))
        # RESIZED using: http://matplotlib.org/users/image_tutorial.html
        out.thumbnail((124, 124, 4), Image.ANTIALIAS)
        prot = self.filenames[i].split("_")[0]
        cl =  self.prots[prot]
        if self.tform:
            out = self.tform(out)
        return out, cl


def enlarge_nd(arr, new_shape):
    new = arr.reshape(new_shape)


def assert_no_incest(train_dir='dataset', test_dir='dataset/test'):
    train = read_dict(dir=train_dir)
    test = read_dict(dir=test_dir)
    train = set(train.values())
    test = set(test.values())
    assert len(train.intersection(test)) == 0


def visualize_kernels(net):
    from torchvision import transforms as tf
    from torch.autograd import Variable
    trainloader = torch.utils.data.DataLoader(ImageLoader(tform=tf.ToTensor()), batch_size=4,
                                              shuffle=True, num_workers=2)
    im, _ = next(iter(trainloader))
    ker = net.conv1
    # ker_one = net.conv1.weight.data
    out = ker(Variable(im))
    out = out.data.numpy()
    im = im.numpy()
    # import pdb; pdb.set_trace()
    plt.imsave("orig.png", im[0], format='png', cmap=plt.cm.gray)
    plt.imsave("ker.png", out[0][0], format='png', cmap=plt.cm.gray)


 
def gen_randim_and_axis(rho, n):
    a_comp = [np.random.randint(0, 3)]
    b_comp = [i for i in range(3) if i not in a_comp]
    a = np.random.rand(3)
    b = np.random.rand(3)
    a[a_comp] = 0
    b[b_comp] = 0
    return a, b, project_fst(a, b, n, rho=rho)
