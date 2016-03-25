import numpy as np
import itertools
from scipy import interpolate
import matplotlib.image as mpimg

def arr_to_int32(arr):
    return arr[2] + (arr[1] * 256) + (arr[0] * (256**2))

def read(path):
    img = mpimg.imread(path)
    rows, cols = len(img), len(img[0])
    hm = np.zeros((rows, cols))
    for i,j in itertools.product(range(rows), range(cols)):
            hm[i,j] = arr_to_int32(img[i,j])
    return hm

def fit(hm, t=None):
    X, Y = np.meshgrid(np.arange(0.0, hm.shape[1], 1.0), np.arange(0.0, hm.shape[0], 1.0))

    if t is not None:
        return interpolate.bisplrep(X, Y, hm, kx=3, ky=3, task=-1, tx=t[0], ty=t[1])
    else:
        return interpolate.bisplrep(X, Y, hm, kx=3, ky=3)

def approx(tck, X, Y):
    return interpolate.bisplev(X, Y, tck).transpose()

def linearMean(a, b, clip=[0, 100], margin=4, order=4):
    # linear approximation
    la = len(a)
    lb = len(b)

    pfa = np.polyfit(range(margin, la - margin), a[margin:la-margin], order)
    pfb = np.polyfit(range(margin, lb - margin), b[margin:lb-margin], order)
    pfc = np.array([pfa, pfb]).mean(axis=0)
    p = np.poly1d(pfc)

    x = np.arange((la + lb) / 2)
    y = p(x)
    y = np.clip(y, *clip)

    return y
