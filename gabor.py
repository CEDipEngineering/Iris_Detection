'''
gabor_threads.py
=========
Sample demonstrates:
- use of multiple Gabor filter convolutions to get Fractalius-like image effect (http://www.redfieldplugins.com/filterFractalius.htm)
- use of python threading to accelerate the computation
Usage
-----
gabor_threads.py [image filename]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from multiprocessing.pool import ThreadPool
from support_functions import *

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process_threaded(img, filters, threadn = 8):
    accum = np.zeros_like(img)
    def f(kern):
        return cv.filter2D(img, cv.CV_8UC3, kern)
    pool = ThreadPool(processes=threadn)
    for fimg in pool.imap_unordered(f, filters):
        np.maximum(accum, fimg, accum)
    return accum

if __name__ == '__main__':
    import os

    img_fn = r"C:\Users\cadud\Repos\Iris_Detection\original\0001\0001_001.bmp"

    filters = build_filters()

    morph_close(img_fn, "tmp_0.bmp")
    img = cv.imread("tmp_0.bmp")

    res2 = process_threaded(img, filters)

    cv.imshow('img', img)
    cv.imshow('result', res2)
    cv.waitKey()
    cv.destroyAllWindows()

    os.remove("tmp_0.bmp")