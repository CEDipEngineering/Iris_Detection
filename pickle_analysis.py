from multiprocessing import process
from pathlib import Path
import os
import pickle
from G6_iris_recognition.feature_vec import engroup
import cv2 as cv
from cv2 import dnn_superres
import numpy as np
import multiprocessing as mp
import time


with open("model_version_0\model.pickle", "rb") as fl:
    model_serial = pickle.loads(fl.read())
with open("model_version_0\model_no_mod.pickle", "rb") as fl:
    model_parallel = pickle.loads(fl.read())

print(model_serial["names"])
print(model_parallel["names"])


# print(modPLSolo["names"][10], modPL["encodings"][0][0][0][0])
# print((modPLSolo["encodings"][0][0][0][0]))
# print(([modSR["encodings"][0][0][0][i] == modPLSolo["encodings"][0][0][0][0] for i in range(len(modSR["encodings"][0][0][0]))]))