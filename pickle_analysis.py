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


with open("model_parallel.pickle", "rb") as fl:
    modPL = pickle.loads(fl.read())
with open("model_parallel_small.pickle", "rb") as fl:
    modPLSmall = pickle.loads(fl.read())
with open("model_parallel_small_2.pickle", "rb") as fl:
    modPLSmall2 = pickle.loads(fl.read())
with open("model.pickle", "rb") as fl:
    modPLSolo = pickle.loads(fl.read())
with open("model_serial.pickle", "rb") as fl:
    modSR = pickle.loads(fl.read())

print(modPLSolo["names"], len(modPLSolo["encodings"]))
# print(modPLSolo["names"][10], modPL["encodings"][0][0][0][0])
# print((modPLSolo["encodings"][0][0][0][0]))
# print(([modSR["encodings"][0][0][0][i] == modPLSolo["encodings"][0][0][0][0] for i in range(len(modSR["encodings"][0][0][0]))]))