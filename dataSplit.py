import os
from sklearn.model_selection import train_test_split
from shutil import copy, copyfile, rmtree
import cv2 as cv

RANDOM_SEED = 42

folders_main = os.listdir("original")
try:
    rmtree("./train")
    rmtree("./test")
finally:
    os.mkdir("./train")
    os.mkdir("./test")

# upscaling
sr = cv.dnn_superres.DnnSuperResImpl_create()
sr.readModel("FSRCNN-small_x3.pb")
sr.setModel("fsrcnn", 3)

for label in folders_main:
    try:
        os.mkdir(f"test/{label}")
    except Exception:
        pass
    try:
        os.mkdir(f"train/{label}")
    except Exception:
        pass

for label in folders_main:
    selected_folder = os.listdir(f"original/{label}")
    X_train, X_test, Y_train, Y_test = train_test_split(selected_folder, [label]*len(selected_folder), train_size=0.75, random_state=RANDOM_SEED)
    upscale = True
    print(f"Working on label {label}")
    for xtr, ytr in zip(X_train, Y_train):
        if not upscale:
            copyfile(f"original/{ytr}/{xtr}",f"train/{ytr}/{xtr}")
        else:
            # Upscale the image
            image = cv.imread(f"original/{ytr}/{xtr}")
            result = sr.upsample(image)
            cv.imwrite(f"train/{ytr}/{xtr}", result)
    for xts, yts in zip(X_test, Y_test):
        if not upscale:
            copyfile(f"original/{yts}/{xts}",f"test/{yts}/{xts}")
        else:
            # Upscale the image
            image = cv.imread(f"original/{yts}/{xts}")
            result = sr.upsample(image)
            cv.imwrite(f"test/{yts}/{xts}", result)