import os
from sklearn.model_selection import train_test_split
from shutil import copyfile

RANDOM_SEED = 42

folders_main = os.listdir("original")
os.mkdir("train")
os.mkdir("test")

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
    for xtr, ytr in zip(X_train, Y_train):
        copyfile(f"original/{ytr}/{xtr}",f"train/{ytr}/{xtr}")
    for xts, yts in zip(X_test, Y_test):
        copyfile(f"original/{yts}/{xts}",f"test/{yts}/{xts}")
