# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from pathlib import Path
import os
Path("./model.pickle").touch()

import G6_iris_recognition
# G6_iris_recognition.iris_model_train("subset","model.pickle")
# for label, image in zip(range(6), range(1,20)):
#     print(f"./subset/{label:04d}/{label:04d}_{image:03d}.bmp")
#     iris_name = G6_iris_recognition.iris_model_test("model.pickle",f"./subset/{label:04d}/{label:04d}_{image:03d}.bmp")
#     print(f"The correct label is {label:04d}, the model predicted {iris_name}")
labels=['0027', '0037', '0038', '0036', '0053', '0001', '0019', '0049', '0022']
for i in range(9):
    iris_name = G6_iris_recognition.iris_model_test("model.pickle",f"subset/0000/0000_000.bmp")
    print(f"The correct label is {labels[i]}, the model predicted {iris_name}")
