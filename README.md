## Iris Recognition Project

The images used are available in https://github.com/lucianosilva-github/images

The machine learning model used is available in https://github.com/lucianosilva-github/visaocomputacional/tree/master/G6_iris_recognition and is an adaptation of the G6 model.

___

### Project specs

For this project we had to use the provided images and G6 version. The goal was to achieve the highest score possible upon testing.

### Files

The files [test_G6.py](./test_G6.py) and [train_G6.py](./train_G6.py) each have a class that use a multiprocessing Pool to create multiple worker processes, which speed up their progress dramatically. Do note, due to bugs.python.org/issue38428, using Ctrl+C to stop  the program will cause the error handling to crash, leaving stranded processes running. This can be avoided by restarting the kernel (e.g. on jupyter lab), or using task manager.

The [G6_master.py](./G6_master.py) file makes use of both train and test modules to automatically train and then test a given model.

### Notes

The FSRCNN-small_x3.pb file is a pre-trained neural network model, made available by OpenCV. It takes in any image, and triples its resolution, as shown in [upscaling.ipynb](./upscaling.ipynb). It was considered as an option, however it ended up quadrupling train/test times and dind't affect score much at all.

We also considered doing inteligent ROI selection using a Hough Transform for circle detection, alongside some other processing to make the image clearer. We ended up not pursuing this path, due to a tip saying it would probably not help very much at all, seeing as G6 does that already.

### Interactivity

Both training and test are separated in the following structure; The test folder contains a series of folders, with unique names, such as '0000', '0001', etc. Inside each folder, all of the images that belong to that sample, such as '0000_000.bmp', '0000_001.bmp', etc.

When starting training, a Tkinter window will pop up, asking you to select a folder. Once selected, that will be considered the root folder for the training data set. For example, if your structure is 'train/0000/0000_000.bmp', 'train/0000/0000_001.bmp', 'train/0001/0001_000.bmp', 'train/0001/0001_001.bmp', you should select the 'train' folder.

The trained model is stored in a '.pickle' file. When running the test script, you will be prompted with another Tkinter window to select said file. If you are using [G6_master.py](./G6_master.py), this prompt will be skipped, as it knows to test the model that was just trained.
