from G6_iris_recognition import iris_model_test, iris_model_train
import os
import pickle
from train_G6 import blurMorph_cleanup, morphClose_cleanup, CLAHE_cleanup, medianSlide_cleanup, cleanup_wrapper



miss = 0
hit = 0 
path_to_tmp_file = "tmp_img.bmp"
# cleanup_options = [medianSlide_cleanup, CLAHE_cleanup, blurMorph_cleanup, morphClose_cleanup]
cleanup_options = []
curr_option_index = "Original"
for folder in os.listdir(f"test/"):
    current_encodings = []
    for file in os.listdir(f"test/{folder}"):
        path_to_image=f"test/{folder}/{file}"
        result = iris_model_test("./model.pickle", path_to_image)
        # if result == "unmatch":
        #     # Tentar todas as opções
        #     curr_option_index = 0
        #     while curr_option_index<len(cleanup_options):
        #         option=cleanup_options[curr_option_index]
        #         curr_option_index += 1
        #         if cleanup_wrapper(path_to_image, path_to_tmp_file, option):
        #                     result = iris_model_test("./model.pickle", path_to_tmp_file)
        #                     if result != "unmatch":
        #                         break
                         
        print(f"img:{file}, result:{result}, folder {folder}, method : {curr_option_index}")
        curr_option_index = "Original"
        if result == folder:
            hit +=1
        else:
            miss +=1
print(f"HIT:{hit}")
print(f"MISS:{miss}")
print(f"TOTAL:{hit/(hit+miss)}")



