from G6_iris_recognition import iris_model_test
import os
from G6_iris_recognition.feature_vec import engroup
import multiprocessing as mp
import time
from support_functions import cleanup_wrapper
import tkinter.filedialog as tkfd
import pickle
from tkinter import Tk

def process_folder(folder, cleanup_options, path_to_tmp_file, pickle_path):  
    ## Processing
    print(f"Starting directory {folder}")
    predictions = []
    for file in os.listdir(f"test/{folder}"):
        processed=False
        ## Tentando imagem original, e se não der, tentar imagem processada
        path_to_image=f"test/{folder}/{file}"   
        prediction = iris_model_test(pickle_path, path_to_image)
        if prediction == "unmatch":
            # Tentar todas as opções
            curr_option_index = 0
            while curr_option_index<len(cleanup_options) and not processed:
                # Select and advance option
                option=cleanup_options[curr_option_index]
                curr_option_index += 1
                # Try option
                if cleanup_wrapper(path_to_image, path_to_tmp_file, option):
                    prediction = iris_model_test(pickle_path, path_to_tmp_file)
                    if prediction != "unmatch":
                        processed = True
                        # print(f"[SUCCESS] Sucessfully processed {file} with method {option.__name__}")
                         # Alguma íris dessa pasta deu certo então
        else:
            processed = True
             # Alguma íris dessa pasta deu certo então
            # print(f"[SUCCESS] Sucessfully processed {file} with standard image")                              
        # if not processed:
            # print(f"[FAIL] Unable to process {file}")
        # Saves last prediction made. If any prediction wasn't unmatch, saves that, otherwise, saves unmatch
        predictions.append(prediction)
    print(f"Directory {folder} finished...")
    try:
        os.remove(path_to_tmp_file)
    finally:
        return {f"{folder}":predictions}

def main(processes=os.cpu_count(), debug = False):
    
    # File dialog for selecting model to test
    Tk().withdraw()
    pickle_path = ""
    pickle_path = tkfd.askopenfilename(title='Indicate model.pickle location')
    while not pickle_path.endswith(".pickle"):
        pickle_path = tkfd.askopenfilename(title='INVALID: Indicate model.pickle location')
    with open(pickle_path, "rb") as fl:
        pickle_data = pickle.loads(fl.read())
    cleanup_options = pickle_data["functions"]
    summary=f"{'='*8}\n SUMMARY \n{'='*8}\n"


    try:
        pool = mp.Pool(processes)
        if debug:
            multiple_results = [pool.apply_async(process_folder, (folder, cleanup_options, f"tmp_img{i}.bmp", pickle_path)) for i, folder in enumerate(os.listdir("test")[:10])]
        else:
            multiple_results = [pool.apply_async(process_folder, (folder, cleanup_options, f"tmp_img{i}.bmp", pickle_path)) for i, folder in enumerate(os.listdir("test"))]
        output = [res.get() for res in multiple_results]        


        data = pickle.loads(open(pickle_path, "rb").read())        
        summary+=f"Known categories: { data['names'] }\n"
        summary+=f"Cleanup options: { [i.__name__ for i in cleanup_options] }\n"
        summary+=f"Training data set: { data['train_set'] }\n\n"
        hit, count = 0, 0
        for p in output:
            folder = list(p.keys())[0]
            values = list(p.values())[0]
            for v in values:
                if v == folder:
                    hit += 1
                count += 1
            summary+=f"{folder}: {values}\n"
        summary += f"Hits: {hit}/{count}, Misses: {count-hit}/{count}, Score: {(hit/count):.03f}\n"

        if "id" in data.keys():
            fn = f"results/test_result{data['id']}.txt"
        else:
            fn = f"results/test_result.txt"
        with open(fn,"w") as f:
            f.write(summary)
        

    finally:
        pool.close()
        pool.join()

if __name__ == "__main__":
    ##=============================================##
    ##=====DO NOT USE CTRL+C TO STOP PROCESS=======##
    ##=======THIS WILL CRASH YOUR TERMINAL=========##
    ##=======IT IS A KNOWN BUG IN mp.Pool==========##
    ##========bugs.python.org/issue38428===========##
    ##=============================================##

    print(f"Begining processing")
    start = time.perf_counter()
    main(processes=7)
    end = time.perf_counter()-start
    print(f"Finished in {end//60} minutes and {end%60} seconds")

