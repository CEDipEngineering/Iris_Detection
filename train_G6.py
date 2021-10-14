from multiprocessing import process
from pathlib import Path
import os
import pickle
from G6_iris_recognition.feature_vec import engroup
import tkinter.filedialog as tkfd
from tkinter import Tk
import multiprocessing as mp
import time
from support_functions import *
from itertools import combinations

## O treinamento foi feito usando o método engroup do G6, e não o iris_model_train, como sugerido.

##=============================================##
##=====DO NOT USE CTRL+C TO STOP PROCESS=======##
##=======THIS WILL CRASH YOUR TERMINAL=========##
##=======IT IS A KNOWN BUG IN mp.Pool==========##
##========bugs.python.org/issue38428===========##
##=============================================##

class Iris_Trainer():

    def __init__(self, processes, sct, id=None):
        Tk().withdraw()
        self.processes = processes
        self.origin = tkfd.askdirectory(title='Indicate folder with training data')
        self.SUCCESS_COUNT_THRESHOLD = sct
        self.id = id

    # Saves trained model to .pickle, as well as training logs
    def save_model(self, encodings, names, functions, train_set, train_log_csv, log):
        # encodings = lista de listas de encondings gerados com engroup
        # names = nome referente a cada lista da matriz encodings
        if self.id is None:
            id = len(os.listdir('results'))//3
        else:
            id = self.id
        data = {"encodings": encodings, "names": names, "functions": functions, "train_set": train_set, "id": id}
        fn_pickle = f"results/model{id}.pickle"
        fn_log = f"results/log{id}.txt"
        fn_csv = f"results/log{id}.csv"
        
        with open(fn_pickle, "wb") as f:
            f.write(pickle.dumps(data))
        with open(fn_log, "w") as log_csv:
            log_csv.write(str(train_log_csv))
        with open(fn_csv, "w") as log_txt:
            log_txt.write(str(log))
        self.pickle_file = fn_pickle
        return fn_pickle

    # Function used for multiprocessing, processes given folder.
    def process_folder(self, folder, cleanup_options, path_to_tmp_file):
        ## Output variables
        names = [""]
        all_encodings = []
        log = ""
        train_log_csv = []
        ## Processing
        print(f"Starting directory {folder}")
        log += f"------------------\nStarting directory {folder}\n------------------\n"
        current_encodings = []
        any_successes_in_folder = False
        success_count = 0
        for file in os.listdir(f"{self.origin}/{folder}"):
            processed=False
            ## Tentando imagem original, e se não der, tentar imagem processada
            path_to_image=f"{self.origin}/{folder}/{file}"   
            iris_encodings_in_image = engroup(path_to_image)
            if iris_encodings_in_image == "invalid image":
                # Tentar todas as opções
                curr_option_index = 0
                while curr_option_index<len(cleanup_options) and not processed:
                    # Select and advance option
                    option=cleanup_options[curr_option_index]
                    curr_option_index += 1
                    # Try option
                    if cleanup_wrapper(path_to_image, path_to_tmp_file, option):
                        iris_encodings_in_image = engroup(path_to_tmp_file)
                        if iris_encodings_in_image != "invalid image":
                            processed = True
                            # print(f"[SUCCESS] Sucessfully processed {file} with method {option.__name__}")
                            log += f"[SUCCESS] Sucessfully processed {file} with method {option.__name__}\n"
                            any_successes_in_folder = True # Alguma íris dessa pasta deu certo então
                            success_count += 1
                            current_encodings.append(iris_encodings_in_image)
            else:
                processed = True
                any_successes_in_folder = True # Alguma íris dessa pasta deu certo então
                # print(f"[SUCCESS] Sucessfully processed {file} with standard image")
                log += f"[SUCCESS] Sucessfully processed {file} with standard image\n"
                success_count += 1
                current_encodings.append(iris_encodings_in_image)                            
            if not processed:
                # print(f"[FAIL] Unable to process {file}")
                log += f"[FAIL] Unable to process {file}\n"
                # if you need all of them to succeed, any failures mean you can stop training on that folder
                if self.SUCCESS_COUNT_THRESHOLD < 0:
                    print(f"Directory {folder} finished...")
                    if (path_to_tmp_file in os.listdir()):
                        os.remove(path_to_tmp_file)
                    return {"names":[], "all_encodings":[], "log":log, "train_log_csv":train_log_csv}
        print(f"Directory {folder} finished...")
        log += f"Folder {folder} finished...\n"
        if any_successes_in_folder:
            # print(f"Apending data...\n{len(current_encodings)} encodings extracted from {folder}")
            log += f"Apending data...\n{len(current_encodings)} encodings extracted from {folder}\n"
            # Guardar nome da pasta como label e lista de encodings como features.
            names[0] = folder
            all_encodings.append(current_encodings)
        else:
            # print("Unable to extract any information from folder...")
            log += ("Unable to extract any information from folder...\n")
            all_encodings.append([])
        train_log_csv.append([folder, len(current_encodings)])
        try:
            os.remove(path_to_tmp_file)
        finally:
            ## Only return results if successful detections exceed threshold (negative threshold means it must capture all irises on folder)
            if self.SUCCESS_COUNT_THRESHOLD < 0:
                if len(all_encodings[0]) < len(os.listdir(f"{self.origin}/{folder}")):
                    return {"names":[], "all_encodings":[], "log":log, "train_log_csv":train_log_csv} 
                return {"names":names[0], "all_encodings":all_encodings[0], "log":log, "train_log_csv":train_log_csv}
            if len(all_encodings[0]) < self.SUCCESS_COUNT_THRESHOLD:
                return {"names":[], "all_encodings":[], "log":log, "train_log_csv":train_log_csv}
            return {"names":names[0], "all_encodings":all_encodings[0], "log":log, "train_log_csv":train_log_csv}

    # Spawns multiprocessing pool and jobs, then gets results.
    def main(self, cleanup_options=[], debug = False):
        names = []
        all_encodings = []

        try:
            pool = mp.Pool(self.processes)
            if debug:
                multiple_results = [pool.apply_async(self.process_folder, (folder, cleanup_options, f"tmp_img{i}.bmp")) for i, folder in enumerate(os.listdir(self.origin)[:10])]
            else:
                multiple_results = [pool.apply_async(self.process_folder, (folder, cleanup_options, f"tmp_img{i}.bmp")) for i, folder in enumerate(os.listdir(self.origin))]
            output = [res.get() for res in multiple_results]        
            log = ""
            train_log_csv = []
            for person in output:
                this_name = person["names"]
                these_encodings = person["all_encodings"]
                log += person["log"]
                train_log_csv += person["train_log_csv"]
                
                if len(these_encodings) != 0:
                    all_encodings.append(these_encodings)
                    names.append(this_name)
            self.save_model(all_encodings, names, cleanup_options, self.origin, log, train_log_csv)
        except Exception as e:
            print(f"Exception encountered during training: {e}")
            pool.close()
            pool.join()

    # Changes ID used to identify output files
    def update_id(self, id):
        self.id = id

    # Outputs current produced pickle file
    def get_pickle_fn(self):
        return self.pickle_file

if __name__ == "__main__":
    ##=============================================##
    ##=====DO NOT USE CTRL+C TO STOP PROCESS=======##
    ##=======THIS WILL CRASH YOUR TERMINAL=========##
    ##=======IT IS A KNOWN BUG IN mp.Pool==========##
    ##========bugs.python.org/issue38428===========##
    ##=============================================##
    
    # Configure functions for attempted cleanup
    cleanup_options = [clahe_test, morphClose_cleanup, medianSlide_cleanup, blurMorph_cleanup, CLAHE_cleanup]
    
    iris_trainer = Iris_Trainer(processes= 7, sct= -1, id=42)

    start = time.perf_counter()
    print(f"Begining processing")
    iris_trainer.main(cleanup_options=cleanup_options)
    end = time.perf_counter()-start
    print(f"Finished in {end//60} minutes and {end%60} seconds")








