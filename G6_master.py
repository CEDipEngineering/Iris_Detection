from train_G6 import Iris_Trainer
from test_G6 import Iris_Tester
import time
from support_functions import *
import os

def main(cleanup_options, trainer, tester):
    ##=============================================##
    ##=====DO NOT USE CTRL+C TO STOP PROCESS=======##
    ##=======THIS WILL CRASH YOUR TERMINAL=========##
    ##=======IT IS A KNOWN BUG IN mp.Pool==========##
    ##========bugs.python.org/issue38428===========##
    ##=============================================##

    # This function imports the training and test modules, and then applies them sequentially.

    # Training
    start = time.perf_counter()
    print(f"Begining training")
    trainer.main(cleanup_options=cleanup_options)
    end = time.perf_counter()-start
    print(f"Finished in {end//60} minutes and {end%60} seconds")

    # Testing
    print(f"Begining testing")
    start = time.perf_counter()
    tester.main(trainer.get_pickle_fn()) # test what was just trained
    end = time.perf_counter()-start
    print(f"Finished in {end//60} minutes and {end%60} seconds")

if __name__ == "__main__":
    
    ##=============================================##
    ##=====DO NOT USE CTRL+C TO STOP PROCESS=======##
    ##=======THIS WILL CRASH YOUR TERMINAL=========##
    ##=======IT IS A KNOWN BUG IN mp.Pool==========##
    ##========bugs.python.org/issue38428===========##
    ##=============================================##

    # All functions available:
    # [CLAHE_cleanup, 
    #  blurMorph_cleanup, 
    #  medianSlide_cleanup, 
    #  morph_close, 
    #  clahe_test, 
    #  detail_enhance, 
    #  erosion, 
    #  morph_open, 
    #  morph_hit_miss, 
    #  median_blur, 
    #  gaussian_blur, 
    #  bilateral_filter, 
    #  erode_dilatate, 
    #  wavelet_morph_open]


    # Configure functions for attempted cleanup
    cleanup_options = [morph_close, blurMorph_cleanup]
    
    # Number of processes the program is allowed to spawn
    process_limit = 7
    # Determine id for the experiment
    id = sum([1 for i in os.listdir("results") if i.endswith(".pickle")]) # Counts previous trials

    iris_trainer = Iris_Trainer(processes = process_limit, sct = -1, id = id)
    iris_tester = Iris_Tester(processes= process_limit)

    start_all = time.perf_counter()
    for i, e in enumerate(cleanup_options):
        start = time.perf_counter()
        iris_trainer.update_id(id = sum([1 for i in os.listdir("results") if i.endswith(".pickle")]))
        main([e], iris_trainer, iris_tester)
        end = time.perf_counter()-start
        print(f"Time for experiment {i} was {end//60} minutes and {end%60} seconds.")
    end_all = time.perf_counter()-start_all
    print(f"Total time for {i+1} experiments was {end_all//60} minutes and {end_all%60} seconds.")