from train_G6 import Iris_Trainer
from test_G6 import Iris_Tester
import time
from support_functions import *
import os

def main():
    ##=============================================##
    ##=====DO NOT USE CTRL+C TO STOP PROCESS=======##
    ##=======THIS WILL CRASH YOUR TERMINAL=========##
    ##=======IT IS A KNOWN BUG IN mp.Pool==========##
    ##========bugs.python.org/issue38428===========##
    ##=============================================##
    
    # Configure functions for attempted cleanup
    cleanup_options = [clahe_test, morphClose_cleanup, medianSlide_cleanup, blurMorph_cleanup, CLAHE_cleanup]
    # Number of processes the program is allowed to spawn
    process_limit = 7
    # Determine id for the experiment
    id = sum([1 for i in os.listdir("results") if i.endswith(".pickle")]) # Counts previous trials

    # Training
    iris_trainer = Iris_Trainer(processes = process_limit, sct = -1, id = id)

    start = time.perf_counter()
    print(f"Begining training")
    iris_trainer.main(cleanup_options=cleanup_options)
    end = time.perf_counter()-start
    print(f"Finished in {end//60} minutes and {end%60} seconds")

    # Testing
    iris_tester = Iris_Tester(processes= process_limit)

    print(f"Begining testing")
    start = time.perf_counter()
    iris_tester.main(iris_trainer.get_pickle_fn()) # test what was just trained
    end = time.perf_counter()-start
    print(f"Finished in {end//60} minutes and {end%60} seconds")

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()-start
    print(f"Full time was {end//60} minutes and {end%60} seconds.")
