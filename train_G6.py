from pathlib import Path
import os
import pickle
from G6_iris_recognition.feature_vec import engroup
import cv2 as cv
import numpy as np

"""
# Template for adding new cleanup methods.
def generic_cleanup(path_to_image, path_to_save_output):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        processed_img = img.copy() # Example, useless
        # Write
        cv.imwrite(path_to_save_output, processed_img)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success
"""

def save_model(encodings, names):
    # encodings = lista de listas de encondings gerados com engroup
    # names = nome referente a cada lista da matriz encodings
    data = {"encodings": encodings, "names": names}
    with open("model.pickle", "wb") as f:
        f.write(pickle.dumps(data))

def cleanup_wrapper(path_to_img, path_to_tmp_file, transform_function, **kwargs):
    # Wrapper for other functions, receives 'transform function' pointer
    # 'transform function' must have following signature: tf(path_to_img, output_path, **kwargs)
    if transform_function(path_to_img, path_to_tmp_file, **kwargs):
        # print(f"Transformation was successfull")
        return True
    else:
        return False

def CLAHE_cleanup(path_to_image, path_to_save_output, cL = 30, tGS = (5,5)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        imgMedian = np.median(img.flatten())
        processed_img = cv.add(img, 127-imgMedian)
        processed_img = cv.medianBlur(processed_img, 3)
        clahe = cv.createCLAHE(clipLimit = cL, tileGridSize= tGS)
        processed_img = clahe.apply(processed_img)
        # Write
        cv.imwrite(path_to_save_output, processed_img)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success

def blurMorph_cleanup(path_to_image, path_to_save_output, kernel = np.ones((7,7),np.uint8), gaussBlurSize=(3,3)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        imgMedian = np.median(img.flatten())
        processed_img = cv.add(img, 127-imgMedian)
        processed_img = cv.GaussianBlur(processed_img,gaussBlurSize,cv.BORDER_DEFAULT)
        processed_img = cv.morphologyEx(processed_img, cv.MORPH_OPEN, kernel)
        # Write
        cv.imwrite(path_to_save_output, processed_img)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success

def medianSlide_cleanup(path_to_image, path_to_save_output):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        imgMedian = np.median(img.flatten())
        processed_img = cv.add(img, 127-imgMedian)
        # Write
        cv.imwrite(path_to_save_output, processed_img)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success

def morphClose_cleanup(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        processed_img = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel, iterations = 1)
        # Write
        cv.imwrite(path_to_save_output, processed_img)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success

def main():
    names = []
    all_encodings = []
    cleanup_options = [blurMorph_cleanup, morphClose_cleanup, CLAHE_cleanup, medianSlide_cleanup]
    log = open("log.txt", "w")
    train_log_csv = []
    path_to_tmp_file = "tmp_img.bmp"
    try:
        for folder in os.listdir("train"):
            print(f"----\nStarting directory {folder}\n----\n")
            log.write(f"------------------\nStarting directory {folder}\n------------------\n")
            current_encodings = []
            any_successes_in_folder = False
            for file in os.listdir(f"train/{folder}"):
                processed=False
                ## Tentando imagem original, e se não der, tentar imagem processada
                path_to_image=f"train/{folder}/{file}"
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
                                print(f"[SUCCESS] Sucessfully processed {file} with method {option.__name__}")
                                log.write(f"[SUCCESS] Sucessfully processed {file} with method {option.__name__}\n")
                                any_successes_in_folder = True # Alguma íris dessa pasta deu certo então
                                current_encodings.append(iris_encodings_in_image)
                else:
                    processed = True
                    any_successes_in_folder = True # Alguma íris dessa pasta deu certo então
                    print(f"[SUCCESS] Sucessfully processed {file} with standard image")
                    log.write(f"[SUCCESS] Sucessfully processed {file} with standard image\n")
                    current_encodings.append(iris_encodings_in_image)                            
                if not processed:
                    print(f"[FAIL] Unable to process {file}")
                    log.write(f"[FAIL] Unable to process {file}\n")

            print(f"Folder {folder} finished...")
            log.write(f"Folder {folder} finished...\n")
            if any_successes_in_folder:
                print(f"Apending data...\n{len(current_encodings)} encodings extracted from {folder}")
                log.write(f"Apending data...\n{len(current_encodings)} encodings extracted from {folder}\n")
                # Guardar nome da pasta como label e lista de encodings como features.
                names.append(folder)
                all_encodings.append(current_encodings)
            else:
                print("Unable to extract any information from folder...")
                log.write("Unable to extract any information from folder...\n")
            train_log_csv.append([folder, len(current_encodings)])
        save_model(all_encodings, names)
    finally:
        log.close()
        with open("log.csv", "w") as log_csv:
            log_csv.write(str(train_log_csv))

if __name__ == "__main__":
    Path("./model.pickle").touch()
    main()














