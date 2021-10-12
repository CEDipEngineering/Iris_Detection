import numpy as np
import cv2 as cv


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

def clahe_test(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        clahe = cv.createCLAHE(clipLimit = 2, tileGridSize= (5,5))
        processed_img = clahe.apply(img)
        closing = cv.morphologyEx(processed_img, cv.MORPH_CLOSE, kernel)
        # Write
        cv.imwrite(path_to_save_output, closing)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success