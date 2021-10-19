import numpy as np
import cv2 as cv
import pywt
from multiprocessing.pool import ThreadPool



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

def medianSlide_cleanup(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        morph_img = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel, iterations = 1)
        imgMedian = np.median(morph_img.flatten())
        processed_img = cv.add(morph_img, 127-imgMedian)
        
        # Write
        cv.imwrite(path_to_save_output, processed_img)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success

def morph_close(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
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

def detail_enhance(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        img_bgr = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
        processed_img = cv.detailEnhance(img_bgr, sigma_s=6, sigma_r=0.15)
        processed_img_gray = cv.cvtColor(processed_img,cv.COLOR_BGR2GRAY)
        # Write
        cv.imwrite(path_to_save_output, processed_img_gray)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success

def erosion(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        processed_img = cv.erode(img,kernel,iterations = 1)
        # Write
        cv.imwrite(path_to_save_output, processed_img)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success

def morph_open(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        processed_img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        # Write
        cv.imwrite(path_to_save_output, processed_img)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success

def morph_hit_miss(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        processed_img = cv.morphologyEx(img, cv.MORPH_HITMISS, kernel)
        # Write
        cv.imwrite(path_to_save_output, processed_img)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success

def median_blur(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        processed_img = cv.medianBlur(img,5)

        # Write
        cv.imwrite(path_to_save_output, processed_img)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success

def gaussian_blur(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        processed_img = cv.GaussianBlur(img,(5,5),0)

        # Write
        cv.imwrite(path_to_save_output, processed_img)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success

def bilateral_filter(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        processed_img = cv.bilateralFilter(img, 5, 75, 75)

        # Write
        cv.imwrite(path_to_save_output, processed_img)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success

def erode_dilatate(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):

    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        eroded = cv.erode(img, kernel, iterations=2)
        processed_img = cv.dilate(eroded, kernel, iterations=1)

        # Write
        cv.imwrite(path_to_save_output, processed_img)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success
        
def wavelet_morph_open(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
        # Transform
        coefs = pywt.dwt2(img,'haar')
        LL, (LH, HL, HH) = coefs
        coefs2 = pywt.dwt2(LL,'haar')
        LL2, (LH2, HL2, HH2) = coefs
        LL2Blur = cv.GaussianBlur(LL2,(5,5),0)
        coefs3 = LL2Blur, (LH2, HL2, HH2)
        processed_img = pywt.idwt2(coefs3,'haar')
        processed_img2 = cv.morphologyEx(processed_img, cv.MORPH_OPEN, kernel)

        # Write
        cv.imwrite(path_to_save_output, processed_img2)
    except:
        # Handle failure
        print(f"ERROR")
        success=False
    finally:
        # Return state
        return success


def gabor_morph(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
    success = True
    try:
        # Read
        filters = _build_filters()

        morph_close(path_to_image, path_to_save_output) # tira cílios
        img = cv.imread(path_to_save_output) 
        res2 = _process_threaded(img, filters) # gabor
        
        # Write
        cv.imwrite(path_to_save_output, res2)
    except Exception as e:
        # Handle failure
        print(f"ERROR {e}")
        success=False
    finally:
        # Return state
        return success

def gabor(path_to_image, path_to_save_output, kernel = np.ones((5,5),np.uint8)):
    success = True
    try:
        # Read
        img = cv.imread(path_to_image) 
        filters = _build_filters()
        res2 = _process_threaded(img, filters) # gabor
        
        # Write
        cv.imwrite(path_to_save_output, res2)
    except Exception as e:
        # Handle failure
        print(f"ERROR {e}")
        success=False
    finally:
        # Return state
        return success

def _build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def _process_threaded(img, filters, threadn = 8):
    accum = np.zeros_like(img)
    def f(kern):
        return cv.filter2D(img, cv.CV_8UC3, kern)
    pool = ThreadPool(processes=threadn)
    for fimg in pool.imap_unordered(f, filters):
        np.maximum(accum, fimg, accum)
    return accum