import cv2
import numpy as np


def canny(f, show=False):
    im = cv2.imread(f)
    e = cv2.Canny(im, 45, 50)

    if show:
        cv2.imshow('Canny', e)
        cv2.waitKey(0)
    else:
        return e


def laplacian(f, show=False):
    im = cv2.imread(f)
    e = cv2.Laplacian(im, cv2.CV_64F)
    e = np.uint8(np.absolute(e))

    if show:
        cv2.imread('Laplacian', e)
        cv2.waitKey(0)
    else:
        return e


def sobel(f, show=False):
    im = cv2.imread(f)

    sobelX = cv2.Sobel(im, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(im, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)

    if show:
        cv2.imshow("Sobel X", sobelX)
        cv2.imshow("Sobel Y", sobelY)
        cv2.imshow("Sobel Combined", sobelCombined)
        cv2.waitKey()
    else:
        return sobelX, sobelY, sobelCombined
