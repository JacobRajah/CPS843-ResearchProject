import numpy as np
import cv2


img = cv2.imread('palm_island_ikonos.jpg', 0)


def convert_to_roberts(img):
    """
    Sample code is provided by programmerall.com
    URL: https://programmerall.com/article/8736477946/
    """
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)

    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    cv2.imshow('Original', img)
    cv2.imshow('Roberts Edge Image', Roberts)

    cv2.waitKey(0)

    cv2.imwrite('Original.jpg', img)
    cv2.imwrite('Roberts Edge Image.jpg', Roberts)


def convert_to_sobel(img):
    """
    Sample code is provided by docs.opencv.org
    URL: https://stackoverflow.com/questions/42802352/sobel-edge-detection-in-python-and-opencv
    """
    sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    abs_grad_x = cv2.convertScaleAbs(sobel_horizontal)
    abs_grad_y = cv2.convertScaleAbs(sobel_vertical)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv2.imshow('Original', img)
    cv2.imshow('Sobel Edge Image (X-direction)', sobel_horizontal)
    cv2.imshow('Sobel Edge Image (Y-direction)', sobel_vertical)
    cv2.imshow('Sobel Edge Image (combined)', grad)

    cv2.waitKey(0)

    cv2.imwrite('Original.jpg', img)
    cv2.imwrite('Sobel Edge Image (X-direction).jpg', sobel_horizontal)
    cv2.imwrite('Sobel Edge Image (Y-direction).jpg', sobel_vertical)
    cv2.imwrite('Sobel Edge Image (combined).jpg', grad)


def convert_to_prewitt(img):
    """
    Sample code is provided by gist.github.com/rahit/
    URL: https://gist.github.com/rahit/c078cabc0a48f2570028bff397a9e154
    """
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(img, -1, kernelx)
    img_prewitty = cv2.filter2D(img, -1, kernely)

    cv2.imshow("Original", img)
    cv2.imshow("Prewitt Edge Image (X-direction)", img_prewittx)
    cv2.imshow("Prewitt Edge Image (Y-direction)", img_prewitty)
    cv2.imshow("Prewitt Edge Image (combined)", img_prewittx + img_prewitty)

    cv2.waitKey(0)

    cv2.imwrite("Original.jpg", img)
    cv2.imwrite("Prewitt Edge Image (X-direction).jpg", img_prewittx)
    cv2.imwrite("Prewitt Edge Image (Y-direction).jpg", img_prewitty)
    cv2.imwrite("Prewitt Edge Image (combined).jpg",
                img_prewittx + img_prewitty)


def convert_to_canny(img):
    """
    Sample code is provided by docs.opencv.org
    URL: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
    """
    edges = cv2.Canny(img, 100, 200)

    cv2.imshow('Original', img)
    cv2.imshow('Canny Edge Image', edges)

    cv2.waitKey(0)

    cv2.imwrite('Original.jpg', img)
    cv2.imwrite('Canny Edge Image.jpg', edges)


if __name__ == '__main__':
    """
    Un-comment each method to perform the desired edge detection and generate result images.
    """
    # convert_to_roberts(img)
    # convert_to_sobel(img)
    convert_to_prewitt(img)
    # convert_to_canny(img)
    pass
