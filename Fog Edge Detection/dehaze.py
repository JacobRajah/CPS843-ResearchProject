"""
This code is taken directly from He-Zhang on GitHub, URL: https://github.com/He-Zhang/image_dehaze
It has been modified and extended to meet the needs of this research topic
"""

import cv2;
import math;
import numpy as np;

# Extract the dark channel image given an input image
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

# Determine the atmospheric light from the brightest regions
# in the DarkChannel image.
def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)

    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

# Determine the transmission value from the dark
# channel of the normalized input image (im / A)
def TransmissionEstimate(im,A,sz):
    omega = 0.95
    im3 = np.empty(im.shape,im.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)

    return t

# Recover the haze free image using the inverted
# haze imaging equations. Lower bound transmission
# is set to 0.1
def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def ExtractEdges(J):
    # Convert the color scale image to gray scale to remove unnecessary noise
    grayscale_img = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)
    # Apply a 5x5 gaussian filter to the grayscale image
    blurred_img = cv2.GaussianBlur(grayscale_img, (5,5), 0)
    edges = cv2.Canny(blurred_img, 30, 150)

    return edges

if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = './image/15.png'

    def nothing(*argv):
        pass

    src = cv2.imread(fn)

    I = src.astype('float64')/255

    patch_size = 30
 
    dark = DarkChannel(I,patch_size)
    A = AtmLight(I,dark)
    te = TransmissionEstimate(I,A,patch_size)
    t = TransmissionRefine(src,te)
    J = Recover(I,t,A,0.1)

    # Save de-fogged image to file to enforce format type for Canny
    cv2.imwrite("./image/J.png", J*255)
        
    edges_no_fog = ExtractEdges(cv2.imread("./image/J.png"))
    edges_fog = ExtractEdges(src)

    cv2.imshow("dark",dark)
    cv2.imshow("t",t)
    cv2.imshow('I',src)
    cv2.imshow('J',J)
    
    cv2.imshow('Edges no Fog', edges_no_fog)
    cv2.imshow('Edges with Fog', edges_fog)

    cv2.waitKey(delay=50000)
    
