#!/usr/bin/env python3
import numpy as np
import convolution as conv
import utils


def gaussian_pyr(img, i_size, num_layers, kernel):
    """
    Compute Gaussian Pyramid
    Input:
    - img: 2D or 3D image matrix
    - i_size: (height, width) of input image
    - num_layers: int indicating the number of layers of the pyramid
                  this includes the original image
    - kernel: gaussian kernel used to smooth the image

    Output:
    - Gaussian pyramid with length = num_layers
    """
    H, W = i_size
    nH, nW = H // 2, W // 2
    pH, pW = H, W
    gPyr = [img]
    while num_layers > 1 and (0 < nH < pH or 0 < nW < pW):
        tmp = gPyr[-1]
        smooth = conv.conv2(tmp, kernel, 'reflect')
        subsample = utils.resample(smooth, (nH, nW))
        gPyr.append(subsample)
        pH, pW = nH, nW
        nH = 1 if (nH == 1) else nH // 2
        nW = 1 if (nW == 1) else nW // 2
        num_layers -= 1
    return gPyr


def laplacian_pyr(gPyr, kernel):
    """
    Compute Laplacian Pyramid using Gaussian Pyramid
    Input:
    - gPyr: Gaussian Pyramid 
    - kernel: the same kernel used for Gaussian Pyramid

    Output:
    - Laplacian pyramid with length = len(gPyr)
    - the last layer of Laplacian Pyramid is the same as
      Gaussian Pyramid's last layer
    """
    lPyr = []
    n = len(gPyr)
    for i in range(n-1):
        big = gPyr[i]
        h, w = big.shape[:2]
        small = utils.resample(gPyr[i+1], (h, w))
        smooth = conv.conv2(small, kernel, 'reflect')
        lPyr.append(big - smooth)
    lPyr.append(gPyr[-1])
    return lPyr
        

def ComputePyr(img, num_layers):
    """
    Given image and number of pyramid layers,
    computes Gaussian and Laplacian Pyramid of the image
    Input:
    - img: 2D or 3D image matrix
    - num_layers: int indicating the number of layers of the pyramids

    Output:
    - (Gaussian Pyramid, Laplacian Pyramid)
    """
    # only accept 2D and 3D images
    if np.ndim(img) == 2:
        H, W = img.shape
    elif np.ndim(img) == 3:
        H, W, _ = img.shape
    else:
        raise ValueError("input image has invalid dimension %s" % (img.shape))

    kernel = utils.gaussian_kernel((9,9), 2)
    gPyr = gaussian_pyr(img, (H, W), num_layers, kernel)
    lPyr = laplacian_pyr(gPyr, kernel)
    return (gPyr, lPyr)
        
        
def PyrBlending(source, target, mask):
    """
    Blend source image onto target. Source is the foreground
    Source, target, and mask have the same height and width
    Source.shape == target.shape
    Number of layers for pyramid is fixed at 6
    Input:
    - source: 2D or 3D source image matrix
    - target: 2D or 3D target image matrix
    - mask: binary matrix with 1 indicating ROI
    
    Output:
    - blended image 
    """
    num_layers = 6
    sH, sW = source.shape[:2]
    tH, tW = target.shape[:2]
    mH, mW = mask.shape[:2]
    kernel = utils.gaussian_kernel((9,9), 2)
    res_pyr = []

    s_gPyr, s_lPyr = ComputePyr(source, num_layers)
    t_gPyr, t_lPyr = ComputePyr(target, num_layers)
    m_gPyr, _ = ComputePyr(mask, num_layers)

    for i in range(num_layers):
        s = s_lPyr[i]
        t = t_lPyr[i]
        m = m_gPyr[i]
        combined = m * s + (1 - m) * t
        res_pyr.append(combined)

    res = res_pyr.pop()
    while len(res_pyr) > 0:
        layer = res_pyr.pop()
        h, w = layer.shape[:2]
        res = utils.resample(res, (h,w))
        res = conv.conv2(res, kernel, 'reflect')
        res += layer

    return res
        
