#!/usr/bin/env python3
import numpy as np
import matplotlib
# to deal with "python not installed as framework" error
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import convolution as conv


def intensity_spread(img, L):
    """
    The intensity transformation function is
    T(x) = (L - 1) * (x - min) / (max - min)
    This is a function written in homework 2
    
    Inputs:
    img: image represented in a 2D array
    L: int; L - 1 is the highest new intensity

    Output:
    an image with intensity transformed to fit in the range [0, L-1]
    """

    pixels = img.flatten()
    min_pix = np.amin(pixels)
    max_pix = np.amax(pixels)
    res = np.zeros(pixels.shape)

    for idx, p in enumerate(pixels):
        res[idx] = (L - 1) * ((p - min_pix) / (max_pix - min_pix))

    return np.reshape(res, img.shape)


def log_shift(data):
    """
    Shift data using transform s = log(1 + abs(d))
    Input:
    - Data: list of data d that needs to be transform
    Output:
    List of transformed data d'
    """
    result = [np.log(1 + np.abs(d.copy())) for d in data]
    return result


def visualization(data, rows, cols, titles, figsize):
    """
    Function to visualize the result images.
    This function is a modified version of the code written by 
    user "swatchai" as an answer on stackoverflow. 
    The code and thread can be found at this link:
    https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645

    Inputs:
    - data: list of images array
    - rows: int; number of rows of the plot
    - cols: int; number of columns of the plot
    - titles: list of subplots' titles
    - figsize: int tuples; the size of the entire figure

    Output: a figure contains several images, each in a separate subplot
    """
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    # plot image on each subplot
    for i, axi in enumerate(ax.flat):
        # i is in range [0, nrows * ncols)
        # axi is equivalent to ax[rowid][colid]
        axi.imshow(data[i])
        axi.set_title(titles[i])
    plt.tight_layout(True)
    plt.show()


def gaussian_kernel(size, sigma):
    """
    - 2D gaussian kernel.
    - mimick the behavior of fspecial('gaussian', [shape], [sigma])
    in MATLAB
    - source: https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (by user ali_m)
    """

    m, n = [(s - 1.) / 2. for s in size]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0: h /= sumh
    return h


# function to resample original image
def resample(img, newSize):
    """
    Function to downsample and upsample image
    Use simple nearest neighbor interpolation
    Input:
    - img: 2D or 3D image matrix
    - newSize: tuple of result size (height, width)

    Output: resampled image
    """
    # use float as datatype to preserve float precision
    # will convert to np.uint8 when display result
    # get the new dimensions
    nH, nW = newSize
    if np.ndim(img) == 2:
        H, W = img.shape
        res = np.zeros((nH, nW), dtype=np.float32)
    elif np.ndim(img) == 3:
        H, W, _ = img.shape
        res = np.zeros((nH, nW, _), dtype=np.float32)
    else:
        raise ValueError("input image has invalid dimension %s" % (img.shape))

    # interpolate the value for the result
    for idx in range(nH * nW):
        i = idx // nW
        j = idx % nW
        orig_i = int((i * H) // nH)
        orig_j = int((j * W) // nW)
        res[i, j] = img[orig_i, orig_j]
    return res


def extract_roi(img, coords):
    """
    Extract ROI using coords of the region
    Input:
    - img: 2D or 3D image matrix
    - coord: the coordinates of the rectangle bounding box
             of the region of interest. Includes top-left point
             and bottom right point. Points follow (x, y) format
    return:
    - image roi
    - the coordinates of the center of roi.
      roi_center follows (y, x) format
    """

    start, end = coords
    roi = img[start[1]:end[1], start[0]:end[0]]
    h, w = roi.shape[:2]
    center = (start[1] + h//2, start[0] + w//2)
    return roi, center


def align_images(src, target, scoords, tcoords, shape):
    """
    Aligns 2 images so that their ROIs overlap
    Assumes the ROIs have the same shape: either rectangle or ellipse
    Coordinates follow format (x, y)

    Input:
    - src: 2D or 3D source image matrix
    - target: 2D or 3D target image matrix
    - scoords: coordinates of the bounding box of source ROI
               [top_left_point, bottom_right_point]
    - tcoords: coordinates of the bounding box of target ROI
               [top_left_point, bottom_right_point]
    - shape: string describing the shape of ROI;
             either "rectangle" or "ellipse"

    Return: aligned source image and mask
    """

    # create black background
    bg = np.zeros(target.shape, dtype=np.float32)    
    # extract ROIs using the regions' coordinates
    src_roi, sr_center = extract_roi(src , scoords)
    t_roi, tr_center = extract_roi(target, tcoords)

    # check to see if we need to resize source
    # such that source and target's areas of ROIs are 
    # approximately the same size
    # if the area of target roi is either at least twice or
    # at most half the area of source roi, resize source
    sr_h, sr_w = src_roi.shape[:2]
    tr_h, tr_w = t_roi.shape[:2]
    sr_area = sr_h * sr_w; tr_area = tr_h * tr_w
    if 2 <= tr_area / sr_area or tr_area / sr_area <= 0.5:
        kernel = gaussian_kernel((9,9), 2)
        # keep track of the ratio
        ratio = [tr_h / sr_h, tr_w / sr_w]
        new_size = [int(src.shape[0] * ratio[0]), int(src.shape[1] * ratio[1])]
        smooth_src = conv.conv2(src, kernel, 'reflect')
        resized_src = resample(smooth_src, new_size)
    else:
        resized_src = src.copy()
        ratio = [1,1]
    new_sr_cntr = [int(sr_center[0] * ratio[0]), int(sr_center[1] * ratio[1])]

    # psate source image on black background at (0,0)
    # translate pasted image so that center of source ROI
    # overlaps center of target ROI
    # M is translation matrix [[1, 0, tx], [0, 1, ty]]
    tx = tr_center[1] - new_sr_cntr[1]
    ty = tr_center[0] - new_sr_cntr[0]
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    bg[0:resized_src.shape[0], 0:resized_src.shape[1]] = resized_src
    shifted = cv2.warpAffine(bg, M, (bg.shape[1], bg.shape[0]))

    # recalculating coordinates of source roi
    # creating aligned mask
    start, end = scoords
    # because coords in scoords have form (x, y)
    # ratio follows (y, x)
    new_start = (start[0] * ratio[1], start[1] * ratio[0])
    new_end = (end[0] * ratio[1], end[1] * ratio[0])
    tmp = np.zeros(target.shape, dtype=np.float32)
    if shape == "rectangle":
        mask = cv2.rectangle(tmp, new_start, new_end, (1,1,1), -1)
    elif shape == "ellipse":
        startx, starty = new_start
        endx, endy = new_end
        centerx = int((startx + endx) // 2)
        centery = int((starty + endy) // 2)
        axlen = (int((endx - startx) // 2), int((endy - starty) // 2))
        mask = cv2.ellipse(tmp, (centerx, centery), axlen, 0, 0, 360, (1,1,1), -1)
    mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

    return shifted, mask
