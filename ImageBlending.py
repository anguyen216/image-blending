#!/usr/bin/env python3
# GUI and ROI selections are adaptation of the following blog posts
# by Adrian Rosebrock
# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
# https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/
from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from LaplacianBlending import PyrBlending
import utils
import math


def click_and_crop(event, x, y, flags, param):
    """
    Adapted from Arian Rosebrock block post's function
    """
    # grab references to the global variables
    global REFPT, CROPPING, TOOL

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that CROPPING is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        REFPT = [(x, y)]
        CROPPING = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the CROPPING operation is finished
        REFPT.append((x, y))
        CROPPING = False
        green = (0, 255, 0)
        
        # draw a rectangle around the region of interest
        if TOOL == "rectangle":
            cv2.rectangle(param[1], REFPT[0], REFPT[1], green, 2)
        # draw an ellipse around the region of interest
        elif TOOL == "ellipse":
            startx = REFPT[0][0]; starty = REFPT[0][1]
            endx = REFPT[1][0]; endy = REFPT[1][1]
            centerx = (startx + endx) // 2
            centery = (starty + endy) // 2
            axlen = ((endx - startx)//2, (endy - starty)//2)
            cv2.ellipse(param[1], (centerx, centery), axlen, 0, 0, 360, green, 2)

        cv2.imshow(param[0], param[1])


def select_img():
    """
    Function to retrieve image file according to user's need
    """
    global SOURCE, OPENED_SOURCE, TARGET
    global SRC_PATH, T_PATH
    path = filedialog.askopenfilename()
    if len(path) > 0:
        if not OPENED_SOURCE:
            OPENED_SOURCE = True
            SRC_PATH = path
            SOURCE = cv2.imread(path)
        else:
            T_PATH = path
            TARGET = cv2.imread(path)


def get_roi(image, wname):
    """
    Adapted from Adrian Rosebrock blog post's code
    """
    global REFPT, CROPPING, SRC_COORDS, T_COORDS
    clone = image.copy()
    param = [wname, image]
    cv2.namedWindow(wname)
    cv2.setMouseCallback(wname, click_and_crop, param)
    cv2.imshow(wname, image)
    cv2.waitKey(0)

    # if there are two reference points, then set the region of interest
    # of the image and display it
    if len(REFPT) == 2:
        if wname == "SOURCE":
            SRC_COORDS = REFPT
        if wname == "TARGET":
            T_COORDS = REFPT

        #print(REFPT)  # for debugging
        cv2.destroyAllWindows()


def close_gui():
    """
    Function to close GUI window
    """
    global LOOP_ACTIVE
    LOOP_ACTIVE = False


def reset_gui():
    """
    Reset the GUI setting in case the user wants to
    reset their progress
    """
    global OPENED_SOURCE, SOURCE, TARGET, REFPT, CROPPING
    global SRC_COORDS, T_COORDS
    OPENED_SOURCE = False
    SOURCE = None; TARGET = None
    REFPT = []; CROPPING = False
    SRC_COORDS = []; T_COORDS = []


def main():
    global SOURCE, TARGET, REFPT, CROPPING, OPENED_SOURCE
    global TOOL, SRC_COORDS, T_COORDS
    global LOOP_ACTIVE
    global SRC_PATH, T_PATH

    # set up GUI and its buttons and dropdown
    root = Tk()
    loop_active = True
    options = ["selection tool", "rectangle", "ellipse"]
    var = StringVar(root)
    var.set(options[0])
    source = None; target = None
    dropdown = OptionMenu(root, var, *options)
    dropdown.pack()
    btn1 = Button(root, text="Select image", command=select_img)
    btn1.pack(side="left", fill="both", expand="yes", padx="10", pady="10")
    btn2 = Button(root, text="Done", command=close_gui)
    btn2.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
    btn3 = Button(root, text="Reset", command=reset_gui)
    btn3.pack(side="right", fill="both", expand="yes", padx="10", pady="10")

    # wait for user's action
    # extract inputs and only perform blending
    # once source, target and their ROIs are given
    while LOOP_ACTIVE:
        root.update()
        TOOL = var.get()
        if TOOL == "selection tool":
            continue

        if SOURCE is None and TARGET is None:
            continue
        elif SOURCE is not None and SRC_COORDS == []:
            source = SOURCE.copy()
            get_roi(SOURCE, "SOURCE")

        elif TARGET is None:
            continue
        elif TARGET is not None and T_COORDS == []:
            target = TARGET.copy()
            get_roi(TARGET, "TARGET")
            # algin images before input into blending algorithm
            aligned, mask = utils.align_images(source, target, SRC_COORDS, \
                                                   T_COORDS, TOOL)
            result = PyrBlending(aligned, target, mask)
            # normalized result and then clip before display
            normalized = result.copy()
            if np.ndim(result) == 2:
                normalized = utils.intensity_spread(result, 256)
            else:
                for i in range(3):
                    normalized[:,:,i] = utils.intensity_spread(result[:,:,i], 256)
            # clip float values for visualization
            normalized = normalized.astype(np.uint8)
            mask[mask == 1] = 255   # for displaying purpose
            # display mask and blended result
            cv2.imshow('MASK', mask)
            cv2.imshow('BLENDED IMAGE', normalized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # save blended result using source and target filename
            sname = SRC_PATH.split("/")[-1][:-4]
            tname = T_PATH.split("/")[-1]
            cv2.imwrite("../results/" + sname + tname, normalized)

            # reset all global vars to get ready for next pair of images
            SOURCE = None; TARGET = None
            OPENED_SOURCE = False
            SRC_COORDS = []
            T_COORDS = []
            

OPENED_SOURCE = False
SOURCE = None; TARGET = None
REFPT = []; CROPPING = False
SRC_COORDS = []; T_COORDS = []
TOOL = None
LOOP_ACTIVE = True
SRC_PATH = None; T_PATH = None
if __name__ == "__main__":
    main()
