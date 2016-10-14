"""helper.py: 

"""
    
__author__           = "Me"
__copyright__        = "Copyright 2016, Me"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Me"
__email__            = ""
__status__           = "Development"

import cv2
import numpy as np

bbox_ = None
current_frame_ = None

def onmouse(event, x, y, flags, params):
    global current_frame_, bbox_
    # Draw Rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox_ = []
        bbox_.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        bbox_.append((x, y))
        cv2.rectangle(current_frame_, bbox_[0], (x,y), 0,2)

def get_bounding_box(frame):
    global current_frame_, bbox_
    current_frame_ = frame.copy()
    title = "Bound eye and press 'q' to quit."
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, onmouse)
    clone = frame.copy()
    while True:
        cv2.imshow(title, current_frame_)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            current_frame_ = clone.copy()
        elif key == ord("q"):
            break
    cv2.waitKey(1)
    cv2.destroyWindow('Bound_eye')
    (r1,c1), (r2,c2) = bbox_
    print( 'User defined bounding box %s' % bbox_ )
    cv2.destroyAllWindows( )
    return bbox_, frame[c1:c2,r1:r2]

def toGrey( frame ):
    return cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )


def merge_contours(cnts, img):
    """Merge these contours together. And create an image"""
    for c in cnts:
        hull = cv2.convexHull(c)
        cv2.fillConvexPoly(img, hull, 0)
    return img


def accept_contour_as_possible_eye( contour, threshold = 0.1 ):
    # The eye has a certain geometrical shape. If it can not be approximated by
    # an ellipse which major/minor < 0.8, ignore it.
    return True
    if len(contour) < 5:
        # Too tiny to be an eye
        return True
    ellipse = cv2.fitEllipse( contour )
    axes = ellipse[1]
    minor, major = axes
    if minor / major > threshold:
        # Cool, also the area of ellipse and contour area cannot ve very
        # different.
        cntArea = cv2.contourArea( contour )
        ellipseArea = np.pi * minor * major 
        if cntArea < 1:
            return False
        return True
    else:
        return False


def compute_open_eye_index(frame):
    """ Accepts a frame and compute the hull-image of eye.

        reutrn hull image and open-eye index. 
        Larger openEyeIndex means that eye was open.
    """
    # Find edge in frame
    s = np.mean(frame)
    edges = cv2.Canny(frame, s + np.std( frame), np.max( frame) - np.std(frame) )
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    cntImg = np.ones(frame.shape)
    merge_contours(cnts[0], cntImg)

    # cool, find the contour again and convert again. Sum up their area.
    im = np.array((1-cntImg) * 255, dtype = np.uint8)
    cnts = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    hullImg = np.ones(frame.shape)
    openEyeVals = []
    for c in cnts[0]:
        c = cv2.convexHull(c)
        if accept_contour_as_possible_eye( c ):
            cv2.fillConvexPoly(hullImg, c, 0, 8)
            openEyeVals.append(cv2.contourArea(c))
    hullImg = np.array((1-hullImg) * 255, dtype = np.uint8)
    openEyeIndex = sum( openEyeVals )
    return hullImg, openEyeIndex

