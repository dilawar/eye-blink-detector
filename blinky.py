#!/usr/bin/env python

"""blinky.py: 

Starting point of blinky

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import extract
import webcam
import pylab
import numpy as np
import cv2
import helper


cap_ = None
box_ = None
template_ = None

def display_frame( frame, delay = 40 ):
    cv2.imshow( 'frame', frame )
    cv2.waitKey( delay )

def clip_frame( frame, box ):
    (r1, c1), (r2, c2 ) = box
    return frame[c1:c2,r1:r2]

def generate_box( (c,r), width, height ):
    if width < 0: width = 10
    if height < 0 : height = 10
    leftCorner = ( max(0,c - width / 2), max(0, r - height / 2 ) )
    rightCorner = (leftCorner[0] + width, leftCorner[1] + height)
    return leftCorner, rightCorner 


def process( args ):
    global cap_
    global box_, template_
    cap_ = cv2.VideoCapture( args.video_device )
    ret, frame = cap_.read()
    frame = helper.toGrey( frame )
    # box_, template_ = helper.get_bounding_box( frame )
    box_ = [ (425, 252), (641, 420 ) ]
    template_ = clip_frame( frame, box_ )
    # Now track the template in video
    while True:
        ret, frame = cap_.read( )
        frame = helper.toGrey( frame )
        print template_.shape
        tr, tc = template_.shape    # Rows and cols in template.
        res = cv2.matchTemplate( frame, template_, cv2.TM_SQDIFF_NORMED )
        minv, maxv, (ctopL, rtopL), maxl = cv2.minMaxLoc( res )
        # (ctopL, rtopL) is the point where we have best match.
        # box = generate_box( minl, tc, tr )
        matchBox = ( ctopL, rtopL ), (ctopL + tc, rtopL + tr )
        print( "Bounding box for result %s" % str(matchBox) )
        resFromTemplateMatch = clip_frame( frame, matchBox )
        resFromClipping = clip_frame( frame, box_ )
        # print( template_.shape, resultFrame.shape )
        display_frame( np.hstack( (resFromClipping, resFromTemplateMatch)), 10 )

        


def main(args):
    # Extract video first
    process( args )

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''description'''
    parser = argparse.ArgumentParser(description=description)
    class Args: pass 
    args = Args()
    parser.add_argument('--video-device', '-f'
        , required = False
        , default = 0
        , help = 'Path of the video file or camera index. default camera 0'
        )
    parser.add_argument('--bbox', '-b'
        , required = False
        , nargs = '+'
        , type = int
        , help = 'Bounding box : topx topy width height'
        )
    parser.parse_args(namespace=args)
    main( args )

