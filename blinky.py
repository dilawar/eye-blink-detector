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
import pylab
import numpy as np
import cv2
import helper
import gnuplotlib as gp


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

def get_region_of_interest( frame, method = 'box' ):
    """ When method == 'template', use template matching algorithm to get 
    the region of interest. Unfortunately it does not work on blinking eye
    """
    global template_ , box_
    if method == 'template':
        tr, tc = template_.shape    # Rows and cols in template.
        res = cv2.matchTemplate( frame, template_, cv2.TM_SQDIFF_NORMED )
        minv, maxv, (ctopL, rtopL), maxl = cv2.minMaxLoc( res )
        # (ctopL, rtopL) is the point where we have best match.
        # box = generate_box( minl, tc, tr )
        matchBox = ( ctopL, rtopL ), (ctopL + tc, rtopL + tr )
        print( "Bounding box for result %s" % str(matchBox) )
        return clip_frame( frame, matchBox )
    else:
        return clip_frame( frame, box_ )


def process( args ):
    global cap_
    global box_, template_
    cap_ = cv2.VideoCapture( args.video_device )
    nFames = cap.get( cv2.CV_CAP_PROP_FRAME_COUNT )
    ret, frame = cap_.read()
    frame = helper.toGrey( frame )
    box_, template_ = helper.get_bounding_box( frame )
    # box_ = [ (425, 252), (641, 420 ) ]
    # template_ = clip_frame( frame, box_ )
    # Now track the template in video
    blinkValues = [ ]
    totalFramesDone = 1
    towrite = []
    while True:
        ret, frame = cap_.read( )
        if not ret:
            continue
        totalFramesDone += 1
        frame = helper.toGrey( frame )
        # print( template_.shape, resultFrame.shape )
        # display_frame( np.hstack( (resFromClipping, resFromTemplateMatch)), 10 )
        roi = get_region_of_interest( frame )
        eyeHull, eyeIndex= helper.compute_open_eye_index( roi )
        display_frame( np.hstack( (roi, eyeHull) ), 1 )

        blinkValues.append( eyeIndex )
        towrite.append( eyeIndex )

        csvFile = '%s_eye_bink_index.csv' % args.video_device 
        with open( csvFile, 'w') as f:
            f.write( '# eye blink values of file %s' % args.video_device )

        if len( blinkValues ) % 100 == 0:
            print( '[INFO] Done %d out of %d frames.' % ( totalFramesDone
                , nFames ) )
            gp.plot( np.array( blinkValues[-1000:] )
                , title = 'Open Eye index - last 1000 frames'
                , cmds = 'set terminal x11'
                )
            with open( csvFile, 'a' ) as f:
                f.write( '\n'.join( towrite ) )
                towrite = []

    # Also write the left-over values.
    with open( csvFile, 'a' ) as f:
        f.write( '\n'.join( towrite ) )
        towrite = []
    print( '[INFO] Done writing data to %s' % csvFile )
    print( ' == All done from me folks' )


def main(args):
    # Extract video first
    process( args )

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Detect eye blinks in given recording.'''
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
    parser.add_argument('--verbose', '-v'
        , required = False
        , action = 'store_true'
        , default = True
        , help = 'Show you whats going on?'
        )
    parser.parse_args(namespace=args)
    main( args )

