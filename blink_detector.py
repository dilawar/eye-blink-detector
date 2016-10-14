#!/usr/bin/env python

"""
Detect eyeblinks in video.

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

def is_a_good_frame( frame ):
    if frame.max( ) < 100 or frame.min() > 150:
        print( '[WARN] not a good frame: too bright or dark' )
        return False
    if frame.mean( ) < 50 or frame.mean() > 200:
        print( '[WARN] not a good frame: not much variation' )
        return False
    return True

def fetch_a_good_frame(  ):
    global cap_
    ret, frame = cap_.read()
    if ret:
        if is_a_good_frame( frame ):
            return frame
        else:
            return fetch_a_good_frame( )
    else:
        print( '[Warn] Failed to fetch a frame' )
        return None

def process( args ):
    global cap_
    global box_, template_
    cap_ = cv2.VideoCapture( args.video_device )
    nFames = cap_.get( cv2.cv.CV_CAP_PROP_FRAME_COUNT )
    fps = float( cap_.get( cv2.cv.CV_CAP_PROP_FPS ) )

    print( '[INFO] FPS = %f' % fps )
    frame = fetch_a_good_frame( )
    frame = helper.toGrey( frame )
    box_, template_ = helper.get_bounding_box( frame )
    # box_ = [ (425, 252), (641, 420 ) ]
    # template_ = clip_frame( frame, box_ )
    # Now track the template in video
    blinkValues = [ ]
    towrite = []
    csvFile = '%s_eye_bink_index.csv' % args.video_device 
    with open( csvFile, 'w') as f:
        f.write( 'time, open-eye\n')
    
    while True:
        totalFramesDone = cap_.get( cv2.cv.CV_CAP_PROP_POS_FRAMES ) 
        if totalFramesDone + 1 >= nFames:
            print( '== All done' )
            break
        frame = fetch_a_good_frame( )
        frame = helper.toGrey( frame )
        # print( template_.shape, resultFrame.shape )
        # display_frame( np.hstack( (resFromClipping, resFromTemplateMatch)), 10 )
        roi = get_region_of_interest( frame )
        eyeHull, eyeIndex = helper.compute_open_eye_index( roi )
        blinkValues.append( eyeIndex )
        towrite.append( '%g,%g' % ((totalFramesDone / fps), eyeIndex ))

        if args.verbose:
            display_frame( np.hstack( (roi, eyeHull) ), 1 )


        if len( blinkValues ) % 100 == 0:
            print( '[INFO] Done %d out of %d frames.' % ( totalFramesDone
                , nFames ) )
            with open( csvFile, 'a' ) as f:
                line = "%s\n" % ('\n'.join( towrite ) )
                print( line )
                f.write( line )
            towrite = []
            if args.verbose:
                gp.plot( np.array( blinkValues[-1000:] )
                    , title = 'Open Eye index - last 1000 frames'
                    , terminal = 'x11'
                    )

    # Also write the left-over values.
    with open( csvFile, 'a' ) as f:
        f.write( '\n'.join( towrite ) )

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
        , default = False
        , help = 'Show you whats going on?'
        )
    parser.parse_args(namespace=args)
    main( args )

