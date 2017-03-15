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

def main(args):
    # Extract video first
    data = webcam.video2csv(args)
    if len(data) == 0:
        print('[WARN] Could not load data. Quitting.')
        return None
    edgyBlinks = extract.find_blinks_using_edge(data)
    outfile = "%s_blinks_using_edges.csv" % args['video_device']
    print("[INFO] Writing to outfile %s" % outfile)
    np.savetxt(outfile, np.array(edgyBlinks).T, delimiter=","
            , header = "time,blinks")

    pixalBlinks = extract.find_blinks_using_pixals(data)
    outfile = "%s_blinks_using_pixals.csv" % args['video_device']
    print("[INFO] Writing to outfile %s" % outfile)
    np.savetxt(outfile, np.array(pixalBlinks).T, delimiter=","
            , header = "time,blinks")


if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Detect eye-blinks in recording (optimized for mouse eye)'''
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
    main(vars(args))

