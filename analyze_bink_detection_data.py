"""analyze_bink_detection_data.py: 

Analyze the blink detect data.

"""
    
__author__           = "Me"
__copyright__        = "Copyright 2016, Me"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Me"
__email__            = ""
__status__           = "Development"

import sys
import os
import matplotlib.pyplot as plt
import pandas
import numpy as np

def merge_continuous_blink( blinkIndex ):
    nDiff = np.diff( blinkIndex )
    newIndices = np.where( nDiff > 1 )[0] + 1
    newIndices = blinkIndex[ newIndices ]
    return np.insert( newIndices, 0, blinkIndex[0] )

def main( ):
    datafile = sys.argv[1]
    print( 'Analyzing detection data' )
    data = pandas.read_csv( datafile )
    ax1 = plt.subplot( 2, 1, 1)
    plt.plot( data['time'], data['value'] )
    # plt.ylim( [ 0, data['value'].mean() + 100 ] )
    mean = data['value'].mean()
    t = data['time'].values
    vals = data['value'].values

    blinkIndex = np.where( vals < mean / 4.0 )[0]
    blinkIndex = merge_continuous_blink( blinkIndex )

    plt.title( 'Raw values of open-eye index' )
    ax2 = plt.subplot( 2, 1, 2, sharex=ax1)
    blinkT = t[blinkIndex]
    plt.plot( blinkT, [1] * len( blinkT ), '|' )
    plt.title( 'Location of blinks' )
    outfile = '%s.png' % datafile 
    plt.savefig( outfile )
    print( 'Plot is saved to %s' % outfile )

if __name__ == '__main__':
    main()
