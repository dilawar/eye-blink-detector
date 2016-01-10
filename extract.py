#!/usr/bin/env python
"""process_blink_csv.py: 

    Process the csv file and extract the blink data.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import numpy as np
import pylab
import sys
from collections import OrderedDict

window_size_ = 7
# The mean area multiplied by this factor. Anything below this belongs to blink
# zone.
threshold_factor_area_ = 0.5

def plot_data(data, nplots = 4):
    global window_size_
    window = np.ones(window_size_) / window_size_
    tvec, yvec = data[:,0], data[:,1]
    pylab.subplot(nplots, 1, 1)
    pylab.plot(tvec, yvec, label="raw data")
    pylab.legend()

    yvec = np.convolve(yvec, window, 'same')
    pylab.subplot(nplots, 1, 2)
    pylab.plot(tvec, yvec, label='Window size = %s' % window_size_)
    pylab.plot([0, tvec[-1]], [0.5*np.mean(yvec)]*2, label = '0.5*Mean pupil size')
    pylab.legend()

    pylab.subplot(nplots, 1, 4)
    # When area reduces to half of eye pupil, it should be considered.
    newY = 0.5*yvec.mean() - yvec
    newY = newY + np.fabs(newY)
    window = np.ones(3*window_size_)/(3*window_size_)

    yy = np.convolve(newY, window, 'same')
    pylab.plot(tvec, yy, label='Blinks')

    pylab.xlabel("Time (seconds)")
    outfile = 'output.png'
    print("[INFO] Writing to %s" % outfile)
    pylab.savefig(outfile)

def plot_records(records):
    for i, k in enumerate(records):
        pylab.subplot(len(records), 1, i+1)
        d = records[k]
        if len(d) < 3:
            pylab.plot(d[0], d[1], label=str(k))
        else:
            pylab.plot(d[0], d[1], d[2], label=str(k))
        pylab.legend()
    pylab.xlabel("Time (seconds)")
    outfile = "output.png"
    print("[INFO] Writing to %s" % outfile)
    pylab.savefig(outfile)


def get_blink(i, yy, threshold = 10.0):
    # Go left and right and set pixals to 0 as long as they are decreasing on
    # the left and right.
    #print("Using index: %s, %s" % (i, yy[i]))
    start = yy[i]
    left, right = [], []
    x = i+1
    while  x < len(yy) and 0.1 < yy[x] <= start:
        start = yy[x]
        yy[x] = 0
        x += 1
        left.append(start)

    start = yy[i]
    x = i - 1
    while x > 0 and 0.1 < yy[x] <= start:
        start = yy[x]
        yy[x] = 0
        x -= 1
        right.append(start)
    yy[i] = 0.0
    w = left + right
    if len(w) == 0:
        return False, 0
    res = sum(w) / len(w)
    if res < threshold:
        return False, 0.0
    return True, res

def find_blinks_using_edge(data, plot = False, **kwargs):
    """Find location of blinks in data"""
    global window_size_
    records = OrderedDict()
    window = np.ones(window_size_)/window_size_
    t, y = data[:,0], data[:,1]
    # Smooth out the vectors.
    yvec = np.convolve(y, window, 'same')
    records['smooth'] = (t, y)
    newY = yvec - yvec.min()
    window = np.ones(window_size_)/(window_size_)
    yy = np.convolve(newY, window, 'same')
    blinks = []
    while yy.max() > yy.mean() + 1.5 * yy.std() :
        i = np.argmax(yy)
        isBlink, a = get_blink(i, yy)
        if isBlink:
            blinks.append((i, a))

    xvec, yvec = [], []
    for i, x in sorted(blinks):
        xvec.append(t[i])
        yvec.append(x)
    return xvec, yvec

def find_blinks_using_pixals(data, plot = False):
    t, y, w = data[:,0], data[:,1], data[:,2]
    # must be odd.
    windowSizeSec = 6
    N = windowSizeSec*32.0
    window = np.ones(N)/N
    try:
        smoothW = np.convolve(w, window, 'valid')
    except Exception as e:
        print('[WARN] Can not convolve with window size %s' % windowSizeSec)
        smoothW = w 
    if plot:
        pylab.subplot(2, 1, 1)
        pylab.plot(t, w, linewidth=0.5, label = "W")

    # Shift because of convolution.
    x = int(N) / 2
    bT, yy = t[x-1:-x], w[x-1:-x] - smoothW

    if plot:
        pylab.plot(bT, smoothW, linewidth=2, label = "Smooth W")
        pylab.legend()
        pylab.subplot(2, 1, 2)

    win = np.ones(2) / 2.0
    yy = np.convolve(yy, win, 'same')
    yy = (yy + np.fabs(yy))
    if plot:
        pylab.plot(bT, yy, linewidth=1, alpha=0.4, label = "W - Smooth W")
        pylab.legend()

    # Find blink in this data.
    blinks = []
    while yy.max() > 10.0:
        i = np.argmax(yy)
        isBlink, a = get_blink(i, yy, 8.0)
        if isBlink:
            blinks.append((i, a))

    xvec, yvec = [], []
    for i, x in sorted(blinks):
        xvec.append(bT[i])
        yvec.append(x)
    return xvec, yvec

def process_csv(csv_file):
    data = np.genfromtxt(csv_file, skiprows=1, delimiter=",")
    d = data #[:1000,:]
    blinkA = find_blinks_using_edge(d)
    print("Total blink using edges: %s" % len(blinkA[0]))
    blinkB = find_blinks_using_pixals(d)
    print("Total blinks using pixals: %s" % len(blinkB[0]))
    pylab.plot(blinkA[0], 1+np.zeros(len(blinkA[0])), '+', lw = 10)
    pylab.plot(blinkB[0], 0.1+np.ones(len(blinkB[0])), '+', lw = 10)
    pylab.legend()
    pylab.ylim(0.6, 1.5)
    pylab.show()


def main():
    csvFile = sys.argv[1]
    process_csv(csvFile)

if __name__ == '__main__':
    main()
