#!/usr/bin/env python
"""Extract blink data and visualize it using matplotlib.


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
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import extract
import sys
import cv2
import webcam
import os
import sys

######################################
# Initialize animation here 


data_ = np.zeros(shape=(1,3))
cap_ = None
box_ = []
fig_ = plt.figure()
fps_ = 0.0

axes_ = {}
lines_ = {}
ax1 = fig_.add_subplot(2, 1, 1)
ax2 = ax1.twinx()
ax3 = fig_.add_subplot(2, 1, 2)
ax4 = ax3.twinx()

# Inset for raw data.
save_video_ = False
writer_ = None
if save_video_:
    fig_ax_ = fig_.add_axes([.7, .55, .2, .2], axisbg='y')
    #writer_ = anim.writers['ffmpeg'](fps = 15)
else:
    cv2.namedWindow('image')

axes_ = { 'raw' : ax1, 'raw_twin' : ax2, 'blink' : ax3, 'blink_twin' : ax4 }
lines_["rawA"] = ax1.plot([], [], color='blue')[0]
lines_["rawB"] = ax2.plot([], [], color='red')[0]
lines_['blinkA'] = ax3.plot([], [], 's', color = 'blue')[0]
lines_['blinkB'] = ax4.plot([], [], 'p', color = 'red')[0]

time_template_ = 'Time = %.1f s'
time_text_ = fig_.text(0.05, 0.9, '', transform=axes_['blink'].transAxes)

tvec_ = []
y1_ = []
y2_ = []
args_ = None

def init():
    global axes_, lines_
    global box_, fps_
    global cap_, args_
    cap_ = cv2.VideoCapture(0)
    fps_ = cap_.get(cv2.cv.CV_CAP_PROP_FPS)
    if fps_ < 1:
        print('[WARN] failed to get FPS. Setting to 15')
        fps_ = 15 

    ret, fstFrame = cap_.read()
    box_ = webcam.get_bounding_box(fstFrame)
    cv2.destroyWindow('Bound_eye')
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return lines_.values()

def update_axis_limits(ax, x, y):
    xlim = ax.get_xlim()
    if x >= xlim[1]:
        ax.set_xlim(xlim[0], x+10)

    ylims = ax.get_ylim()
    if y >= ylims[1]:
        ax.set_ylim(ylims[0], y+1)

def animate(i):
    global data_
    global time_text_
    global box_
    global tvec_, y1_, y2_
    global cap_
    global fig_ax_

    t = float(i) / fps_
    ret, img = cap_.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (x0, y0), (x1, y1) = box_
    try:
        frame = img[y0:y1,x0:x1]
    except Exception as e:
        print('[WARN] Frame %s dropped' % i)
        return lines_.values(), time_text_

    if save_video_:
        fig_ax_.imshow(frame[::2,::2], interpolation='nearest')
    else:
        cv2.imshow('image', frame)
        cv2.waitKey(1)

    inI, outI, edge, pixal = webcam.process_frame(frame)
    cv2.imshow('Convex hull of eye', outI)

    tvec_.append(t); y1_.append(edge); y2_.append(pixal)
    update_axis_limits(axes_['raw'], t, edge)
    update_axis_limits(axes_['raw_twin'], t, pixal)

    lines_['rawA'].set_data(tvec_, y1_)
    lines_['rawB'].set_data(tvec_, y2_)
    
    #return lines_.values(), time_text_ 

    if i % int(fps_) == 0 and i > int(fps_)*5:
        data_ = np.array((tvec_, y1_, y2_)).T
        try:
            tA, bA = extract.find_blinks_using_edge(data_[:,:])
        except Exception as e:
            print('[WARN] Failed to detect blink data using egdes in frame %s' % i)
            tA, bA = [], []
        try:
            tB, bB = extract.find_blinks_using_pixals(data_[:,:])
        except Exception as e:
            print('[WARN] Failed to detect blink using pixals in frame %s' % i)
            tB, bB = [], []
        update_axis_limits(axes_['blink'], t, 1)
        update_axis_limits(axes_['blink_twin'], t, 1)
        lines_['blinkA'].set_data(tA, 0.9*np.ones(len(tA)))
        lines_['blinkB'].set_data(tB, np.ones(len(tB)))

    time_text_.set_text(time_template_ % t)
    return lines_.values(), time_text_

def get_blinks( ):
    global ani_, cap_
    global save_video_
    ani_ = anim.FuncAnimation(fig_
        , animate
        , interval = 1
        , init_func=init
        , blit = False
        )

    if save_video_:
        print("Writing to video file output.mp4")
        ani_.save('output.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
    plt.show( )

def main():
    global data_, args_
    try:
        get_blinks()
    except Exception as e:
        print('[ERR] Failed %s' % e)
    cap_.release()

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Detect eyeblink in live camera feed'''
    parser = argparse.ArgumentParser(description=description)
    class Args: pass 
    args = Args()
    parser.parse_args(namespace=args)
    args_ = vars(args)
    main()
