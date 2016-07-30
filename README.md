First frame of video/camera is presented to the user to select the box where
your eyes it. Use mouse to left-click at a point, drag the cursor to another
point, and release the left-button. You won't see a box unless you are done
dragging. Once box is visible press `q`. The process will start.

# blinky.py

Process a given video for blinks. Very fast.

1. python blinky.py -f video_file.webm
2. Select a region on frame where eye it.
3. Let it run till video is over. A `csv` files will be produced with blink
   location. Two algorithms are used: one based on edge detection and other one
   is based on couting dark-pixals.


# blinky_gui.py

Gui version of blinky.py. Only for testing and demo purposes. Very slow.


# blinky_webcam.py 

Process the live feed from given camera index ( default 0 ). Only for demo
purpose. 

## Dependencies

- gnuplotlib. Use the command `sudo pip install gnuplotlib` 
- gnuplot

