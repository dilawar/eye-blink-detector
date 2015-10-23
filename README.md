# blinky.py

Process a given video for blinks. Very fast.

1. python blinky.py -f video_file.webm
2. Select a region on frame where eye it.
3. Let it run till video is over. CVF files will be produced with blink
   location. Two algorithms are used: one based on edge detection and other one
   is based on couting dark-pixals.


# blinky_gui.py

Gui version of blinky.py. Only for testing. Very slow.

For more details see the [project
page](https://dilawar.github.io/eye-blink-detector).
