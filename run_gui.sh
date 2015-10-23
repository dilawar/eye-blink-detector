#!/bin/bash
if [ $# -lt 1 ]; then
    echo "USAGE: $0 video_file"
    exit
fi
python ./blinky_gui.py -f "$1"
