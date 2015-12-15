#!/bin/bash
if [ $# -lt 1 ]; then
    echo "USAGE: $0 filename"
    exit
fi
python ./blinky.py -f $1
