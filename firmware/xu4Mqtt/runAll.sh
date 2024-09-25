#!/bin/bash
#
sleep 60
kill $(pgrep -f 'python3 nanoReader.py 0')
sleep 5
python3 nanoReader.py 0 &
sleep 5
kill $(pgrep -f 'python3 GPSReader.py')
sleep 5
python3 GPSReader.py &
sleep 5
python3 ipReader.py
sleep 5

Start-Process python GPSReader.py
Start-Process python ipReader.py