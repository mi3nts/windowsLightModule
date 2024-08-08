#!/bin/bash
#
sleep 60
kill $(pgrep -f 'python3 nanoReader.py 0')
sleep 5
python3 nanoReader.py 0 &
sleep 5
kill $(pgrep -f 'python3 nanoReader.py 1')
sleep 5
python3 nanoReader.py 1 &
sleep 5
kill $(pgrep -f 'python3 nanoReader.py 2')
sleep 5
python3 nanoReader.py 2 &
sleep 5
kill $(pgrep -f 'ips7100ReaderV1.py')
sleep 5
python3 ips7100ReaderV1.py &
sleep 5
kill $(pgrep -f 'python3 ozoneQLMDuoReader.py 0')
sleep 5
python3 ozoneQLMDuoReader.py 0 &
sleep 5
kill $(pgrep -f 'python3 ozoneQLMDuoReader.py 1')
sleep 5
python3 ozoneQLMDuoReader.py 1 &
sleep 5
kill $(pgrep -f 'python3 rainReader.py')
sleep 5
python3 rainReader.py &
sleep 5
kill $(pgrep -f 'python3 GPSReader.py')
sleep 5
python3 GPSReader.py &
sleep 5
python3 ipReader.py
sleep 5