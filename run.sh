#!/bin/bash

python3 trt_ssd.py --model ssd_mobilenet_v2_coco --rtsp "rtsp://admin:admin_8856@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

