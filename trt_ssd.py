"""trt_ssd.py

This script demonstrates how to do real-time object detection with
TensorRT optimized Single-Shot Multibox Detector (SSD) engine.
"""

import requests

import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization


WINDOW_NAME = 'TrtSsdDemo'
INPUT_HW = (300, 300)
SUPPORTED_MODELS = [
    'ssd_mobilenet_v1_coco',
    'ssd_mobilenet_v1_egohands',
    'ssd_mobilenet_v2_coco',
    'ssd_mobilenet_v2_egohands',
    'ssd_inception_v2_coco',
    'ssdlite_mobilenet_v2_coco',
]


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'SSD model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('-m', '--model', type=str,
                        default='ssd_mobilenet_v1_coco',
                        choices=SUPPORTED_MODELS)
    args = parser.parse_args()
    return args

def counting(clss):
    p = clss.count(1)
    bi = clss.count(2)
    return p, bi

def get_time():
    now = time.localtime()
    id_num = "%04d%02d%02d%02d%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    date_str = "%04d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday)
    time_str = "%02d:%02d" % (now.tm_hour, now.tm_min)
    
    return int(id_num), date_str, time_str, now.tm_min
        
def insert(table_name, item):
    url = 'https://g0iktfy78k.execute-api.ap-northeast-2.amazonaws.com/v1'
    obj = {
        'TableName': table_name,
        'Item' : {
            'id' : item[0],
            'date' : item[1],
            'time' : item[2],
            'people' : item[3],
            'cycle' : item[4]
        }
    }

    x = requests.post(url, json = obj)
    print(x, obj)

def loop_and_detect(cam, trt_ssd, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_ssd: the TRT SSD object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    second = 1
    count = 0
    ave_p = 0
    ave_bi = 0
    p_count = 0
    bi_count = 0
    
    id_num, date_str, time_str, m = get_time()
    prev_time = int(m / 10)

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        boxes, confs, clss = trt_ssd.detect(img, conf_th)

        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)

        count = count + 1
        if count >= int(fps) :
            count = 0
            second = second + 1
            p, bi = counting(clss)
            ave_p = ave_p + p
            ave_bi = ave_bi + bi
            print(p, bi, ave_p, ave_bi, p_count, bi_count)
        
        if second % 11 == 0 :
            second = 1
            p_count = p_count + ave_p / 10
            bi_count = bi_count + ave_bi / 3
            ave_p = ave_bi = 0

            id_num, date_str, time_str, m = get_time()
            
            print(id_num, date_str, time_str, m, prev_time)

        if prev_time != int(m / 10) :
            insert('seongbuk', [id_num, date_str, time_str, int(p_count), int(bi_count)])
            prev_time = int(m / 10)
            p_count = bi_count = ave_p = ave_bi = second = 0
            
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc

        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.model.split('_')[-1])
    trt_ssd = TrtSSD(args.model, INPUT_HW)

    open_window(
        WINDOW_NAME, 'Camera TensorRT SSD Demo',
        cam.img_width, cam.img_height)
    vis = BBoxVisualization(cls_dict)
    loop_and_detect(cam, trt_ssd, conf_th=0.3, vis=vis)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
