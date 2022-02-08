# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import src._init_paths

import logging
import os
import os.path as osp
import cv2
import time
import torch
import numpy as np
import uuid

from src.lib.opts import opts
from src.lib.tracking_utils import visualization as vis
from src.lib.tracker.multitracker import JDETracker
from src.lib.tracking_utils.log import logger

logger.setLevel(logging.INFO)

def pre_process(img0, width=1088, height=608):

    def letterbox(img, height=608, width=1088,
                color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, ratio, dw, dh

    img0 = cv2.resize(img0, (width, height))
    # Padded resize
    img, _, _, _ = letterbox(img0, height=height, width=width)

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    return img0, img

def video(opt, videocapture, res, show_image=True, frame_rate=30, write_video_out = True, video_out_fname = None):
    width, height = res
    tracker = JDETracker(opt, frame_rate=frame_rate)

    results = []
    frame_id = 0
    while True:
        # print("Reading frame")
        ret, img0 = videocapture.read()
        if not ret:
            break
        # img0 = cv2.rotate(img0, cv2.cv2.ROTATE_90_CLOCKWISE)

        # print("Network preprocessing and inference...wait")
        start = time.time()
        img0, img = pre_process(img0, width=width, height=height)
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        # print("Network preprocessing and inference: done")

        elapsed = (time.time() - start)
        fps = 1. / elapsed
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps;{})'.format(frame_id, fps, elapsed))

        if frame_id > 100 and not write_video_out:
            logger.info("Debug stop")
            break
    
        # save results
        # results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image is not None or write_video_out:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=fps)
        if show_image:
            cv2.imshow('online_im', online_im)
            q = cv2.waitKey(1)
            if q == ord('q'):
                return
        if write_video_out:
            cv2.imwrite(video_out_fname.format(frame_id), online_im)

        frame_id += 1



def demo(opt):
    # The resolution should be multiples of 32. 
    # 864x480 and 576x320 are OK. 
    # The results of 864x480 do not decrease much and it can get 40+ FPS.
    
    cap = cv2.VideoCapture(video_file)
    _, frame = cap.read()
    w,h,_ =frame.shape
    res = (576,320)

    folder_name = os.path.join("/video_out", str(uuid.uuid4()))
    os.makedirs(folder_name)

    fname = os.path.join(folder_name, "{0}.jpg")

    video(
        opt, 
        cap, 
        res, 
        show_image=False,
        write_video_out=True,
        video_out_fname=fname)



if __name__ == '__main__':
    import sys
    video_file = sys.argv[1]
    sys.argv.pop(1)

    [sys.argv.append(s) for s in "mot --load_model models/all_dla34.pth --conf_thres 0.4".split(" ")]
    opt = opts().init()
    demo(opt)
