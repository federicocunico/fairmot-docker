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
    def letterbox(
        img, height=608, width=1088, color=(127.5, 127.5, 127.5)
    ):  # resize a rectangular image to a padded rectangular
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (
            round(shape[1] * ratio),
            round(shape[0] * ratio),
        )  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(
            img, new_shape, interpolation=cv2.INTER_AREA
        )  # resized, no border
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # padded rectangular
        return img, ratio, dw, dh

    img0 = cv2.resize(img0, (width, height))
    # Padded resize
    img, _, _, _ = letterbox(img0, height=height, width=width)

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    return img0, img


def birdview(
    source_res, frame, bbox_tlwh_coords, bbox_id, inference_res, orig_frame=None
):
    t, l, w, h = bbox_tlwh_coords
    inference_width, inference_height = inference_res

    t = t / inference_width * source_res[1]
    w = w / inference_width * source_res[1]

    l = l / inference_height * source_res[0]
    h = h / inference_height * source_res[0]

    center_bot = (int(t + (w / 2)), int(l + h))
    if orig_frame is not None:
        cv2.circle(orig_frame, center_bot, 10, (255, 0, 0), -1)

    src = np.float32(center_bot).reshape(-1, 1, 2)
    pt = cv2.perspectiveTransform(src, H).reshape(-1)

    pt = int(round(pt[0])), int(round(pt[1]))

    # offsety = 850
    # offsetx = -100
    offsety = 0
    offsetx = 0
    color = vis.get_color(abs(int(bbox_id)))
    cv2.circle(frame, (pt[0], pt[1]), 10, color, -1)
    cv2.circle(frame, (pt[0], pt[1]), 10, (0, 0, 255))

    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (pt[0] + 8, pt[1] - 8)
    fontScale = 1
    fontColor = (0, 0, 255)
    thickness = 1
    lineType = 2

    cv2.putText(
        frame,
        str(bbox_id),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )


def video(
    opt,
    videocapture,
    res,
    show_image=True,
    frame_rate=30,
    write_video_out=False,
    video_out_fname=None,
    birdview_fname=None,
):
    width, height = res
    tracker = JDETracker(opt, frame_rate=frame_rate)

    results = []
    frame_id = 0
    while True:

        # birdview_frame = np.zeros((200, 500, 3), dtype=np.uint8) + 255
        birdview_frame = cv2.imread(plane_file)

        # print("Reading frame")
        ret, orig = videocapture.read()
        if not ret:
            break
        source_res = orig.shape
        img0 = orig.copy()

        ##### DEBUG
        # pts1 = np.load("../data/pts1__ice.npy")
        # pts2 = np.load("../data/pts2__ice.npy")
        # for j, pt in enumerate(pts1):
        #     cv2.circle(img0, (int(pt[0]), int(pt[1])), 20, (255,0,0), -1)
        #     cv2.circle(birdview_frame, (int(pts2[j][0]), int(pts2[j][1])), 20, (0,255,0), -1)
        #####

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

        elapsed = time.time() - start
        fps = 1.0 / elapsed
        if frame_id % 20 == 0:
            logger.info(
                "Processing frame {} ({:.2f} fps;{})".format(frame_id, fps, elapsed)
            )

        if frame_id > 100 and not write_video_out:
            logger.info("Debug stop")
            break

        for i, bb in enumerate(online_tlwhs):
            bb_id = online_ids[i]

            birdview(source_res, birdview_frame, bb, bb_id, res, orig)
            # birdview(img0, bb, bb_id)

        # save results
        # results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image is not None or write_video_out:
            online_im = vis.plot_tracking(
                img0, online_tlwhs, online_ids, frame_id=frame_id, fps=fps
            )
        if show_image:
            cv2.imshow("online_im", online_im)
            # cv2.imshow("orig", orig)
            cv2.imshow("bw", birdview_frame)
            q = cv2.waitKey(1)
            if q == ord("q"):
                return
        if write_video_out:
            cv2.imwrite(video_out_fname.format(frame_id), online_im)
            cv2.imwrite(birdview_fname.format(frame_id), birdview_frame)

        frame_id += 1


def demo(opt):
    # The resolution should be multiples of 32.
    # 864x480 and 576x320 are OK.
    # The results of 864x480 do not decrease much and it can get 40+ FPS.

    cap = cv2.VideoCapture(video_file)
    _, frame = cap.read()
    w, h, _ = frame.shape
    res = (576, 320)

    # cv2.namedWindow("bw", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("bw", 500, 500)

    # cv2.namedWindow("orig", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("orig", 500, 500)

    write_video_out = True

    if write_video_out:

        unique_name = str(uuid.uuid4())
        # frame_folder_name = os.path.join("..", "video_out", unique_name, "frames")
        # bview_folder_name = os.path.join("..", "video_out", unique_name, "bview")
        frame_folder_name = os.path.join("/video_out", unique_name, "frames")
        bview_folder_name = os.path.join("/video_out", unique_name, "bview")

        if not os.path.isdir(frame_folder_name):
            os.makedirs(frame_folder_name)

        if not os.path.isdir(bview_folder_name):
            os.makedirs(bview_folder_name)

        frame_fname = os.path.join(frame_folder_name, "{0}.jpg")
        bview_fname = os.path.join(bview_folder_name, "{0}.jpg")
    else:
        frame_fname = None
        bview_fname = None

    video(
        opt,
        cap,
        res,
        show_image=False,
        write_video_out=write_video_out,
        video_out_fname=frame_fname,
        birdview_fname=bview_fname,
    )


if __name__ == "__main__":
    import sys

    video_file = sys.argv[1]
    plane_file = sys.argv[2]
    h_file = sys.argv[3]
    sys.argv.pop(1)  # pop video file
    sys.argv.pop(1)  # pop plane file
    sys.argv.pop(1)  # pop h file

    H = np.load(h_file)
    assert H.shape == (3, 3), "Homography must be 3x3"

    [
        sys.argv.append(s)
        for s in "mot --load_model models/all_dla34.pth --conf_thres 0.4".split(" ")
    ]
    opt = opts().init()
    demo(opt)
