#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SmartDoc 2017 reference creation tool.
(c) 2017 L3i - Univ. La Rochelle
    joseph (dot) chazalon (at) univ-lr (dot) fr

Sample usage:
python create_reference.py -d \
  /path/to/dataset/screen01/ground-truth.png \
  /path/to/dataset/screen01/input.mp4 \
  /path/to/dataset/screen01

"""

# ==============================================================================
# Imports
import logging
import argparse
import os
import os.path
import sys
from time import time
import json

# ==============================================================================
import cv2
import numpy as np

# ==============================================================================
from utils.args import *
from utils.log import *

from trackers.SIFT_BFTracker import SIFT_BFTracker

# ==============================================================================
logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
PROG_VERSION = "1.0"
PROG_NAME = "SmartDoc17 Reference Creator"

# ==============================================================================
def main(argv):
    # Option parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Semi-automated reference frame detection and extraction for ICDAR 2017 SmartDoc competition.', 
        version=PROG_VERSION)

    parser.add_argument('-d', '--debug', 
        action="store_true", 
        help="Activate debug output.")

    # parser.add_argument('-f', '--force', 
    #     action="store_true", 
    #     default=False,
    #     help="Force overwriting of existing files in output directory.")

    parser.add_argument('ground_truth_image', 
        action=StoreValidFilePath,
        help="Path to ground truth image which will be used to detect document region in frames.")
    
    parser.add_argument('sample_video', 
        action=StoreValidFilePath,
        help='Path to video input sample in which the document will be detected, and from which the reference frame will be extracted.')

    parser.add_argument('output_dir', 
        action=StoreExistingOrCreatableDir,
        help="Path to output directory (will be created if it does not exist) were result files will be generated.")

    # -----------------------------------------------------------------------------
    args = parser.parse_args()
    initLogger(logger, debug=args.debug)

    # --------------------------------------------------------------------------
    # Definition of output paths
    out_path_log = os.path.join(args.output_dir, "create_reference.log")
    out_path_json = os.path.join(args.output_dir, "sample.json")
    out_path_frame_extracted = os.path.join(args.output_dir, "reference_frame_%02d_extracted.png")
    out_path_frame_extracted_viz = os.path.join(args.output_dir, "reference_frame_%02d_extracted_viz.png")
    out_path_frame_dewarped = os.path.join(args.output_dir, "reference_frame_%02d_dewarped.png")


    # init logger to store a log copy in output dir
    fh = logging.FileHandler(out_path_log)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    programHeader(logger, PROG_NAME, PROG_VERSION)
    dumpArgs(args, logger)

    # --------------------------------------------------------------------------
    # Prepare process
    logger.debug("Starting up")
    frames = cv2.VideoCapture(args.sample_video)

    # --------------------------------------------------------------------------
    # Load and initialize tracker
    logger.debug("Creating tracker...")
    tracker = SIFT_BFTracker()
    logger.debug("Tracker created.")

    logger.debug("Configuring tracker with model '%s'", args.ground_truth_image)
    tracker.reconfigureModel(args.ground_truth_image)
    logger.debug("Tracker model configuration complete.")



    win_name_gt = "Ground truth"
    cv2.namedWindow(win_name_gt, cv2.WINDOW_NORMAL)
    img_gt = cv2.imread(args.ground_truth_image)
    cv2.imshow(win_name_gt, img_gt)

    # Let's go
    # --------------------------------------------------------------------------
    logger.debug("--- Process started. ---")
    # --------------------------------------------------------------------------
    win_name_frame = 'Tracker output'
    cv2.namedWindow(win_name_frame, cv2.WINDOW_NORMAL)

    (frame_width, frame_height) = (None, None)
    detection_complete = False
    can_read_frame = True
    fidx = -1
    tl = bl = br = tr = None
    frame = None
    draw_mat = None
    while not detection_complete and can_read_frame:
        can_read_frame, frame = frames.read()
        fidx += 1
        if not can_read_frame:
            logger.error("End of stream (or read error) reached at frame %02d.", fidx)

        # res_data = [(frameid, framemat, coords)]
        draw_mat = frame.copy()
        (fh, fw, ch) = frame.shape

        # TODO remove?
        if (frame_width, frame_height) != (fw, fh):
            (frame_width, frame_height) = (fw, fh)
            logger.debug("Reinitializing tracker with frame size (w=%.3f; h=%.3f)" % (fw, fh))
            tracker.reinitFrameSize(frame_width, frame_height)

        (rejected, tl, bl, br, tr) = tracker.processFrame(frame)

        # TODO extract display function
        if not rejected:
            logger.info("frame %04d: A " 
                        "tl:(%4d,%4d) bl:(%4d,%4d) br:(%4d,%4d) tr:(%4d,%4d)",
                        fidx, tl.x, tl.y, bl.x, bl.y, br.x, br.y, tr.x, tr.y)
        else:
            logger.info("frame %04d: R", fidx)

        # viz
        if not rejected:
                dbgq = np.int32([[tl.x, tl.y],
                                 [bl.x, bl.y],
                                 [br.x, br.y],
                                 [tr.x, tr.y]])
                cv2.polylines(draw_mat, [dbgq], True, (0, 255, 0), 2)

                for pt in (tl, bl, br, tr):
                    cv2.putText(draw_mat, pt.name.upper(), (int(pt.x), int(pt.y)), cv2.FONT_HERSHEY_PLAIN, 2, (64, 255, 64), 2)
        else:
            cv2.circle(draw_mat, (frame_width/2, frame_height/2), 20, (0, 0, 255), 10)


        action_chosen = None
        logger.info("Press <q> to save and quit or <SPACE> to select next frame.")
        while action_chosen is None:
            cv2.imshow(win_name_frame, draw_mat)
            key = cv2.waitKey(20) # required otherwise display thread show nothing
            if key & 0xFF == ord('q'):
                action_chosen = "quit"
            if key & 0xFF == ord(' '):
                action_chosen = "continue"

        # TODO add <e> for edit mode

        if action_chosen == "quit":
            detection_complete = True


    # Output file
    results = {
        "target_image_shape": {"x_len": img_gt.shape[1], "y_len": img_gt.shape[0]},
        "input_video_shape": {"x_len": frame_width, "y_len": frame_height},
        "reference_frame_id": fidx,
        "object_coord_in_ref_frame": {
            "top_left": {"x": tl.x, "y": tl.y},
            "bottom_left": {"x": bl.x, "y": bl.y},
            "bottom_right": {"x": br.x, "y": br.y},
            "top_right": {"x": tr.x, "y": tr.y},
        }
    }


    with open(out_path_json, "wb") as output:
        json.dump(results, output, indent=2)

    # write output out_path_json
    logger.debug("SegResult file generated: %s" % out_path_json)

    # save ref. frame extracted
    cv2.imwrite(out_path_frame_extracted%fidx, frame)
    cv2.imwrite(out_path_frame_extracted_viz%fidx, draw_mat)

    # dewarp and save dewarped image
    shape_object = np.float32([[tl.x, tl.y],
                               [bl.x, bl.y],
                               [br.x, br.y],
                               [tr.x, tr.y]])
    shape_target = np.float32([[0, 0],
                               [0, img_gt.shape[0]-1],
                               [img_gt.shape[1]-1, img_gt.shape[0]-1],
                               [img_gt.shape[1]-1, 0]])

    trans = cv2.getPerspectiveTransform(shape_object, shape_target)
    frame_dewarped = cv2.warpPerspective(frame, trans, (img_gt.shape[1], img_gt.shape[0]))
    cv2.imwrite(out_path_frame_dewarped%fidx, frame_dewarped)

    # --------------------------------------------------------------------------
    logger.debug("--- Process complete. ---")
    # --------------------------------------------------------------------------
    logger.info("Press any key to exit.")
    cv2.waitKey()
    cv2.destroyWindow(win_name_frame)


# ==============================================================================
# ==============================================================================
if __name__ == "__main__":
    ret = main(sys.argv)
    if ret is not None:
        sys.exit(ret)
