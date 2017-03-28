#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Imports
import logging

from Tracker import *
from utils.log import *

import cv2
import numpy as np

# ==============================================================================
logger = logging.getLogger(__name__)
initLogger(logger, debug=True)

# ==============================================================================
class AbstractPOITracker(Tracker):
    def __init__(self, detector, matcher,
                 num_pyrdown_model=0,
                 num_pyrdown_frames=0,
                 num_of_matches=15,
                 second_match_tresh=0.75):
        super(AbstractPOITracker, self).__init__(
                num_pyrdown_model=num_pyrdown_model,
                num_pyrdown_frames=num_pyrdown_frames)

        self.detector = detector
        self.matcher = matcher
        self.num_of_matches = num_of_matches
        self.second_match_tresh = second_match_tresh

    def reconfigureModel(self, tracker_model):
        # Clears the train descriptor collection.
        self.matcher.clear()

        Cimg = cv2.imread(tracker_model)
        Cimg = self._autoPyrDownModel(Cimg)
        Cgray = cv2.cvtColor(Cimg, cv2.COLOR_BGR2GRAY)
        (xmax, ymax) = (Cimg.shape[1], Cimg.shape[0])
        tl = (1, 1)
        bl = (1, ymax)
        br = (xmax, ymax)
        tr = (xmax, 1)
        self.mdl_quad = np.float32([tl, bl, br, tr])
        # print Cquad
        (Ckeyp,Cdesc) = self.detector.detectAndCompute(Cgray,None)
        # self.matcher.add(np.uint8([Cdesc]))
        self.matcher.add([Cdesc])
        self.mdl_keyp = Ckeyp

    def processFrame(self, frame):
        self._rejectCurrent = True
        img = frame
        # -----

        img = self._autoPyrDownFrame(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (keypoints,descriptors) = self.detector.detectAndCompute(gray,None)
        if descriptors is None:
            logger.debug("R no descriptors")
        else:
            # matches = self.matcher.knnMatch(np.uint8(descriptors), k = 2)
            matches = self.matcher.knnMatch(descriptors, k = 2)
            matches = [m[0] for m in matches if len(m) >= 2 and m[0].distance < m[1].distance * self.second_match_tresh]
            # print len(matches)
            if len(matches) < self.num_of_matches:
                logger.debug("R: not enough matches (%d < %d)", len(matches), self.num_of_matches)
            else:
                pt00 = [self.mdl_keyp[m.trainIdx].pt for m in matches]
                pt10 = [keypoints[m.queryIdx].pt for m in matches]
                pt0, pt1 = np.float32((pt00, pt10))
                H, s = cv2.findHomography(pt0, pt1, cv2.RANSAC, 3.0)

                s = s.ravel() != 0
                # print s.sum()
                if s.sum() < self.num_of_matches:
                    logger.debug("R: not enough RANSAC inliers (%d < %d, got %d matches before)", s.sum(), self.num_of_matches, len(matches))
                else:
                    pt0, pt1 = pt0[s], pt1[s]
                    q = cv2.perspectiveTransform(self.mdl_quad.reshape(1, -1, 2), H).reshape(-1, 2)

                    self._rejectCurrent = False
                    self._tl = Pt(name="tl", x=self._scaleCoord(q[0][0]), y=self._scaleCoord(q[0][1]))
                    self._bl = Pt(name="bl", x=self._scaleCoord(q[1][0]), y=self._scaleCoord(q[1][1]))
                    self._br = Pt(name="br", x=self._scaleCoord(q[2][0]), y=self._scaleCoord(q[2][1]))
                    self._tr = Pt(name="tr", x=self._scaleCoord(q[3][0]), y=self._scaleCoord(q[3][1]))
                    
        return (self._rejectCurrent, self._tl, self._bl, self._br, self._tr)
