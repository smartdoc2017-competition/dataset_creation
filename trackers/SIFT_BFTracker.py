#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Imports
import logging

from AbstractPOITracker import *

import cv2

# ==============================================================================
logger = logging.getLogger(__name__)

# ==============================================================================
class SIFT_BFTracker(AbstractPOITracker):
    def __init__(self):
        param_SIFT_nf=0
        param_SIFT_no=10
        param_SIFT_ct=0.04
        param_SIFT_et=10.0
        param_SIFT_si=1.6
        detector = cv2.SIFT(nfeatures=param_SIFT_nf,
                            nOctaveLayers=param_SIFT_no,
                            contrastThreshold=param_SIFT_ct,
                            edgeThreshold=param_SIFT_et,
                            sigma=param_SIFT_si)

        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        super(SIFT_BFTracker, self).__init__(detector, 
                                          matcher, 
                                          num_pyrdown_model=1, 
                                          num_of_matches=15)

