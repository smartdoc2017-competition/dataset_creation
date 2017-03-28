#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Imports
import logging
from collections import namedtuple

import cv2

# ==============================================================================
Pt = namedtuple("Pt", ["name", "x", "y"])
# validnames = ["tl","bl","br","tr"]
logger = logging.getLogger(__name__)

# ==============================================================================
def multiPyrDown(img, num_pyrdown=1):
    res = img
    for q in range(num_pyrdown):
        res = cv2.pyrDown(res)
    return res

# ==============================================================================
# TODO add counters to facilitate statistics accumulation
class Tracker(object):
    """
    Tracker is an interface for every tracker callable by `process_sample`.
    Additionally to the interface methods, it also provided utility methods for 
    child classes.

    The constructor of the concrete tracker classes (which will be effectively
    used to process samples) MUST be parameter-free and MUST properly call 
    Tracker's constructor.
    """
    # Interface
    # --------------------------------------------------------------------------

    def reconfigureModel(self, tracker_model): # TODO allow multiple models?
        """
        Tracker x str ---> None
        Will be called once before processing each test sequence with the appropriate
        model of the object to track.
        """
        pass

    def reinitFrameSize(self, frame_width, frame_height):
        """
        Tracker x int x int ---> None
        Will be called once before processing each test sequence with the appropriate
        size of the image frames which will be provided.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height


    def processFrame(self, frame_data):
        """
        Tracker x FrameData ---> tuple(rejected:bool, tl:models.models.Pt, bl:models.models.Pt, br:models.models.Pt, tr:models.models.Pt)
        Will be called once with each frame to process.
        You MUST override this method in child class.

        Frame data may contain more than raw image data.
        """
        raise NotImplementedError()

    def getName(self):
        """
        Tracker ---> str
        Return the name used to generate output directory for the results of this tracker.

        Overwrite this method if your tracker can be initialized with various configurations,
        so as to differentiate (and avoid to overwrite) the results produced.

        Example:
            return super(MyTracker, self).getName() + "_" + str(self._param1) + "_" + str(self._param2)
        """
        return self.__class__.__name__.split(".")[-1]

    # Utility methods
    # --------------------------------------------------------------------------
    def __init__(self, 
                 num_pyrdown_model=0,
                 num_pyrdown_frames=0):
        self._rejectCurrent = True
        self._tl = None
        self._bl = None
        self._br = None
        self._tr = None

        self._num_pyrdown_model  = num_pyrdown_model # could be dynamic according to frame size (to fit in frame's harmonics)
        self._num_pyrdown_frames = num_pyrdown_frames


    def _autoPyrDownModel(self, img):
        return multiPyrDown(img, self._num_pyrdown_model)

    def _autoPyrDownFrame(self, img):
        return multiPyrDown(img, self._num_pyrdown_frames)
            
    def _scaleCoord(self, coord):
        return coord * 2**self._num_pyrdown_frames


