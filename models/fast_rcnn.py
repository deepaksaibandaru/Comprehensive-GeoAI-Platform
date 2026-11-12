"""Compatibility export for the latest Breain.ai detection implementation.
The current unified factory implements the Fast R-CNN UI model through the
native Faster R-CNN detector configuration. The lower-level ROI FastRCNN
class from rcnn.py is also exported here for direct use.
"""
from .rcnn import FastRCNN
__all__ = ["FastRCNN"]
