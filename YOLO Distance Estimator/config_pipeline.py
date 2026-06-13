import cv2
import depthai as dai
import numpy as np

def configure():
    pipeline = dai.Pipeline()

    topLeft_distFrame = dai.Point2f(0.0, 0.0)
    bottomRight_distFrame = dai.Point2f(0.1, 0.1)


    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 10
    config.depthThresholds.upperThreshold = 10000
    config.roi = dai.Rect(topLeft_distFrame, bottomRight_distFrame)

    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    #cam_rgb = pipeline.create(dai.node.Camera).build()
    stereo = pipeline.create(dai.node.StereoDepth)


    monoLeftOut = monoLeft.requestOutput((1280, 720))
    monoRightOut = monoRight.requestOutput((1280, 720))
    #cam_out = cam_rgb.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)

    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)

    stereo.setRectification(True)
    stereo.setExtendedDisparity(True)

    return config, topLeft_distFrame, bottomRight_distFrame, stereo, monoLeft, monoRight, pipeline
