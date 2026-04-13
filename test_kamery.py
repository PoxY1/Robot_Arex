#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np


def configure_cam(cam, size_x: int, size_y: int, fps: float):
    cap = dai.ImgFrameCapability()
    cap.size.fixed((size_x, size_y))
    cap.fps.fixed(fps)


    return cam.requestOutput(cap, True)


pipeline = dai.Pipeline()
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)
#xout = pipeline.create(dai.node.XLinkOut)
#xout.setStreamName("depth")
monoLeftOut = configure_cam(monoLeft, 640, 400, 60)
monoRightOut = configure_cam(monoRight, 640, 400, 60)




# Linking
#monoLeftOut = monoLeft.requestFullResolutionOutput()
#monoRightOut = monoRight.requestFullResolutionOutput()

#monoLeftOut = monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
#monoRightOut = monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# monoLeftOut = pipeline.create(dai.node.MonoCamera)
# monoLeftOut.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
#
# monoRightOut = pipeline.create(dai.node.MonoCamera)
# monoRightOut.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

monoLeftOut.link(stereo.left)
monoRightOut.link(stereo.right)

stereo.setRectification(True)
stereo.setExtendedDisparity(True)
stereo.setLeftRightCheck(True)

disparityQueue = stereo.disparity.createOutputQueue()

colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
colorMap[0] = [0, 0, 0]  # to make zero-disparity pixels black

with pipeline:
    pipeline.start()
    maxDisparity = 100
    while pipeline.isRunning():
        disparity = disparityQueue.get()


        # intrinsics = disparity.getTransformation().getSourceIntrinsicMatrix()
        # print('Focal length in pixels:', intrinsics[0][0])


        assert isinstance(disparity, dai.ImgFrame)
        npDisparity = disparity.getFrame()
        maxDisparity = max(maxDisparity, np.max(npDisparity))
        colorizedDisparity = cv2.applyColorMap(((npDisparity / maxDisparity) * 255).astype(np.uint8), colorMap)
        cv2.imshow("disparity", colorizedDisparity)

        cv2.circle(colorizedDisparity, (200, 200), 50, (255, 255, 255), 10)
        key = cv2.waitKey(1)

        if key == ord('q'):
            pipeline.stop()
            break
        if key == ord('='):
            maxDisparity = maxDisparity + 100
            print(maxDisparity)
        if key == ord('-'):
            maxDisparity = maxDisparity - 100
            print(maxDisparity)