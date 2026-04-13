#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from pathlib import Path

color = (0, 255, 0)

# Create pipeline
pipeline = dai.Pipeline()
# Config
# topLeft = dai.Point2f(0.4, 0.4)
# bottomRight = dai.Point2f(0.6, 0.6)
topLeft = dai.Point2f(0.2, 0.2)
bottomRight = dai.Point2f(0.3, 0.3)

# Define sources and outputs
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

# Linking
monoLeftOut = monoLeft.requestOutput((1280, 720))
monoRightOut = monoRight.requestOutput((1280, 720))
monoLeftOut.link(stereo.left)
monoRightOut.link(stereo.right)

stereo.setRectification(True)
stereo.setExtendedDisparity(True)

stepSize = 0.05

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 10
config.depthThresholds.upperThreshold = 10000
calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)


xoutSpatialQueue = spatialLocationCalculator.out.createOutputQueue()
outputDepthQueue = spatialLocationCalculator.passthroughDepth.createOutputQueue()

stereo.depth.link(spatialLocationCalculator.inputDepth)


inputConfigQueue = spatialLocationCalculator.inputConfig.createInputQueue()

with pipeline:
    pipeline.start()
    while pipeline.isRunning():


        spatialData = xoutSpatialQueue.get().getSpatialLocations()

        print("Use WASD keys to move ROI!")
        outputDepthIMage : dai.ImgFrame = outputDepthQueue.get()

        frameDepth = outputDepthIMage.getCvFrame()
        frameDepth = outputDepthIMage.getFrame()
        print("Median depth value: ", np.median(frameDepth))

        depthFrameColor = cv2.normalize(frameDepth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        for depthData in spatialData:


            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            if int(depthData.spatialCoordinates.z) > 600:
                xmin = int(roi.topLeft().x)+20
                ymin = int(roi.topLeft().y)+20
                xmax = int(roi.bottomRight().x)-20
                ymax = int(roi.bottomRight().y)-20
            else:
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)


            depthMin = depthData.depthMin
            depthMax = depthData.depthMax

            fontType = cv2.FONT_HERSHEY_TRIPLEX

            # if int(depthData.spatialCoordinates.z) > 800:
            #     topLeft = dai.Point2f(0.1, 0.1)
            #     bottomRight = dai.Point2f(0.15, 0.15)
            # else:
            #     topLeft = dai.Point2f(0.2, 0.2)
            #     bottomRight = dai.Point2f(0.30, 0.30)

            # cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            # if int(depthData.spatialCoordinates.z) > 600:
            #     cv2.rectangle(depthFrameColor, (xmin+20, ymin+20), (xmax-20, ymax-20), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            # else:
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)


            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)




        # Show the frame
        cv2.imshow("depth", depthFrameColor)

        key = cv2.waitKey(1)
        if key == ord('q'):
            pipeline.stop()
            break

        stepSize = 0.05

        newConfig = False

        if key == ord('q'):
            break
        elif key == ord('w'):
            if topLeft.y - stepSize >= 0:
                topLeft.y -= stepSize
                bottomRight.y -= stepSize
                newConfig = True
        elif key == ord('a'):
            if topLeft.x - stepSize >= 0:
                topLeft.x -= stepSize
                bottomRight.x -= stepSize
                newConfig = True
        elif key == ord('s'):
            if bottomRight.y + stepSize <= 1:
                topLeft.y += stepSize
                bottomRight.y += stepSize
                newConfig = True
        elif key == ord('d'):
            if bottomRight.x + stepSize <= 1:
                topLeft.x += stepSize
                bottomRight.x += stepSize
                newConfig = True
        elif key == ord('1'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
            print('Switching calculation algorithm to MEAN!')
            newConfig = True
        elif key == ord('2'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MIN
            print('Switching calculation algorithm to MIN!')
            newConfig = True
        elif key == ord('3'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MAX
            print('Switching calculation algorithm to MAX!')
            newConfig = True
        elif key == ord('4'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MODE
            print('Switching calculation algorithm to MODE!')
            newConfig = True
        elif key == ord('5'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
            print('Switching calculation algorithm to MEDIAN!')
            newConfig = True

        if newConfig:
            #config.roi = dai.Rect(topLeft, bottomRight)
            if int(depthData.spatialCoordinates.z) > 600:

                topLeft.x = topLeft.x + 0.1
                bottomRight.x = bottomRight.x - 0.1
                topLeft.y = topLeft.y - 0.1
                bottomRight.y = bottomRight.y + 0.1



                config.roi = dai.Rect(topLeft, bottomRight)
            else:
                config.roi = dai.Rect(topLeft, bottomRight)

            config.calculationAlgorithm = calculationAlgorithm
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            inputConfigQueue.send(cfg)
            if int(depthData.spatialCoordinates.z) > 600:
                topLeft.x = topLeft.x - 0.1
                bottomRight.x = bottomRight.x + 0.1
                topLeft.y = topLeft.y + 0.1
                bottomRight.y = bottomRight.y - 0.1
            newConfig = False
