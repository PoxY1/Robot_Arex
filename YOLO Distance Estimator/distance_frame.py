import cv2
import depthai as dai
import config_pipeline
config, topLeft_distFrame, bottomRight_distFrame, stereo, monoLeft, monoRight, pipeline = config_pipeline.configure()
windowResolution = [1280, 720]
#yoloResolution = [640, 640]
yoloResolution = windowResolution
def configure(windowRes, yoloRes):
    stepSize = 0.05
    windowResolution = windowRes
    yoloResolution = yoloRes
    #color_distFrame = (255, 255, 255) #bialy
    color_distFrame = (0, 255, 0) #zielony

    #topLeft_distFrame = dai.Point2f(0.0, 0.0)
    #bottomRight_distFrame = dai.Point2f(0.1, 0.1)

    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

    calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
    #calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MODE


    spatialLocationCalculator.inputConfig.setWaitForMessage(False)
    spatialLocationCalculator.initialConfig.addROI(config)


    xoutSpatialQueue = spatialLocationCalculator.out.createOutputQueue()
    outputDepthQueue = spatialLocationCalculator.passthroughDepth.createOutputQueue()

    stereo.depth.link(spatialLocationCalculator.inputDepth)

    inputConfigQueue = spatialLocationCalculator.inputConfig.createInputQueue()


    return xoutSpatialQueue, outputDepthQueue, calculationAlgorithm, inputConfigQueue
def calculate(spatialData):
    for depthData in spatialData:
        roi = depthData.config.roi
        #roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
        xmin = int(roi.topLeft().x)
        ymin = int(roi.topLeft().y)
        xmax = int(roi.bottomRight().x)
        ymax = int(roi.bottomRight().y)
    return depthData
#pipeline.start()
def run():
    xoutSpatialQueue, outputDepthQueue, calculationAlgorithm, inputConfigQueue = configure(windowResolution,
                                                                                           yoloResolution)
    if pipeline.isRunning():
    #def distanceMeasure(x_top,y_top,x_bottom,y_bottom):
        topLeft_distFrame = dai.Point2f(windowResolution[0] / 2, windowResolution[1] / 2)
        bottomRight_distFrame = dai.Point2f(windowResolution[0] / 2.5, windowResolution[1] / 2.5)
        config.roi = dai.Rect(topLeft_distFrame, bottomRight_distFrame)
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.addROI(config)
        inputConfigQueue.send(cfg)

        spatialData = xoutSpatialQueue.get().getSpatialLocations()
        outputDepthImage: dai.ImgFrame = outputDepthQueue.get()
        frameDepth = outputDepthImage.getCvFrame()
        frameDepth = outputDepthImage.getFrame()

        depthData = calculate(spatialData)

        print(f"Z: {int(depthData.spatialCoordinates.z)} mm")


        config.roi = dai.Rect(topLeft_distFrame, bottomRight_distFrame)
        #config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.addROI(config)
        inputConfigQueue.send(cfg)
    else:
        pipeline.start()

for i in range(100):
    run()