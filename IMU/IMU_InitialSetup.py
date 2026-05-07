import depthai as dai
import cv2
import numpy as np
import time

freq = 100
Ts = 1/freq

with dai.Pipeline() as pipeline:
    imu = pipeline.create(dai.node.IMU)

    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, freq)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, freq)

    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    imuQueue = imu.out.createOutputQueue(maxSize=50, blocking=False)

    pipeline.start()

    while pipeline.isRunning():
        try:
            imuData = imuQueue.get()
        except KeyboardInterrupt:
            break
        assert isinstance(imuData, dai.IMUData)
        imuPackets = imuData.packets

        for imuPacket in imuPackets:
            acceleroValues = imuPacket.acceleroMeter
            gyroValues = imuPacket.gyroscope
            rotationVector = imuPacket.rotationVector

            acceleroTs = acceleroValues.getTimestamp()
            gyroTs = gyroValues.getTimestamp()
            rotationTs = rotationVector.getTimestamp()

            imuF = "{:.06f}"
            tsF  = "{:.03f}"

        okno_danych = np.zeros((300, 800, 3), dtype="uint8")
        cv2.imshow("IMU Data", okno_danych)

        key = cv2.waitKey(1)
        if key == ord('q'):
            pipeline.stop()
            break

        if round(time.time(), 1) > round(timeDisplay,1) + 0.2:
            #print("Czas trwania: ", round(time.time()-timeStart,1))
            timeDisplay = time.time()

        acceleration_vector = [acceleroValues.x, acceleroValues.y, acceleroValues.z]
        gyroscope_vector = [gyroValues.x, gyroValues.y, gyroValues.z]