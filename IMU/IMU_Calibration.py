import depthai as dai
import cv2
import numpy as np
import time

freq = 100
Ts = 1/freq
calibrationAxis = ["staticXpositive","staticXnegative","staticYpositive","staticYnegative","staticZpositive","staticZnegative","rotationX","rotationY","rotationZ"]
calibrationBool = False
calibrationNumber = 8
flaga = 0

timeStart = time.time()
timeDisplay = time.time()

#accValues = np.array
#gyroValues
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
        if key == ord('f'):
            calibrationBool = not calibrationBool
            print("Stan kalibracji:", calibrationBool)
            if calibrationBool:
                timeDisplay = time.time()
                timeStart = time.time()
                print("Rozpoczynanie kalibracji ", calibrationAxis[calibrationNumber])
                flaga = True


        if round(time.time(), 1) > round(timeDisplay,1) + 0.2 and calibrationBool:
            print("Czas trwania: ", round(time.time()-timeStart,1))
            timeDisplay = time.time()

        acceleration_vector = [acceleroValues.x, acceleroValues.y, acceleroValues.z]
        gyroscope_vector = [gyroValues.x, gyroValues.y, gyroValues.z]

        if not calibrationBool:
            if flaga:
                nazwaPliku = str(calibrationAxis[calibrationNumber]) + "_acce" + ".csv"
                #print(nazwaPliku)
                np.savetxt(nazwaPliku, accelerometerValues, delimiter=",")  # Eksport do pliku

                nazwaPliku = str(calibrationAxis[calibrationNumber]) + "_gyro" + ".csv"
                #print(nazwaPliku)
                np.savetxt(nazwaPliku, gyroscopeValues, delimiter=",")  # Eksport do pliku
                calibrationNumber += 1
                flaga = False
            accelerometerValues = np.zeros([1,3])
            gyroscopeValues = np.zeros([1,3])

        if calibrationBool:
            accelerometerValues = np.append(accelerometerValues, [acceleration_vector], axis=0)
            gyroscopeValues = np.append(gyroscopeValues,[gyroscope_vector], axis=0)

        #np.savetxt(calibrationAxis[calibrationNumber], acceleration_vector, delimiter=",")  # Eksport do pliku

        #print(acceleration_vector)