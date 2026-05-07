#!/usr/bin/env python3
import depthai as dai
import cv2
import numpy as np
import time
import math
def timeDeltaToMilliS(delta) -> float:
    return delta.total_seconds()*1000


def windowPutText(x_pos,title,val1,val2,val3):
    wyswietlane_x = "X: " + str(val1)
    wyswietlane_y = "Y: " + str(val2)
    wyswietlane_z = "Z: " + str(val3)


    cv2.putText(okno_danych, title, (x_pos, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(okno_danych, wyswietlane_x, (x_pos, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(okno_danych, wyswietlane_y, (x_pos, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(okno_danych, wyswietlane_z, (x_pos, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return 0

# Create pipeline
spowolnienie_wyswietlania = 0
spowolnienie_wyswietlania_max = 10
freq = 400
Ts = 1/freq

orientacja_x = 0.0
orientacja_y = 0.0
orientacja_z = 0.0
predkosc_x = 0.0
predkosc_y = 0.0
predkosc_z = 0.0
polozenie_x = 0.0
polozenie_y = 0.0
polozenie_z = 0.0

okno_danych = np.zeros((300,300,3), dtype="uint8")

with dai.Pipeline() as pipeline:
    # Define sources and outputs
    imu = pipeline.create(dai.node.IMU)

    # enable ACCELEROMETER_RAW at 500 hz rate
    #imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, freq) ##Stary link z akcelerometrem
    # enable GYROSCOPE_RAW at 400 hz rate
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, freq)

    #imu.enableIMUSensor(dai.IMUSensor.GRAVITY, freq) ## do testa

    imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, freq)

    # it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
    # above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
    imu.setBatchReportThreshold(1)
    # maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
    # if lower or equal to batchReportThreshold then the sending is always blocking on device
    # useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
    imu.setMaxBatchReports(10)

    imuQueue = imu.out.createOutputQueue(maxSize=50, blocking=False)
    pipeline.stop()
    pipeline.start()
    baseTs = None
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
            #g_values = imuPacket.gravity
            acceleroTs = acceleroValues.getTimestamp()
            gyroTs = gyroValues.getTimestamp()
            rotationTs = rotationVector.getTimestamp()

            #imuF = "{:.06f}"
            imuF = "{:.06f}"
            tsF  = "{:.03f}"

            #print(f"Accelerometer timestamp: {acceleroTs}")
            #print(f"Latency [ms]: {dai.Clock.now() - acceleroValues.getTimestamp()}")
            latency = dai.Clock.now() - acceleroValues.getTimestamp()
            print(f"Accelerometer linear [m/s^2]: x: {imuF.format(acceleroValues.x)} y: {imuF.format(acceleroValues.y)} z: {imuF.format(acceleroValues.z)}")
            print(f"Raw: {acceleroValues.x}")
            #print(f"Gyroscope timestamp: {gyroTs}")
            #print(f"Gyroscope [rad/s]: x: {imuF.format(gyroValues.x)} y: {imuF.format(gyroValues.y)} z: {imuF.format(gyroValues.z)} ")
            #print(f"Rotation vector: accuracy: {imuF.format(rotationVector.rotationVectorAccuracy)} i: {imuF.format(rotationVector.i)} j: {imuF.format(rotationVector.j)} k: {imuF.format(rotationVector.k)} ")

            orientacja_x = orientacja_x + float(imuF.format(gyroValues.x))
            orientacja_y = orientacja_y + float(imuF.format(gyroValues.y))
            orientacja_z = orientacja_z + float(imuF.format(gyroValues.z))

            #print(f"X: {orientacja_x} Y: {orientacja_y} Z: {orientacja_z}")
            opoznienie = float(latency.microseconds)/1000000

            predkosc_x = round(predkosc_x + float(imuF.format(acceleroValues.x)) * Ts,3)
            predkosc_y = round(predkosc_y + float(imuF.format(acceleroValues.y)) * Ts,3)
            predkosc_z = round(predkosc_z + float(imuF.format(acceleroValues.z)) * Ts,3)
            print(f"Predkosc X: {predkosc_x} Y: {predkosc_y} Z: {predkosc_z}")

            #Polozenie blednie obliczane - znalezc przyczyne nieodpowiedniego przeskalowania
            polozenie_x = round(polozenie_x + predkosc_x * Ts,5)
            polozenie_y = round(polozenie_y + predkosc_y * Ts,5)
            polozenie_z = round(polozenie_z + predkosc_z * Ts,5)
            print(f"Polozenie X: {polozenie_x} Y: {polozenie_y} Z: {polozenie_z}")
            polozenie = [polozenie_x, polozenie_y, polozenie_z]

            #x_kwadrat = float(imuF.format(acceleroValues.x))*float(imuF.format(acceleroValues.x))
            #y_kwadrat = float(imuF.format(acceleroValues.y))*float(imuF.format(acceleroValues.y))
            #z_kwadrat = float(imuF.format(acceleroValues.z))*float(imuF.format(acceleroValues.z))
            #print(f"X: {x_kwadrat} Y: {y_kwadrat} Z: {z_kwadrat}")
            #grawitacja = math.sqrt(x_kwadrat + y_kwadrat + z_kwadrat)
            #print(grawitacja)

            key = cv2.waitKey(1)
            if key == ord('q'):
                pipeline.stop()
                break

            print()
            print()
        if(spowolnienie_wyswietlania > spowolnienie_wyswietlania_max): #Funkcja do wyświetlania co 10go odczytu
            okno_danych = np.zeros((300, 800, 3), dtype="uint8")

            #polozenie_wyswietlane = "Polozenie X: " + str(polozenie_x) + " Y: " + str(polozenie_y) + " Z: " + str(polozenie_z)

            windowPutText(50, "Polozenie", polozenie_x,polozenie_y,polozenie_z)

            windowPutText(200, "Predkosc", predkosc_x, predkosc_y, predkosc_z)

            windowPutText(350, "Przyspieszenie",imuF.format(acceleroValues.x), imuF.format(acceleroValues.y), imuF.format(acceleroValues.z))

            cv2.imshow("IMU Data", okno_danych)
            spowolnienie_wyswietlania = 0
        spowolnienie_wyswietlania += 1