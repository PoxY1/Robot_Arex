#!/usr/bin/env python3
import depthai as dai
import time
import math
def timeDeltaToMilliS(delta) -> float:
    return delta.total_seconds()*1000

# Create pipeline

freq = 100
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

with dai.Pipeline() as pipeline:
    # Define sources and outputs
    imu = pipeline.create(dai.node.IMU)

    # enable ACCELEROMETER_RAW at 500 hz rate
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, freq)
    # enable GYROSCOPE_RAW at 400 hz rate
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, freq)
    # it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
    # above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
    imu.setBatchReportThreshold(1)
    # maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
    # if lower or equal to batchReportThreshold then the sending is always blocking on device
    # useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
    imu.setMaxBatchReports(10)

    imuQueue = imu.out.createOutputQueue(maxSize=50, blocking=False)

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

            acceleroTs = acceleroValues.getTimestamp()
            gyroTs = gyroValues.getTimestamp()

            imuF = "{:.06f}"
            tsF  = "{:.03f}"

            print(f"Accelerometer timestamp: {acceleroTs}")
            print(f"Latency [ms]: {dai.Clock.now() - acceleroValues.getTimestamp()}")
            latency = dai.Clock.now() - acceleroValues.getTimestamp()
            print(f"Accelerometer [m/s^2]: x: {imuF.format(acceleroValues.x)} y: {imuF.format(acceleroValues.y)} z: {imuF.format(acceleroValues.z)}")
            #print(f"Gyroscope timestamp: {gyroTs}")
            print(f"Gyroscope [rad/s]: x: {imuF.format(gyroValues.x)} y: {imuF.format(gyroValues.y)} z: {imuF.format(gyroValues.z)} ")

            orientacja_x = orientacja_x + float(imuF.format(gyroValues.x))
            orientacja_y = orientacja_y + float(imuF.format(gyroValues.y))
            orientacja_z = orientacja_z + float(imuF.format(gyroValues.z))

            #print(f"X: {orientacja_x} Y: {orientacja_y} Z: {orientacja_z}")
            opoznienie = float(latency.microseconds)/1000000

            predkosc_x = predkosc_x + (float(imuF.format(acceleroValues.x)) + 9.05) * opoznienie
            predkosc_y = predkosc_y + float(imuF.format(acceleroValues.y)) * opoznienie
            predkosc_z = predkosc_z + float(imuF.format(acceleroValues.z)) * opoznienie


            #Polozenie blednie obliczane - znalezc przyczyne nieodpowiedniego przeskalowania
            polozenie_x = polozenie_x + predkosc_x * opoznienie
            polozenie_y = polozenie_y + predkosc_y * opoznienie
            polozenie_z = polozenie_z + predkosc_z * opoznienie
            print(f"X: {polozenie_x} Y: {polozenie_y} Z: {polozenie_z}")


            x_kwadrat = float(imuF.format(acceleroValues.x))*float(imuF.format(acceleroValues.x))
            y_kwadrat = float(imuF.format(acceleroValues.y))*float(imuF.format(acceleroValues.y))
            z_kwadrat = float(imuF.format(acceleroValues.z))*float(imuF.format(acceleroValues.z))
            #print(f"X: {x_kwadrat} Y: {y_kwadrat} Z: {z_kwadrat}")
            grawitacja = math.sqrt(x_kwadrat + y_kwadrat + z_kwadrat)
            print(grawitacja)
            print()
            print()