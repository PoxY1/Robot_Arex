import depthai as dai
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.signal as ss
import math
import csv
import pandas as pd
from imucal.management import load_calibration_info


def windowPutText(window, x_pos,y_pos,title,val1,val2,val3):
    wyswietlane_x = "X: " + str(val1)
    wyswietlane_y = "Y: " + str(val2)
    wyswietlane_z = "Z: " + str(val3)


    cv2.putText(window, title, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(window, wyswietlane_x, (x_pos, y_pos+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(window, wyswietlane_y, (x_pos, y_pos+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(window, wyswietlane_z, (x_pos, y_pos+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return 0

def readFile(filename):
    data = []  # empty list
    with open(filename) as file:
        reader = csv.reader(file)
        for line in reader:
            data.append(line)
    return data
first = True
freq = 400
Ts = 1/freq

displayPeriod = 0.1
medSize = 11

loaded_cal_info = load_calibration_info('imu1/Ferraris/imu1_2020-08-12_13-21.json')




timeStart = round(time.time(),3)
timeNow = round(time.time(),1) #Do nadawania czestotliwosci wyswietlania odczytow z IMU

accelerationPlotValue = [0.0]
accelerationTimestamp = [0.0]
accelerationFiltered = [0.0]*3
predkosc = [0.0]*3
polozenie = [0.0]*3

accelerationFilterWindow = np.zeros((medSize,3),dtype=float)

#accelerationFilterWindow = [[0.0]*3]*11
#accelerationFilterWindow = np.append(accelerationFilterWindow,[[1.0,2.0,3.0]],0)
#accelerationFilterWindow = np.delete(accelerationFilterWindow,1,0)


#print(accelerationFilterWindow)
#print(accelerationFilterWindow)
#print("Okrojona wartosc",accelerationFilterWindow[:,0])

with dai.Pipeline() as pipeline:
    imu = pipeline.create(dai.node.IMU)

    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, freq)
    imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, freq)

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


            #print(acceleration_vector)
            #print()
            #print()

            key = cv2.waitKey(1)
            if key == ord('q'):
                pipeline.stop()
                break

        acceleration_vector = [acceleroValues.x, acceleroValues.y, acceleroValues.z]
        gyroscope_vector = [gyroValues.x, gyroValues.y, gyroValues.z]
        #IMU_pack = [acceleration_vector, gyroscope_vector]
        if first:
            IMU_dataPack = pd.DataFrame({'gyr_x': [gyroValues.x], 'gyr_y': [gyroValues.y], 'gyr_z': [gyroValues.z],
                                         'acc_x': [acceleroValues.x],'acc_y': [acceleroValues.y], 'acc_z': [acceleroValues.z]})
        # else:
        #     IMU_dataPack.append({'gyr_x': [gyroValues.x], 'gyr_y': [gyroValues.y], 'gyr_z': [gyroValues.z],
        #                          'acc_x': [acceleroValues.x],'acc_y': [acceleroValues.y], 'acc_z': [acceleroValues.z]})

        calibrated_data = loaded_cal_info.calibrate_df(IMU_dataPack, "m/s^2", "deg/s")

        print(calibrated_data)
        accelerationFilterWindow = np.append(accelerationFilterWindow, [acceleration_vector], 0)
        accelerationFilterWindow = np.delete(accelerationFilterWindow, 0, 0)

        # accelerationFiltered[0] = ss.medfilt(accelerationFilterWindow[:,0], kernel_size=medSize)
        # accelerationFiltered[1] = ss.medfilt(accelerationFilterWindow[:,1], kernel_size=medSize)
        # accelerationFiltered[2] = ss.medfilt(accelerationFilterWindow[:,2], kernel_size=medSize)

        accelerationFiltered[0] = sum(accelerationFilterWindow[:,0])/medSize
        accelerationFiltered[1] = sum(accelerationFilterWindow[:,1])/medSize
        accelerationFiltered[2] = sum(accelerationFilterWindow[:,2])/medSize
        #print("Okno:",accelerationFilterWindow[:,0])
        #print("Przefiltrowane:",accelerationFiltered[0])

        # predkosc[0] += round(float(imuF.format(acceleroValues.x)) * Ts, 3)
        # predkosc[1] += round(float(imuF.format(acceleroValues.y)) * Ts, 3)
        # predkosc[2] += round(float(imuF.format(acceleroValues.z)) * Ts, 3)
        predkosc[0] += accelerationFiltered[0] * Ts
        predkosc[1] += accelerationFiltered[1] * Ts
        predkosc[2] += accelerationFiltered[2] * Ts


        #print(f"Predkosc X: {predkosc_x} Y: {predkosc_y} Z: {predkosc_z}")

        # Polozenie blednie obliczane - znalezc przyczyne nieodpowiedniego przeskalowania
        polozenie[0] += predkosc[0] * Ts
        polozenie[1] += predkosc[1] * Ts
        polozenie[2] += predkosc[2] * Ts
        #print(f"Polozenie X: {polozenie_x} Y: {polozenie_y} Z: {polozenie_z}")


        accelerationPlotValue.append(acceleration_vector[0])
        accelerationTimestamp.append(round(time.time(),3) - timeStart)

        if round(time.time(),1) > timeNow+displayPeriod:
            timeNow = round(time.time(), 1)

            okno_danych = np.zeros((800, 800, 3), dtype="uint8")

            #windowPutText(okno_danych, 30,20,"Akcelerometr:",acceleration_vector[0],acceleration_vector[1],acceleration_vector[2])
            windowPutText(okno_danych, 30, 20, "Akcelerometr:", accelerationFiltered[0], accelerationFiltered[1], accelerationFiltered[2])

            windowPutText(okno_danych, 30, 120, "Predkosc:", predkosc[0], predkosc[1], predkosc[2])

            windowPutText(okno_danych, 30, 220, "Polozenie:", polozenie[0], polozenie[1], polozenie[2])

            cv2.imshow("IMU Data", okno_danych)