from djitellopy import tello
import numpy as np
#import time
import cv2
from cv2 import aruco
import math
import time
import csv
import socket
import KeyPressModule as kp

# # get the cam.....
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# #
# #
# # # for tello
# def send_command(command):
#     sock.sendto(command.encode('utf-8'), ("192.168.10.1", 8889))
kp.init()
tello = tello.Tello()
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()
print(tello.get_battery())
# send_command("command")
# time.sleep(5)
# send_command("streamon")
# time.sleep(5)

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

calib_path = ""
camera_matrix = np.loadtxt(calib_path + 'cameraMatrix_webcam.txt', delimiter=',')
camera_distortion = np.loadtxt(calib_path + 'cameraDistortion_webcam.txt', delimiter=',')

# --- 180 deg rotation matrix around the x axis
R_flip = np.zeros((3, 3), dtype=np.float32)
R_flip[0, 0] = 1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0

def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    
    if kp.getKey("LEFT"): lr = -speed
    elif kp.getKey("RIGHT"): lr = speed

    if kp.getKey("UP"): fb = speed
    elif kp.getKey("DOWN"): fb = -speed

    if kp.getKey("w"): ud = speed
    elif kp.getKey("s"): ud = -speed

    if kp.getKey("a"): yv = speed
    elif kp.getKey("d"): yv = -speed

    if kp.getKey("SPACE"):  tello.land()
    if kp.getKey("1"):  tello.takeoff()
    if kp.getKey("0"):  tello.emergency()
    return [lr, fb, ud, yv]

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:

        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))

    return image


aruco_type = "DICT_4X4_100"

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters_create()
# cv2.CAP_PROP_FPS = 30
print(cv2.CAP_PROP_FPS)
#cap = frame_read.frame
#cap = cv2.VideoCapture('challengeA.mp4')

#cap = cv2.VideoCapture(0)

##from the Tello camera
#cap = cv2.VideoCapture("udp://@0.0.0.0:11111", cv2.CAP_FFMPEG)
#fps = cap.get(5)
#print('Frames per second : ', fps,'FPS')
# cap.set(cv2.CAP_PROP_FPS, 1)
#frame_count = cap.get(1)
#print('Frame count : ', frame_count)

# cap = cv2.VideoCapture('14.mp4')
# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#          print("The event is : " + event)
#cap.set(cv2.CAP_PROP_FPS, 60)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
header = ['Vx', 'Vy', 'Vz', 'pitch', 'roll', 'Yaw']
# open the file in the write mode
with open('example.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)
f.close()
# f = open('path/to/csv_file', 'w')
i = 0  # frame counter
frameTime = 0  # time of each frame in ms, you can add logic to change this value.
while True:
    i = i + 1  # increment counter
    if i % 3 == 0:  # display only one third of the frames, you can change this parameter according to your needs
       # ret, img = frame_read.frame
        img = frame_read.frame
        #h, w, _ = img.shape

        #width = 1000
        #height = int(width * (h / w))
        #img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        font = cv2.FONT_HERSHEY_PLAIN

        corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
        # print("corners:  ", corners)
        # print("ids:  ", ids)
        # print("rejected:  ", rejected)
        if ids is not None:
            # print("I am here!!!!!!!")
            ret = aruco.estimatePoseSingleMarkers(corners, 100, camera_matrix, camera_distortion)
            # print("ret", ret)
            # -- Unpack the output, get only the first
            rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

            # -- Draw the detected marker and put a reference frame over it
            aruco.drawDetectedMarkers(img, corners)
            aruco.drawAxis(img, camera_matrix, camera_distortion, rvec, tvec, 10)

            # -- Print the tag position in camera frame
            str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f angle=%4.0f" % (tvec[0], tvec[1], tvec[2], math.degrees((tvec[0])/(tvec[2])))
            cv2.putText(img, str_position, (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # -- Obtain the rotation matrix tag->camera
            R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
            R_tc = R_ct.T

            # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
            roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)

            # -- Print the marker's attitude respect to camera frame
            str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
                math.degrees(roll_marker), math.degrees(pitch_marker),
                math.degrees(yaw_marker))
            cv2.putText(img, str_attitude, (0, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # -- Now get Position and attitude f the camera respect to the marker
            pos_camera = -R_tc * np.matrix(tvec).T

            str_position = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f" % (pos_camera[0], pos_camera[1], pos_camera[2])
            cv2.putText(img, str_position, (0, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # -- Get the attitude of the camera respect to the frame
            roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_tc)
            str_attitude = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
                math.degrees(roll_camera), math.degrees(pitch_camera),
                math.degrees(yaw_camera))
            cv2.putText(img, str_attitude, (0, 250), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            #data = [tvec[0], tvec[1], tvec[2], math.degrees(pitch_marker),  math.degrees(roll_marker), math.degrees(yaw_marker)]
            #with open('example.csv', 'a', encoding='UTF8', newline='') as f:
               # writer = csv.writer(f)
                # write the header
                # writer.writerow(header)
                # write multiple rows
                #writer.writerow(data)
            # --- Display the frame

        detected_markers = aruco_display(corners, ids, rejected, img)
        cv2.imshow("Image", detected_markers)
        #cv2.imshow("Image", img)

        # header = ['Vx', 'Vy', 'Vz', 'pitch', 'roll', 'Yaw']



        # # create the csv writer
        # writer = csv.writer(f)
        #
        # # write a row to the csv file
        # writer.writerow(row)

        key = cv2.waitKey(1) & 0xFF
        vals = getKeyboardInput()
        tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])
       # sleep(0.05)
        if key == ord("q"):
            break
#send_command("streamoff")  # for tello

cap.release()
cv2.destroyAllWindows()
