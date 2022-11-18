import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math
from StateMachine.DroneObject import DroneObject
from djitellopy import tello

# --- Define Tag
id_to_find = 1
marker_size = 9.5  # - [cm]

tello = tello.Tello()
tello.connect()
print(tello.get_battery())
tello.takeoff()
tello.streamon()
frame_read = tello.get_frame_read()


#def Movment():
   # lr, fb, ud, yv = 0, 0, 0, 0
speed = 30
MIN_DISTANCE = 200

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

# --- Get the camera calibration path
calib_path = ""
camera_matrix = np.loadtxt(calib_path + 'cameraMatrix_webcam.txt', delimiter=',')
camera_distortion = np.loadtxt(calib_path + 'cameraDistortion_webcam.txt', delimiter=',')

# --- 180 deg rotation matrix around the x axis
R_flip = np.zeros((3, 3), dtype=np.float32)
R_flip[0, 0] = 1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
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


# Sends command for Take off and Land when it detects marker in desinated orientations
def StateTransition(orientation):
    if (abs(orientation) > 180):
        print("invalid orientation, orientation should be between -180 and 180")
        return

    if (abs(orientation) > 170):
        tello.takeoff()


    elif (abs(orientation) < 20):
        tello.land()
    return

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

# Draws HUD on the screen
def Draw(frame, Distance, angle, Center):
    cv2.putText(frame, ('Distance %d' % Distance), (600, 45), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255),
                2, cv2.LINE_AA)
    cv2.circle(frame, (int(Center[0]), int(Center[1])), 2, (255, 0, 0), thickness=1)

    cv2.putText(frame, ('Orientation %d' % angle), (600, 60), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0),
                2,
                cv2.LINE_AA)
    cv2.putText(frame, ('Center %d,%d' % Center), (600, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2,
                cv2.LINE_AA)

    return


# if __name__ == "__main__":
#    # Set up Drone object, and establish connection and streaming
#     drone = DroneObject()
#     drone.setup()
#     frame_read = drone.tello.get_frame_read()
#     time.sleep(5)
#
#     tello = tello.Tello()
#     tello.connect()
#     tello.streamon()
#     frame_read = tello.get_frame_read()


    # --- Define the aruco dictionary
aruco_type = "DICT_4X4_100"
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters_create()
    #aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    #parameters = aruco.DetectorParameters_create()
    # -- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

while True:

        # -- Read the camera frame
        frame = frame_read.frame
       # (H, W) = frame.shape[:2]
        H, W, _ = frame.shape
        # -- Convert in gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # -- remember, OpenCV stores color images in Blue, Green, Red

        # -- Find all the aruco markers in the image
        #corners, ids, rejected = cv2.aruco.detectMarkers(frame=gray, arucoDict, parameters=arucoParams,
#                                                     cameraMatrix=camera_matrix, distCoeff=camera_distortion)

        corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

        if ids is not None:  # when we detect something
            # -- ret = [rvec, tvec, ?]
            # -- array of rotation and position of each marker in camera frame
            # -- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
            # -- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame
            print(corners)
            ret = aruco.estimatePoseSingleMarkers(corners, 100, camera_matrix, camera_distortion)

            # -- Unpack the output, get only the first
            rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

            # -- Draw the detected marker and put a reference frame over it
            aruco.drawDetectedMarkers(frame, corners)
            aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)

            # -- Print the tag position in camera frame
            str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f" % (tvec[0], tvec[1], tvec[2])
            cv2.putText(frame, str_position, (0, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # -- Obtain the rotation matrix tag->camera
            R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
            R_tc = R_ct.T

            # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
            roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)

            # -- Print the marker's attitude respect to camera frame
            str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
                math.degrees(roll_marker), math.degrees(pitch_marker),
                math.degrees(yaw_marker))
            cv2.putText(frame, str_attitude, (0, 20), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # -- Now get Position and attitude f the camera respect to the marker
            pos_camera = -R_tc * np.matrix(tvec).T

            str_position = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f" % (pos_camera[0], pos_camera[1], pos_camera[2])
            cv2.putText(frame, str_position, (0, 80), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # -- Get the attitude of the camera respect to the frame
            roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_tc)
            str_attitude = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
                math.degrees(roll_camera), math.degrees(pitch_camera),
                math.degrees(yaw_camera))

            cv2.putText(frame, str_attitude, (0, 60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


            #Calculated data from orientation and position vectors that are required by the statemachine
            Distance = tvec[2]
            Angle_yaw = math.degrees(yaw_marker)
            Angle_pitch = math.degrees(pitch_marker)
            Angle_roll = math.degrees(roll_marker)
            #print (Angle)
            coordinates = tuple(corners[0])
            centerY = int((coordinates[0][0][1] + coordinates[0][2][1]) / 2)
            centerX = int((coordinates[0][0][0] + coordinates[0][2][0]) / 2)
            Center = (centerX, centerY)
            Tilt = math.degrees(pitch_marker) / 400 + 1

            if abs(Distance) > 200:
                #f_b_speed = int(speed)
                tello.send_rc_control(0, speed, 0 ,0)
            elif  abs(Distance) <= MIN_DISTANCE:
                u_d_speed = 0
                l_r_speed = 0
                f_b_speed = 0

            if tvec[0] > 0:
                tello.send_rc_control(speed, 0, 0, 0)
            elif tvec[0] < 0:
                tello.send_rc_control(-speed, 0, 0, 0)

            if tvec[1] > 0:
                tello.send_rc_control(0, 0, speed, 0)
            elif tvec[1] < 0:
                tello.send_rc_control(0, 0, -speed, 0)

            if Angle_pitch > 0:
                tello.send_rc_control(0, 0, 0, speed)
            elif Angle_pitch < 0:
                tello.send_rc_control(0, 0, 0, -speed)

            # detected_markers = aruco_display(corners, ids, rejected, frame)
            # center_x, center_y = detected_markers[0]
            # image, x_distance, y_distance, distance = detect_distance_from_image_center(image, center_x,
            #                                                                             center_y)
            #movment = Movment()
            #l_r_speed = int((speed * Center[0]) / (W // 2))
            #u_d_speed = int((speed * Center[1] / (H // 2)) * -1)

            # l_r_speed = int((MAX_SPEED * x_distance) / (W // 2))
            # u_d_speed = int((MAX_SPEED * y_distance / (H // 2)) * -1)

            # if abs(Distance) <= MIN_DISTANCE:
            #     u_d_speed = 0
            #     l_r_speed = 0
            # else:
            #     tello.send_rc_control(l_r_speed, 0, u_d_speed, 0)
            # drone.set_parameter(Center[0], Center[1], Distance, Tilt)
            # StateTransition(Angle)
            # drone.action()
            #Draw(frame, Distance, Angle, Center)
        # --- Display the frame
        detected_markers = aruco_display(corners, ids, rejected, frame)
        cv2.imshow("Image", detected_markers)

        #cv2.putText(frame, str(drone.state), (600, 90), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #cv2.imshow('frame', frame)

        # --- use 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('l'):
            time.sleep(3)
            tello.land()
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit()


