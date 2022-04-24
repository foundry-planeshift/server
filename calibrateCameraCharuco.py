import numpy as np
from cv2 import cv2
import cv2.aruco as aruco
import json

from planeshift.webcamVideostream import WebcamVideoStream

# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 16
CHARUCOBOARD_COLCOUNT = 30
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_1000)

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=25,
        markerLength=19,
        dictionary=ARUCO_DICT)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 0, 0)
line_type = 2

if __name__ == "__main__":

    corners_all = []  # Corners discovered in all images processed
    ids_all = []  # Aruco ids corresponding to corners discovered
    image_size = (1920, 1080)

    capture = WebcamVideoStream()

    nr_of_calibration_images_wanted = 60
    images_captured = 0
    frame_nr = 0
    avg_img = np.zeros((1080, 1920, 3))
    avg_num = 10
    while images_captured < nr_of_calibration_images_wanted:
        img = capture.read()
        if img is None:
            continue
        
        frame_nr += 1

        # if frame_nr % avg_num == 0:
            
            # avg_img = np.divide(avg_img, avg_num)
            # img = avg_img.astype('uint8')


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # if frame_nr % 30 == 0:
        # Find aruco markers in the query image
        corners, ids, rejected = aruco.detectMarkers(
            image=gray,
            dictionary=ARUCO_DICT)

        response = 0
        if len(corners) > 0 or ids is not None:

            # Outline the aruco markers found in our query image
            img = aruco.drawDetectedMarkers(
                image=img,
                corners=corners)

            # Get charuco corners and ids from detected aruco markers
            response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=CHARUCO_BOARD)

            # If a Charuco board was found, let's collect image/corner points
            # Requiring at least 20 squares
            if response > 150:
                if frame_nr % 1 == 0:
                    # Add these corners and ids to our calibration arrays
                    corners_all.append(charuco_corners)
                    ids_all.append(charuco_ids)
                    images_captured += 1

                    # print(f"Camera image {images_captured}/{nr_of_calibration_images_wanted}.")

                # Draw the Charuco board we've detected to show our calibrator the board was properly detected
                img = aruco.drawDetectedCornersCharuco(
                    image=img,
                    charucoCorners=charuco_corners,
                    charucoIds=charuco_ids)

            # print(f"\rMarkers detected: {response} | Camera image {images_captured}/{nr_of_calibration_images_wanted}.", end='')

        image_text = f"Markers detected: {len(corners)} | Camera image {images_captured}/{nr_of_calibration_images_wanted}"
        text_size, _ = cv2.getTextSize(image_text, font, font_scale, line_type)
        text_w, text_h = text_size
        cv2.rectangle(img, (0, 0), (text_w, 70), (255, 255, 255), -1)
        cv2.putText(img, image_text, (0, 30), font, font_scale, font_color, line_type)
        cv2.putText(img, f"FPS: {capture.fps}", (0, 60), font, font_scale, font_color, line_type)

        cv2.imshow('img',img)
        cv2.waitKey(1)


    calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print(f"Distortion: {distCoeffs}")
    print(f"Camera matrix: {cameraMatrix}")

    json_export = {
        "distortion": distCoeffs.tolist(),
        "camera_matrix": cameraMatrix.tolist()
    }

    with open('calibration/calibration.json', 'w') as outfile:
        json.dump(json_export, outfile)

    print("Exported calibration.json")

    cv2.destroyAllWindows()

