import json
import time

import numpy as np
import cv2
# from cv2 import cv2
from cv2 import aruco as aruco
from cv2 import cuda as cuda
from copy import copy

from ctypes import c_bool
from multiprocessing import Process, Array, Manager, Lock, Value, Event

from planeshift.arucoMarker import ArucoMarker, MarkerType, CALIBRATION_MARKER_TYPES
from planeshift.utils import putBorderedText

from enum import Enum

class Mode(Enum):
    CALIBRATION = 0
    DETECTION = 1

MODE_NAMES = {
    Mode.CALIBRATION: "calibration",
    Mode.DETECTION: "detection"
}

class PlaneShift:

    def __init__(self, device: str, resolution: tuple, debug: bool = False):

        self.device = device
        self.debug = debug
        self.resolution = resolution

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_color = (255, 255, 255)
        self.line_type = 2

        arr = np.zeros((3, 3), dtype='float64')
        arr.shape = arr.size
        self._warp_matrix = Array('d', arr)

        self.process_manager = Manager()
        self._lock = Lock()
        self._warp_size = self.process_manager.list()

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        self.aruco_detector_params = aruco.DetectorParameters_create()

        self.aruco_markers = {}

        self._mp_calibration_markers = self.process_manager.list()
        self._roi_selected = Value(c_bool, False)

        self.capture = None

        self._mp_mode = Value('i', Mode.DETECTION.value)

        self._all_tokens = {}
        self._mp_all_tokens = self.process_manager.list()
        self._mp_player_tokens = self.process_manager.list()

        self._stream_image_size = (480, 640, 3)
        self._mp_original_image = Array('B', self.create_array(self._stream_image_size, np.uint8))
        self._original_image = np.zeros(self._stream_image_size)

        self._mp_roi_image = Array('B', self.create_array(self._stream_image_size, np.uint8))
        self._roi_image = np.zeros(self._stream_image_size)
        
        self._camera_exposure = Value('i', 700)

    def set_mode(self, mode: Mode):
        mode_name = MODE_NAMES[mode]
        print(f"PlaneShift: Switching to mode {mode_name}")
        self._mp_mode.value = mode.value

    def load_camera_calibration(self, calibration_file):
        with open(calibration_file) as fr:
            c = json.load(fr)

            self._mtx = np.array(c['camera_matrix'])
            self._dist = np.array(c['distortion'])

    def set_camera_exposure(self, exposure):
        self._camera_exposure.value = exposure

    def original_image(self):
        mp_image = np.frombuffer(self._mp_original_image.get_obj(), dtype=np.uint8).reshape(self._stream_image_size)

        self._original_image = mp_image

        return self._original_image

    def _set_original_image(self, image):
        with self._lock:
            original_image = np.frombuffer(self._mp_original_image.get_obj(), dtype=np.uint8)
            original_image.shape = (480, 640, 3)
            resized_annotated_image = cv2.resize(image, (640, 480))
            original_image[...] = resized_annotated_image

    def roi_image(self):
        return np.frombuffer(self._mp_roi_image.get_obj(), dtype=np.uint8).reshape((480, 640, 3))

    def all_tokens(self):
        return self._mp_all_tokens

    def warp_matrix(self):
        return np.frombuffer(self._warp_matrix.get_obj(), dtype=float).reshape((3, 3))

    def set_warp_matrix(self, M):
        mp_M = np.frombuffer(self._warp_matrix.get_obj(), dtype='float64').reshape((3, 3))
        mp_M.shape = (3, 3)
        mp_M[...] = M.astype('float32')

    def create_array(self, size, dtype):
        image = np.zeros(size, dtype=dtype)
        image.shape = image.size
        return Array('B', image)

    def find_markers(self, image: np.array) -> list:

        detected = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_detector_params)

        if len(detected[0]) == 0:
            return []

        detected_markers = [cv2.UMat.get(d) for d in detected[0]]
        ids = cv2.UMat.get(detected[1])

        aruco_markers = []
        for i in range(0, len(detected[0])):
            (polygon, id) = [detected_markers[i].astype(int), ids[i]]

            marker = ArucoMarker(id[0], polygon[0])
            aruco_markers.append(marker)

        return aruco_markers

    def find_max_height_width(self, pts):
        (tl, tr, br, bl) = pts

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(tl - bl)
        height_b = np.linalg.norm(tr - br)
        max_height = int(max(height_a, height_b))

        return max_width, max_height


    def select_battlearea(self, debug_image=None) -> bool:
        markers = self.find_calibration_markers(self.all_tokens())
        if markers is None or len(markers) < 4:
            return False

        self._mp_calibration_markers[:] = markers
        topleft_qrcode, topright_qrcode, bottomright_qrcode, bottomleft_qrcode = markers

        pts = np.float32([topleft_qrcode.inner_tl(),
                                  topright_qrcode.inner_tr(),
                                  bottomright_qrcode.inner_br(),
                                  bottomleft_qrcode.inner_bl()])
        if debug_image is not None:
            cv2.polylines(debug_image, np.array([pts]).astype(int), True, color=(0, 255, 0), thickness=5)

        max_width, max_height = self.find_max_height_width(pts)

        dst = np.array([
            [0, 0],
            [max_width-1, 0],
            [max_width-1, max_height-1],
            [0, max_height-1]
        ], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M, _ = cv2.findHomography(pts, dst, cv2.RANSAC, 5.0)

        self.set_warp_matrix(M)
        self._warp_size[:] = [max_width, max_height]
        self._roi_selected.value = True

        return True

    def calibration_marker(self, token: MarkerType):
        return self._mp_calibration_markers[token.value]

    def find_calibration_markers(self, tokens):
        return sorted([t for t in tokens if MarkerType.has_value(t.id)], key=lambda x: x.id)

    def find_player_tokens(self, tokens):
        return [t for t in tokens if not MarkerType.has_value(t.id)]


    def draw_tokens(self, image, tokens, color=(0, 255, 255)):
        for token in tokens:
            cv2.polylines(image, np.array([token.inner_polygon()], dtype=np.int), True, color=color, thickness=5)

    def draw_roi_area(self, image, tokens, roi_selected):
        markers = self.find_calibration_markers(tokens)
        if markers is None or len(markers) < 4:
            return False

        topleft_qrcode, topright_qrcode, bottomright_qrcode, bottomleft_qrcode = markers

        pts = np.float32([topleft_qrcode.inner_tl(),
                                  topright_qrcode.inner_tr(),
                                  bottomright_qrcode.inner_br(),
                                  bottomleft_qrcode.inner_bl()])

        color = (0, 255, 0) if roi_selected else (0, 0, 255)
        cv2.polylines(image, np.array([pts]).astype(int), True, color=color, thickness=5)

    def start(self):
        self.process = Process(target=self.run).start()
        return self

    def run(self):
        print("Starting PlaneShift process")


        capture = cv2.VideoCapture(self.device)
        if not capture.isOpened():
            print(f"Cannot open camera {self.device}. Exiting.")
            exit()

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        capture.set(cv2.CAP_PROP_FOURCC, fourcc)

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        while True:
            mode = Mode(self._mp_mode.value)
            if mode == Mode.CALIBRATION:
                self._calibration(capture)
            elif mode == Mode.DETECTION:
                self._detection(capture)

    def _detection(self, capture):
        print("PlaneShift: Entering mode 'detection'")

        fps = 0
        while Mode(self._mp_mode.value) == Mode.DETECTION:

            start = time.time()

            image = capture.read()[1]
            if image is None:
                continue

            umat_image = cv2.UMat(image)

            annotated_image = copy(umat_image.get())

            putBorderedText(annotated_image, f"FPS: {fps}", (0, 25), self.font)

            all_tokens = self.find_markers(umat_image)
            self._mp_all_tokens[:] = all_tokens
            self.draw_tokens(annotated_image, all_tokens)

            calibration_markers = self.find_calibration_markers(all_tokens)
            self.draw_roi_area(annotated_image, calibration_markers, False)
            if self._mp_calibration_markers:
                self.draw_roi_area(annotated_image, self._mp_calibration_markers, self._roi_selected.value)

            self._set_original_image(annotated_image)

            if self.debug:
                resized = cv2.resize(annotated_image, (1200, 900))
                cv2.imshow("Original image", resized)

            if self._roi_selected.value:
                warped_image = cv2.warpPerspective(umat_image, self.warp_matrix(), self._warp_size)
                warped_image_shape = cv2.UMat.get(warped_image).shape

                warped_annotated_image = copy(warped_image.get())

                player_tokens = self.find_player_tokens(all_tokens)
                warped_markers = [t.transform(self.warp_matrix()) for t in player_tokens]

                self.draw_tokens(warped_annotated_image, warped_markers)

                self.draw_roi_area(annotated_image, warped_markers, self._roi_selected.value)



                if len(warped_markers) > 0:
                    self._mp_player_tokens[:] = []
                    for marker in warped_markers:

                        centroid = marker.centroid()

                        relative_centroid = np.round(np.divide(centroid, np.flip(warped_image_shape[:2])), 4)

                        string = f"ID: {marker.id} Position: ({relative_centroid})"

                        data = {
                            "id": int(marker.id),
                            "coordinates": relative_centroid.tolist()
                        }
                        self._mp_player_tokens.append(json.dumps(data))
                        # print(data)

                        putBorderedText(warped_annotated_image, string, marker.tl(), self.font)


                if self.debug:
                    resized_warped = cv2.resize(warped_annotated_image, (1200, 900))
                    cv2.imshow("Battle area", resized_warped)

                with self._lock:
                    roi_image = np.frombuffer(self._mp_roi_image.get_obj(), dtype=np.uint8)
                    roi_image.shape = (480, 640, 3)
                    resized_roi_image = cv2.resize(warped_annotated_image, (640, 480))
                    roi_image[...] = resized_roi_image

            elapsed = time.time() - start
            fps = int(1/elapsed)

            if self.debug:
                cv2.waitKey(1)

        print("PlaneShift: Exiting mode 'detection'")

    def _calibration(self, capture):
        print("PlaneShift: Entering mode 'calibration'")
        corners_all = []  # Corners discovered in all images processed
        ids_all = []  # Aruco ids corresponding to corners discovered

        CHARUCOBOARD_ROWCOUNT = 16
        CHARUCOBOARD_COLCOUNT = 30
        CHARUCO_BOARD = aruco.CharucoBoard_create(
            squaresX=CHARUCOBOARD_COLCOUNT,
            squaresY=CHARUCOBOARD_ROWCOUNT,
            squareLength=25,
            markerLength=19,
            dictionary=self.aruco_dict)

        nr_of_calibration_images_wanted = 60
        images_captured = 0
        frame_nr = 0
        while (Mode(self._mp_mode.value) == Mode.CALIBRATION) and (images_captured < nr_of_calibration_images_wanted):
            img = capture.read()[1]
            if img is None:
                continue

            frame_nr += 1

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find aruco markers in the query image
            corners, ids, rejected = aruco.detectMarkers(
                image=gray,
                dictionary=self.aruco_dict)

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
                if response > 230:
                    if frame_nr % 1 == 0:
                        # Add these corners and ids to our calibration arrays
                        corners_all.append(charuco_corners)
                        ids_all.append(charuco_ids)
                        images_captured += 1

                    # Draw the Charuco board we've detected to show our calibrator the board was properly detected
                    img = aruco.drawDetectedCornersCharuco(
                        image=img,
                        charucoCorners=charuco_corners,
                        charucoIds=charuco_ids)

            image_text = f"Markers detected: {len(corners)} | Camera image {images_captured}/{nr_of_calibration_images_wanted}"
            putBorderedText(img, image_text, (0, 30), self.font)
            # putBorderedText(img, f"FPS: {capture.fps}", (0, 60), self.font)

            self._set_original_image(img)
            if self.debug:
                resized = cv2.resize(img, (1920, 1080))
                cv2.imshow("Original image", resized)
                cv2.waitKey(1)



        calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=corners_all,
            charucoIds=ids_all,
            board=CHARUCO_BOARD,
            imageSize=self.resolution,
            cameraMatrix=None,
            distCoeffs=None
        )

        print(f"Distortion: {distCoeffs}")
        print(f"Camera matrix: {cameraMatrix}")

        # Load calibration parameters
        self._dist = distCoeffs
        self._mtx = cameraMatrix

        json_export = {
            "distortion": distCoeffs.tolist(),
            "camera_matrix": cameraMatrix.tolist()
        }

        with open('calibration/calibration.json', 'w') as outfile:
            json.dump(json_export, outfile)

        print("Exported calibration.json")

        self.set_mode(Mode.DETECTION)


if __name__ == '__main__':
    planeshift = PlaneShift()
    planeshift.run()