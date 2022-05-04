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
    SELECT_BATTLEAREA = 2

MODE_NAMES = {
    Mode.CALIBRATION: "calibration",
    Mode.DETECTION: "detection",
    Mode.SELECT_BATTLEAREA: "select battlearea"
}

class PlaneShift:

    def __init__(self, device, debug=False):

        self.device = device
        self.debug = debug

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
        self.aruco_detector_params.cornerRefinementMethod = 0

        self.aruco_markers = {}

        self._calibration_markers = None
        self._roi_selected = Value(c_bool, False)

        self.capture = None

        self._mp_mode = Value('i', Mode.SELECT_BATTLEAREA.value)
        # self._mp_mode = Value('i', Mode.DETECTION.value)

        self._all_tokens = {}
        self._mp_all_tokens = self.process_manager.dict()
        self._mp_player_token_locations = self.process_manager.list()

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

    def find_markers(self, image: np.array) -> dict:

        detected = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_detector_params)

        if len(detected[0]) == 0:
            return {}

        detected_markers = [cv2.UMat.get(d) for d in detected[0]]
        ids = cv2.UMat.get(detected[1])
        # detected_markers = [d for d in detected[0]]
        # ids = detected[1]

        aruco_markers = {}
        for i in range(0, len(detected[0])):
            (polygon, id) = [detected_markers[i].astype(int), ids[i]]

            marker = ArucoMarker(id[0], polygon[0])
            aruco_markers[id[0]] = marker

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


    def select_roi(self, debug_image=None) -> bool:
        markers = self.find_calibration_markers(self.all_tokens().values())
        if markers is None or len(markers) < 4:
            return False

        self._calibration_markers = markers
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

        self.set_mode(Mode.DETECTION)

        return True

    def calibration_marker(self, token: MarkerType):
        return self._calibration_markers[token.value]

    def find_calibration_markers(self, tokens):
        return sorted([t for t in tokens if MarkerType.has_value(t.id)], key=lambda x: x.id)

    def find_player_tokens(self, tokens):
        player_keys =  set(tokens.keys()) - set(CALIBRATION_MARKER_TYPES)
        return [tokens.get(k) for k in player_keys]


    def draw_tokens(self, image, tokens, color=(0, 255, 255)):
        for token in tokens:
            cv2.polylines(image, np.array([token.inner_polygon()], dtype=np.int), True, color=color, thickness=5)

    def draw_roi_area(self, image, tokens, roi_selected):
        markers = self.find_calibration_markers(tokens)
        if markers is None or len(markers) < 4:
            return False

        self._calibration_markers = markers
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

        # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
        # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)

        while True:
            mode = Mode(self._mp_mode.value)
            if mode == Mode.CALIBRATION:
                self._calibration(capture)
            elif mode == Mode.DETECTION:
                self._detection(capture)
            elif mode == Mode.SELECT_BATTLEAREA:
                self._select_battlearea(capture)

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

            self._set_original_image(annotated_image)

            if self._roi_selected.value:
                warped_image = cv2.warpPerspective(umat_image, self.warp_matrix(), self._warp_size)
                warped_image_shape = cv2.UMat.get(warped_image).shape

                warped_annotated_image = copy(warped_image.get())

                tokens = self.find_markers(warped_image)
                with self._lock:
                    self._mp_all_tokens.update(tokens)

                markers = self.find_player_tokens(tokens)
                self.draw_tokens(warped_annotated_image, markers)

                warped_markers = [t.transform(np.linalg.inv(self.warp_matrix())) for t in markers]
                self.draw_tokens(annotated_image, warped_markers)

                self.draw_roi_area(annotated_image, markers, self._roi_selected.value)

                if self.debug:
                    resized = cv2.resize(annotated_image, (1200, 900))
                    cv2.imshow("Original image", resized)

                if len(markers) > 0:
                    self._mp_player_token_locations[:] = []
                    for marker in markers:

                        centroid = marker.centroid()

                        relative_centroid = np.round(np.divide(centroid, np.flip(warped_image_shape[:2])), 4)

                        string = f"ID: {marker.id} Position: ({relative_centroid})"

                        data = {
                            "id": int(marker.id),
                            "coordinates": relative_centroid.tolist()
                        }
                        self._mp_player_token_locations.append(json.dumps(data))

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
        image_size = (1920, 1080)

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

            # img = cv2.UMat(img)

            frame_nr += 1

            # if frame_nr % avg_num == 0:

                # avg_img = np.divide(avg_img, avg_num)
                # img = avg_img.astype('uint8')


            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # if frame_nr % 30 == 0:
            # Find aruco markers in the query image
            corners, ids, rejected = aruco.detectMarkers(
                image=gray,
                dictionary=self.aruco_dict)

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
            imageSize=image_size,
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

    def _select_battlearea(self, capture):
        print("PlaneShift: Entering mode 'select battlearea'")

        fps = 0
        while Mode(self._mp_mode.value) == Mode.SELECT_BATTLEAREA:

            start = time.time()

            image = capture.read()[1]
            if image is None:
                continue

            umat_image = cv2.UMat(image)

            annotated_image = copy(umat_image.get())

            tokens = self.find_markers(umat_image)

            with self._lock:
                self._mp_all_tokens.update(tokens)

            calibration_tokens = [t for t in tokens.values() if MarkerType.has_value(t.id)]

            putBorderedText(annotated_image, f"FPS: {fps}", (0, 25), self.font)

            self.draw_tokens(annotated_image, calibration_tokens)
            self.draw_roi_area(annotated_image, tokens.values(), self._roi_selected.value)

            if self.debug:
                resized = cv2.resize(annotated_image, (1200, 900))
                cv2.imshow("Original image", resized)

            self._set_original_image(annotated_image)

            if self.debug:
                if cv2.waitKey(1) > 0:
                    print("Selecting battle area")
                    self.select_roi()

            elapsed = time.time() - start
            fps = int(1/elapsed)

        print("PlaneShift: Exiting mode 'select battlearea'")

if __name__ == '__main__':
    planeshift = PlaneShift()
    planeshift.run()