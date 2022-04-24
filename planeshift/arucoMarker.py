import json
import numpy as np
import cv2
from enum import Enum

class MarkerType(Enum):
    TOPLEFT = 0
    TOPRIGHT = 1
    BOTTOMRIGHT = 2
    BOTTOMLEFT = 3

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

CALIBRATION_MARKER_TYPES = [MarkerType.TOPLEFT, MarkerType.TOPRIGHT, MarkerType.BOTTOMRIGHT, MarkerType.BOTTOMLEFT]

class ArucoMarker:

    MARKER_SIZE = 1.94 # cm
    # MARKER_SIZE = 1.34 # cm
    BORDER_SIZE = 0.3  # cm

    def __init__(self, id: int, polygon: np.array):
        self.id = id
        self._polygon = polygon

        self._transformed_polygon = None

        self._inner_size = (np.linalg.norm(self.inner_tl() - self.inner_tr()),
                            np.linalg.norm(self.inner_tl() - self.inner_bl()))

        self._size = (np.linalg.norm(self.tl() - self.tr()) + self.border_width()*2,
                      np.linalg.norm(self.tl() - self.bl()) + self.border_height()*2)

    def to_json(self):
        data = {
            "id": int(self.id),
            # "coordinates": self.tvec().tolist()
        }

        return json.dumps(data)

    def is_calibration_marker(self):
        return MarkerType(self.id) in CALIBRATION_MARKER_TYPES

    def draw(self, image, mtx, dist):
        cv2.aruco.drawDetectedMarkers(image, np.array([[self.inner_polygon()]]).astype('float32'))
        # cv2.aruco.drawAxis(image, mtx, dist, self.rvec, self.tvec(), 5)  # Draw axis

    def transform(self, transformation_matrix: np.array):
        return ArucoMarker(self.id, cv2.perspectiveTransform(np.array([self._polygon.astype('float32')]), transformation_matrix).astype(int)[0])

    def transformed(self):
        return self._transformed_polygon

    def inner_tl(self):
        return self._polygon[0]

    def inner_tr(self):
        return self._polygon[1]

    def inner_br(self):
        return self._polygon[2]

    def inner_bl(self):
        return self._polygon[3]

    def inner_center(self):
        xs = [p[0] for p in self.inner_polygon()]
        ys = [p[1] for p in self.inner_polygon()]
        return np.array([np.sum(xs)/len(xs), np.sum(ys)/len(ys)])

    def tl(self):
        return self._polygon[0] + [-self.border_width(), -self.border_height()]

    def tr(self):
        return self._polygon[1] + [self.border_width(), -self.border_height()]

    def br(self):
        return self._polygon[2] + [self.border_width(), self.border_height()]

    def bl(self):
        return self._polygon[3] + [-self.border_width(), self.border_height()]

    def centroid(self):
        polygon_x = [p[0] for p in self.inner_polygon()]
        polygon_y = [p[1] for p in self.inner_polygon()]
        centroid = np.array(
            [sum(polygon_x) / len(self.inner_polygon()),
             sum(polygon_y) / len(self.inner_polygon())],
            dtype=int)

        return centroid

    def border_width(self):
        return ((self._inner_size[0] * self.BORDER_SIZE / self.MARKER_SIZE)).astype(int)

    def border_height(self):
        return (self._inner_size[1] * (self.BORDER_SIZE / self.MARKER_SIZE)).astype(int)

    def inner_size(self):
        return self._inner_size

    def size(self):
        return self._size

    def inner_polygon(self):
        return self._polygon

    def polygon(self):
        return np.array([self.tl(), self.tr(), self.br(), self.bl()])