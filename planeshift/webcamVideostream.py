import time
from threading import Thread
from cv2 import cv2

class WebcamVideoStream:
	def __init__(self):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture("/dev/video2")

		self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
		self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
		self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
		self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
		# self.stream.set(cv2.CAP_PROP_FPS, 30)

		self.fps = 0

		(self.grabbed, self.frame) = self.stream.read()
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

		self.start()

	def set(self, property, value):
		self.stream.set(property, value)

	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			start = time.time()

			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

			elapsed = time.time() - start
			self.fps = int(1/elapsed)

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True