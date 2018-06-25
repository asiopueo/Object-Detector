import cv2
import imutils

camera = cv2.VideoCapture(0)

while True:
	(grabber, frame) = camera.read()
	#frame = imutils.resize(frame, width=600)

	cv2.imshow('Frame', frame)
	cv2.waitKey(1)


camera.release()
cv2.destroyAllWindows()







