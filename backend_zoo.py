# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
from database_module import create_table, insert_data, update, get_id

def entry_exit(filename, min_area = 500) : 
	create_table()
	# otherwise, we are reading from a video file
	vs = cv2.VideoCapture(filename)
	 
	# initialize the first frame in the video stream
	firstFrame = None
	prev = 0
	global t1, t2 # for time init and later seeing the difference in how long the room was occupied
	global id_
	# loop over the frames of the video
	while True:
		# grab the current frame and initialize the occupied/unoccupied
		# text
		ret, frame = vs.read()
		if not ret : 
			break
		text = "Unoccupied"
		occupied = 0

		# if the frame could not be grabbed, then we have reached the end
		# of the video
		if frame is None:
			break
	 
		# resize the frame, convert it to grayscale, and blur it
		frame = imutils.resize(frame, width=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
	 
		# if the first frame is None, initialize it
		if firstFrame is None:
			firstFrame = gray
			continue
		# compute the absolute difference between the current frame and
		# first frame
		frameDelta = cv2.absdiff(firstFrame, gray)
		thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	 
		# dilate the thresholded image to fill in holes, then find contours
		# on thresholded image
		thresh = cv2.dilate(thresh, None, iterations=2)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]

		# loop over the contours
		for c in cnts:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < min_area :  
				continue
			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			if occupied == 0 :
				occupied = 1 
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			text = "Occupied"
		

		# draw the text and timestamp on the frame
		cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
			(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		# show the frame and record if the user presses a key
		cv2.imshow("Security Feed", frame)
		#cv2.imshow("Thresh", thresh)
		#cv2.imshow("Frame Delta", frameDelta)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the `q` key is pressed, break from the lop
		if key == ord("q"):
			break
	 	
		if prev != occupied : 
			if occupied == 0 : 
				t2 = datetime.datetime.now()
				print("frame exit: ", t2)
				id_ = get_id()
				update(id_, t1, t2)
			else : 
				t1 = datetime.datetime.now()
				print("frame entry: ", t1)
				insert_data(t1, 0)
		prev = occupied # previous frames state 

	# cleanup the camera and close any open windows
	cv2.destroyAllWindows()