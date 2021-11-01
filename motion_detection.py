import cv2 as cv
import os
import numpy as np
import importlib
import imutils
import time

start_time = time.time()

def track(track):
    pass

# create trackbars
cv.namedWindow("Thresh", flags= cv.WINDOW_FREERATIO)
cv.createTrackbar( "Threshold", "Thresh", 500, 1000, track)

baseFrame = None

def getThreshold():
    return cv.getTrackbarPos("Threshold", "Thresh")


def grabFirstFrame(frame):
    global baseFrame
    if baseFrame is None:
        print('Grabbing first frame...')
        baseFrame = frame


def grabCurrentFrameForNextComparison(frame):
    global baseFrame
    baseFrame = frame

def detectMotion(frame):
    global baseFrame
    # compute the absolute difference between the current frame and
	# first frame
    print('Detecting motion...')
    if baseFrame is None:
        baseFrame = frame

    displayFrame = frame.copy()

    frameDelta = cv.absdiff(baseFrame, frame)
    thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]

    dilated = cv.dilate(thresh, None, iterations=3)

    contouring = dilated.copy()

    contours = cv.findContours(contouring, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

	# loop over the contours
    for c in contours:
        (x, y, w, h) = cv.boundingRect(c)
        print(cv.boundingRect(c))
        displayFrame = frame.copy()

		# if the contour is too small, ignore it
  
        if cv.contourArea(c) < getThreshold():
            continue
        # if the contour area is larger than our supplied --min-area , we’ll draw the bounding box 
        # surrounding the foreground and motion region
        cv.rectangle(displayFrame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv.putText(displayFrame, "Status: {}".format('Movement'), (10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2)
        cv.putText(thresh, "Threshold: {}".format(getThreshold()), (10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2)


        print('Definging bounding box...')

    #cv.drawContours(frame, contours, -1, (0,255,0), 3)
    canvas = np.hstack((thresh, frameDelta, displayFrame))
    cv.imshow('Threshold, FrameDelta, Source', canvas)
    #cv.imshow("Thresh", thresh)
    #cv.imshow("Delta", frameDelta)
    #cv.imshow("Feed", displayFrame)




def main():
    video = cv.VideoCapture(0)

    while video.isOpened():
        _, frame = video.read()
        if frame is None:
            break

        resized = cv.resize(frame, (640, 480), interpolation= cv.INTER_AREA)
        gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (3, 3), 0)
       
        detectMotion(gray)
        grabCurrentFrameForNextComparison(gray)

        if cv.waitKey(20) & 0xFF == ord("s"):
            cv.destroyAllWindows()
            print("Initializing selection...")
            if cv.waitKey(20) & 0xFF == ord('q'):
                break
            r = cv.selectROI("Selection", canvas, False, False)
            print("Image slected!")
            canvas = canvas[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]
            cv.destroyAllWindows()
            cv.imshow("Cropped", canvas)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break


main()

print("Goodbye after")
print("--- %s seconds ---" % (time.time() - start_time))
cv.destroyAllWindows()