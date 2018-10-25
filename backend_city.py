import cv2
import numpy as np
from tkinter import messagebox

def draw_lines (img, lines) : 
    for line in lines : 
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [0,255,0], 3 )

def check_lines(ave_list) : 
    same_lines = 0 
    lines1 , lines2 = ave_list 
    for line1 in lines1 : 
        pts = line1[0]
        for line2 in lines2 :   
            cords = line2[0]
            if (pts[0] == cords[0] and pts[1] == cords[1] and pts[2] == cords[2] and pts[3] == cords[3]) : 
                same_lines = same_lines + 1 
    return same_lines 


## -- initialize global variables -- ## 

# capture the video frames 
def detect_discrepancy(filename) : 
    cap = cv2.VideoCapture(filename)
    frame_no = 0  
    ave_same = 0 
    ave_list = list()
    while(1):

        ret, frame = cap.read()
        frame_no = frame_no + 1 
        if not ret :    
            exit(0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #detecting white lanes of a road .... 
        sensitivity = 50
        lower_white = np.array( [0,0,255 - sensitivity] )
        upper_white = np.array([255 , 255 - sensitivity , 255])
        mask = cv2.inRange( hsv , lower_white , upper_white)

        edges = cv2.Canny(mask, 75, 150)
        edges = cv2.GaussianBlur(edges ,(5,5) , 0)
        try : 
            #detected lines 
            lines = cv2.HoughLinesP(edges, 1, np.pi/180 , 100 , minLineLength = 20)
            ave_list.append(lines)
            #lines detected are drawn to the frames... 
            draw_lines(frame,lines)
            if (frame_no%2 == 0 and len(lines) > 0) :  #do this if lines were detected... 
                same = check_lines(ave_list)
                ave_list = []
                ave_same = ( ave_same*(frame_no/2 - 1) + same ) / (frame_no/2) #computing the average 
        # if the number of same lines falls below a threshold and lines are being detected (i.e not due to traffic) ->  high chance of an earthquake
                if same < 5 and len(lines) > 10 : 
                    messagebox.showinfo("ALERT!!!!", " This feed needs attention ASAP")                        
        ## if average needs to be verified, uncomment these lines 
        # if (frame_no%50 == 0 ) : 
        #     print(ave_same)           
        except : 
            pass
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()




"""  
On analyzing the video ->
On a noraml functioning day, we see that on an average the number of same lines ~ 38 to 45 between 2 frames
but as the earthquake starts, due to the shifting of the lines, the average keeps dropping and it even drops below
32 due to the constant shaking of everything, thus everything is deviated, and we dont get the same lines... 

ASSUMPTIONS - 
Lanes are demarkated by white color
if not, apt color can be given to InRange function... 

"""