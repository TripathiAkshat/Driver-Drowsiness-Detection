import cv2
import dlib
from scipy.spatial import distance
from playsound import playsound
import time
import csv



f = open('Data.csv', 'r+')
f.truncate(0)
f = open('Data1.csv', 'r+')
f.truncate(0)

fieldnames = ['Timestap','Symptoms']

def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

cap = cv2.VideoCapture(0)#Change as per the camera no

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
with open ('Data.csv' , 'a' , newline='') as csvfile:
    thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
    thewriter.writeheader()
with open ('Data1.csv' , 'a' , newline='') as csvfile:
    thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
    thewriter.writeheader()    

while True:
    
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	leftEye.append((x,y))
        	next_point = n+1
        	if n == 41:
        		next_point = 36
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	#cv2.line(frame,(x,y),(x2,y2),(255,255,255),1)

        for n in range(42,48):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	rightEye.append((x,y))
        	next_point = n+1
        	if n == 47:
        		next_point = 42
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	#cv2.line(frame,(x,y),(x2,y2),(255,255,255),1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        cv2.putText(frame,current_time,(450,50),
        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)

        
       
            
            
        if EAR<0.20 :
            cv2.putText(frame,"Feeling Sleepy ???",(20,100),
        		cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),4)
            cv2.putText(frame,"Please wash your face !!!",(20,400),
        		cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4)
            print("Drowsiness Symptopms detected") 
            playsound('Alert.mp3')
            with open ('Data.csv' , 'a' , newline='') as csvfile:
               thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
               thewriter.writerow({'Timestap':current_time,'Symptoms':"Drowsiness Symptopms detected"  })
            with open ('Data1.csv' , 'a' , newline='') as csvfile:
                     thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
                     thewriter.writerow({'Timestap':current_time,'Symptoms': 1 }) 
               

        else :                   
                with open ('Data1.csv' , 'a' , newline='') as csvfile:
                     thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
                     thewriter.writerow({'Timestap':current_time,'Symptoms': 0 })             
                

   
    cv2.imshow("Drowsiness Detection System", frame)  
    

    key = cv2.waitKey(3)
    time.sleep(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()