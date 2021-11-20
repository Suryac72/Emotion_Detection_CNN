#EMOTION DETECTION USING CNN,KERAS,TENSORFLOW AND OPENCV.
#************* ALL EXPLAINATION ARE PRESENT IN THE FORM OF COMMENTS IN EACH LINE OF THE CODE KINDLY GO THROUGH IT.*******************#

#importing libraries
from keras.models import load_model 
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# taking the haarcascade_frontalface_default.xml file from the git repo of the opencv for applying cascade classifier on our trained model

face_classifier = cv2.CascadeClassifier(r'F:\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')

#fetching our trained CNN model which we used for emotion detection.
classifier =load_model(r'F:\Emotion_Detection_CNN-main\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise'] #categorizing the emotion labels and stored into the list named emotion_labels.

cap = cv2.VideoCapture(0) #capturing the image using system webcam for emotion detection.


while True:
    _, frame = cap.read()  #reading captured image, if there is no captured image is present then it prints No faces in the frame otherwise it analyzes the face and predicts the status of the emotion of that particular image which is shown in the frame.
    labels = [] # list of labels in which value of status of image ie, sad,happy,angry etc is stored in that list
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #converting frame image into grayscale because in our dataset it uses grayscale image for emotion detection.
    faces = face_classifier.detectMultiScale(gray) 

    for (x,y,w,h) in faces: # in for loop parameters x ---> horizontal, y---> vertical,w----> width, h--->height. In that what area of the frame which we havae to detect the protions of th frame of emotion detection.
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) # creating the rectangle for the face protion of the image where it shows the label i.e, status of image or expression of the image.
        roi_gray = gray[y:y+h,x:x+w] #conszing the portion of the image which portion which we have to detect.
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA) #resizeing the image in the dimension 48x48 because in our emotion model of CNN, we take the size of sample image is 48x48.

        if np.sum([roi_gray])!=0:
            #standarizing the region of intrest 'roi' by means of dividing it by 255 and converting it into an array and setting its axis to 0 which is Y-AXIS.
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            #storing the value of prediction of classifier into the variable prediction.
            prediction = classifier.predict(roi)[0]
            #generating labels of expression based on the result of models after analyzing the image present in the frame.
            label=emotion_labels[prediction.argmax()]
            #initilzing the coordinates of image as (x,y)
            label_position = (x,y)
            #putting text as labels corresponding to the frame image face status
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            #printing the labels of expression at each instance into the console screen.
            print(label)
        else:
            #putting text to the frame 'No Faces' if it is unable to detect any faces into the frame
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    #for exit we have to click 'q' button to quit the frame 
    if cv2.waitKey(1) & 0xFF == ord('q'):  #wait key of q is 1 
        break

cap.release()  #Releasing all the taken resources 
cv2.destroyAllWindows() #destroying captured window frame 