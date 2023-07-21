import webbrowser
from time import sleep

import cv2
import numpy as np
import pyjokes
import pyttsx3
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array

face_classifier = cv2.CascadeClassifier(r'C:\Users\Admin\Desktop\ML\Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\Admin\Desktop\ML\Emotion_Detection_CNN-main/model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

l1=' '


while l1!="Sad":
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    label=' '
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
              
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if label=="Sad":
            l1=label
            lable =' '
            computer = pyttsx3.init()
            text= pyjokes.get_joke(language="en",category="all")
            # joke = pyjokes.get_joke() 
            computer.say(text)
            print(text)
            computer.runAndWait()
            # print("Here is the Playlist as your Current mood")
            webbrowser.open("https://open.spotify.com/playlist/2XLDEbpTJQYWY4jfLMAnli")
            
            pass
#cap.release()
cv2.destroyAllWindows()