from flask import Flask
import json
from flask import jsonify
import cv2
#import urllib.request
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
from flask_cors import CORS
from flask import render_template, request
import dlib
from scipy.spatial import distance
from imutils import face_utils
import pyrebase
import os
import tempfile
from PIL import Image
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

app = Flask(__name__)
CORS(app)
cred = credentials.Certificate('teamkalm-29130-firebase-adminsdk-dhlje-415faee97d.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
config = {
  #firebase config
}
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
db.collection(u'driverDrowsy').document(u'session').set({u'imgs': []})
path_on_cloud = "drowsy"
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
EYE_ASPECT_RATIO_THRESHOLD = 0.2
EYE_ASPECT_RATIO_CONSEC_FRAMES = 2
counter = 0
arr = []
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

@app.route("/")
def home():
    return '<html><body><form action = "/uploader" method = "POST" enctype = "multipart/form-data"><input type = "file" name = "file" /><input type = "submit"/></form></body></html>'

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
  res = ''
  if request.method == 'POST':      
    global counter
    global arr
    f = request.files['file'].read()
    # convert string data to numpy array
    npimg = np.fromstring(f, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    print("req",request)
    
    # plt.imshow(img)
    # plt.show()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Detect facial points through detector function
    faces = detector(gray, 0)
    
    #Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw rectangle around each face detected
    # for (x,y,w,h) in face_rectangle:
    #   cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    #Detect facial points
    for face in faces:
      shape = predictor(gray, face)
      shape = face_utils.shape_to_np(shape)

      #Get array of coordinates of leftEye and rightEye
      leftEye = shape[lStart:lEnd]
      rightEye = shape[rStart:rEnd]

      #Calculate aspect ratio of both eyes
      leftEyeAspectRatio = eye_aspect_ratio(leftEye)
      rightEyeAspectRatio = eye_aspect_ratio(rightEye)

      eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

      #Use hull to remove convex contour discrepencies and draw eye shape around eyes
      # leftEyeHull = cv2.convexHull(leftEye)
      # rightEyeHull = cv2.convexHull(rightEye)
      # cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
      # cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
			
      if (eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
        print(eyeAspectRatio)
        res = "drowsy"
        counter += 1
        im = Image.fromarray(img)
        temp = tempfile.NamedTemporaryFile(delete=False)
        print(temp.name)
        im.save(temp.name+'.jpg')
        resp = {"downloadTokens": "null"}
        filename = temp.name.replace("/","%2F")
        resp = storage.child(path_on_cloud+temp.name+'.jpg').put(temp.name+'.jpg')
        url = "https://firebasestorage.googleapis.com/v0/b/teamkalm-29130.appspot.com/o/drowsy"+filename+".jpg"+"?alt=media&token="+resp["downloadTokens"]
        print(resp,url)
        arr.append(url)
        db.collection(u'driverDrowsy').document(u'session').set({u'imgs': arr})
        os.remove(temp.name+'.jpg')
        
        # if (counter >= 2):
		      #   storage.child(path_on_cloud).put(img)
        #   print('close')
        #   res = "drowsy"
        
      else:
        res = "open"
        counter = 0
        print('open')      
  return res

if __name__ == '__main__':
  app.run(host='0.0.0.0',port=8080)
