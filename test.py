import cv2
import dlib
import numpy as np
from tensorflow.python import tf2
from keras.models import load_model
from time import sleep, time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db



cred = credentials.Certificate("./drowsydrivingmonitor-firebase-adminsdk-j4msb-7462370037.json")
firebase_admin.initialize_app(cred,
{
    'databaseURL':"https://drowsydrivingmonitor-default-rtdb.asia-southeast1.firebasedatabase.app"
})

eyeOpen = True
yawn    = False

appStartTimeRef = db.reference('app_on_start_time')
alertRef = db.reference('alert')


print("\n\n")
print("App start time : ", appStartTimeRef.get())
print("Alert : ", alertRef.get())
print("\n\n")
def im_preprocess(img):
    x = [img]
    X = np.array(x).reshape(-1,224,224,3)
    X= X/255
    return X

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)
classifier_drawsin =load_model('drawsiness_model.h5')
classifer_yawn = load_model('yawning_model.h5')

classes_1 = {0:'close',1:'open'}
classes_2 = {0:'yawn',1:'No_yawn'}

count = 0

while True:
    
    # Capture the image from the webcam
    ret, image = cap.read()
    # Convert the image color to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect the face
    rects = detector(gray, 1)
    # Detect landmarks for each face
    for rect in rects:
        # Get the landmark points
        shape = predictor(gray, rect)

	# Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")

        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)

        shape = shape_np
        eye_image = image[shape[19][1]:shape[1][1], shape[1][0]:shape[15][0]]
        mouth = image[shape[2][1]:shape[6][1], shape[2][0]:shape[14][0]]

        height = eye_image.shape[0]
        width = eye_image.shape[1]
        width_cutoff = width // 2
        s1 = eye_image[:, :width_cutoff]

        eye = cv2.resize(s1, (224, 224), interpolation = cv2.INTER_AREA)
        mouth = cv2.resize(mouth, (224, 224), interpolation = cv2.INTER_AREA)


        
        X = im_preprocess(eye)
        y_pred_1 = classifier_drawsin.predict(X)
        class_1 = [np.argmax(element) for element in y_pred_1]
        
        eyeOpen = class_1[0]==1
        


        if(eyeOpen==True):
            count = 0
        else:
            count+=1
        print("Count",count)


        if(count>=5):
            count = 0
            alertRef.set(True)

        


        cv2.putText(img=image, text=classes_1[class_1[0]], org=(150, 250) ,fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3,color=(0, 255, 0),thickness=3)
        
        #cv2.imwrite('eye.jpg',eye_image)


        X2 = im_preprocess(mouth)
        
        y_pred_2 = classifer_yawn.predict(X2)
        class_2 = [np.argmax(element) for element in y_pred_2]
       
        
        yawn = class_2[0]==1


        cv2.putText(img=image, text=classes_2[class_2[0]], org=(100, 250) ,fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2,color=(255, 0, 0),thickness=2)
        #cv2.imwrite('mouth.jpg',mouth)
        #for i, (x, y) in enumerate(shape):

            #cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
		
    # Display the image
    cv2.imshow('Landmark Detection', image)
    # Press the escape button to terminate the code
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
