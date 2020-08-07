import os
import argparse
import cv2
import numpy as np
import sys
from flask import Flask, render_template, Response
#from servodrive import ServoMotor
from videostream import VideoStream
from flask_basicauth import BasicAuth
import time
import threading
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import importlib.util





cred = credentials.Certificate('drone-detection-js-firebase-adminsdk-4xh9r-3ba93b9ccd.json')
# Initialize the app with a service account, granting admin privileges
cred = credentials.Certificate('firestore-1add2-firebase-adminsdk-7vjg4-6c20413010.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# App Globals (do not edit)
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'pi'
app.config['BASIC_AUTH_PASSWORD'] = 'pi'
app.config['BASIC_AUTH_FORCE'] = True

basic_auth = BasicAuth(app)
last_epoch = 0

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--savedir', help='saveimage folder',
                    default='DroneImg')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
imgfolder = args.savedir
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5
# Initialize frame rate calculation


@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')

def gen(videostream):
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    dnum=0
    time_appear_drone = 0
    zero_count=0
    leftmotor_count=0
    rightmotor_count=0
    rectangule_color = (10, 255, 0)
    frame1 = videostream.read()
    rows, cols, _ = frame1.shape
    print(frame1.shape)
    x_medium = int(cols / 2)
    x_center = int(cols / 2)
    y_medium = int(rows / 2)
    y_center = int(rows / 2)
    while True:
        t1 = cv2.getTickCount()
        D=0
        distance = 0        
        if videostream.stopped:
            break
        
        frame1 = videostream.read()
        frame = frame1.copy()
        capimg = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        
        
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        
        
        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        #print(num)
        now = time.gmtime(time.time())
        y = str(now.tm_year)
        mo = int(now.tm_mon)
        d = int(now.tm_mday)
        h = int(now.tm_hour + 9)
        mi = int(now.tm_min)
        sec = int(now.tm_sec)
        if mo < 10:
            mo = str(0) + str(mo)
        if d < 10:
            d = str(0) + str(d)
        if h >12:
            h = h-12
        if mi < 10:
            mi = str(0) + str(mi)
        date =str(y)+'-'+str(mo)+'-'+str(d)+'/'+str(h)+':'+str(mi)
        boxthickness = 3
        linethickness = 2
        
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
            
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), rectangule_color, boxthickness)
                
                
                

                    
                               
                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        
        
        text = "NoD is : {} ".format(num)        
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(frame, text, (210, 50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,'Distance: {0:.1f}m'.format(distance),(10,80),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,'DAY: ' + date,(15,460),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,0),1,cv2.LINE_AA)
        
        
            
        
        
        
        ret, jpeg = cv2.imencode('.jpg',frame)
        if jpeg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            print("frame is none")
       
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= (1/time1)+10   

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoStream().start()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

    
    









