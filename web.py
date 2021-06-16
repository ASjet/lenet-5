import cv2
import numpy as np
from flask import Flask, render_template, Response
from nn import cnn
from cv import config, ip, recognition

dgt = 0

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def gen():
    global dgt
    net = recognition.loadModel()
    cap = cv2.VideoCapture(config.camera_id)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if(ret):
            sel = ip.cut(frame, 256)
            prep = ip.getDigit(sel)
            got, obj = ip.detect(prep)
            if(got):
                img = obj
                dgt = np.argmax(net.feedforward(obj/255.0))
            else:
                img = prep
            print('\rNum: ',dgt, sep=' ', end='')
            image = cv2.imencode('.jpg', sel)[1].tobytes()
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
