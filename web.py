import cv2
from flask import Flask, render_template, Response
from nn import cnn
from cv import config, ip, recognition

dgt = 0

app = Flask(__name__)

@app.route('/')
def index():
    dgt = 1
    return render_template('index.html', digit=dgt)


def gen():
    # video_path = 'S:/ITFS/Movies&Drama/Short/ALIKE.mp4'
    # vid = cv2.VideoCapture(video_path)
    global dgt
    net = recognition.loadModel()
    cap = cv2.VideoCapture(config.camera_id)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if(ret):
            prep = ip.process(frame)
            got, obj = ip.detect(prep)
            if(got):
                dgt = net.feedforward(obj)
                image = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()