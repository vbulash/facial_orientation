import os
from flask import Flask, jsonify, render_template, Response, request
import cv2
from facial_orientation import FacialOrientation
from dotenv import dotenv_values
import requests
import pusher
import json

app = Flask(__name__)

def gen_frames():
    global sid, env
    
    cap = cv2.VideoCapture(0)
    if (cap is None or not cap.isOpened()):
        raise Exception("Невозможно открыть источник видео {}".format(cap))
    Face = FacialOrientation(cap=cap, angle_deviation=10,
                             show_fps=False, show_coords=False, blur_background=True)

    is_save = False
    while (cap.isOpened() and not is_save):
        buffer, is_save, message = Face.get_frame()
        if (not is_save):
            ret, png = cv2.imencode('.png', buffer)
            frame = png.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

            if (len(message) > 0):
                pusher_client.trigger(
                    'face-channel-' + sid, 'face-event', {'message': message[0]})
            else:
                pusher_client.trigger(
                    'face-channel-' + sid, 'face-event', {'message': ''})

    cap.release()
    if (is_save):
        # print(buffer)
        image = buffer.decode('ascii')
        # TODO pusher не может принимать файлы больше 10K - необходимо передавать буфер через request
        # pusher_client.trigger(
        #     'face-save-channel-' + sid, 'face-save-event', image)
        r = requests.post(env.get('SHOT_DONE_URL'), {'uuid': sid, 'photo': image})
        print(r)
        pusher_client.trigger(
            'face-save-channel-' + sid, 'face-save-event', True)
        


@app.route('/', methods=['GET'])
def index():
    global sid, pkey, env, pusher_client, exiting

    sid = request.args.get('sid', '')
    pkey = request.args.get('pkey', '')
    env = dotenv_values(".env")
    pusher_client = pusher.Pusher(
        app_id=env.get('PUSHER_APP_ID'),
        key=env.get('PUSHER_APP_KEY'),
        secret=env.get('PUSHER_APP_SECRET'),
        cluster=env.get('PUSHER_APP_CLUSTER')
    )
    exiting = False

    return render_template('index.html')


@app.route('/cv2_feed')
def cv2_feed():
    return Response(
        gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Компактный способ выхода без устаревшего werkzeug - https://stackoverflow.com/questions/15562446/how-to-stop-flask-application-without-using-ctrl-c
@app.route("/exit", methods=['GET'])
def exit_app():
    global exiting
    exiting = True
    return "Done"


@app.teardown_request
def teardown(exception):
    global exiting
    if exiting:
        os._exit(0)


if __name__ == "__main__":
    app.run()
