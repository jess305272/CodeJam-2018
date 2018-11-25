from flask import Flask, render_template, Response, jsonify, request
from Frontend.camera import VideoCamera
import time
from pygame import mixer
import random

import Frontend.text2speech as myTTS

from MLPackage.ImageCaptioning import ImageCaption

try:
    import cv2
except ImportError:
    cv2 = None

app = Flask(__name__,
            template_folder='./Frontend/templates',
            static_folder='./Frontend/static'
            )

video_camera = None
global_frame = None
AIModel = None

import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\Google\codejam18.json'


@app.route('/')
def hello():
    return render_template('home.html')


def playMusic(filename):
    mixer.init()
    mixer.music.load(filename)
    mixer.music.play()

@app.route('/', methods=['POST'])
def text2speech():
    text = request.form.get('text', 'No text')
    speech = myTTS.getSpeech(text)
    filename = 'audios/speech%d.mp3' % time.time()
    myTTS.saveMp3(speech, filename)

    playMusic(filename)
    return render_template('home.html')


@app.route('/backend')
def backend():
    return "Backend"


@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera

    print('reached record_status')

    if cv2:

        if video_camera == None:
            video_camera = VideoCamera()

        json = request.get_json()

        status = json['status']

        print('so far so good')

        if video_camera is None:
            return

        if status == "true":
            video_camera.start_record()
            print('start_record')
            return jsonify(result="started")
        else:
            video_camera.stop_record()
            print('stop_record')
            return jsonify(result="stopped")


@app.route('/video_stream')
def video_stream():
    global video_camera
    global global_frame

    if cv2:

        if video_camera == None:
            video_camera = VideoCamera()

        while True:

            if video_camera is None:
                return

            frame = video_camera.get_frame()

            if frame != None:
                global_frame = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')


elevatorMusic = os.listdir('elevator')

@app.route('/snapshot_status', methods=['POST'])
def snapshot_status():
    print('ran snapshot_status')
    global video_camera
    caption = 'default caption'
    result = 'default result'

    if cv2:

        if video_camera == None:
            video_camera = VideoCamera()

        json = request.get_json()

        status = json['status']

        if video_camera is None:
            return

        if status == "true":
            frameTime = time.time()
            frameName = 'frames/frame%d.png' % frameTime
            cv2.imwrite(frameName, video_camera.take_snapshot())
            print('Picture taken')

            r = random.randint(0, len(elevatorMusic)-1)
            playMusic('%s/%s' % ('elevator', elevatorMusic[r]))

            # run image captioning
            global AIModel
            if AIModel is None:
                AIModel = ImageCaption()
            caption = AIModel.predict_captions(frameName)
            result = caption

            # run text-to-speech
            audioTime = time.time()
            audioName = 'audios/audio%d.mp3' % audioTime
            speech = myTTS.getSpeech(caption)
            myTTS.saveMp3(speech, audioName)

            mixer.init()
            mixer.music.load(audioName)
            mixer.music.play()
            print('Time: %d' % (audioTime - frameTime))

            print(result)

    return render_template('home.html', caption=result)


@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    global AIModel

    if AIModel is None:
        AIModel = ImageCaption()

    # file = request.files['image']
    # file.save('./test.png')

    caption = AIModel.predict_captions('./test.png')

    return render_template('result.html', caption=caption)


@app.route('/upload', methods=['POST'])
def upload():
    global AIModel
    if AIModel is None:
        AIModel = ImageCaption()

    file = request.files['image']

    file.save('./test.png')

    caption = AIModel.predict_captions('./test.png')


    # run text-to-speech
    audioTime = time.time()
    audioName = 'audios/audio%d.mp3' % audioTime
    speech = myTTS.getSpeech(caption)
    myTTS.saveMp3(speech, audioName)

    mixer.init()
    mixer.music.load(audioName)
    mixer.music.play()
    # print('Time: %d' % (audioTime - frameTime))


    return render_template('home.html', caption=caption)


if __name__ == '__main__':
    app.run()
