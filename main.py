from flask import Flask, render_template, Response, jsonify, request
from Frontend.camera import VideoCamera
import simpleaudio as sa

import Frontend.text2speech as myTTS

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

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\Google\codejam-2018-43e63ace5e4e.json'

@app.route('/')
def hello():
	return render_template('home.html')


@app.route('/', methods=['POST'])
def text2speech():
	text = request.form.get('text', 'No text')
	speech = myTTS.getSpeech(text)
	audio = myTTS.getMp3(speech)
	# myTTS.saveMp3(speech, 'output')
	sa.play_buffer(audio, 2, 2, 44100)

	# return text
	return render_template('home.html')


# @app.route('/frontend')
# def frontend():
#     return render_template('home.html')

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


@app.route('/snapshot_status')
def snapshot_status():
	print('ran snapshot_status')
	img_counter = 0

	global video_camera
	global global_frame

	if cv2:

		if video_camera == None:
			video_camera = VideoCamera()

		json = request.get_json()

		status = json['status']

		if video_camera is None:
			return

		if status == "true":
			cv2.imwrite('frame.png', global_frame)
			print('picture taken')
			img_counter += 1


@app.route('/video_viewer')
def video_viewer():
	return Response(video_stream(),
	                mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
	app.run()
