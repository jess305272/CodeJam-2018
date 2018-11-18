from flask import Flask, render_template, Response, jsonify, request
from Frontend.camera import VideoCamera
from Frontend import text2speech

app = Flask(__name__,
            template_folder='./Frontend/templates',
            static_folder='./Frontend/static/css'
            )

video_camera = None
global_frame = None


@app.route('/')
def hello():
    return render_template('home.html')


@app.route('/text2speech', methods=['POST'])
def text2speech():
    text = request.form.get('text', 'No text')
    print(text)
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

    if video_camera == None:
        video_camera = VideoCamera()

    json = request.get_json()

    status = json['status']  # if video_camera is None:
    # 	return

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        return jsonify(result="stopped")


@app.route('/video_stream')
def video_stream():
    global video_camera
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera()

    while True:

        # if video_camera is None:
        #     return

        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')


@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
