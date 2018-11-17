from flask import Flask, render_template
app = Flask(__name__,
            template_folder='./Frontend/templates',
            static_folder = './Frontend/static/css'
            )


import os
from Frontend import camera

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/frontend')
def frontend():
    return render_template('index.html')

@app.route('/backend')
def backend():
    return "Backend"

if __name__ == '__main__':
    app.run()