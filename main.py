from flask import Flask, render_template
app = Flask(__name__,
            template_folder='./Frontend/templates',
            static_folder = './Frontend/static/css'
            )

from Frontend import text2speech

@app.route('/')
def hello():
    text2speech.main()
    return render_template('home.html')

@app.route('/frontend')
def frontend():
    return render_template('index.html')

@app.route('/backend')
def backend():
    return "Backend"

if __name__ == '__main__':
    app.run()