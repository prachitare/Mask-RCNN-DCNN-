from flask import Flask, render_template, url_for, request, redirect
#import sqlalchemy
from werkzeug import secure_filename
import os
import mask

IMAGE_FOLDER = os.path.join('static', 'photos')
file_name = 1

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/upload')
def upload():
	return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
	if request.method == 'POST':
		f = request.files['imgfile']
		global imgname
		imgname = secure_filename(f.filename)
		imgpath = 'static/photos/'+imgname
		f.save(imgpath)
		return redirect('show')

# @app.route('/mask')
# def mask():
	

@app.route('/show')
def show():
	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], imgname)
	masked_image = 'masked_'+full_filename
	return render_template('show.html', user_image = masked_image)

if __name__ == '__main__':
    app.run()