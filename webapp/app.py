from flask import Flask, render_template, request, redirect, url_for, send_file, flash, redirect, session
import os
import zipfile
from werkzeug.utils import secure_filename
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import shutil
import logging
from statistics import mode
import matplotlib.pyplot as plt
import nibabel as nib

tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

UPD_FLD = 'static/uploads/'

app = Flask(__name__)

UPLOAD_FOLDER = os.path.dirname(os.path.realpath(__file__))
ALLOWED_EXTENSIONS = set(['zip','txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPD_FLD
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'This is your secret key to utilize session in Flask'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize(self,img):
	img_rib = np.array(img.dataobj)
	a = np.min(img_rib)
	img_array = zoom(img_rib, (1,1,0.50),cval=a)
	return img_array.astype(np.uint8)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analysis')
def analysis():
	return render_template('analysis.html')

@app.route('/download', methods=['POST'])
def download_image():
    return send_file(os.path.join(os.getcwd(),"test0_segmented.jpg"), as_attachment=True)
      
@app.route('/', methods=['POST'])
def upload_image():
	kvasir = False
	brain_mri = False
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		if(filename.endswith(".zip")):
			brain_mri = tf.keras.models.load_model("static/models/skull3_20.h5")
			count = 0
			mri = []
			with zipfile.ZipFile(os.path.join(app.config['UPLOAD_FOLDER'], filename), mode="r") as archive:
				for info in archive.infolist():
					if count:
						img = nib.load(os.path.join(file_path,info.filename))
						img = np.expand_dims(resize(img),axis=-1)
						mri.append(img)

			mri_np = np.array(mri)
			pred = brain_mri.predict(mri_np,steps=int(len(mri_np)//1))
			pred = np.around(pred)
			pred = "Demented" if pred == 1 else "Non Demented"
			flash(pred,"result")
		else:
			flash('Image successfully uploaded and displayed below')
			labels = ['AbdomenCT','BreastMRI','ChestCT','CXR','Hand','HeadCT','Kvasir-Capsule']
			model = tf.keras.models.load_model("static/models/mednist.h5")
			IMAGE_WIDTH=64
			IMAGE_HEIGHT=64
			img = cv2.imread(file_path,cv2.IMREAD_COLOR)
			img = cv2.resize(img, (IMAGE_WIDTH,IMAGE_HEIGHT))
			img = img.astype("float64")/255
			img = img.reshape(-1,64,64,3)
			pred = model.predict(img)
			name = labels[np.argmax(pred)]
			flash(name,"result")
			
			if name == "ChestCT":
				DenseNet_Path = 'static/models/chest_CT_SCAN-DenseNet201.h5'
				ResNet_Path = 'static/models/chest_CT_SCAN-ResNet50.h5'
				EffNet_Path = 'static/models/chest_CT_SCAN-EfficientNet.h5'
				ResNet_model = tf.keras.models.load_model(ResNet_Path)
				DenseNet_model = tf.keras.models.load_model(DenseNet_Path)
				EffNet_model = tf.keras.models.load_model(EffNet_Path)
				my_image = load_img(file_path, target_size=(460, 460)) 
				my_image = img_to_array(my_image)
				my_image = my_image.reshape((1,460, 460,3))
				#my_image_dense = my_image.copy()

				keys = ['adeno','large','normal','squamous']
				res = []

				ResNet_res = np.argmax(ResNet_model.predict(my_image)) 
				#ResNet_val = np.max(ResNet_model.predict(my_image))

				EffNet_res = np.argmax(EffNet_model.predict(my_image))
				#EffNet_val = np.max(EffNet_model.predict(my_image))

				DenseNet_res = np.argmax(DenseNet_model.predict(my_image))
				#DenseNet_val = np.max(EffNet_model.predict(my_image))

				res.append(ResNet_res)
				res.append(EffNet_res)
				res.append(DenseNet_res)
				flash(keys[mode(res)],"result")
			
			elif name == "Kvasir-Capsule":
				md = tf.keras.models.load_model("static/models/unet.h5")
				test_img = cv2.imread(file_path, cv2.IMREAD_COLOR)       
				test_img = cv2.resize(test_img, (512, 512))
				test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
				test_img = np.expand_dims(test_img, axis=0)
				prediction = md.predict(test_img)
				prediction_image = prediction.reshape((512,512))
				plt.imsave('test0_segmented.jpg', prediction_image, cmap='gray')  
				shutil.copy('test0_segmented.jpg', UPD_FLD)
				kvasir = True
		return render_template('upload.html', filename=filename, kvasir=kvasir,brain_mri=brain_mri)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display')
def display_img():
	return redirect(url_for('static', filename='uploads/test0_segmented.jpg'), code=301)
        
@app.route('/segregation', methods=['GET', 'POST'])
def segregation():
    labels = ['AbdomenCT','BreastMRI','ChestCT','CXR','Hand','HeadCT','Kvasir-Capsule']
    model = tf.keras.models.load_model("static/models/mednist.h5")
    IMAGE_WIDTH=64
    IMAGE_HEIGHT=64

    folder_name = "Segregated"
    newpath = os.path.join(os.getcwd(),folder_name)
    if not os.path.isdir(newpath):
        os.mkdir(newpath) 

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            count = 0

            with zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, filename), mode="r") as archive: 
                 for info in archive.infolist():
                    if count:

                        print(os.path.join(UPLOAD_FOLDER,info.filename))
                        img = cv2.imread(info.filename,cv2.IMREAD_COLOR)
                        img = cv2.resize(img, (IMAGE_WIDTH,IMAGE_HEIGHT))
                        img = img.astype("float64")/255
                        img = img.reshape(-1,64,64,3)
                        pred = model.predict(img)
                        folder = labels[np.argmax(pred)]
                        dest_path = os.path.join(newpath, folder)

                        if not os.path.isdir(dest_path):
                            os.mkdir(dest_path)
                        shutil.copy(info.filename, dest_path)
                        print(f"Filename: {info.filename}")   

                    count += 1

            shutil.make_archive(folder_name, 'zip', newpath)  
            return send_file(os.path.join(os.getcwd(),"Segregated.zip"), as_attachment=True) 
        
    return render_template('segregation.html')

if __name__ == "__main__":
    app.run(debug=True)