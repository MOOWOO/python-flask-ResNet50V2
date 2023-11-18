# https://github.com/jrosebr1/simple-keras-rest-api/blob/master/run_keras_server.py
# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import tensorflow as tf
from keras.applications import ResNet50
from keras.applications.resnet50 import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import flask
import io
from tensorflow import keras
from gevent.pywsgi import WSGIServer
import pandas as pd
from PIL import Image, ImageOps

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	global class_names
	model = keras.models.load_model("./Hair_loss/ResNet50V2")
	
	#model = ResNet50(weights="imagenet")
	#model = keras.models.load_model("./converted_keras/keras_model.h5", compile=False)
	class_names = open("./Hair_loss/ResNet50V2/labels.txt", "r").readlines()

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):

			arr_data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image)).convert("RGB")

			size = (224, 224)
			image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
			image_array = np.asarray(image)
			
			# Normalize the image
			normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

			# Load the image into the array
			arr_data[0] = normalized_image_array

			# classify the input image and then initialize the list
			# of predictions to return to the client
			prediction = model.predict(arr_data)

			index = np.argmax(prediction)
			class_name = class_names[index]
			confidence_score = float(prediction[0][index])

			# Print prediction and confidence score
			print("Class:", class_name[2:], end="")
			print("Confidence Score:", confidence_score)

			# arr_predict = np.array(prediction[0]) #list to ndarray
			# data["predictions"] = arr_predict.tolist() # ndarray to list
			data["class"] = class_name[2:].replace("\n", "")
			data["confidence"] = confidence_score
			
			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	# app.run()
	http_server = WSGIServer(('', 5000), app)
	http_server.serve_forever()
