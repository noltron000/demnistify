# time stuff
from datetime import datetime

# Firebase stuff
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Make a flask API for our DL Model
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify
import numpy as np
from werkzeug.datastructures import FileStorage
from PIL import Image
from keras.models import model_from_json
import tensorflow as tf

# Firebase: get credentials and options
cred = credentials.Certificate('firebase-sdk.json')
params = {
	'databaseURL': 'https://test-app-e0c0d.firebaseio.com/'
}

# Firebase: init app
firebase_admin.initialize_app(cred, params)
ref = db.reference('/')

# Flask stuff
app = Flask(__name__)
api = Api(app, version='1.0', title='MNIST Classification', description='CNN for Mnist')
ns = api.namespace('Make_School', description='Methods')
my_model = 'my_model.h5'

single_parser = api.parser()
single_parser.add_argument('file', location='files', type=FileStorage, required=True)

model = load_model(my_model)
graph = tf.get_default_graph()

# Model reconstruction from JSON file
# with open('model_architecture.json', 'r') as f:
# 	model = model_from_json(f.read())
#
# Load weights into the new model
model.load_weights(my_model)

@ns.route('/prediction')
class CNNPrediction(Resource):
	"""Uploads your data to the CNN"""
	@api.doc(parser=single_parser, description='Upload an mnist image')
	def post(self):
		args = single_parser.parse_args()
		image_file = args.file
		image_file.save('image.png')
		img = Image.open('image.png')
		image_red = img.resize((28, 28))
		image = img_to_array(image_red)
		# print(image.shape)
		x = image.reshape(1, 28, 28, 1)
		x = x/255

		# This is not good, because this code implies that the model will be
		# loaded each and every time a new request comes in.
		# model = load_model(my_model)

		with graph.as_default():
			out = model.predict(x)
		print(out[0])
		print(np.argmax(out[0]))
		r = np.argmax(out[0])

		# save r to database
		my_object = {
			'prediction': str(r),
			'filename': image_file.filename,
			'time': str(datetime.now())
		}

		ref.push(my_object)

		# return object
		return my_object

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000)
