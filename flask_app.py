from flask import Flask, render_template, request, jsonify, abort, send_file
from depth_est import est_depth
from PIL import Image
from flask_cors import CORS
import cloudinary
import cloudinary.uploader

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def index():
	print('potato activated')
	return 'potato'

@app.route('/depthEst', methods=["POST"])
def process_image():
	print('================== depthEst======================')
	try:
		file = request.files['image']
		# Read the image via file.stream
		img = Image.open(file.stream)
		img.save('original_img.jpg')

		print('-- gotten image')
		pilImage_path = 'tempimg.jpg'
		est_depth(img, pilImage_path)

		print('-- processed')

		# my cloudinary acc (wilbert's)
		cloudinary.config(
		  cloud_name= "dpuh9sp7u",
		  api_key= "449866115973213",
		  api_secret= "Yvc_McCnAkbwavJAKIe9zr8B0RM"
		)
		upload_result = cloudinary.uploader.upload(pilImage_path, public_id = "dl_result")

		print('completed')
		return {'path':upload_result['url']}
	except Exception as e: 
		print(e)
		return 'WHERE IMAGE?! (but something could have went wrong, contact gabriel)'



if __name__ == '__main__':
    app.run(debug=True)