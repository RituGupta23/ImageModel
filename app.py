from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('image_model.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return "Welcome to the Image Classification API!"

@app.route('/image', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file directly to Render's persistent disk
    uploads_dir = '/mnt/data/uploads'  # Persistent disk directory on Render
    
    # Ensure uploads directory exists within the persistent disk
    img_path = os.path.join(uploads_dir, file.filename)

    # Save file directly, no need for os.makedirs
    file.save(img_path)

    # Preprocess the image and make predictions
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    
    # Return prediction result
    prediction_class = 'Class 1' if prediction[0] > 0.5 else 'Class 0'
    return jsonify({
        'prediction': prediction_class,
        'confidence': float(prediction[0])
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
