from flask import Flask, request, jsonify
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import gdown

app = Flask(__name__)

# URL của file model trên Google Drive
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n'

# Đường dẫn lưu file model
MODEL_PATH = 'best_weights_model.keras'

# Kiểm tra xem model đã có trong thư mục chưa, nếu chưa thì tải về
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Set upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocessing the image (example, adjust as per model input)
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

        # Predict
        prediction = model.predict(img_array)
        label = 'nodule' if prediction[0][0] > 0.5 else 'non-nodule'
        
        return jsonify({'prediction': label})
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
