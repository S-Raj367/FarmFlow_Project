import os
import flask
from flask import render_template, Flask, request, jsonify, redirect
import joblib  # Replaced pickle with joblib
import pickle
import numpy as np
import warnings
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
from openai import OpenAI  # Import OpenAI SDK
from transformers import pipeline
import traceback
import threading
from huggingface_hub import InferenceClient
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
CORS(app)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['upload_folder'] = 'uploads'
app.config['height_folder'] = 'uploads/height'

# Ensure upload directories exist
os.makedirs(app.config['upload_folder'], exist_ok=True)
os.makedirs(app.config['height_folder'], exist_ok=True)

# Replaced .pkl with .joblib for model and scaler loading
model = pickle.load(open("trained_models/crop_recommendation_svm.pkl", "rb"))
scaler = pickle.load(open("trained_models/crop_recommendation_scaler.pkl", "rb"))

# Replaced .h5 with .pb for model loading
disease_model = load_model('trained_models/plant_disease_model.h5')

plant_list = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
       'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango',
       'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas',
       'pomegranate', 'rice', 'watermelon']

plant_disease_class = ['Apple-scab :https://www2.ipm.ucanr.edu/agriculture/apple/Apple-scab/', 
    'Apple-Black-rot :https://extension.umn.edu/plant-diseases/black-rot-apple#prune-correctly-1767010', 
    'Apple-Cedar-Rust :https://www.planetnatural.com/pest-problem-solver/plant-disease/cedar-apple-rust/', 
    'Apple-healthy :None', 'Blueberry-healthy :None', 
    'Cherry-Powdery-mildew :https://www2.ipm.ucanr.edu/agriculture/cherry/Powdery-Mildew/ ', 
    'Cherry-healthy :None', 
    'Corn-Cercospora-leaf-spot :https://www.pioneer.com/us/agronomy/gray_leaf_spot_cropfocus.html ', 
    'Corn-Common-rust :http://ipm.ucanr.edu/PMG/r113100811.html', 
    'Corn-Northern-Leaf-Blight :https://www.extension.purdue.edu/extmedia/bp/bp-84-w.pdf', 
    'Corn-healthy :None',
    'Grape-Black-rot: https://www.missouribotanicalgarden.org/gardens-gardening/your-garden/help-for-the-home-gardener/advice-tips-resources/pests-and-problems/diseases/fruit-spots/black-rot-of-grapes.aspx', 
    'Grape-Black-Measles :https://www2.ipm.ucanr.edu/agriculture/grape/esca-black-measles/',
    'Grape-Leaf-blight_(Isariopsis_Leaf_Spot) :https://www.sciencedirect.com/science/article/abs/pii/S0261219414001598',
    'Grape-healthy:None', 
    'Orange-Haunglongbing-(Citrus_greening) :https://www.aphis.usda.gov/aphis/resources/pests-diseases/hungry-pests/the-threat/citrus-greening/citrus-greening-hp', 
    'Peach-Bacterial-spot ', 'Peach-healthy',
    'Pepper-bell-Bacterial-spot', 'Pepper-bell-healthy', 
    'Potato-Early-blight :https://www.ag.ndsu.edu/publications/crops/early-blight-in-potato', 
    'Potato-Late-blight :https://www.apsnet.org/edcenter/disandpath/oomycete/pdlessons/Pages/LateBlight.aspx', 
    'Potato-healthy :None', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry-Leaf-scorch : https://content.ces.ncsu.edu/leaf-scorch-of-strawberry', 
    'Strawberry-healthy :None', 'Tomato-Bacterial-spot :https://hort.extension.wisc.edu/articles/bacterial-spot-of-tomato/',
    'Tomato-Early-blight :https://extension.umn.edu/diseases/early-blight-tomato',
    'Tomato-Late-blight :https://content.ces.ncsu.edu/tomato-late-blight', 
    'Tomato-Leaf-Mold :https://www.canr.msu.edu/news/tomato-leaf-mold-in-hoophouse-tomatoes',
    'Tomato-Septoria-leaf-spot :https://www.missouribotanicalgarden.org/gardens-gardening/your-garden/help-for-the-home-gardener/advice-tips-resources/pests-and-problems/diseases/fungal-spots/septoria-leaf-spot-of-tomato.aspx', 
    'Tomato-Spider-mites(Two-spotted_spider_mite) :https://pnwhandbooks.org/insect/vegetable/vegetable-pests/hosts-pests/tomato-spider-mite',
    'Tomato-Target-Spot :https://apps.lucidcentral.org/pppw_v10/text/web_full/entities/tomato_target_spot_163.htm', 
    'Tomato-Yellow-Leaf-Curl-Virus :https://www2.ipm.ucanr.edu/agriculture/tomato/tomato-yellow-leaf-curl/', 
    'Tomato-mosaic-virus :https://extension.umn.edu/disease-management/tomato-viruses', 'Tomato-healthy :None']

def allowed_files(filename):
    allowed_extensions = ['jpg', 'jpeg', 'png']
    #abc.jpg --> ['abc', 'jpg']
    ext = filename.split('.')[-1]
    if ext.lower() in allowed_extensions:
        return True
    else:
        return False

'''
 API endpoint for recommending the crop based on 7 features
 model used is Support Vector Machine with RBF kernel
'''
@app.route('/api/predict-crop', methods=['POST'])
def recommend_crop():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        print("Received data:", data)
        
        # Extract the required parameters
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_values = []
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
            input_values.append(data[field])
        
        X = np.array(input_values).reshape(1, -1)
        print("Input shape:", X.shape)
        
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled).tolist()
        prediction_index = predictions[0]
        
        if isinstance(prediction_index, (int, np.integer)) and 0 <= prediction_index < len(plant_list):
            plant_pred = plant_list[prediction_index]
            return jsonify({"crop": plant_pred})
        else:
            return jsonify({"error": f"Invalid prediction index: {prediction_index}"}), 500
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add a simple test endpoint to verify the server is running
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok', 'message': 'Server is running'})


'''
API endpoint for plant disease prediction for 38 different disease
The model used is MobileNetV2 and dataset link: https://www.kaggle.com/vipoooool/new-plant-diseases-dataset
'''
@app.route('/api/predict-disease', methods=['POST'])
def predict_disease():
    try:
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({"error": "No file provided"}), 400
       
        file = request.files['file']
        print("Received file:", file.filename)
        if file.filename == "":
            print("Empty filename")
            return jsonify({"error": "No file selected"}), 400
       
        if file and allowed_files(file.filename):
            filepath = os.path.join(app.config['upload_folder'], file.filename)
            file.save(filepath)
            print("File saved to:", filepath)
           
            # Load and preprocess the image
            image = cv2.imread(filepath)
            if image is None:
                print("Failed to read image")
                return jsonify({"error": "Invalid image file"}), 400
            
            # Resize the image to match the model's expected input shape
            input_shape = (256, 256)  # Match the model's input shape from summary
            image = cv2.resize(image, input_shape, interpolation=cv2.INTER_NEAREST)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
            x = np.expand_dims(image, axis=0)  # Add batch dimension
            print("Input shape:", x.shape)
           
            # Use the Keras model for prediction
            predictions = disease_model.predict(x)
            print("Model predictions:", predictions)
           
            y = np.argmax(predictions, axis=1)[0]
            print("Predicted class index:", y)
            class_val = plant_disease_class[y]
            confidence = float(predictions[0][y] * 100)  # Convert to percentage
            json_op = {
                "disease": class_val,
                "confidence": confidence,
            }
            return jsonify(json_op)
        else:
            print("Invalid file type")
            return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        print(f"Error in /api/predict-disease: {str(e)}")
        return jsonify({"error": str(e)}), 500
'''
API endpoint to estimate height of plant from images using image processing
'''
# disease_model.summary()
# print(disease_model.input_shape)

# sample_image = np.random.rand(224, 224, 3)  # Example input
# sample_image = np.expand_dims(sample_image, axis=0)
# predictions = disease_model.predict(sample_image)
# print(predictions)
# Height estimation endpoint
@app.route('/api/predict-height', methods=['POST'])
def height():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_files(file.filename):
            filepath = os.path.join(app.config['height_folder'], file.filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            if image is None:
                return jsonify({"error": "Invalid image file"}), 400

            BASE_HEIGHT = 38.5
            image_array = np.array(image)
            blurred_frame = cv2.blur(image_array, (5, 5), 0)
            hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

            low_green = np.array([30, 10, 50])
            high_green = np.array([135, 255, 200])

            green_mask = cv2.inRange(hsv_frame, low_green, high_green)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if not contours:
                return jsonify({"error": "No contours found"}), 400

            biggest = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            blank_mask = np.zeros(image_array.shape, dtype=np.uint8)
            cv2.fillPoly(blank_mask, [biggest], (255, 255, 255))
            blank_mask = cv2.cvtColor(blank_mask, cv2.COLOR_BGR2GRAY)
            result = cv2.bitwise_and(image_array, image_array, mask=blank_mask)
            result = np.array(result)

            positions = np.nonzero(result)
            top = positions[0].min()
            bottom = positions[0].max()

            ratio = (bottom - top) / image.shape[0]
            height = ratio * BASE_HEIGHT

            return jsonify({"height": str(height)})
        else:
            return jsonify({"error": "Invalid file type"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

'''
API endpoint to estimate price of 7 different plants from historical(2012-2019) data
the problem is of time series prediction and model used is RandomforestRegressor
'''

@app.route('/api/predict-price', methods=["POST"])
def price():
    base = {
        "coconut": 5100,
        "cotton": 3600,
        "black_gram": 2800,
        "maize": 1175,
        "moong": 3500,
        "jute": 1675,
        "wheat": 1350
    }

    data = request.get_json()
    if not data or 'crop' not in data:
        return jsonify({"error": "Missing crop field"}), 400

    crop = data['crop'].lower()
    if crop not in base:
        return jsonify({"error": f"Invalid crop: {crop}"}), 400

    try:
        # Load the appropriate model based on the crop
        model_path = f"trained_models/{crop}_price_model.joblib"
        price_model = joblib.load(model_path)

        # Prepare input data (exclude 'crop' from the input values)
        input_values = [data[key] for key in data if key != 'crop']
        X = np.array(input_values).reshape(1, -1)

        # Predict the price
        predictions = price_model.predict(X)
        price = round((predictions[0] * base[crop]) / 100, 2)

        return jsonify({"price": price})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    






# @app.route('/api/chatbot', methods=['POST'])
# def chatbot():
#     data = request.json
#     user_message = data.get("message", "")

#     if not user_message:
#         return jsonify({"error": "No message provided"}), 400

#     # Tokenize input and generate a response
#     inputs = tokenizer(user_message, return_tensors="pt").to("cuda")
#     output = model.generate(**inputs, max_new_tokens=100)
#     response = tokenizer.decode(output[0], skip_special_tokens=True)

#     return jsonify({"response": response})


# Initialize Groq client (temporary hardcoded key)
groq_client = Groq(api_key="API_KEY")  # Replace with your key

@app.route('/api/farm-chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' field"}), 400

        # Call Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an agricultural expert."},
                {"role": "user", "content": data["message"]}
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=150
        )

        return jsonify({"response": chat_completion.choices[0].message.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(port=5000, debug=True)