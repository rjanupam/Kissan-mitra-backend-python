import io
import pickle

import numpy as np
import requests
import torch
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

import env
from utils.disease import disease_classes, disease_dic
from utils.fertilizer import fertilizer_dic, fertilizer_preprocess
from utils.model import ResNet9
from utils.yields import yield_preprocess

# model paths
crop_model_path = "models/crop_recommendere.pkl"
yield_model_path = "models/yield_predictor.pkl"
fertilizer_model_path = "models/fertilizer_recommender.pkl"
disease_model_path = "models/disease_teller.pth"

# open or initialize models
with open(crop_model_path, "rb") as f:
    crop_model = pickle.load(f)

# with open(yield_model_path, "rb") as f:
#    model = pickle.load(f)
#    yield_model, yield_encoder, yield_scaler = (
#        model["model"],
#        model["one_hot_encoder"],
#        model["scaler"],
#    )


with open(fertilizer_model_path, "rb") as f:
    fertilizer_model = pickle.load(f)

disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(
    torch.load(disease_model_path, map_location=torch.device("cpu"), weights_only=True)
)
disease_model.eval()


# Get weather and humidity for a city
def weather_fetch(city_name):
    api_key = env.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


# Transform the image to tensor and predict disease
def predict_image(img, model=disease_model):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # get predictions from model
    yb = model(img_u)
    # pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # retrieve the class label
    return prediction


# create app
app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({"msg": "namaste user"})


@app.route("/crop-predict", methods=["POST"])
def crop_prediction():
    if request.method == "POST":
        try:
            data = request.get_json()
            N = int(data["nitrogen"])
            P = int(data["phosphorous"])
            K = int(data["pottasium"])
            ph = float(data["ph"])
            rainfall = float(data["rainfall"])
            city = data.get("city")

            temperature, humidity = weather_fetch(city)
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = crop_model.predict(input_data)

            return jsonify({"prediction": prediction[0]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/yield-predict", methods=["POST"])
def yield_prediction():
    if request.method == "POST":
        try:
            data = request.get_json()
            N = int(data["nitrogen"])
            P = int(data["phosphorous"])
            K = int(data["pottasium"])
            ph = float(data["ph"])
            rainfall = float(data["rainfall"])
            area = float(data["area"])
            season = data.get("season")
            state = data.get("state")
            temperature = data.get("temperature")

            input_data = np.array(
                [state, season, crop, N, P, K, ph, rainfall, temperature, area]
            )

            X_input = yield_preprocess(yield_model, yield_encoder, yield_scaler)

            prediction = yield_model.predict(X_input)

            return jsonify({"prediction": prediction[0]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/fertilizer-predict", methods=["POST"])
def fertilizer_prediction():
    if request.method == "POST":
        try:
            data = request.get_json()
            N = int(data["nitrogen"])
            P = int(data["phosphorous"])
            K = int(data["pottasium"])
            moisture = int(data["moisture"])
            soil_type = data.get("soil_type")
            crop = data.get("crop")
            temperature = int(data["temperature"])
            humidity = int(data["humidity"])

            input_data = np.array(
                [temperature, humidity, moisture, soil_type, crop, N, P, K]
            )

            X_input = fertilizer_preprocess(input_data, fertilizer_model)

            prediction = fertilizer_model.predict(X_input)

            fertilizer_decoder = {
                0: "10-26-26",
                1: "14-35-14",
                2: "17-17-17",
                3: "20-20",
                4: "28-28",
                5: "DAP",
                6: "Urea",
            }

            decoded_prediction = fertilizer_decoder[prediction[0]]

            return jsonify(
                {
                    "prediction": decoded_prediction,
                    "description": fertilizer_dic[decoded_prediction],
                }
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/disease-predict", methods=["POST"])
def disease_prediction():
    if request.method == "POST":
        try:
            if "file" not in request.files:
                return jsonify({"error": "couldn't upload file"}), 500
            file = request.files.get("file")
            if not file:
                jsonify({"error": "no file found"}), 500

            img = file.read()

            prediction = predict_image(img)

            prediction = disease_dic.get(prediction, "Unknown Disease")
            return jsonify({"prediction": prediction})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)

