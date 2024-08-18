Kissan Mitra Backend
Kissan Mitra is a backend service for a comprehensive agricultural support system, providing functionalities for crop recommendation, yield prediction, fertilizer recommendation, and disease prediction. The backend is built using Python and Flask, with various machine learning models to support its features.

Features
Crop Prediction: Recommends crops based on soil and environmental parameters.
Yield Prediction: Estimates crop yield based on various agricultural factors.
Fertilizer Recommendation: Suggests the right type of fertilizer for optimal crop growth.
Disease Prediction: Identifies plant diseases from images.
Setup
Prerequisites
Python 3.x
Anaconda or any Python environment manager
Required Python packages (listed below)
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/Amanw-25/Kissan-mitra-backend-python.git
cd Kissan-mitra-backend-python
Set up the Python environment:

Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Models
The project uses pre-trained models for various predictions:

Crop Recommendation Model: models/crop_recommender.pkl
Yield Prediction Model: models/yield_predictor.pkl
Fertilizer Recommendation Model: models/fertilizer_recommender.pkl
Disease Prediction Model: models/disease_teller.pth
Ensure these model files are located in the models/ directory.

Running the Application
Start the Flask server:

bash
Copy code
python app.py
API Endpoints:

Crop Prediction:

Endpoint: /crop-predict
Method: POST
Payload Example:
json
Copy code
{
  "nitrogen": 50,
  "phosphorous": 50,
  "pottasium": 50,
  "ph": 7,
  "rainfall": 200,
  "city": "Bhopal"
}
Yield Prediction:

Endpoint: /yield-predict
Method: POST
Payload Example:
json
Copy code
{
  "temperature": 26,
  "rainfall": 100,
  "state": "karnataka",
  "season": "rabi",
  "crop": "rice",
  "nitrogen": 50,
  "pottasium": 50,
  "phosphorous": 50,
  "pH": 6.5,
  "area": 1000
}
Fertilizer Recommendation:

Endpoint: /fertilizer-predict
Method: POST
Payload Example:
json
Copy code
{
  "nitrogen": 50,
  "phosphorous": 50,
  "pottasium": 50,
  "moisture": 30,
  "soil_type": "Loamy",
  "crop": "Wheat",
  "temperature": 25,
  "humidity": 60
}
Disease Prediction:

Endpoint: /disease-predict
Method: POST
File Upload: An image file of the plant leaf
Contributing
If you want to contribute to the project:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -am 'Add new feature').
Push to the branch (git push origin feature-branch).
Create a new Pull Request.
