from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained objects
model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
encoder = joblib.load(os.path.join(BASE_DIR, "encoder.joblib"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()

        # Get input values
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])

        # Arrange features in EXACTLY the same order
        # used while training the model
        features = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        # Apply the scaler used during training
        features_scaled = scaler.transform(features)

        # Predict encoded class
        prediction = model.predict(features_scaled)

        # Convert encoded class back to species name
        species = encoder.inverse_transform(prediction)[0]

        return jsonify({
            "success": True,
            "species": species
        })

    except KeyError as e:
        return jsonify({
            "success": False,
            "error": f"Missing input field: {e}"
        }), 400

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": f"Invalid input value: {e}"
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)