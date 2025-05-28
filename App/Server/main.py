from server import Server
from MenuGenerator import MenuGenerator
import torch
import os
import json
import joblib

MODEL_PATH = "model_weights.pth"
FOOD_NAMES_PATH = "Food_names.json"

if __name__ == "__main__":
    # Load the food names
    print("Loading food names...")
    with open(FOOD_NAMES_PATH, "r") as f:
        food_data = json.load(f)
    food_names = food_data["foods"]

    # Load the vectorizers and models
    print("Loading vectorizers and models...")
    char_vectorizer = joblib.load("char_vectorizer.pkl")
    char_nn = joblib.load("char_nn.pkl")
    word_vectorizer = joblib.load("word_vectorizer.pkl")
    word_nn = joblib.load("word_nn.pkl")

    # Load the model
    print("Loading model...")
    model = MenuGenerator()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True))
    model.eval()

    # Create the server instance with the model
    server = Server(model, food_names, char_vectorizer, char_nn, word_vectorizer, word_nn)

    # Run the server
    print("Starting server...")
    port = int(os.environ.get("PORT", 5000))
    server.run(port=port)
