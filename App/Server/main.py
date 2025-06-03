from server import Server
import os
import json
import joblib

FOOD_NAMES_PATH = "Food_names.json"
FOODS_DATA_PATH = "FoodsByID.json"
FOODS_ALTERNATIVES_PATH = "FoodAlternatives.json"
MEALS_PATH = "Meals.json"

if __name__ == "__main__":
    # Load the food names
    with open(FOOD_NAMES_PATH, "r") as f:
        food_data = json.load(f)
    food_names = food_data["foods"]

    # Load the vectorizers and models
    char_vectorizer = joblib.load("char_vectorizer.pkl")
    char_nn = joblib.load("char_nn.pkl")
    word_vectorizer = joblib.load("word_vectorizer.pkl")
    word_nn = joblib.load("word_nn.pkl")

    # Load the foods data
    with open(FOODS_DATA_PATH, "r") as f:
        foods_data = json.load(f)
    with open(FOODS_ALTERNATIVES_PATH, "r") as f:
        foods_alternatives = json.load(f)
    with open(MEALS_PATH, "r") as f:
        meals = json.load(f)

    # Create the server instance with the model
    server = Server(food_names, char_vectorizer, char_nn, word_vectorizer, word_nn, foods_data, foods_alternatives, meals)

    # Run the server
    port = int(os.environ.get("PORT", 5000))
    server.run(port=port)
