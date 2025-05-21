import json

def generate_data(start, end):
    template = {
        "Initial": {
            "Calories": 0,
            "Carbohydrate": 0,
            "Sugars": 0,
            "Fat": 0,
            "Protein": 0,
            "Vegetarian": 0,
            "Vegan": 0,
            "Contains eggs": 0,
            "Contains milk": 0,
            "Contains peanuts or nuts": 0,
            "Contains fish": 0,
            "Contains sesame": 0,
            "Contains soy": 0,
            "Contains gluten": 0
        },
        "Menu": {
            "Calories": 0,
            "Calories1": 0,
            "Calories2": 0,
            "Calories3": 0,
            "Calories MSE": 0,
            "Carbohydrate": 0,
            "Sugars": 0,
            "Fat": 0,
            "Protein": 0,
            "Fruit": 0,
            "Vegetable": 0,
            "Cheese": 0,
            "Meat": 0,
            "Cereal": 0,
            "Vegetarian": 0,
            "Vegan": 0,
            "Contains eggs": 0,
            "Contains milk": 0,
            "Contains peanuts or nuts": 0,
            "Contains fish": 0,
            "Contains sesame": 0,
            "Contains soy": 0,
            "Contains gluten": 0
        }
    }
    
    data = {str(i): template for i in range(start, end + 1)}
    
    with open("templates.json", "w") as f:
        json.dump(data, f, indent=4)

generate_data(6101, 6500)

print("Templates generated successfully!")
