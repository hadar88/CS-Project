import requests

def convert_to_dict(data):
    days = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    meals = ["breakfast", "lunch", "dinner"]

    nested_data = data.get("output", [])
    structured_dict = {}

    for i, group in enumerate(nested_data):
        group_key = days[i]
        structured_dict[group_key] = {}

        for j, sub_group in enumerate(group):
            sub_group_key = meals[j]
            structured_dict[group_key][sub_group_key] = {}

            for k, pair in enumerate(sub_group):
                structured_dict[group_key][sub_group_key][int(pair[0])] = round(pair[1])

    return structured_dict

SERVER_URL = "http://127.0.0.1:5000"

data = {'calories': 2826.6875, 'carbohydrates': 326.16, 'sugar': 27.18, 'fat': 72.48, 'protein': 190.26, 'vegetarian': 0, 'vegan': 0, 'eggs': 1, 'milk': 1, 'nuts': 1, 'fish': 1, 'sesame': 1, 'soy': 1, 'gluten': 1}  # Example input data

response = requests.post(f"{SERVER_URL}/predict", json=data)

if response.status_code == 200:
    result = response.json()
    result = convert_to_dict(result)
    print("Prediction:", result)
else:
    print(response.json())




