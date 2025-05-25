import json
import torch

FOODS_DATA_PATH = "../../Data/layouts/FoodsByID.json"

def menu_dict_to_tensor(menu_dict: dict):
    """This function is used to convert the menu dictionary to a tensor."""

    ten = []
    max_foods = max(len(menu_dict[day][meal]) for day in menu_dict for meal in menu_dict[day])
    for day in menu_dict:
        d = []
        for meal in menu_dict[day]:
            m = []
            for food in menu_dict[day][meal]:
                f = [int(food), menu_dict[day][meal][food]]
                m.append(f)
            while len(m) < max_foods:
                m.append([0, 0])  # Padding with zeros
            d.append(m)
        ten.append(d)
    return torch.tensor(ten)

def menu_tensor_to_dict(menu: torch.Tensor):
    """This function is used to convert the menu tensor to a dictionary."""

    days = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    meals = ["breakfast", "lunch", "dinner"]
    menu_dict = {}
    for i, day in enumerate(days):
        menu_dict[day] = {}
        for j, meal in enumerate(meals):
            menu_dict[day][meal] = {}
            for food in menu[i][j]:
                food_id, amount = food.tolist()
                if food_id != 0 and amount != 0:  # Ignore padding
                    menu_dict[day][meal][str(food_id)] = amount
    return menu_dict

def transform(menu: torch.Tensor, food_data, device):
    """This function is used to transform the menu to the menu data."""

    output = torch.zeros(23, dtype=torch.int32, device=device)

    output[14], output[15] = 1, 1  # vegetarian, vegan

    daily_calories = torch.zeros(7, device=device)

    total_calories, carbs, sugars, fats, proteins = 0, 0, 0, 0, 0

    meal_calories = torch.zeros(3, device=device)

    menu = menu.to(device)  # Move menu tensor to GPU

    for didx, day in enumerate(menu):
        for midx, meal in enumerate(day):
            for food in meal:
                food_id, food_amount = food.int().tolist()

                if food_id == 0:
                    continue

                food_amount /= 100

                food_nut = food_data[str(food_id)]

                food_calories = food_nut["Calories"] * food_amount

                daily_calories[didx] += food_calories

                total_calories += food_calories

                carbs += food_nut["Carbohydrate"] * food_amount
                sugars += food_nut["Sugars"] * food_amount
                fats += food_nut["Fat"] * food_amount
                proteins += food_nut["Protein"] * food_amount

                meal_calories[midx] += food_calories

                output[9] += food_nut["Fruit"]
                output[10] += food_nut["Vegetable"]
                output[11] += food_nut["Cheese"]
                output[12] += food_nut["Meat"]
                output[13] += food_nut["Cereal"]

                output[14] *= food_nut["Vegetarian"]
                output[15] *= food_nut["Vegan"]

                output[16] |= food_nut["Contains eggs"]
                output[17] |= food_nut["Contains milk"]
                output[18] |= food_nut["Contains peanuts or nuts"]
                output[19] |= food_nut["Contains fish"]
                output[20] |= food_nut["Contains sesame"]
                output[21] |= food_nut["Contains soy"]
                output[22] |= food_nut["Contains gluten"]

    output[0] = total_calories // 7
    output[1] = meal_calories[0] // 7
    output[2] = meal_calories[1] // 7
    output[3] = meal_calories[2] // 7
    output[4] = int((1 / 7) * sum((dcal - (total_calories / 7)) ** 2 for dcal in daily_calories))
    output[5] = carbs // 7
    output[6] = sugars // 7
    output[7] = fats // 7
    output[8] = proteins // 7

    return output.clone().detach().float().to(device)

def transform_batch(menu_batch: torch.Tensor, food_data, device):
    return torch.stack([transform(menu, food_data, device) for menu in menu_batch])

def transform_batch2(menu_batch: torch.Tensor, food_data, device):
    return torch.stack([transform2(menu, food_data, device) for menu in menu_batch])

def transform2(menu: torch.Tensor, food_data, device, bound_fn=lambda x: x):
    output = torch.zeros(23, dtype=torch.float32, device=device)

    output[14] = 1.0
    output[15] = 1.0
    daily_calories = torch.zeros(7, device=device)
    
    for didx, day in enumerate(menu):
        for midx, meal in enumerate(day):
            for food in meal:
                # food: [id, amount]
                food_id = int(bound_fn(food[0]).item())
                food_amount = food[1].item() / 100.0
                
                if food_id == 0:
                    continue
                    
                food_nut = food_data[str(food_id)]
                food_calories = food_nut["Calories"] * food_amount
                
                # Update accumulators
                daily_calories[didx] += food_calories
                output[0] = output[0] + food_calories
                output[midx + 1] = output[midx + 1] + food_calories
 
                output[5] = output[5] + food_nut["Carbohydrate"] * food_amount
                output[6] = output[6] + food_nut["Sugars"] * food_amount
                output[7] = output[7] + food_nut["Fat"] * food_amount
                output[8] = output[8] + food_nut["Protein"] * food_amount
                output[9] = output[9] + food_nut["Fruit"]
                output[10] = output[10] + food_nut["Vegetable"]
                output[11] = output[11] + food_nut["Cheese"]
                output[12] = output[12] + food_nut["Meat"]
                output[13] = output[13] + food_nut["Cereal"]
                output[14] = output[14] * food_nut["Vegetarian"]
                output[15] = output[15] * food_nut["Vegan"]
                output[16] = torch.maximum(output[16], torch.tensor(float(food_nut["Contains eggs"]), device=menu.device))
                output[17] = torch.maximum(output[17], torch.tensor(float(food_nut["Contains milk"]), device=menu.device))
                output[18] = torch.maximum(output[18], torch.tensor(float(food_nut["Contains peanuts or nuts"]), device=menu.device))
                output[19] = torch.maximum(output[19], torch.tensor(float(food_nut["Contains fish"]), device=menu.device))
                output[20] = torch.maximum(output[20], torch.tensor(float(food_nut["Contains sesame"]), device=menu.device))
                output[21] = torch.maximum(output[21], torch.tensor(float(food_nut["Contains soy"]), device=menu.device))
                output[22] = torch.maximum(output[22], torch.tensor(float(food_nut["Contains gluten"]), device=menu.device))
    
    # Division operations
    output_final = torch.clone(output)
    output_final[:4] = output[:4] / 7.0
    
    # Calculate daily calorie variance
    output_final[4] = torch.mean((daily_calories - output_final[0]) ** 2)
    
    output_final[5:9] = output[5:9] / 7.0
    
    # Make it require gradients at the end of all operations
    return output_final.requires_grad_(True).int()

def check_menu(ten):
    foods = open(FOODS_DATA_PATH, "r")
    data = json.load(foods)

    menu = menu_tensor_to_dict(ten)        
    temp_menu = {}
    for day in menu:
        thisday = {}
        for meal in menu[day]:
            thismeal = {}
            for food in menu[day][meal]:
                amount = menu[day][meal][food]
                food_id = food[:-2]
                food_name = data[food_id]["Name"]
                thismeal[food_name] = amount
            thisday[meal] = thismeal
        temp_menu[day] = thisday

    with open("check_menu.json", "w") as f:
        json.dump(temp_menu, f, indent=4)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data = transform2(ten, data, device)
    # values = ["Calories", "Calories1", "Calories2", "Calories3", "Calories MSE", "Carbohydrate", 
    #             "Sugars", "Fat", "Protein", "Fruit", "Vegetable", "Cheese", "Meat", "Cereal", 
    #             "Vegetarian", "Vegan", "Contains eggs", "Contains milk", "Contains peanuts or nuts", 
    #             "Contains fish", "Contains sesame", "Contains soy", "Contains gluten"]
    # for i, v in enumerate(data):
    #     print(f"{values[i]}: {v.item():.0f}")

if __name__ == "__main__":
    ten = torch.tensor([[[[195.0000,  72.8652],
          [ 70.0000,  82.1609],
          [ 20.0000,  67.9633],
          [ 17.0000, 120.1937],
          [ 17.0000,  35.2280],
          [  0.0000,   0.0000],
          [201.0000,   5.5175],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]],

         [[196.0000,  37.9129],
          [ 25.0000, 122.0889],
          [ 25.0000, 138.6339],
          [102.0000,  39.2521],
          [208.0000,  65.4394],
          [103.0000,  67.9323],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]],

         [[ 57.0000,  82.7044],
          [ 10.0000,  17.7364],
          [205.0000,  93.8205],
          [147.0000,  18.4788],
          [117.0000, 112.5597],
          [176.0000,  62.7684],
          [  0.0000,   6.3454],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]]],


        [[[108.0000,  75.7072],
          [  4.0000,  95.2259],
          [ 71.0000,  64.9434],
          [ 24.0000,  52.4722],
          [201.0000,  53.2095],
          [  0.0000,  13.1947],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]],

         [[194.0000, 212.0798],
          [ 33.0000,  45.5795],
          [ 67.0000,  46.5932],
          [ 30.0000,  71.9665],
          [ 24.0000,  63.8938],
          [ 30.0000,  65.7390],
          [112.0000,  37.9434],
          [208.0000,  46.1516],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]],

         [[ 77.0000, 157.9688],
          [ 78.0000, 103.8433],
          [ 10.0000,  17.7707],
          [ 34.0000,  26.5792],
          [121.0000,  44.6433],
          [207.0000, 141.7262],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]]],


        [[[ 54.0000, 242.5891],
          [ 17.0000,  28.5167],
          [  4.0000,  84.1748],
          [ 80.0000,  54.5394],
          [ 15.0000,  33.7786],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]],

         [[ 42.0000,  51.2465],
          [ 33.0000,  54.7478],
          [ 67.0000,  65.9749],
          [195.0000,  57.3183],
          [ 83.0000,  45.4481],
          [208.0000,  79.2422],
          [ 15.0000,  36.7930],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]],

         [[ 21.0000,  51.1688],
          [ 95.0000,  23.3394],
          [213.0000, 191.0968],
          [ 95.0000,  57.5799],
          [ 30.0000,  67.8689],
          [  0.0000,  20.1168],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]]],


        [[[ 33.0000,  43.3929],
          [ 61.0000,  52.2753],
          [ 61.0000,  91.4006],
          [ 80.0000,  48.6495],
          [201.0000,  97.7489],
          [  0.0000,  16.5358],
          [  0.0000,  39.4045],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]],

         [[ 87.0000,  91.6125],
          [ 88.0000,  57.1842],
          [ 33.0000,  51.2253],
          [ 90.0000,  51.4133],
          [ 91.0000,  32.0800],
          [ 73.0000,  48.8369],
          [ 93.0000,  50.7090],
          [201.0000,  51.0456],
          [138.0000,  35.4000],
          [  0.0000,   0.0000]],

         [[ 94.0000, 151.9811],
          [ 21.0000,  93.0546],
          [ 95.0000,  23.1677],
          [ 95.0000,  30.0546],
          [127.0000,  34.6077],
          [214.0000,  75.5555],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]]],


        [[[127.0000,  70.1987],
          [ 61.0000, 116.2103],
          [200.0000,  59.0559],
          [201.0000,  55.2064],
          [205.0000,  32.6199],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]],

         [[108.0000,  77.0150],
          [195.0000,  65.9727],
          [ 80.0000,  80.4162],
          [201.0000,  78.3637],
          [ 80.0000,  47.6488],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,  15.4341],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]],

         [[126.0000, 379.3835],
          [ 57.0000, 141.8787],
          [ 10.0000,  16.8826],
          [138.0000,  36.6463],
          [117.0000, 110.8106],
          [  0.0000,  24.7680],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]]],


        [[[ 64.0000,  38.6753],
          [ 36.0000,  57.5725],
          [ 92.0000,  54.5052],
          [205.0000, 207.9473],
          [201.0000,  66.3696],
          [  0.0000,   4.5613],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]],

         [[ 98.0000,  25.2268],
          [ 33.0000,  40.1409],
          [ 73.0000,  87.8315],
          [ 30.0000,  81.0332],
          [208.0000,  80.9879],
          [138.0000,  47.3951],
          [102.0000,  99.9945],
          [  0.0000,  10.3158],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]],

         [[101.0000, 168.9517],
          [102.0000,  93.9566],
          [103.0000,  68.9578],
          [ 10.0000,  40.6872],
          [104.0000,  40.8949],
          [ 53.0000, 202.6727],
          [ 81.0000,   5.2029],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]]],


        [[[ 90.0000,  44.9896],
          [ 42.0000,  86.2854],
          [ 70.0000,  57.1521],
          [ 37.0000,  49.1608],
          [  0.0000,   7.9126],
          [  0.0000,   0.0000],
          [200.0000,  17.2218],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]],

         [[ 87.0000,  73.0662],
          [ 76.0000,  35.2448],
          [ 33.0000,  34.6412],
          [ 90.0000,  40.9504],
          [ 88.0000,  55.8194],
          [ 33.0000,  47.6917],
          [138.0000,  39.8623],
          [208.0000,  30.6030],
          [138.0000,  13.1295],
          [  0.0000,   0.0000]],

         [[194.0000, 202.1490],
          [ 41.0000,  82.0541],
          [ 24.0000,  40.1820],
          [ 72.0000,  54.8401],
          [121.0000,  44.1549],
          [102.0000,  49.1918],
          [  0.0000, 105.0770],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000],
          [  0.0000,   0.0000]]]])
    
    check_menu(ten)
