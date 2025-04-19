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
                if food_id != 0:  # Ignore padding
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

if __name__ == "__main__":
    # Example usage

    menu_dict = {
        "sunday": {"breakfast": {"1": 180, "2": 28, "3": 150, "4": 34, "5": 28}, "lunch": {"6": 113, "7": 94, "8": 38, "9": 16, "10": 14, "11": 15}, "dinner": {"12": 113, "13": 130}},
        "monday": {"breakfast": {"14": 30, "1": 180, "15": 28, "16": 182, "17": 32}, "lunch": {"18": 113, "19": 14, "20": 30, "52": 30, "15": 28}, "dinner": {"6": 170, "21": 156}},
        "tuesday": {"breakfast": {"22": 170, "23": 40, "5": 28, "24": 120, "25": 150}, "lunch": {"26": 170, "27": 28, "28": 45, "29": 155, "30": 130}, "dinner": {"31": 170, "32": 28, "33": 30, "7": 20, "34": 15, "35": 20}},
        "wednesday": {"breakfast": {"22": 70, "36": 60, "37": 120, "38": 28, "39": 150, "30": 130}, "lunch": {"6": 170, "7": 94, "40": 40, "41": 40, "42": 40, "43": 15, "10": 14, "44": 2, "52": 30}, "dinner": {"26": 170, "45": 320}},
        "thursday": {"breakfast": {"52": 30, "61": 100, "17": 16, "53": 240, "46": 30}, "lunch": {"47": 113, "27": 28, "48": 32, "35": 20, "33": 30, "3": 150, "15": 28}, "dinner": {"6": 113, "49": 100, "50": 14}},
        "friday": {"breakfast": {"51": 30, "3": 56, "52": 30, "53": 60, "29": 155, "30": 130}, "lunch": {"6": 85, "28": 45, "22": 170, "55": 150}, "dinner": {"56": 170, "57": 120}},
        "saturday": {"breakfast": {"58": 136, "59": 18, "60": 30, "42": 40, "2": 28, "3": 160, "53": 240, "4": 38, "61": 50}, "lunch": {"56": 170, "57": 120, "37": 120}, "dinner": {"6": 113, "63": 100, "62": 15}}
    }

    foods = open(FOODS_DATA_PATH, "r")
    data = json.load(foods)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ten = torch.tensor([[[[[24.7251, 21.8422],
            [20.2098,  0.0000],
            [13.5923, 38.3061],
            [21.0261,  7.3387],
            [13.7382, 52.4606],
            [ 8.5545, 21.2482],
            [18.1183, 14.5507],
            [26.0516, 14.6971],
            [17.7127, 16.9934],
            [19.9651, 15.0946]],

            [[27.3392, 12.9963],
            [14.6118, 11.7356],
            [23.1820, 21.9667],
            [26.3321, 24.9973],
            [20.7667,  8.9025],
            [20.6361, 31.4491],
            [22.7500, 13.0351],
            [20.7971, 17.3625],
            [23.8740,  4.2911],
            [45.0321, 23.6374]],

            [[24.6408,  0.0000],
            [24.9140, 42.7165],
            [29.1588, 18.8793],
            [17.9495, 51.3707],
            [20.9338, 23.1527],
            [84.3686, 45.7167],
            [24.1501,  0.0000],
            [18.5495,  9.9085],
            [36.8568,  0.0000],
            [23.7292, 17.4119]]],


            [[[21.7226,  0.0000],
            [24.8239, 18.3086],
            [17.9845,  4.1908],
            [16.2955,  0.0000],
            [24.9316,  0.0000],
            [14.7707,  3.0663],
            [27.9465, 10.9985],
            [22.3880, 13.4821],
            [20.4170, 34.5249],
            [27.0602, 32.3638]],

            [[16.9330, 13.0883],
            [23.2792,  7.6294],
            [21.8348, 22.5848],
            [27.9277,  8.5610],
            [19.4706,  5.3703],
            [27.1263, 24.2191],
            [10.8774,  6.8077],
            [20.1560, 24.3540],
            [27.8026, 21.6405],
            [15.3443,  3.8012]],

            [[12.2099,  5.7560],
            [18.2767, 29.3988],
            [19.0580, 30.8444],
            [42.0666, 28.2931],
            [26.0403, 14.6516],
            [26.7143, 17.1283],
            [18.5047,  0.0000],
            [19.1227, 24.3444],
            [42.9286, 15.6138],
            [22.5554,  0.0000]]],


            [[[27.5079, 12.9490],
            [43.1941, 14.8523],
            [21.0484, 27.9952],
            [20.0056, 23.0520],
            [17.4264, 21.9869],
            [20.3246, 24.2620],
            [27.6393,  7.6371],
            [27.3123,  8.2627],
            [17.5047,  8.4383],
            [ 8.8964, 21.9446]],

            [[27.0003,  0.0000],
            [40.3277, 25.4157],
            [12.3987, 32.3906],
            [20.6631,  0.0000],
            [20.5861, 41.0583],
            [32.4265,  0.0000],
            [19.9276, 19.4163],
            [27.1270, 19.3892],
            [ 9.0926, 28.7138],
            [22.7354, 44.8555]],

            [[20.0805,  0.0000],
            [21.7916, 16.4552],
            [26.1043, 14.3198],
            [28.0263,  8.9762],
            [38.4883, 15.1261],
            [29.7713, 28.7019],
            [46.4464, 15.8995],
            [20.5526, 24.2803],
            [11.6583, 17.9970],
            [23.1508, 28.0984]]],


            [[[23.0332,  9.1994],
            [14.6770, 21.0757],
            [25.0712, 12.5195],
            [23.0573, 13.1739],
            [12.2505,  0.0000],
            [25.2483, 31.0022],
            [19.6562,  0.0000],
            [26.9047, 31.5881],
            [23.9741, 17.6309],
            [25.2769, 17.0482]],

            [[18.2059, 15.1640],
            [23.8331, 11.8971],
            [21.6739, 29.5171],
            [25.0825, 24.0550],
            [17.3878, 10.7532],
            [22.4476, 16.2005],
            [36.6614,  0.0000],
            [21.2615, 15.3880],
            [27.4047,  5.3885],
            [23.3023, 29.3043]],

            [[34.5039,  5.6001],
            [19.8067, 28.1158],
            [23.6731, 30.1630],
            [25.0695, 12.5124],
            [24.5085, 11.2826],
            [23.3798, 20.4582],
            [10.5047,  0.0000],
            [22.4573, 11.8531],
            [21.0666,  8.1221],
            [34.2744, 15.2537]]],


            [[[40.1918, 29.5984],
            [15.2152, 27.9996],
            [22.6588,  0.0000],
            [21.1757, 14.1124],
            [12.5138,  0.0000],
            [33.9734, 13.8114],
            [14.3070, 22.1259],
            [29.7886, 27.2137],
            [27.9313, 50.1765],
            [19.6145, 16.6973]],

            [[22.3383,  0.0000],
            [24.2204,  9.8928],
            [29.8110, 22.3119],
            [14.8976, 30.0450],
            [18.6718,  0.0000],
            [21.0605, 16.0451],
            [20.7844, 12.1784],
            [ 7.1707, 25.5324],
            [21.9192,  6.0474],
            [30.0074, 17.3491]],

            [[28.0183,  6.8354],
            [29.8417,  9.7583],
            [27.6209, 13.3881],
            [20.9872,  6.1210],
            [29.7995,  0.0000],
            [ 8.6700,  6.5829],
            [25.9411, 15.6550],
            [19.0807,  0.0000],
            [22.0232,  0.0000],
            [24.3153, 29.8153]]],


            [[[24.2973,  2.0305],
            [21.3864, 22.0085],
            [16.0568, 26.4590],
            [14.6386, 26.9865],
            [30.4508, 18.7999],
            [21.4261,  2.6214],
            [27.7016, 22.0408],
            [29.2361,  0.0000],
            [18.9564, 19.3016],
            [15.0473,  0.0000]],

            [[20.9966,  1.4636],
            [ 9.7844, 12.5111],
            [29.7456,  2.7906],
            [26.9487, 13.8081],
            [22.8266, 13.3858],
            [25.9805, 12.9089],
            [11.6529,  0.0000],
            [13.8335,  3.1992],
            [28.1613, 39.6997],
            [49.2281, 12.2277]],

            [[15.4549, 17.9111],
            [23.2725, 12.9398],
            [12.3385,  0.0000],
            [21.9929,  0.0000],
            [14.1712, 20.3432],
            [11.0821,  3.3074],
            [30.0399,  3.3315],
            [26.5599,  6.1440],
            [26.5970,  7.5128],
            [25.1608, 27.7941]]],


            [[[14.6759, 32.3789],
            [15.5657, 24.2816],
            [34.1050, 18.2086],
            [27.3771,  2.8466],
            [45.9783, 20.3788],
            [19.3179, 11.0421],
            [30.1028, 17.7936],
            [36.9979, 21.7227],
            [23.2902, 28.3206],
            [21.5158,  0.0000]],

            [[13.6581, 51.3819],
            [ 8.7741, 31.0792],
            [33.4961, 43.7025],
            [14.8971,  0.0000],
            [21.5257,  0.0000],
            [21.4873, 30.7684],
            [24.2222, 28.3950],
            [29.6572, 24.2437],
            [23.4649,  0.0000],
            [22.0320, 24.8325]],

            [[25.2443, 15.6961],
            [18.0585, 24.2008],
            [29.9583,  0.0000],
            [26.8653,  0.0000],
            [27.4951, 13.4272],
            [13.5491, 12.7010],
            [21.3453, 27.3568],
            [25.2820, 25.0529],
            [24.5345,  6.4614],
            [10.7857, 29.9506]]]]])

    # data = transform_batch2(ten, data, device)
    # printing = ["Calories", "Calories1", "Calories2", "Calories3", "Calories MSE", "Carbohydrate", 
    #             "Sugars", "Fat", "Protein", "Fruit", "Vegetable", "Cheese", "Meat", "Cereal", 
    #             "Vegetarian", "Vegan", "Contains eggs", "Contains milk", "Contains peanuts or nuts", 
    #             "Contains fish", "Contains sesame", "Contains soy", "Contains gluten"]
    # for i, v in enumerate(data[0]):
    #     print(f"{printing[i]}: {v.item():.0f}")
