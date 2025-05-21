# This script adds nutritional data to the FoodsByName.json and FoodsByID.json files.
# The nutritional data is taken from the FoodDataDict.json file.
# The script will print an error message if it cannot find the nutritional data for a food.

import json

foods_by_name_file = open("../../layouts/FoodsByName.json", "a+")
foods_by_id_file = open("../../layouts/FoodsByID.json", "w")

foods_by_name_file.seek(0)
foods_by_id_file.seek(0)

foods_by_name: dict[str, dict] = json.load(foods_by_name_file)
foods_by_id: dict[str, dict] = dict()

foods_data_file = open("../../food_data/FoodData.json")
food_data = json.load(foods_data_file)
foods_data_file.close()

id = 1

for food_name in foods_by_name:
    data = food_data.get(food_name, None)

    if data is None:
        print(f"{food_name} : is not in the dict!")
    else:
        foods_by_name[food_name]["Protein"] = data["Protein"]
        foods_by_name[food_name]["Fat"] = data["Fat"]
        foods_by_name[food_name]["Carbohydrate"] = data["Carbohydrate"]
        foods_by_name[food_name]["Calories"] = data["Calories"]
        foods_by_name[food_name]["Sugars"] = data["Sugars"]

    foods_by_name[food_name]["ID"] = id

    foods_by_id[str(id)] = foods_by_name[food_name].copy()
    foods_by_id[str(id)]["Name"] = food_name
    del foods_by_id[str(id)]["ID"]

    id += 1

foods_by_name_file.seek(0)
foods_by_name_file.truncate(0)

json.dump(foods_by_name, foods_by_name_file, indent=4)
json.dump(foods_by_id, foods_by_id_file, indent=4)

foods_by_name_file.close()
foods_by_id_file.close()

# One line

db = open("../../layouts/FoodsByName.json")

dictionary = json.load(db)

db.close()

onelinedb = open("../../layouts/FoodsByNameOneLine.json", "w")

onelinedb.writelines("{\n")

for food in dictionary:
    onelinedb.write('"' + food + '": ')
    onelinedb.write(json.dumps(dictionary[food]["ID"]))
    onelinedb.write(',\n')

onelinedb.writelines(["}\n"])

onelinedb.close()
