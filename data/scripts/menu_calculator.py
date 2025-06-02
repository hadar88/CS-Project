import json

foodbyid = open("../layouts/FoodsByID.json", "r")
data = json.load(foodbyid)
foodbyid.close()

menus = open("../layouts/menusById.json", "r")
menudata = json.load(menus)
menus.close()

## fiil the data

begin = 6401
end = 6500

for menu_id in range(begin, end + 1):
    days = menudata[str(menu_id)]

    daily_calories = [0, 0, 0, 0, 0, 0, 0]

    Calories = 0
    Calories1 = 0
    Calories2 = 0
    Calories3 = 0
    Calories_MSE = 0
    Carbohydrate = 0
    Sugars = 0
    Fat = 0
    Protein = 0
    Fruit = 0
    Vegetable = 0
    Cheese = 0
    Meat = 0
    Cereal = 0
    Vegetarian = 1
    Vegan = 1
    Contains_eggs = 0
    Contains_milk = 0
    Contains_peanuts_or_nuts = 0
    Contains_fish = 0
    Contains_sesame = 0
    Contains_soy = 0
    Contains_gluten = 0

    for i, day in enumerate(days):
        breakfast = days[day]["breakfast"]
        lunch = days[day]["lunch"]
        dinner = days[day]["dinner"]

        for meal in [breakfast, lunch, dinner]:
            for id in meal:
                grams = meal[id]
                food = data[id]
                daily_calories[i] += food["Calories"] * (grams / 100)
                Calories = Calories + food["Calories"] * (grams / 100)
                Carbohydrate = Carbohydrate + food["Carbohydrate"] * (grams / 100)
                Sugars = Sugars + food["Sugars"] * (grams / 100)
                Fat = Fat + food["Fat"] * (grams / 100)
                Protein = Protein + food["Protein"] * (grams / 100)

                if meal == breakfast:
                    Calories1 = Calories1 + food["Calories"] * (grams / 100)
                if meal == lunch:
                    Calories2 = Calories2 + food["Calories"] * (grams / 100)
                if meal == dinner:
                    Calories3 = Calories3 + food["Calories"] * (grams / 100)
                
                if food["Fruit"] == 1:
                    Fruit = Fruit + 1
                if food["Vegetable"] == 1:
                    Vegetable = Vegetable + 1
                if food["Cheese"] == 1:
                    Cheese = Cheese + 1
                if food["Meat"] == 1:
                    Meat = Meat + 1
                if food["Cereal"] == 1:
                    Cereal = Cereal + 1
                if food["Vegetarian"] == 0:
                    Vegetarian = 0
                if food["Vegan"] == 0:
                    Vegan = 0
                if food["Contains eggs"] == 1:
                    Contains_eggs = 1
                if food["Contains milk"] == 1:
                    Contains_milk = 1
                if food["Contains peanuts or nuts"] == 1:
                    Contains_peanuts_or_nuts = 1
                if food["Contains fish"] == 1:
                    Contains_fish = 1
                if food["Contains sesame"] == 1:
                    Contains_sesame = 1
                if food["Contains soy"] == 1:
                    Contains_soy = 1
                if food["Contains gluten"] == 1:
                    Contains_gluten = 1

    ###

    Calories = Calories / 7
    Calories1 = Calories1 / 7
    Calories2 = Calories2 / 7
    Calories3 = Calories3 / 7

    Calories_MSE = 1/7 * sum([(daily_calories[i] - Calories) ** 2 for i in range(7)])

    Carbohydrate = Carbohydrate / 7
    Sugars = Sugars / 7
    Fat = Fat / 7
    Protein = Protein / 7

    Calories = round(Calories, 3)
    Calories1 = round(Calories1, 3)
    Calories2 = round(Calories2, 3)
    Calories3 = round(Calories3, 3)
    Calories_MSE = round(Calories_MSE, 3)
    Carbohydrate = round(Carbohydrate, 3)
    Sugars = round(Sugars, 3)
    Fat = round(Fat, 3)
    Protein = round(Protein, 3)

    menusinput = open("../layouts/MenusInput.json", "r+")
    menusinput.seek(0)

    menus = json.load(menusinput)
    data1 = menus[str(menu_id)]
    Menu = data1.get("Menu")
    Menu["Calories"] = Calories
    Menu["Calories1"] = Calories1
    Menu["Calories2"] = Calories2
    Menu["Calories3"] = Calories3
    Menu["Calories MSE"] = Calories_MSE
    Menu["Carbohydrate"] = Carbohydrate
    Menu["Sugars"] = Sugars
    Menu["Fat"] = Fat
    Menu["Protein"] = Protein
    Menu["Fruit"] = Fruit
    Menu["Vegetable"] = Vegetable
    Menu["Cheese"] = Cheese
    Menu["Meat"] = Meat
    Menu["Cereal"] = Cereal
    Menu["Vegetarian"] = Vegetarian
    Menu["Vegan"] = Vegan
    Menu["Contains eggs"] = Contains_eggs
    Menu["Contains milk"] = Contains_milk
    Menu["Contains peanuts or nuts"] = Contains_peanuts_or_nuts
    Menu["Contains fish"] = Contains_fish
    Menu["Contains sesame"] = Contains_sesame
    Menu["Contains soy"] = Contains_soy
    Menu["Contains gluten"] = Contains_gluten

    menusinput.seek(0)
    menusinput.truncate()
    json.dump(menus, menusinput, indent=4)
    menusinput.close()
    print(f'menu_id {menu_id} calculated successfully!')

# Move data from the "menu" section to the "Initial" section

menus_input = open("../layouts/menusInput.json", "a+")

menus_input.seek(0)

menus = json.load(menus_input)

for number in menus:
    data = menus[number]
    initial = data.get("Initial")
    menu = data.get("Menu")
    
    initial["Calories"] = menu["Calories"]
    initial["Carbohydrate"] = menu["Carbohydrate"]
    initial["Sugars"] = menu["Sugars"]
    initial["Fat"] = menu["Fat"]
    initial["Protein"] = menu["Protein"]
    initial["Vegetarian"] = menu["Vegetarian"]
    initial["Vegan"] = menu["Vegan"]
    initial["Contains eggs"] = menu["Contains eggs"]
    initial["Contains milk"] = menu["Contains milk"]
    initial["Contains peanuts or nuts"] = menu["Contains peanuts or nuts"]
    initial["Contains fish"] = menu["Contains fish"]
    initial["Contains sesame"] = menu["Contains sesame"]
    initial["Contains soy"] = menu["Contains soy"]
    initial["Contains gluten"] = menu["Contains gluten"]

menus_input.seek(0)
menus_input.truncate()
json.dump(menus, menus_input, indent=4)

menus_input.close()

# menusById to menusByName

# menus = open("../layouts/menusById.json", "r")
# menusById = json.load(menus)
# menusName = {}
# menus.close()

# foods = open("../layouts/FoodsByID.json", "r")
# foodsByid = json.load(foods)
# foods.close()

# for menuid in menusById:
#     for day in menusById[menuid]:
#         for meal in menusById[menuid][day]:
#             for foodid in menusById[menuid][day][meal]:
#                 if foodid not in foodsByid:
#                     print("Food not found: " + foodid)
#                     continue
#                 foodname = foodsByid[foodid]["Name"]
#                 amount = menusById[menuid][day][meal][foodid]

#                 if menuid not in menusName:
#                     menusName[menuid] = {}
#                 if day not in menusName[menuid]:
#                     menusName[menuid][day] = {}
#                 if meal not in menusName[menuid][day]:
#                     menusName[menuid][day][meal] = {}

#                 menusName[menuid][day][meal][foodname] = amount

# menusByName = open("../layouts/menusByName.json", "w")
# json.dump(menusName, menusByName, indent=4)
# menusByName.close()

# db = open("../layouts/menusByName.json")
# dict = json.load(db)
# db.close()

# onelinedb = open("../layouts/menusByName.json", "w")

# onelinedb.writelines("{\n")

# for id in dict:
#     onelinedb.writelines('\t"' + id + '": {\n')
#     for day in dict[id]:
#         onelinedb.writelines('\t\t"' + day + '": ')
#         onelinedb.writelines(json.dumps(dict[id][day]))
#         if day != "saturday":
#             onelinedb.writelines(",\n")
#         else:
#             onelinedb.writelines("\n")
#     onelinedb.writelines("\t},\n")

# onelinedb.writelines("}")
# onelinedb.close()