import json

## input ##

menu_id = input("Id: ")
check = input("Test: ")

###########

foodbyid = open("../layouts/FoodsByID.json", "r")
data = json.load(foodbyid)
foodbyid.close()

menus = open("../layouts/menusById.json", "r")
menus_data = json.load(menus)
menus.close()

menu = menus_data[menu_id]

for day in menu:
    for meal in menu[day]:
        for id in menu[day][meal]:
            food = data[id]
            if food[check] == 1:
                print(id + " - " + food["Name"])



