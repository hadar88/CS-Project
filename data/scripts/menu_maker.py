################################################################################
# This sctipt is used to convert a menu to a new menu with a desired property. #
# I assumed a new format for the food data.                                    #
# First run "reformat_foodsbid.py" to reformat the food data to the new format.#
################################################################################

import json
import random
import copy

# ! Paths relative to the root folder
MENUS_FILE = "../layouts/menusById.json"
FOODS_FILE = "../layouts/FoodsByID.json"  
ALTER_FILE = "../layouts/FoodAlternatives.json" 

w = open("new_menus.json", "w")

def read_menu(id: int):
    with open(MENUS_FILE, "r") as f:
        menus = json.load(f)
        return menus[str(id)]


def read_food(id: int):
    with open(FOODS_FILE, "r") as f:
        foods = json.load(f)
        return foods[str(id)]
    

def get_alternatives(food_id: int, property: str):
    with open(ALTER_FILE, "r") as f:
        alternatives = json.load(f)
        return alternatives[str(food_id)][property]


def fix_menu(property: str, menu_id: int, exhaustive: bool = False) -> dict | list[dict]:
    """
    Covert some menu to a new menu with the desired property.
    For examle non-vegetarian to vegetarian, or one with lactose to one without lactose.

    Args:
        property (str): The problematic property (e.g. "Vegetarian", "Vegan", "Lactose").
        menu_id (int): The id of the menu to convert.
        exhaustive (bool, optional): Whether to try all alternatives or just a random one. Defaults to False.

    Raises:
        ValueError: When there are no alternatives for the property.

    Returns:
        dict | list[dict]: The new menu or a list of new menus if exhaustive.
    """

    invalid_menus = [read_menu(menu_id)]
    fixed_menus = []
    fixed_menus_set = set()

    while invalid_menus:
        current_menu = invalid_menus.pop()

        try:
            fixed_one_step = fix_menu_one_step(current_menu, property, 1 if property in ["Vegetarian", "Vegan"] else 0, exhaustive)
        except ValueError as e:
            print(f"ERROR: in menu {menu_id} {e}")
            continue

        if not fixed_one_step:
            menu_str = json.dumps(current_menu, sort_keys=True)

            if menu_str not in fixed_menus_set:
                fixed_menus_set.add(menu_str)
                fixed_menus.append(current_menu)
                print_menu(current_menu)

                if len(fixed_menus) > 25:
                    return fixed_menus
        else:
            invalid_menus.extend(fixed_one_step)

    return fixed_menus


def fix_menu_one_step(menu: dict, property, desired_value, exhaustive):
    fixed_menus = []

    # day is the name of the day (e.g. "Monday")
    for day in menu:
        # meal is the name of the meal (e.g. "Breakfast")
        for meal in menu[day]:
            meal_foods = menu[day][meal]

            for food_id in meal_foods:
                food = read_food(food_id)

                if food[property] != desired_value:
                    alternatives = get_alternatives(food_id, property)

                    if alternatives == []:
                        raise ValueError(
                            f"No {property} alternatives for food {food_id}"
                        )
                                                        
                    if alternatives[0] == -1:
                        new_menu = copy.deepcopy(menu)
                        new_meal = copy.deepcopy(meal_foods)
                        del new_meal[food_id]
                        new_menu[day][meal] = new_meal
                        fixed_menus.append(new_menu)
                        return fixed_menus

                    amount = meal_foods[food_id]

                    if exhaustive:
                        for alternative in alternatives:
                            # For each alternative, create a new menu with the alternative food.
                            new_menu = copy.deepcopy(menu)
                            new_meal = copy.deepcopy(meal_foods)
                            del new_meal[food_id]
                            new_meal[str(alternative)] = amount
                            new_menu[day][meal] = new_meal
                            fixed_menus.append(new_menu)

                        return fixed_menus
                    else:
                        # Choose a random alternative and replace the food with it.
                        new_menu = copy.deepcopy(menu)
                        new_meal = copy.deepcopy(meal_foods)
                        alternative = random.choice(alternatives)
                        del new_meal[food_id]
                        new_meal[str(alternative)] = amount
                        new_menu[day][meal] = new_meal
                        fixed_menus.append(new_menu)  
                        return fixed_menus 
                    
    return fixed_menus


def print_menu(menu: dict):
    global current_menu_id
    w.write(f'"{current_menu_id}": ') 
    w.write("{\n")
    current_menu_id = current_menu_id + 1
    for day in menu:
        w.write("\t")
        w.write(f'"{day}": ')
        w.write(json.dumps(menu[day]))
        if day != "saturday":
            w.write(",\n")
            
    w.write("\n},\n")
 

properties = ["Vegetarian", "Vegan", "Contains eggs", "Contains milk", "Contains peanuts or nuts", "Contains fish", "Contains sesame", "Contains soy", "Contains gluten"]

me = menus = open("../layouts/menusById.json", "r")
da = json.load(me)
current_menu_id = da.__len__() + 1
me.close()

menu_to_fix = input("Enter the menu id to fix: ")

for i, name in enumerate(properties):
    print(f"{i}) {name}")

property_to_fix = int(input("Enter the property to fix: "))

org = read_menu(menu_to_fix)

fix_menu(properties[property_to_fix], int(menu_to_fix), exhaustive=True)

print(f"\nDone menu {menu_to_fix}!")

w.close()