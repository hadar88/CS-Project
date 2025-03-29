# Menus: 

## `add_nutritional_data.py`
 
This script adds nutritional data to the `FoodsByName.json` and `FoodsByID.json` files.
It also build a json file `FoodsByNameOneLine.json` which contains foods and their id's and put it in the file.
[View Script](./add_nutritional_data.py)

## `build.py`

This script build templates of menus for the file `MenusInput.json` with a range of id's.
[View Script](./build.py)

## `find_in_menu.py` 

This script finds the foods in a certain menu that are vegetarian, vegan or contain some kind of allergan.
[View Script](./find_in_menu.py)

## `menu_calculator.py`

This script calculates the nutritional data of all the menus and puts it in the file `MenusInput.json`. 
It also moves data in the file `MenusInput.json` from the `Menu` section to the `Initial` section.
It also creates a dictionary with the same structure as `menusById`, but with the food names instead of the food ids.
[View Script](./menu_calculator.py)

## `menu_maker.py`

This sctipt is used to convert a menu to a new menu with a desired property.
[View Script](./menu_maker.py)


