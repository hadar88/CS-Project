The project contains 4 folders:

1. **data**: contains datasets and code for data processing.
    * The dataset of the foods and their nutritional values is in `data/food_data/foodData.json`
    [View Script](./data/food_data/foodData.json).
    * The file `data/scripts/search.py` [View Script](./data/scripts/search.py) is used to build a model for searching the 10 most relevant foods based on a given query.
    * Most of the files were used for the model which we did not use.

2. **App**: contains 2 parts of the application in different folders.
    * TheApp: the code for the application:
        * The code is in `App/TheApp/main.py` [View Script](./App/TheApp/main.py).
        In this file, each window has its own class and there is option to see what windows are in the application in the class `WindowManager` at the end of the file.
        * All the data of the users is stored in `App/TheApp/usersData.json` [View File](./App/TheApp/usersData.json).
    * Server: the code for the server:
        * The code is in `App/Server/server.py` [View Script](./App/Server/server.py) and it contains the functions for handling requests from the client such as `make_graph` that generates a graph of the user's history weights and `generate_menu` which is the algorithm for generating a menu based on the user's nutritional values.
        * The file `main.py` [View Script](./App/Server/main.py) is used to run the server.
        * The files `char_nn.pkl`, `char_vectorizer.pkl`, `word_nn.pkl`, and `word_vectorizer.pkl` are used for searching the 10 most relevant foods based on a given query.
        * The file `App/Server/Meals.json` [View File](./App/Server/Meals.json) contains the foods that relevant to each meal.
        * The file `/App/Server/FoodAlternatives.json` [View File](./App/Server/FoodAlternatives.json) contains the vegetarian and vegan alternatives for each food if needed.
        * The file `App/Server/Food_names.json` [View File](./App/Server/Food_names.json) contains the names of the foods that are used in the application.
        * The file `App/Server/FoodsByID.json` [View File](./App/Server/FoodsByID.json) contains the foods that are used in the menu generation algorithm and their nutritional values.
3. **docs**: Contains all the knowledge for calculating the nutritional values for the person.

4. **MenuCreator**: contains the code and the data for the menu generation algorithm and the tests for it.
    Now all the necessary code and files are in the `App/Server` folder, so this folder is not used anymore.

5. **Model**: Contains all the code that used to build the model but it is not use anymore.
    