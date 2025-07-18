<p align="center">
   <img src="./App/TheApp/logo.png" alt="NutiPlan Logo" width="300"/><br>
</p>

<p align="center">
   <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white"/>
   <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
   <img src="https://img.shields.io/badge/Buildozer-FF6F00?style=for-the-badge&logo=android&logoColor=white"/>
   <img src="https://img.shields.io/badge/Kivy-FFDD00?style=for-the-badge&logo=python&logoColor=black"/>
   <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white"/>
   <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
</p>

## Project Structure
The project contains 6 folders:

### `data` folder
All data related files are in the `data` folder, which contains datasets and code for data processing.
 * [`data/food_data/FoodData.json`](./data/food_data/FoodData.json) contains the dataset of foods and their nutritional values.
 * [`data/scripts/search.py`](./data/scripts/search.py) is used to build a model for searching the 10 most relevant foods based on a given query.
 * Most of the files were used for the model that we ultimately didn’t end up using.

### `App` folder
Contains 2 parts of the application in different folders.  

#### `TheApp` folder 
Contains the code for the application:
  * [`App/TheApp/main.py`](./App/TheApp/main.py) is the app's code. In this file, each window has its own class and there is an option to see what windows are in the application in the class `WindowManager` at the end of the file.
  * [`App/TheApp/usersData.json`](./App/TheApp/usersData.json) stores the users' data.
    
#### `Server` folder
Contains the code for the server:
  * [`App/Server/server.py`](./App/Server/server.py) is the server's code and it contains the functions for handling requests from the client such as `make_graph` that generates a graph of the user's historical weight and `generate_menu` which is the algorithm for generating a menu based on the user's nutritional values.
  * [`main.py`](./App/Server/main.py) is used to run the server.
  * `char_nn.pkl`, `char_vectorizer.pkl`, `word_nn.pkl`, and `word_vectorizer.pkl` files are all used for searching the 10 most relevant foods based on a given query.
  * [`App/Server/Meals.json`](./App/Server/Meals.json) contains the foods that are relevant to each meal.
  * [`App/Server/FoodAlternatives.json`](./App/Server/FoodAlternatives.json) contains the vegetarian and vegan alternatives for each food, if needed.
  * [`App/Server/Food_names.json`](./App/Server/Food_names.json) contains the names of the foods that are used in the application.
  * [`App/Server/FoodsByID.json`](./App/Server/FoodsByID.json) contains the foods that are used in the menu generation algorithm and their nutritional values.

### `docs` folder
Contains all the knowledge for calculating the nutritional values for the person.

### <s>`MenuCreator` folder</s>
Contains the code and the data for the menu generation algorithm and the tests for it.
All the necessary code and files are also in the `App/Server` folder, so this folder is not used anymore.

### <s>`Model` folder</s>
Contains all the code that was used to build the model, but it is not use anymore.
    
### `Presentation` folder
Contains the presentation and poster that were used to introduce the project.
