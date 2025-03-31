import json
import time
import requests
import datetime
import threading
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.spinner import Spinner
from kivy.uix.checkbox import CheckBox

#######################################################################

DATA_PATH = "data.json"
f = open(DATA_PATH, "r")
data = json.load(f)
f.close()

#######################################################################

FOODS_MENU_DATA_PATH = "FoodsByID.json"
f = open(FOODS_MENU_DATA_PATH, "r")
foods_menu_data = json.load(f)
f.close()

#######################################################################

FOODS_DICT_PATH = "FoodData.json"
f = open(FOODS_DICT_PATH, "r")
foods_dict = json.load(f)
f.close()

#######################################################################

UNITS_PATH = "Units.json"
f = open(UNITS_PATH, "r")
units = json.load(f)
f.close()

#######################################################################

def calculate_nutritional_data(goal, activity_type, activity_level, daily_calories):    
    x1, x2, x3, x4 = 0.55, 0.05, 0.3, 0.225

    if goal == "Lose weight":
        x3 = 0.25  
        x4 = 0.25  
    elif goal == "Gain weight":
        x3 = 0.35  
        x4 = 0.3    

    if "cardio" in activity_type:
        x1 = 0.6  
        x3 = max(x3 - 0.05, 0.25) 
    if "strength" in activity_type or "muscle" in activity_type:
        x1 = max(x1 - 0.05, 0.45)  
        x4 = min(x4 + 0.1, 0.35)  

    if activity_level == "sedentary":
        x1 = max(x1 - 0.05, 0.45)
        x3 = max(x3 - 0.05, 0.25)
    elif activity_level == "lightly active":
        x1 = min(x1 + 0.02, 0.55)
        x3 = min(x3 + 0.02, 0.275)
    elif activity_level == "moderately active":
        x1 = min(x1 + 0.05, 0.6)
        x3 = min(x3 + 0.05, 0.3)
    elif activity_level == "active":
        x1 = min(x1 + 0.07, 0.62)
        x3 = min(x3 + 0.07, 0.325)
    elif activity_level == "extremely active":
        x1 = min(x1 + 0.1, 0.65)
        x3 = min(x3 + 0.1, 0.35)

    x1 = max(0.45, min(x1, 0.65))
    x2 = max(0, min(x2, 0.1))
    x3 = max(0.25, min(x3, 0.35))
    x4 = max(0.1, min(x4, 0.35))

    sum = x1 + x2 + x3 + x4
    x1 /= sum
    x2 /= sum
    x3 /= sum
    x4 /= sum

    carbohydrates = (daily_calories * x1) / 4
    sugars = (daily_calories * x2) / 4
    fats = (daily_calories * x3) / 9
    proteins = (daily_calories * x4) / 4

    carbohydrates = round(carbohydrates, 2)
    sugars = round(sugars, 2)
    fats = round(fats, 2)
    proteins = round(proteins, 2)
    
    return [carbohydrates, sugars, fats, proteins]

def bmi(weight, height):
    height /= 100
    return round(weight / (height ** 2), 2)

def check_bmi(weight, height):
    b = bmi(weight, height)
    if b < 16:
        return "Severely underweight"
    elif b < 18.5:
        return "Underweight"
    elif b < 25:
        return "Healthy"
    elif b < 30:
        return "Overweight"
    elif b < 40:
        return "Obese"
    else:
        return "Extremely obese"

def bmr(weight, height, age, gender):
    if(gender == "Male"):
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161
    
def amr(weight, height, age, gender, activity_level):
    bmr_value = bmr(weight, height, age, gender)
    if activity_level == "sedentary":
        return bmr_value * 1.2
    elif activity_level == "lightly active":
        return bmr_value * 1.375
    elif activity_level == "moderately active":
        return bmr_value * 1.55
    elif activity_level == "active":
        return bmr_value * 1.725
    elif activity_level == "extremely active":
        return bmr_value * 1.9
    else:
        return 1

def ideal_body_weight(height, gender):
    inch = 0.3937
    height = height * inch

    if(gender == "Male"):
        i = round(50 + 2.3 * (height - 60))
        return i
    else:
        i = round(45.5 + 2.3 * (height - 60))
        return i

def weight_change(current_weight, goal_weight):
    return abs(current_weight - goal_weight)

def time_of_change(current_weight, goal_weight):
    c = weight_change(current_weight, goal_weight)
    return round(260 * c / current_weight)

def weekly_change(current_weight, goal_weight, goal_time):
    c = weight_change(current_weight, goal_weight)
    return round(1000 * c / goal_time)

def daily_calories_change(current_weight, goal_weight, goal_time):
    w = weekly_change(current_weight, goal_weight, goal_time)
    return round(9 * w / 7)

def calculate_calories(current_weight, goal_weight, goal_time, height, age, gender, activity_level):
    a = amr(current_weight, height, age, gender, activity_level)
    p = daily_calories_change(current_weight, goal_weight, goal_time)
    i = 1 if current_weight <= goal_weight else -1
    return a + i * p

def get_vector(current_weight, goal_weight, goal_time, height, age, gender, goal, cardio, strength, muscle, activity, vegetarian, vegan, eggs, milk, nuts, fish, sesame, soy, gluten):
    activity_type = []
    if cardio == "1":
        activity_type.append("cardio")
    if strength == "1":
        activity_type.append("strength")
    if muscle == "1":
        activity_type.append("muscle")

    c = calculate_calories(current_weight, goal_weight, goal_time, height, age, gender, activity)

    result = calculate_nutritional_data(goal, activity_type, activity, c)

    vec = {}
    vec["calories"] = c
    vec["carbohydrates"] = result[0]
    vec["sugar"] = result[1]
    vec["fat"] = result[2]
    vec["protein"] = result[3]
    
    
    if vegetarian == "1":
        vec["vegetarian"] = 1
    else:
        vec["vegetarian"] = 0
    if vegan == "1":
        vec["vegan"] = 1
    else:
        vec["vegan"] = 0
    if eggs == "1":
        vec["eggs"] = 0
    else:
        vec["eggs"] = 1
    if milk == "1":
        vec["milk"] = 0
    else:
        vec["milk"] = 1        
    if nuts == "1":
        vec["nuts"] = 0
    else:
        vec["nuts"] = 1
    if fish == "1":
        vec["fish"] = 0
    else:
        vec["fish"] = 1
    if sesame == "1":
        vec["sesame"] = 0
    else:
        vec["sesame"] = 1
    if soy == "1":
        vec["soy"] = 0
    else:
        vec["soy"] = 1
    if gluten == "1":
        vec["gluten"] = 0
    else:
        vec["gluten"] = 1
        
    return vec

#######################################################################

def get_meal(day, meal):
    menu = data["menu"]
    m = menu[day][meal]
    a = {}
    for i in m:
        if int(i) != 0 and int(m[i]) != 0:
            key = str(i)
            if key in foods_menu_data:
                name = foods_menu_data[key]["Name"]
                a[name] = m[i]
            else:
                print(f"Warning: Key {key} not found in foods_menu_data")
    return a

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
                structured_dict[group_key][sub_group_key][int(pair[0])] = round(pair[1], 2)

    return structured_dict

#######################################################################

def check_time():
    while True:
        now = datetime.datetime.now()
        if now.weekday() == 5 and now.hour == 23 and now.minute == 59:
            print("It's the time!")
            time.sleep(60)  # Wait a minute to avoid printing multiple times in the same minute
        time.sleep(1)  # Check every second

#######################################################################

class ColoredLabel(Label):
    def __init__(self, color=(0, 0, 0, 1), text_color=(0, 0, 0, 1), **kw):
        super(ColoredLabel, self).__init__(**kw)
        self.color = text_color  # Set the text color
        with self.canvas.before:
            self.bg_color = Color(*color)  # Use the provided background color
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_bg, pos=self._update_bg)

    def _update_bg(self, instance, value):
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos

#######################################################################

class LoginWindow(Screen):
    def __init__(self, **kw):
        super(LoginWindow, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1)  
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.logo = Image(
            source = "logo.png", size_hint = (0.3, 0.3), pos_hint = {"x": 0.35, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.userName = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Username", 
            size_hint=(0.8, 0.1), 
            pos_hint={"x": 0.1, "top": 0.68}
        )
        self.window.add_widget(self.userName)

        self.password = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Password", 
            size_hint=(0.8, 0.1), 
            pos_hint={"x": 0.1, "top": 0.56}, 
            password=True
        )
        self.window.add_widget(self.password)

        self.showpassword = ColoredLabel(
            text = "Show password", 
            font_size = 40, 
            size_hint = (0.2, 0.05), 
            pos_hint = {"x": 0.3, "top": 0.45},
            color=(1, 1, 1, 1),
            text_color=(0, 1, 0, 1)
        )
        self.window.add_widget(self.showpassword)

        self.showpasswordInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.475},
            color=(1, 0, 0, 1),
            on_press = self.show_password
        )
        self.window.add_widget(self.showpasswordInput)

        self.loginButton = Button(
            text = "Login", 
            font_size = 100, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.39},
            on_press = self.login
        )
        self.window.add_widget(self.loginButton)

        self.errorMassage = ColoredLabel(
            text = "", 
            font_size = 50, 
            size_hint = (0.8, 0.05), 
            pos_hint = {"x": 0.1, "top": 0.27},
            color=(1, 1, 1, 1),
            text_color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.errorMassage)

        self.createAccountButton = Button(
            text = "Create account", 
            font_size = 100, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.2},
            on_press = self.createAccount
        )
        self.window.add_widget(self.createAccountButton)

        ###
        
        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def login(self, instance):
        username = self.userName.text
        password = self.password.text
        if(data["username"] == username and data["password"] == password and username != "" and password != ""):
            self.errorMassage.text = ""
            if(data["logincompleted"]):
                data["stage"] = "main"
                with open(DATA_PATH, "w") as file:
                    json.dump(data, file)
                self.manager.current = "main"
            else:
                self.manager.current = data["stage"]
        else:
            self.errorMassage.text = "Invalid username or password"

    def createAccount(self, instance):
        self.errorMassage.text = ""
        self.manager.current = "createAccount"
        self.showpasswordInput.active = False

    def show_password(self, instance):
        self.password.password = not self.password.password

################################

class MainWindow(Screen):
    def __init__(self, **kw):
        super(MainWindow, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1)  
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.temp = ColoredLabel(
            text = "Main", 
            font_size = 50, 
            size_hint = (0.4, 0.4), 
            pos_hint = {"x": 0.3, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.temp)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
    
    def on_enter(self):
        # get all the data from the json file and display it
        # use get_meal(day, meal) to get the meals
        days = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
        meals = ["breakfast", "lunch", "dinner"]

        meal = get_meal("sunday", "breakfast")
        self.temp.text = str(meal)
        
        pass

################################

class PersonalDataWindow(Screen):
    def __init__(self, **kw):
        super(PersonalDataWindow, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.home = Button(
            background_normal="home.png",
            size_hint=(0.1125, 0.07), 
            pos_hint={"x": 0, "top": 1},
            on_press=self.go_home
        )
        self.window.add_widget(self.home)

        self.temp = ColoredLabel(
            text = "Personal data", 
            font_size = 50, 
            size_hint = (0.4, 0.4), 
            pos_hint = {"x": 0.3, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.temp)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def go_home(self, instance):
        self.manager.current = "main"

################################

class StatisticsWindow(Screen):
    def __init__(self, **kw):
        super(StatisticsWindow, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.home = Button(
            background_normal="home.png",
            size_hint=(0.1125, 0.07), 
            pos_hint={"x": 0, "top": 1},
            on_press=self.go_home
        )
        self.window.add_widget(self.home)

        self.temp = ColoredLabel(
            text = "Statistics", 
            font_size = 50, 
            size_hint = (0.4, 0.4), 
            pos_hint = {"x": 0.3, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.temp)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def go_home(self, instance):
        self.manager.current = "main"

################################

class MenuWindow(Screen):
    def __init__(self, **kw):
        super(MenuWindow, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.home = Button(
            background_normal="home.png",
            size_hint=(0.1125, 0.07), 
            pos_hint={"x": 0, "top": 1},
            on_press=self.go_home
        )
        self.window.add_widget(self.home)

        self.temp = ColoredLabel(
            text = "Menu", 
            font_size = 50, 
            size_hint = (0.4, 0.4), 
            pos_hint = {"x": 0.3, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.temp)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def go_home(self, instance):
        self.manager.current = "main"
        
################################

class WeeklymenuWindow(Screen):
    def __init__(self, **kw):
        super(WeeklymenuWindow, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.home = Button(
            background_normal="home.png",
            size_hint=(0.1125, 0.07), 
            pos_hint={"x": 0, "top": 1},
            on_press=self.go_home
        )
        self.window.add_widget(self.home)

        self.temp = ColoredLabel(
            text = "Weekly menu", 
            font_size = 50, 
            size_hint = (0.4, 0.4), 
            pos_hint = {"x": 0.3, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.temp)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def go_home(self, instance):
        self.manager.current = "main"

################################

class DictionaryWindow(Screen):
    def __init__(self, **kw):
        super(DictionaryWindow, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.home = Button(
            background_normal="home.png",
            size_hint=(0.1125, 0.07), 
            pos_hint={"x": 0, "top": 1},
            on_press=self.go_home
        )
        self.window.add_widget(self.home)

        self.temp = ColoredLabel(
            text = "Dictionary", 
            font_size = 50, 
            size_hint = (0.4, 0.4), 
            pos_hint = {"x": 0.3, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.temp)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def go_home(self, instance):
        self.manager.current = "main"

################################

class CreateAccountWindow(Screen):
    def __init__(self, **kw):
        super(CreateAccountWindow, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.title = ColoredLabel(
            text = "Create an account", 
            font_size = 100, 
            size_hint = (0.8, 0.2), 
            pos_hint = {"x": 0.1, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.title)

        self.userName = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Username", 
            size_hint=(0.8, 0.1), 
            pos_hint={"x": 0.1, "top": 0.68}
        )
        self.window.add_widget(self.userName)

        self.password = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Password", 
            size_hint=(0.8, 0.1), 
            pos_hint={"x": 0.1, "top": 0.56},
            password=True 
        )
        self.window.add_widget(self.password)

        self.showpassword = ColoredLabel(
            text = "Show password", 
            font_size = 40, 
            size_hint = (0.2, 0.05), 
            pos_hint = {"x": 0.3, "top": 0.45},
            color=(1, 1, 1, 1),
            text_color=(0, 1, 0, 1)
        )
        self.window.add_widget(self.showpassword)

        self.showpasswordInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.475},
            color=(1, 0, 0, 1),
            on_press = self.show_password
        )
        self.window.add_widget(self.showpasswordInput)

        self.login = Button(
            text = "Login", 
            font_size = 90, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.39},
            on_press = self.log_in
        )
        self.window.add_widget(self.login)

        self.errorMassage = ColoredLabel(
            text = "", 
            font_size = 50, 
            size_hint = (0.8, 0.05), 
            pos_hint = {"x": 0.1, "top": 0.27},
            color=(1, 1, 1, 1),
            text_color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.errorMassage)

        self.submit = Button(
            text = "Submit", 
            font_size = 90, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.2},
            on_press = self.registration
        )
        self.window.add_widget(self.submit)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def log_in(self, instance):
        self.userName.text = ""
        self.password.text = ""
        self.manager.current = "login"
        self.errorMassage.text = ""
        self.showpasswordInput.active = False

    def registration(self, instance):
        username = self.userName.text
        password = self.password.text
        if(username == "" or password == ""):
            self.errorMassage.text = "Cannot leave fields empty"
        else:
            self.userName.text = ""
            self.password.text = ""
            data["username"] = username
            data["password"] = password
            data["stage"] = "registration1"
            with open(DATA_PATH, "w") as file:
                json.dump(data, file)
            self.manager.current = "registration1"
            self.errorMassage.text = ""
            data["logincompleted"] = False
            with open(DATA_PATH, "w") as file:
                json.dump(data, file)

    def show_password(self, instance):
        self.password.password = not self.password.password

################################

class Registration1Window(Screen):
    def __init__(self, **kw):
        super(Registration1Window, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.title = ColoredLabel(
            text = "Registration", 
            font_size = 150, 
            size_hint = (0.775, 0.2), 
            pos_hint = {"x": 0.1125, "top": 0.95},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.title)

        self.weightLabel = ColoredLabel(
            text = "Weight:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.weightLabel)

        self.weightInput = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Kg", 
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.7},
            input_filter="float"
        )
        self.window.add_widget(self.weightInput)

        self.heightLabel = ColoredLabel(
            text = "Height:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.58},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.heightLabel)

        self.heightInput = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "cm", 
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.58},
            input_filter="int"
        )
        self.window.add_widget(self.heightInput)

        self.AgeLabel = ColoredLabel(
            text = "Age:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.46},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.AgeLabel)

        self.AgeInput = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "years", 
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.46},
            input_filter="int"
        )
        self.window.add_widget(self.AgeInput)

        self.genderLabel = ColoredLabel(
            text = "Gender:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.34},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.genderLabel)

        self.genderInput = Spinner(
            text="Select a gender",
            values=("Male", "Female"),
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.34},
            font_size = 50  

        )
        self.window.add_widget(self.genderInput)

        self.errorMessage = ColoredLabel(
            text = "", 
            font_size = 50, 
            size_hint = (0.8, 0.06), 
            pos_hint = {"x": 0.1, "top": 0.22},
            color=(1, 1, 1, 1),
            text_color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.errorMessage)

        self.nextPage = Button(
            text = "Next page", 
            font_size = 50, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.14},
            on_press = self.next
        )
        self.window.add_widget(self.nextPage)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self, instance):
        weight_input = self.weightInput.text
        height_input = self.heightInput.text
        age_input = self.AgeInput.text
        gender_input = self.genderInput.text
        if(weight_input == "" or height_input == "" or age_input == "" or gender_input == "Select a gender"):
            self.errorMessage.text = "Please fill in all fields"
        else:
            data["weight"] = weight_input
            data["height"] = height_input
            data["age"] = age_input
            data["gender"] = gender_input
            data["stage"] = "registration2"
            with open(DATA_PATH, "w") as file:
                json.dump(data, file)
            self.manager.current = "registration2"
            self.errorMessage.text = ""

################################

class Registration2Window(Screen):
    def __init__(self, **kw):
        super(Registration2Window, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.back = Button(
            background_normal="back.png",
            size_hint=(0.1125, 0.07), 
            pos_hint={"x": 0, "top": 1},
            on_press=self.previous
        )
        self.window.add_widget(self.back)

        self.title = ColoredLabel(
            text = "Registration", 
            font_size = 150, 
            size_hint = (0.775, 0.2), 
            pos_hint = {"x": 0.1125, "top": 0.95},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.title)

        self.activityLabel = ColoredLabel(
            text = "I am:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.activityLabel)

        self.activityInput = Spinner(
            text="Level of activity",
            values=("sedentary", 
                    "lightly active", 
                    "moderately active", 
                    "active", 
                    "extremely active"),
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.7},
            font_size = 40  
        )
        self.window.add_widget(self.activityInput)

        self.activityTypeLabel = ColoredLabel(
            text = "Types of activity:", 
            font_size = 60, 
            size_hint = (0.6, 0.1), 
            pos_hint = {"x": 0.2, "top": 0.58},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.activityTypeLabel)

        self.cardioLabel = ColoredLabel(
            text = "Cardio", 
            font_size = 50, 
            size_hint = (0.2, 0.05), 
            pos_hint = {"x": 0.3, "top": 0.46},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.cardioLabel)

        self.cardioInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.485},
            color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.cardioInput)

        self.strengthLabel = ColoredLabel(
            text = "Strength", 
            font_size = 50, 
            size_hint = (0.2, 0.05), 
            pos_hint = {"x": 0.3, "top": 0.39},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.strengthLabel)

        self.strengthInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.415},
            color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.strengthInput)

        self.muscleLabel = ColoredLabel(
            text = "Muscle", 
            font_size = 50, 
            size_hint = (0.2, 0.05), 
            pos_hint = {"x": 0.3, "top": 0.32},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.muscleLabel)

        self.muscleInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.345},
            color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.muscleInput)

        self.errorMessage = ColoredLabel(
            text = "", 
            font_size = 50, 
            size_hint = (0.8, 0.1), 
            pos_hint = {"x": 0.1, "top": 0.255},
            color=(1, 1, 1, 1),
            text_color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.errorMessage)

        self.nextPage = Button(
            text = "Next page", 
            font_size = 50, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.14},
            on_press = self.next
        )
        self.window.add_widget(self.nextPage)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self, instance):
        activity = self.activityInput.text
        cardio = self.cardioInput.active
        strength = self.strengthInput.active
        muscle = self.muscleInput.active
        if(activity == "Level of activity"):
            self.errorMessage.text = "Please select a level of activity"
        else:
            data["activity"] = activity
            data["cardio"] = "1" if cardio else "0"
            data["strength"] = "1" if strength else "0"
            data["muscle"] = "1" if muscle else "0"
            data["stage"] = "registration3"
            with open(DATA_PATH, "w") as file:
                json.dump(data, file)
            self.manager.current = "registration3"
            self.errorMessage.text = ""

    def previous(self, instance):
        self.manager.current = "registration1"
        self.errorMessage.text = ""

################################

class Registration3Window(Screen):
    def __init__(self, **kw):
        super(Registration3Window, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.back = Button(
            background_normal="back.png",
            size_hint=(0.1125, 0.07), 
            pos_hint={"x": 0, "top": 1},
            on_press=self.previous
        )
        self.window.add_widget(self.back)

        self.title = ColoredLabel(
            text = "Registration", 
            font_size = 150, 
            size_hint = (0.775, 0.2), 
            pos_hint = {"x": 0.1125, "top": 0.95},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.title)

        self.goalLabel = ColoredLabel(
            text = "Goal:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.6},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.goalLabel)

        self.goalInput = Spinner(
            text="Goal",
            values=("Lose weight", "Maintain weight", "Gain weight"),
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.6},
            font_size = 40  
        )
        self.window.add_widget(self.goalInput)

        self.dietLabel = ColoredLabel(
            text = "Diet:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.46},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.dietLabel)

        self.dietInput = Spinner(
            text="Diet",
            values=("Vegetarian", "Vegan", "None"),
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.46},
            font_size = 40  
        )
        self.window.add_widget(self.dietInput)

        self.errorMessage = ColoredLabel(
            text = "", 
            font_size = 50, 
            size_hint = (0.8, 0.1), 
            pos_hint = {"x": 0.1, "top": 0.3},
            color=(1, 1, 1, 1),
            text_color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.errorMessage)

        self.nextPage = Button(
            text = "Next page", 
            font_size = 50, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.14},
            on_press = self.next
        )
        self.window.add_widget(self.nextPage)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self, instance):
        goal_input = self.goalInput.text
        diet_input = self.dietInput.text
        if(goal_input == "Goal" or diet_input == "Diet"):
            self.errorMessage.text = "Please select a goal and diet"
        else:
            data["goal"] = goal_input
            if(diet_input == "Vegetarian"):
                data["vegetarian"] = "1"
                data["vegan"] = "0"
            elif(diet_input == "Vegan"):
                data["vegetarian"] = "1"
                data["vegan"] = "1"
            else:
                data["vegetarian"] = "0"
                data["vegan"] = "0"
            data["stage"] = "registration4"
            with open(DATA_PATH, "w") as file:
                json.dump(data, file)
            self.manager.current = "registration4"
            self.errorMessage.text = ""

    def previous(self, instance):
        self.manager.current = "registration2"
        self.errorMessage.text = ""

################################

class Registration4Window(Screen):
    def __init__(self, **kw):
        super(Registration4Window, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.back = Button(
            background_normal="back.png",
            size_hint=(0.1125, 0.07), 
            pos_hint={"x": 0, "top": 1},
            on_press=self.previous
        )
        self.window.add_widget(self.back)

        self.title = ColoredLabel(
            text = "Registration", 
            font_size = 150, 
            size_hint = (0.775, 0.2), 
            pos_hint = {"x": 0.1125, "top": 0.95},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.title)

        self.allergiesLabel = ColoredLabel(
            text = "Allergies:", 
            font_size = 60, 
            size_hint = (0.6, 0.1), 
            pos_hint = {"x": 0.2, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.allergiesLabel)

        self.eggAllergyLabel = ColoredLabel(
            text = "Eggs", 
            font_size = 50, 
            size_hint = (0.2, 0.04), 
            pos_hint = {"x": 0.3, "top": 0.57},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.eggAllergyLabel)

        self.eggAllergyInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.6},
            color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.eggAllergyInput)

        self.milkAllergyLabel = ColoredLabel(
            text = "Milk", 
            font_size = 50, 
            size_hint = (0.2, 0.04), 
            pos_hint = {"x": 0.3, "top": 0.51},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.milkAllergyLabel)

        self.milkAllergyInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.54},
            color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.milkAllergyInput)

        self.nutAllergyLabel = ColoredLabel(
            text = "Nuts", 
            font_size = 50, 
            size_hint = (0.2, 0.04), 
            pos_hint = {"x": 0.3, "top": 0.45},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.nutAllergyLabel)

        self.nutAllergyInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.48},
            color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.nutAllergyInput)

        self.fishAllergyLabel = ColoredLabel(
            text = "Fish", 
            font_size = 50, 
            size_hint = (0.2, 0.04), 
            pos_hint = {"x": 0.3, "top": 0.39},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.fishAllergyLabel)

        self.fishAllergyInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.42},
            color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.fishAllergyInput)

        self.sesameAllergyLabel = ColoredLabel(
            text = "Sesame", 
            font_size = 50, 
            size_hint = (0.2, 0.04), 
            pos_hint = {"x": 0.3, "top": 0.33},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )   
        self.window.add_widget(self.sesameAllergyLabel)

        self.sesameAllergyInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.36},
            color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.sesameAllergyInput)

        self.soyAllergyLabel = ColoredLabel(
            text = "Soy", 
            font_size = 50, 
            size_hint = (0.2, 0.04), 
            pos_hint = {"x": 0.3, "top": 0.27},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.soyAllergyLabel)

        self.soyAllergyInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.3},
            color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.soyAllergyInput)

        self.glutenAllergyLabel = ColoredLabel(
            text = "Gluten", 
            font_size = 50, 
            size_hint = (0.2, 0.04), 
            pos_hint = {"x": 0.3, "top": 0.21},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.glutenAllergyLabel)

        self.glutenAllergyInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.24},
            color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.glutenAllergyInput)

        self.nextPage = Button(
            text = "Next page", 
            font_size = 50, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.14},
            on_press = self.next
        )
        self.window.add_widget(self.nextPage)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self, instance):
        egg_allergy = self.eggAllergyInput.active
        milk_allergy = self.milkAllergyInput.active
        nut_allergy = self.nutAllergyInput.active
        fish_allergy = self.fishAllergyInput.active
        sesame_allergy = self.sesameAllergyInput.active
        soy_allergy = self.soyAllergyInput.active
        gluten_allergy = self.glutenAllergyInput.active
        
        data["eggs allergy"] = "1" if egg_allergy else "0"
        data["milk allergy"] = "1" if milk_allergy else "0"
        data["nuts allergy"] = "1" if nut_allergy else "0"
        data["fish allergy"] = "1" if fish_allergy else "0"
        data["sesame allergy"] = "1" if sesame_allergy else "0"
        data["soy allergy"] = "1" if soy_allergy else "0"
        data["gluten allergy"] = "1" if gluten_allergy else "0"

        data["stage"] = "registration5"
        with open(DATA_PATH, "w") as file:
            json.dump(data, file)
        self.manager.current = "registration5"

    def previous(self, instance):
        self.manager.current = "registration3"

################################

class Registration5Window(Screen):
    def __init__(self, **kw):
        super(Registration5Window, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.idealBodyWeight = 0

        self.back = Button(
            background_normal="back.png",
            size_hint=(0.1125, 0.07), 
            pos_hint={"x": 0, "top": 1},
            on_press=self.previous
        )
        self.window.add_widget(self.back)

        self.title = ColoredLabel(
            text = "Registration", 
            font_size = 150, 
            size_hint = (0.775, 0.2), 
            pos_hint = {"x": 0.1125, "top": 0.95},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.title)

        self.title2 = ColoredLabel(
            text = "Target weight", 
            font_size = 80, 
            size_hint = (0.6, 0.1), 
            pos_hint = {"x": 0.2, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.title2)

        self.suggestedWeight = ColoredLabel(
            text = "Suggested weight: " + str(self.idealBodyWeight) + " kg", 
            font_size = 60, 
            size_hint = (0.9, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.57},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.suggestedWeight)

        self.goalweightLabel = ColoredLabel(
            text = "Weight:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.44},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.goalweightLabel)

        self.goalweightInput = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Kg", 
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.44},
            input_filter="int"
        )
        self.window.add_widget(self.goalweightInput)

        self.errorMessage = ColoredLabel(
            text = "", 
            font_size = 50, 
            size_hint = (0.8, 0.1), 
            pos_hint = {"x": 0.1, "top": 0.29},
            color=(1, 1, 1, 1),
            text_color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.errorMessage)

        self.nextPage = Button(
            text = "Next page", 
            font_size = 50, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.14},
            on_press = self.next
        )
        self.window.add_widget(self.nextPage)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self, instance):
        if(self.goalweightInput.text == ""):
            self.errorMessage.text = "Please fill in the field"
        else:
            self.errorMessage.text = ""
            data["goal weight"] = self.goalweightInput.text
            data["stage"] = "registration6"
            with open(DATA_PATH, "w") as file:
                json.dump(data, file)
            self.manager.current = "registration6"

    def previous(self, instance):
        self.manager.current = "registration4"
        self.errorMessage.text = ""

    def on_enter(self):
        idealBodyWeight = ideal_body_weight(int(data["height"]), data["gender"])
        self.suggestedWeight.text = "Suggested weight: " + str(idealBodyWeight) + " kg"

################################

class Registration6Window(Screen):
    def __init__(self, **kw):
        super(Registration6Window, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.time = 0

        self.back = Button(
            background_normal="back.png",
            size_hint=(0.1125, 0.07), 
            pos_hint={"x": 0, "top": 1},
            on_press=self.previous
        )
        self.window.add_widget(self.back)

        self.title = ColoredLabel(
            text = "Registration", 
            font_size = 150, 
            size_hint = (0.775, 0.2), 
            pos_hint = {"x": 0.1125, "top": 0.95},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.title)

        self.title2 = ColoredLabel(
            text = "Time of the process", 
            font_size = 80, 
            size_hint = (0.8, 0.1), 
            pos_hint = {"x": 0.1, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.title2)

        self.suggestedTime = ColoredLabel(
            text = "Suggested time: " + str(self.time) + " weeks", 
            font_size = 60, 
            size_hint = (0.9, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.57},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.suggestedTime)

        self.timeLabel = ColoredLabel(
            text = "Time:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.44},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.timeLabel)

        self.timeInput = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Weeks", 
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.44},
            input_filter="int"
        )
        self.window.add_widget(self.timeInput)

        self.errorMessage = ColoredLabel(
            text = "", 
            font_size = 50, 
            size_hint = (0.8, 0.1), 
            pos_hint = {"x": 0.1, "top": 0.29},
            color=(1, 1, 1, 1),
            text_color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.errorMessage)

        self.nextPage = Button(
            text = "Next page", 
            font_size = 50, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.14},
            on_press = self.next
        )
        self.window.add_widget(self.nextPage)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self, instance):
        if(self.timeInput.text == ""):
            self.errorMessage.text = "Please fill in the field"
        else:
            self.errorMessage.text = ""
            data["goal time"] = self.timeInput.text
            data["stage"] = "loading"
            with open(DATA_PATH, "w") as file:
                json.dump(data, file)
            self.manager.current = "loading"
            data["logincompleted"] = True
            with open(DATA_PATH, "w") as file:
                json.dump(data, file)

    def previous(self, instance):
        self.manager.current = "registration5"
        self.errorMessage.text = ""

    def on_enter(self):
        self.time = time_of_change(int(data["weight"]), int(data["goal weight"]))
        self.suggestedTime.text = "Suggested time: " + str(self.time) + " weeks"

################################

class LoadingWindow(Screen):
    def __init__(self, **kw):
        super(LoadingWindow, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)    

        ###

        self.vector = {}

        self.loading = ColoredLabel(
            text = "Loading...", 
            font_size = 150, 
            size_hint = (0.6, 0.6), 
            pos_hint = {"x": 0.2, "top": 0.8},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.loading)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self):
        data["stage"] = "main"
        with open(DATA_PATH, "w") as file:
            json.dump(data, file)
        self.manager.current = "main"

    def on_enter(self):
        current_weight_temp = int(data["weight"])
        goal_weight_temp = int(data["goal weight"])
        goal_time_temp = int(data["goal time"])
        height_temp = int(data["height"])
        age_temp = int(data["age"])
        gender_temp = data["gender"]
        goal_temp = data["goal"]
        cardio_temp = data["cardio"]
        strength_temp = data["strength"]
        muscle_temp = data["muscle"]
        activity_temp = data["activity"]
        vegetarian_temp = data["vegetarian"]
        vegan_temp = data["vegan"]
        egg_allergy_temp = data["eggs allergy"]
        milk_allergy_temp = data["milk allergy"]
        nuts_allergy_temp = data["nuts allergy"]
        fish_allergy_temp = data["fish allergy"]
        sesame_allergy_temp = data["sesame allergy"]
        soy_allergy_temp = data["soy allergy"]
        gluten_allergy_temp = data["gluten allergy"]

        self.vector = get_vector(current_weight_temp, goal_weight_temp, goal_time_temp, height_temp, age_temp, 
                                            gender_temp, goal_temp, cardio_temp, strength_temp, muscle_temp, activity_temp,
                                            vegetarian_temp, vegan_temp, egg_allergy_temp, milk_allergy_temp, nuts_allergy_temp,
                                            fish_allergy_temp, sesame_allergy_temp, soy_allergy_temp, gluten_allergy_temp)
        
        data["calories"] = self.vector["calories"]
        data["carbohydrates"] = self.vector["carbohydrates"]
        data["sugar"] = self.vector["sugar"]
        data["fat"] = self.vector["fat"]
        data["protein"] = self.vector["protein"]

        with open(DATA_PATH, "w") as file:
            json.dump(data, file)

        self.build_menu()

    def build_menu(self):
        try:
            server_url = "https://cs-project-m5hy.onrender.com/"

            requests.get(server_url + "wakeup")

            response = requests.post(server_url + "predict", json=self.vector)

            if response.status_code == 200:
                result = response.json()
                result = convert_to_dict(result)
                data["menu"] = result
            else:
                print("Error: " + str(response.status_code))
                
        except Exception as e:
            print("Error: " + str(e))

        self.next()

################################
        
class WindowManager(ScreenManager):
    def __init__(self, **kw):
        super(WindowManager, self).__init__(**kw)

        self.add_widget(LoginWindow(name = "login"))
        self.add_widget(LoadingWindow(name = "loading"))
        self.add_widget(MainWindow(name = "main"))
        self.add_widget(PersonalDataWindow(name = "personalData"))
        self.add_widget(StatisticsWindow(name = "statistics"))
        self.add_widget(MenuWindow(name = "menu"))
        self.add_widget(WeeklymenuWindow(name = "weeklyMenu"))
        self.add_widget(DictionaryWindow(name = "dictionary"))
        self.add_widget(CreateAccountWindow(name = "createAccount"))
        self.add_widget(Registration1Window(name = "registration1"))
        self.add_widget(Registration2Window(name = "registration2"))
        self.add_widget(Registration3Window(name = "registration3"))
        self.add_widget(Registration5Window(name = "registration5"))
        self.add_widget(Registration4Window(name = "registration4"))
        self.add_widget(Registration6Window(name = "registration6"))
        
class MainApp(App):
    def build(self):
        # threading.Thread(target=check_time, daemon=True).start()
        return WindowManager()

if __name__ == "__main__":
    MainApp().run()