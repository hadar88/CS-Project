import json
import requests
from datetime import datetime, timedelta
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle, Line, RoundedRectangle
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.uix.checkbox import CheckBox
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView 
from kivy.uix.stencilview import StencilView
from kivy.clock import Clock

#######################################################################

BUTTON_BG = (0.29, 0.051, 0.051, 1)         # dark bordeaux
BUTTON_TEXT = (0.973, 0.804, 0.816, 1)      # light bordeaux

LABEL_BG = (0.031, 0.145, 0.404, 1)         # dark blue
LABEL_TEXT = (0.733, 0.871, 0.984, 1)       # light blue

SPINNER_BG = (0.106, 0.369, 0.125, 1)       # dark green
SPINNER_DD_BG = (0.263, 0.635, 0.294, 1)    # dark green
SPINNER_TEXT = (0.784, 0.902, 0.788, 1)     # light green

#######################################################################

class Info(object):
    def __init__(self, **kwargs):
        super(Info, self).__init__(**kwargs)
        self.password = ""
        self.weight = ""
        self.height = ""
        self.age = ""
        self.gender = ""
        self.activity = ""
        self.cardio = ""
        self.strength = ""
        self.muscle = ""
        self.goal = ""
        self.vegetarian = ""
        self.vegan = ""
        self.goal_weight = ""
        self.goal_time = ""
        
info = None
current_username = None
menu_request_window = "main"

#######################################################################

USERS_DATA_PATH = "usersData.json"
f = open(USERS_DATA_PATH, "r")
users_data = json.load(f)
f.close()

#######################################################################

FOODS_ID_NAME_PATH = "FoodsIDName.json"
f = open(FOODS_ID_NAME_PATH, "r")
foods_id_name = json.load(f)
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

    if goal == "Lose Weight":
        x3 = 0.25
        x4 = 0.25
    elif goal == "Gain Weight":
        x3 = 0.35
        x4 = 0.3

    if "cardio" in activity_type:
        x1 = 0.6
        x3 = max(x3 - 0.05, 0.25)
    if "strength" in activity_type or "muscle" in activity_type:
        x1 = max(x1 - 0.05, 0.45)
        x4 = min(x4 + 0.1, 0.35)

    if activity_level == "Sedentary":
        x1 = max(x1 - 0.05, 0.45)
        x3 = max(x3 - 0.05, 0.25)
    elif activity_level == "Lightly active":
        x1 = min(x1 + 0.02, 0.55)
        x3 = min(x3 + 0.02, 0.275)
    elif activity_level == "Moderately active":
        x1 = min(x1 + 0.05, 0.6)
        x3 = min(x3 + 0.05, 0.3)
    elif activity_level == "Active":
        x1 = min(x1 + 0.07, 0.62)
        x3 = min(x3 + 0.07, 0.325)
    elif activity_level == "Extremely active":
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
    return bmi_decs_and_color(b)
    
def bmi_decs_and_color(bmi_val):
    if bmi_val < 16:
        return ("Severely underweight", (0, 0, 1, 1))
    elif bmi_val < 18.5:
        return ("Underweight", (0, 1, 0.9, 1))
    elif bmi_val < 25:
        return ("Healthy", (0, 0.5, 0, 1))
    elif bmi_val < 30:
        return ("Overweight", (1, 0.9, 0, 1))
    elif bmi_val < 40:
        return ("Obese", (1, 0.5, 0, 1))
    else:
        return ("Extremely obese", (1, 0, 0, 1))

def bmr(weight, height, age, gender):
    if(gender == "Male"):
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

def amr(weight, height, age, gender, activity_level):
    bmr_value = bmr(weight, height, age, gender)
    if activity_level == "Sedentary":
        return bmr_value * 1.2
    elif activity_level == "Lightly active":
        return bmr_value * 1.375
    elif activity_level == "Moderately active":
        return bmr_value * 1.55
    elif activity_level == "Active":
        return bmr_value * 1.725
    elif activity_level == "xtremely active":
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
    if goal_time <= 0:
        return 0
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

def get_vector(current_weight, goal_weight, goal_time, height, age, gender, goal, cardio, strength, muscle, activity, vegetarian, vegan):
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

    return vec

#######################################################################

def get_meal(day, meal):
    global current_username
    menu = users_data[current_username]["menu"]
    m = menu[day][meal]
    a = {}
    for i in m:
        if int(i) != 0 and int(m[i]) != 0:
            key = str(i)
            if key in foods_id_name:
                name = foods_id_name[key]
                a[name] = m[i]
            else:
                print(f"Warning: Key {key} not found in foods_id_name")
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
                structured_dict[group_key][sub_group_key][int(pair[0])] = round(pair[1])

    return structured_dict

#######################################################################

def clean_self_menu(username):
    self_menu = users_data[username]["self_menu"]

    for day in self_menu:
        for meal in self_menu[day]:
            self_menu[day][meal] = {}

    users_data[username]["self_menu"] = self_menu
    with open(USERS_DATA_PATH, "w") as file:
        json.dump(users_data, file)

def add_food(day, meal, food_name, amount):
    global current_username
    self_menu = users_data[current_username]["self_menu"]
    self_menu[day][meal][food_name] = amount

    calories_today = users_data[current_username]["calories today"]
    carbohydrates_today = users_data[current_username]["carbohydrates today"]
    sugar_today = users_data[current_username]["sugar today"]
    fat_today = users_data[current_username]["fat today"]
    protein_today = users_data[current_username]["protein today"]

    food = foods_dict[food_name]
    calories_today += food["Calories"] * amount / 100
    carbohydrates_today += food["Carbohydrate"] * amount / 100
    sugar_today += food["Sugars"] * amount / 100
    fat_today += food["Fat"] * amount / 100
    protein_today += food["Protein"] * amount / 100

    users_data[current_username]["calories today"] = round(calories_today, 2)
    users_data[current_username]["carbohydrates today"] = round(carbohydrates_today, 2)
    users_data[current_username]["sugar today"] = round(sugar_today, 2)
    users_data[current_username]["fat today"] = round(fat_today, 2)
    users_data[current_username]["protein today"] = round(protein_today, 2)
    users_data[current_username]["self_menu"] = self_menu

    with open(USERS_DATA_PATH, "w") as file:
        json.dump(users_data, file)

def remove_food(day, meal, food_name):
    global current_username
    self_menu = users_data[current_username]["self_menu"]
    food = foods_dict[food_name]

    calories_today = users_data[current_username]["calories today"]
    carbohydrates_today = users_data[current_username]["carbohydrates today"]
    sugar_today = users_data[current_username]["sugar today"]
    fat_today = users_data[current_username]["fat today"]
    protein_today = users_data[current_username]["protein today"]

    calories_today -= food["Calories"] * self_menu[day][meal][food_name] / 100
    carbohydrates_today -= food["Carbohydrate"] * self_menu[day][meal][food_name] / 100
    sugar_today -= food["Sugars"] * self_menu[day][meal][food_name] / 100
    fat_today -= food["Fat"] * self_menu[day][meal][food_name] / 100
    protein_today -= food["Protein"] * self_menu[day][meal][food_name] / 100

    if calories_today < 0:
        calories_today = 0
    if carbohydrates_today < 0:
        carbohydrates_today = 0
    if sugar_today < 0:
        sugar_today = 0
    if fat_today < 0:
        fat_today = 0
    if protein_today < 0:
        protein_today = 0

    del self_menu[day][meal][food_name]

    users_data[current_username]["calories today"] = round(calories_today, 2)
    users_data[current_username]["carbohydrates today"] = round(carbohydrates_today, 2)
    users_data[current_username]["sugar today"] = round(sugar_today, 2)
    users_data[current_username]["fat today"] = round(fat_today, 2)
    users_data[current_username]["protein today"] = round(protein_today, 2)
    users_data[current_username]["self_menu"] = self_menu

    with open(USERS_DATA_PATH, "w") as file:
        json.dump(users_data, file)

#######################################################################

class ColoredLabel(Label):
    def __init__(self, color=(0, 0, 0, 1), text_color=(0, 0, 0, 1), **kw):
        super(ColoredLabel, self).__init__(**kw)
        self.color = text_color 
        with self.canvas.before:
            self.bg_color = Color(*color)
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_bg, pos=self._update_bg)

    def _update_bg(self, instance, value):
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos

class ColoredLabel1(Label):
    def __init__(self, color=(0, 0, 0, 1), text_color=(0, 0, 0, 1), border_color=(0, 0, 0, 1), border_width=2, **kw):
        super(ColoredLabel1, self).__init__(**kw)
        self.color = text_color 
        self.border_color = border_color
        self.border_width = border_width

        with self.canvas.before:
            self.bg_color = Color(*color) 
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)
            self.border_color_instruction = Color(*self.border_color) 
            self.border_line = Line(rectangle=(self.x, self.y, self.width, self.height), width=self.border_width)

        self.bind(size=self._update_bg, pos=self._update_bg)

    def _update_bg(self, instance, value):
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos
        self.border_line.rectangle = (self.x, self.y, self.width, self.height)

class RoundedStencilView(StencilView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(0, 0, 0, 0.7) 
            self.bg_rect = RoundedRectangle(size=self.size, pos=self.pos, radius=[(20, 20), (20, 20), (20, 20), (20, 20)])
        self.bind(size=self._update_bg, pos=self._update_bg)

    def _update_bg(self, instance, value):
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos

class CustomSpinnerOption(SpinnerOption):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = SPINNER_DD_BG
        self.color = SPINNER_TEXT

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
            font_size = 70,
            hint_text = "Username",
            size_hint=(0.8, 0.1),
            pos_hint={"x": 0.1, "top": 0.68},
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
            halign="center",
        )
        self.userName.bind(size=self._update_text_padding1)
        self.window.add_widget(self.userName)
        with self.userName.canvas.before:
            Color(0, 0, 0, 1) 
            self.border = Line(rectangle=(self.userName.x, self.userName.y, self.userName.width, self.userName.height), width=1.0)
        self.userName.bind(size=self._update_border, pos=self._update_border)

        self.password = TextInput(
            multiline = False,
            font_size = 70,
            hint_text = "Password",
            size_hint=(0.8, 0.1),
            pos_hint={"x": 0.1, "top": 0.56},
            password=True,
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
            halign="center"
        )
        self.password.bind(size=self._update_text_padding2)
        self.window.add_widget(self.password)
        with self.password.canvas.before:
            Color(0, 0, 0, 1)  
            self.border2 = Line(rectangle=(self.password.x, self.password.y, self.password.width, self.password.height), width=1.0)
        self.password.bind(size=self._update_border2, pos=self._update_border2)

        self.showpassword = ColoredLabel(
            text = "[b]Show Password[/b]",
            font_size = 40,
            size_hint = (0.2, 0.05),
            pos_hint = {"x": 0.3, "top": 0.45},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup=True
        )
        self.window.add_widget(self.showpassword)

        self.showpasswordInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.475},
            color= LABEL_BG,
            on_press = self.show_password
        )
        self.window.add_widget(self.showpasswordInput)

        self.loginButton = Button(
            text = "[b]Login[/b]",
            font_size = 100,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.39},
            markup=True,
            on_press = self.login
        )
        self.window.add_widget(self.loginButton)

        self.errorMessage = ColoredLabel(
            text = "",
            font_size = 50,
            size_hint = (0.8, 0.05),
            pos_hint = {"x": 0.1, "top": 0.27},
            color=(1, 1, 1, 1),
            text_color=(0.718, 0.11, 0.11, 1),
            markup=True
        )
        self.window.add_widget(self.errorMessage)

        self.createAccountButton = Button(
            text = "[b]Create Account[/b]",
            font_size = 100,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.2},
            markup=True,
            on_press = self.createAccount
        )
        self.window.add_widget(self.createAccountButton)

        ###

        self.add_widget(self.window)

    def _update_text_padding1(self, instance, value):
        instance.padding_y = [(instance.height - instance.line_height) / 2, 0]

    def _update_text_padding2(self, instance, value):
        instance.padding_y = [(instance.height - instance.line_height) / 2, 0]

    def _update_border(self, instance, value):
        self.border.rectangle = (instance.x, instance.y, instance.width, instance.height)

    def _update_border2(self, instance, value):
        self.border2.rectangle = (instance.x, instance.y, instance.width, instance.height)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def login(self, instance):
        username = self.userName.text
        password = self.password.text
        global current_username

        if username in users_data:
            if users_data[username]["password"] == password:
                self.errorMessage.text = ""
                current_username = username
                self.manager.current = "main"
            else:
                current_username = None
                self.errorMessage.text = "[b]Invalid Password[/b]"
        else:
            current_username = None
            self.errorMessage.text = "[b]Invalid Username[/b]"

    def createAccount(self, instance):
        self.errorMessage.text = ""
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

        self.logo = Image(
            source = "logo.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.45, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.personalDataButton = Button(
            text = "[b]Personal Data[/b]",
            font_size = 100,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.9},
            on_press = self.personalData,
            markup=True
        )
        self.window.add_widget(self.personalDataButton)

        self.StatisticsButton = Button(
            text = "[b]Statistics[/b]",
            font_size = 100,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.725},
            on_press = self.statistics,
            markup=True
        )
        self.window.add_widget(self.StatisticsButton)

        self.MenuButton = Button(
            text = "[b]Menu[/b]",
            font_size = 100,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.55},
            on_press = self.menu,
            markup=True
        )
        self.window.add_widget(self.MenuButton)

        self.foodTrackerButton = Button(
            text = "[b]Food Tracker[/b]",
            font_size = 100,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.375},
            on_press = self.foodTracker,
            markup=True
        )
        self.window.add_widget(self.foodTrackerButton)

        self.dictionaryButton = Button(
            text = "[b]Dictionary[/b]",
            font_size = 100,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.2},
            on_press = self.dictionary,
            markup=True
        )
        self.window.add_widget(self.dictionaryButton)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def on_enter(self):
        pass

    def personalData(self, instance):
        self.manager.current = "personalData"

    def statistics(self, instance):
        self.manager.current = "statistics"

    def menu(self, instance):
        self.manager.current = "menu"

    def foodTracker(self, instance):
        self.manager.current = "foodTracker"

    def dictionary(self, instance):
        self.manager.current = "dictionary"

################################

class PersonalDataWindow(Screen):
    def __init__(self, **kw):
        super(PersonalDataWindow, self).__init__(**kw)
        Window.bind(on_keyboard=self.on_keyboard)
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

        self.logo = Image(
            source = "logo.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.45, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.title = ColoredLabel(
            text = "[b]Personal Data[/b]",
            font_size = 100,
            size_hint = (0.8, 0.05),
            pos_hint = {"x": 0.1, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup=True
        )
        self.window.add_widget(self.title)

        self.weightLabel = ColoredLabel(
            text = "Weight: ",
            font_size = 50,
            size_hint = (0.32, 0.05),
            pos_hint = {"x": 0.04, "top": 0.8},
            color = LABEL_BG,
            text_color = LABEL_TEXT
        )
        self.window.add_widget(self.weightLabel)

        self.weightupdateInput = TextInput(
            multiline = False,
            font_size = 40,
            text = "", 
            size_hint=(0.35, 0.05),
            pos_hint={"x": 0.4, "top": 0.8},
            input_filter="int",
            disabled = True, 
            halign="center",
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
        )
        self.weightupdateInput.bind(size=self._update_text_padding)
        self.window.add_widget(self.weightupdateInput)
        with self.weightupdateInput.canvas.before:
            Color(0, 0, 0, 1)
            self.border = Line(rectangle=(self.weightupdateInput.x, self.weightupdateInput.y, self.weightupdateInput.width, self.weightupdateInput.height), width=1.0)
        self.weightupdateInput.bind(size=self._update_border, pos=self._update_border)

        self.weightupdateButton = Button(
            background_normal = "pencil.png",
            size_hint=(0.1, 0.05),
            pos_hint = {"x": 0.8, "top": 0.8},
            on_press = self.weightupdate
        )
        self.window.add_widget(self.weightupdateButton)

        self.heightLabel = ColoredLabel(
            text = "Height: ",
            font_size = 50,
            size_hint = (0.32, 0.05),
            pos_hint = {"x": 0.04, "top": 0.8 - 1/12},
            color = LABEL_BG,
            text_color = LABEL_TEXT
        )
        self.window.add_widget(self.heightLabel)

        self.heightupdateInput = TextInput(
            multiline = False,
            font_size = 40,
            text = "",
            size_hint=(0.35, 0.05),
            pos_hint={"x": 0.4, "top": 0.8 - 1/12},
            input_filter="float",
            disabled = True,
            halign="center",
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
        )
        self.heightupdateInput.bind(size=self._update_text_padding)
        self.window.add_widget(self.heightupdateInput)
        with self.heightupdateInput.canvas.before:
            Color(0, 0, 0, 1) 
            self.border2 = Line(rectangle=(self.heightupdateInput.x, self.heightupdateInput.y, self.heightupdateInput.width, self.heightupdateInput.height), width=1.0)
        self.heightupdateInput.bind(size=self._update_border2, pos=self._update_border2)

        self.heightupdateButton = Button(
            background_normal = "pencil.png",
            size_hint=(0.1, 0.05),
            pos_hint = {"x": 0.8, "top": 0.8 - 1/12},
            on_press = self.heightupdate
        )
        self.window.add_widget(self.heightupdateButton)

        self.targetweightLabel = ColoredLabel(
            text = "Target weight: ",
            font_size = 50,
            size_hint = (0.32, 0.05),
            pos_hint = {"x": 0.04, "top": 0.8 - 2/12},
            color = LABEL_BG,
            text_color = LABEL_TEXT
        )
        self.window.add_widget(self.targetweightLabel)

        self.targetweightupdateInput = TextInput(
            multiline = False,
            font_size = 40,
            text = "",
            size_hint=(0.35, 0.05),
            pos_hint={"x": 0.4, "top": 0.8 - 2/12},
            input_filter="float",
            disabled = True,
            halign="center",
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
        )
        self.targetweightupdateInput.bind(size=self._update_text_padding)
        self.window.add_widget(self.targetweightupdateInput)
        with self.targetweightupdateInput.canvas.before:
            Color(0, 0, 0, 1)  
            self.border3 = Line(rectangle=(self.targetweightupdateInput.x, self.targetweightupdateInput.y, self.targetweightupdateInput.width, self.targetweightupdateInput.height), width=1.0)
        self.targetweightupdateInput.bind(size=self._update_border3, pos=self._update_border3)

        self.targetweightupdateButton = Button(
            background_normal = "pencil.png",
            size_hint=(0.1, 0.05),
            pos_hint = {"x": 0.8, "top": 0.8 - 2/12},
            on_press = self.targetweightupdate
        )
        self.window.add_widget(self.targetweightupdateButton)

        self.activityLabel = ColoredLabel(
            text = "Activity: ",
            font_size = 50,
            size_hint = (0.32, 0.05),
            pos_hint = {"x": 0.04, "top": 0.8 - 3/12},
            color = LABEL_BG,
            text_color = LABEL_TEXT
        )
        self.window.add_widget(self.activityLabel)

        self.activityupdateInput = Spinner(
            font_size=40,
            text = "",
            values=("Sedentary",
            "Lightly active",
            "Moderately active",
            "Active",
            "Extremely active"),
            size_hint=(0.35, 0.05),
            pos_hint={"x": 0.4, "top": 0.8 - 3/12},
            disabled=True,
            background_disabled_normal="",
            disabled_color = SPINNER_TEXT,
            background_color = SPINNER_BG,
            color = SPINNER_TEXT,
            option_cls = CustomSpinnerOption
        )
        self.window.add_widget(self.activityupdateInput)

        self.activityupdateButton = Button(
            background_normal = "pencil.png",
            size_hint=(0.1, 0.05),
            pos_hint = {"x": 0.8, "top": 0.8 - 3/12},
            on_press = self.activityupdate
        )
        self.window.add_widget(self.activityupdateButton)

        self.errorMessage = ColoredLabel(
            text = "",
            font_size = 50,
            size_hint = (0.8, 0.06),
            pos_hint = {"x": 0.1, "top": 0.8 - 4/12},
            color=(1, 1, 1, 1),
            text_color=(0.718, 0.11, 0.11, 1),
            markup=True
        )
        self.window.add_widget(self.errorMessage)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def _update_text_padding(self, instance, value):
        instance.padding_y = [(instance.height - instance.line_height) / 2, 0]

    def go_home(self, instance):
        self.weightupdateInput.disabled = True
        self.heightupdateInput.disabled = True
        self.targetweightupdateInput.disabled = True
        self.activityupdateInput.disabled = True
        self.weightupdateButton.background_normal = "pencil.png"
        self.heightupdateButton.background_normal = "pencil.png"
        self.targetweightupdateButton.background_normal = "pencil.png"
        self.activityupdateButton.background_normal = "pencil.png"

        self.manager.current = "main"

    def weightupdate(self, instance):
        if self.weightupdateInput.disabled:
            self.weightupdateButton.background_normal = "vee.png"
            self.weightupdateInput.disabled = not self.weightupdateInput.disabled
        elif(self.weightupdateInput.text == "" or float(self.weightupdateInput.text) < 40): 
            self.errorMessage.text = "[b]Weight must be greater than 40 kg[/b]"
        else:
            global current_username
            users_data[current_username]["weight"] = self.weightupdateInput.text
            day = datetime.now().date().isoformat()
            if users_data[current_username]["history_times"] and day in users_data[current_username]["history_times"]:
                users_data[current_username]["history_weight"] = users_data[current_username]["history_weight"][:-1] + [float(users_data[current_username]["weight"])]
                users_data[current_username]["history_bmi"] = users_data[current_username]["history_bmi"][:-1] + [bmi(float(users_data[current_username]["weight"]), float(users_data[current_username]["height"]))]
            else:
                users_data[current_username]["history_weight"] = users_data[current_username]["history_weight"] + [float(users_data[current_username]["weight"])]
                users_data[current_username]["history_bmi"] = users_data[current_username]["history_bmi"] + [bmi(float(users_data[current_username]["weight"]), float(users_data[current_username]["height"]))]
                users_data[current_username]["history_times"] = users_data[current_username]["history_times"] + [day]
            
            with open(USERS_DATA_PATH, "w") as file:
                json.dump(users_data, file)

            self.weightupdateButton.background_normal = "pencil.png"
            self.weightupdateInput.disabled = not self.weightupdateInput.disabled
            self.errorMessage.text = ""
        
    def heightupdate(self, instance):
        if self.heightupdateInput.disabled:
            self.heightupdateButton.background_normal = "vee.png"
            self.heightupdateInput.disabled = not self.heightupdateInput.disabled
        elif(self.heightupdateInput.text == "" or int(self.heightupdateInput.text) < 140 or int(self.heightupdateInput.text) > 250):
            self.errorMessage.text = "[b]Height must be between 140 and 250 cm[/b]"
        else:
            global current_username
            users_data[current_username]["height"] = self.heightupdateInput.text
            day = datetime.now().date().isoformat()
            if users_data[current_username]["history_times"] and day in users_data[current_username]["history_times"]:
                users_data[current_username]["history_weight"] = users_data[current_username]["history_weight"][:-1] + [float(users_data[current_username]["weight"])]
                users_data[current_username]["history_bmi"] = users_data[current_username]["history_bmi"][:-1] + [bmi(float(users_data[current_username]["weight"]), float(users_data[current_username]["height"]))]
            else:
                users_data[current_username]["history_weight"] = users_data[current_username]["history_weight"] + [float(users_data[current_username]["weight"])]
                users_data[current_username]["history_bmi"] = users_data[current_username]["history_bmi"] + [bmi(float(users_data[current_username]["weight"]), float(users_data[current_username]["height"]))]
                users_data[current_username]["history_times"] = users_data[current_username]["history_times"] + [day]
            with open(USERS_DATA_PATH, "w") as file:
                json.dump(users_data, file)
            self.heightupdateButton.background_normal = "pencil.png"
            self.heightupdateInput.disabled = not self.heightupdateInput.disabled
            self.errorMessage.text = ""

    def targetweightupdate(self, instance):
        if self.targetweightupdateInput.disabled:
            self.targetweightupdateButton.background_normal = "vee.png"
            self.targetweightupdateInput.disabled = not self.targetweightupdateInput.disabled
        elif(self.targetweightupdateInput.text == "" or int(self.targetweightupdateInput.text) < 40):
            self.errorMessage.text = "[b]Weight must be greater than 40 kg[/b]"
        else:
            global current_username
            users_data[current_username]["goal weight"] = self.targetweightupdateInput.text
            with open(USERS_DATA_PATH, "w") as file:
                json.dump(users_data, file)
            self.targetweightupdateButton.background_normal = "pencil.png"
            self.targetweightupdateInput.disabled = not self.targetweightupdateInput.disabled
            self.errorMessage.text = ""

    def activityupdate(self, instance):
        if self.activityupdateInput.disabled:
            self.activityupdateButton.background_normal = "vee.png"
        else:
            users_data[current_username]["activity"] = self.activityupdateInput.text
            with open(USERS_DATA_PATH, "w") as file:
                json.dump(users_data, file)
            self.activityupdateButton.background_normal = "pencil.png"
        self.activityupdateInput.disabled = not self.activityupdateInput.disabled
    
    def on_enter(self):
        global current_username
        Window.bind(on_keyboard=self.on_keyboard)
        self.weightupdateInput.text = users_data[current_username]["weight"]
        self.heightupdateInput.text = users_data[current_username]["height"]
        self.targetweightupdateInput.text = users_data[current_username]["goal weight"]
        self.activityupdateInput.text = users_data[current_username]["activity"]

    def on_leave(self):
        Window.unbind(on_keyboard=self.on_keyboard)

    def on_keyboard(self, window, key, *args):
        if key == 27:
            if self.manager.current == "personalData":
                self.go_home(self)
                return True
        return False

    def _update_border(self, instance, value):
        self.border.rectangle = (instance.x, instance.y, instance.width, instance.height)

    def _update_border2(self, instance, value):
        self.border2.rectangle = (instance.x, instance.y, instance.width, instance.height)

    def _update_border3(self, instance, value):
        self.border3.rectangle = (instance.x, instance.y, instance.width, instance.height)

################################

class StatisticsWindow(Screen):
    def __init__(self, **kw):
        super(StatisticsWindow, self).__init__(**kw)
        Window.bind(on_keyboard=self.on_keyboard)
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

        self.logo = Image(
            source = "logo.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.45, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.title = ColoredLabel(
            text = "[b]Statistics[/b]",
            font_size = 100,
            size_hint = (0.8, 0.05),
            pos_hint = {"x": 0.1, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup=True
        )
        self.window.add_widget(self.title)

        self.weightHeightLabel = ColoredLabel(
            text = "",
            font_size = 60,
            size_hint = (0.4, 0.05),
            pos_hint = {"x": 0.3, "top": 0.8},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.weightHeightLabel)

        self.bmiLabel = ColoredLabel(
            text = "",
            font_size = 60,
            size_hint = (0.6, 0.05),
            pos_hint = {"x": 0.2, "top": 0.725},
            color=(1, 1, 1, 1),
            text_color=(0, 1, 0, 1)
        )
        self.window.add_widget(self.bmiLabel)
        
        self.caloriesLabel = ColoredLabel(
            text = "",
            font_size = 60,
            size_hint = (0.6, 0.05),
            pos_hint = {"x": 0.2, "top": 0.65},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.caloriesLabel)

        self.carbohydratesLabel = ColoredLabel(
            text = "",
            font_size = 60,
            size_hint = (0.6, 0.05),
            pos_hint = {"x": 0.2, "top": 0.575},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.carbohydratesLabel)

        self.sugarLabel = ColoredLabel(
            text = "",
            font_size = 60,
            size_hint = (0.6, 0.05),
            pos_hint = {"x": 0.2, "top": 0.5},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.sugarLabel)

        self.fatLabel = ColoredLabel(
            text = "",
            font_size = 60,
            size_hint = (0.6, 0.05),
            pos_hint = {"x": 0.2, "top": 0.425},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.fatLabel)

        self.proteinLabel = ColoredLabel(
            text = "",
            font_size = 60,
            size_hint = (0.6, 0.05),
            pos_hint = {"x": 0.2, "top": 0.35},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.proteinLabel)

        self.graphWeight = Image(
            source = "",
            size_hint = (0.6, 0.3),
            pos_hint = {"x": 0.2, "top": 0.3},
        )
        self.window.add_widget(self.graphWeight)

        self.toastLabel = RoundedStencilView(
            size_hint=(0.6, 0.06),
            pos_hint={"x": 0.2, "top": 1},
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.toastLabel)
        self.toast = ColoredLabel(
            text="",
            font_size=40,
            size_hint=(0.6, 0.06),
            pos_hint={"x": 0.2, "top": 1},
            color=(1, 1, 1, 0),
            text_color=(1, 1, 1, 1),
            halign="center",
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.toast)

        ###

        self.add_widget(self.window)

    def hide_toast(self):
        self.toastLabel.pos_hint = {"x": 0.2, "top": 1}
        self.toast.pos_hint = {"x": 0.2, "top": 1}
        self.toastLabel.opacity = 0
        self.toastLabel.disabled = True
        self.toast.opacity = 0
        self.toast.disabled = True

    def show_toast(self, message):
        self.toastLabel.pos_hint = {"x": 0.2, "top": 0.2}
        self.toast.pos_hint = {"x": 0.2, "top": 0.2}
        self.toastLabel.opacity = 1
        self.toastLabel.disabled = False
        self.toast.opacity = 1
        self.toast.disabled = False
        self.toast.text = message

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def go_home(self, instance):
        self.manager.current = "main"

    def on_enter(self): 
        Window.bind(on_keyboard=self.on_keyboard)

        global current_username
        bmi_temp, bmi_color = check_bmi(float(users_data[current_username]["weight"]), float(users_data[current_username]["height"]))
        self.bmiLabel.text = "BMI: " + str(bmi(float(users_data[current_username]["weight"]), float(users_data[current_username]["height"]))) + " " + str(bmi_temp)
        self.bmiLabel.color = bmi_color

        calories = f"{int(users_data[current_username]['calories'])}"
        carbohydrates = f"{int(users_data[current_username]['carbohydrate'])}"
        sugar = f"{users_data[current_username]['sugar']:.2f}"
        fat = f"{users_data[current_username]['fat']:.2f}"
        protein = f"{users_data[current_username]['protein']:.2f}"

        calories_today = f"{int(users_data[current_username]['calories today'])}"
        carbohydrates_today = f"{int(users_data[current_username]['carbohydrates today'])}"
        sugar_today = f"{users_data[current_username]['sugar today']:.2f}"
        fat_today = f"{users_data[current_username]['fat today']:.2f}"
        protein_today = f"{users_data[current_username]['protein today']:.2f}"

        history_weight = users_data[current_username]["history_weight"]
        history_bmi = users_data[current_username]["history_bmi"]
        history_times = users_data[current_username]["history_times"]

        self.weightHeightLabel.text = users_data[current_username]["weight"]+ " Kg | " + users_data[current_username]["height"] + " cm"
        self.caloriesLabel.text = calories_today + "/" + calories + " Kcal Calories today"
        self.carbohydratesLabel.text = carbohydrates_today + "/" + carbohydrates + " g Carbohydrates today"
        self.sugarLabel.text = sugar_today + "/" + sugar + " g Sugar today"
        self.fatLabel.text = fat_today + "/" + fat + " g Fat today"
        self.proteinLabel.text = protein_today + "/" + protein + " g Protein today"

        try:
            server_url = "https://cs-project-m5hy.onrender.com/"

            wgraph_data = {
                "weights": history_weight,
                "bmis": history_bmi,
                "times": history_times
            }

            requests.get(server_url + "wakeup")

            response = requests.get(server_url + "wgraph", json=wgraph_data)

            if response.status_code == 200:
                with open("weight_history.png", "wb") as file:
                    file.write(response.content)
                self.graphWeight.source = "weight_history.png"
                self.graphWeight.reload()
            else:
                print("Error:", response.json())

        except Exception as e:
            self.show_toast("No internet connection. Retrying...")
            Clock.schedule_once(lambda _: self.hide_toast(), 3)
            Clock.schedule_once(lambda dt: self.on_enter(), 3)

    def on_leave(self):
        Window.unbind(on_keyboard=self.on_keyboard)

    def on_keyboard(self, window, key, *args):
        if key == 27:
            if self.manager.current == "statistics":
                self.go_home(self)
                return True
        return False

################################

class MenuWindow(Screen):
    def __init__(self, **kw):
        super(MenuWindow, self).__init__(**kw)
        self._adjust_label_event = None
        Window.bind(on_keyboard=self.on_keyboard)
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

        self.logo = Image(
            source = "logo.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.45, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.title = ColoredLabel(
            text = "[b]Menu[/b]",
            font_size = 100,
            size_hint = (0.8, 0.05),
            pos_hint = {"x": 0.1, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup=True
        )
        self.window.add_widget(self.title)

        self.dayInput = Spinner(
            font_size=40,
            text="",
            values=("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"),
            size_hint=(0.375, 0.05),
            pos_hint={"x": 0.1, "top": 0.8},
            background_color = SPINNER_BG,
            color = SPINNER_TEXT,
            option_cls=CustomSpinnerOption
        )
        self.dayInput.bind(text=self._update_meal)
        self.window.add_widget(self.dayInput)

        self.mealInput = Spinner(
            font_size=40,
            text="",
            values=("Breakfast", "Lunch", "Dinner"),
            size_hint=(0.375, 0.05),
            pos_hint={"x": 0.525, "top": 0.8},
            background_color = SPINNER_BG,
            color = SPINNER_TEXT,
            option_cls=CustomSpinnerOption
        )
        self.mealInput.bind(text=self._update_meal)
        self.window.add_widget(self.mealInput)

        scroll_view = ScrollView(
            size_hint=(0.8, 0.5),
            pos_hint={"x": 0.1, "top": 0.7},
            do_scroll_x=False,
            do_scroll_y=True
        )
        self.menuLabel = ColoredLabel(
            text="",
            font_size=45,
            size_hint=(1, 1),
            height=0,
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup=True,
            halign="left",
            valign="top"
        )
        self.menuLabel.bind(
            size=self._adjust_label_height,
            text=self._adjust_label_height
        )
        scroll_view.add_widget(self.menuLabel)
        self.window.add_widget(scroll_view)

        self.newMenuButton = Button(
            text = "New Menu",
            font_size = 40,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.3, 0.05),
            pos_hint = {"x": 0.35, "top": 0.15},
            on_press = self.newMenu
        )
        self.window.add_widget(self.newMenuButton)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def go_home(self, instance):
        self.manager.current = "main"

    def on_enter(self):
        Window.bind(on_keyboard=self.on_keyboard)
        day = datetime.now().strftime("%A")
        self.dayInput.text = day
        hour = int(datetime.now().strftime("%H"))
        if hour < 11:
            self.mealInput.text = "Breakfast"
        elif 11 <= hour < 17:
            self.mealInput.text = "Lunch"
        else:
            self.mealInput.text = "Dinner"
        
        self._update_meal(self.dayInput, self.dayInput.text)
        
    def on_leave(self):
        Window.unbind(on_keyboard=self.on_keyboard)

    def on_keyboard(self, window, key, *args):
        if key == 27:
            if self.manager.current == "menu":
                self.go_home(self)
                return True
        return False
        
    def _update_meal(self, instance, value):
        day = self.dayInput.text.lower()
        meal = self.mealInput.text.lower()

        if day != "" and meal != "":
            foods = get_meal(day, meal)
            text = '\n'.join([f"[b]{food}[/b] ({amount}g)" for food, amount in foods.items()])
            wrapped_text = self.wrap(text, 45)
            if self.menuLabel.text != wrapped_text: 
                self.menuLabel.text = wrapped_text

    def wrap(self, text: str, k: int):

        sentences = text.split("\n")

        lines = []

        current_line = []
        current_line_len = 0
        
        for sentence in sentences:
            words = sentence.strip(' ').split()

            if words:
                for word in words:
                    if not current_line:
                        current_line.append(word)
                        current_line_len += len(word)
                    elif current_line_len + len(word) + 1 <= k:
                        current_line.append(word)
                        current_line_len += len(word) + 1
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_line_len = len(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = []
                    current_line_len = 0

                lines.append("")

            wrapped_text = '\n'.join(lines)

        return wrapped_text

    def newMenu(self, instance):
        global menu_request_window
        menu_request_window = "menu"

        self.manager.current = "loading"

    def _adjust_label_height(self, *args):
        if self._adjust_label_event:
            self._adjust_label_event.cancel()
        self._adjust_label_event = Clock.schedule_once(self._apply_label_height, 0.1)

    def _apply_label_height(self, *args):
        self.menuLabel.height = self.menuLabel.texture_size[1]
        self.menuLabel.text_size = (self.menuLabel.width, None)
        self.menuLabel.size_hint_y = None

################################

class FoodTrackerWindow(Screen):
    def __init__(self, **kw):
        super(FoodTrackerWindow, self).__init__(**kw)
        Window.bind(on_keyboard=self.on_keyboard)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.temp_food = ""

        self.home = Button(
            background_normal="home.png",
            size_hint=(0.1125, 0.07),
            pos_hint={"x": 0, "top": 1},
            on_press=self.go_home
        )
        self.window.add_widget(self.home)

        self.logo = Image(
            source = "logo.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.45, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.title = ColoredLabel(
            text = "[b]Food Tracker[/b]",
            font_size = 100,
            size_hint = (0.8, 0.05),
            pos_hint = {"x": 0.1, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup = True
        )
        self.window.add_widget(self.title)

        self.dayInput = Spinner(
            font_size=40,
            text="",
            values=("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"),
            size_hint=(0.375, 0.05),
            pos_hint={"x": 0.1, "top": 0.8},
            background_color = SPINNER_BG,
            color = SPINNER_TEXT,
            option_cls=CustomSpinnerOption
        )
        self.dayInput.bind(text=self._update_meal)
        self.window.add_widget(self.dayInput)

        self.mealInput = Spinner(
            font_size=40,
            text="",
            values=("Breakfast", "Lunch", "Dinner"),
            size_hint=(0.375, 0.05),
            pos_hint={"x": 0.525, "top": 0.8},
            background_color = SPINNER_BG,
            color = SPINNER_TEXT,
            option_cls=CustomSpinnerOption
        )
        self.mealInput.bind(text=self._update_meal)
        self.window.add_widget(self.mealInput)

        self.input = TextInput(
            multiline = False,
            font_size = 40,
            hint_text = "Add food",
            size_hint=(0.3, 0.06),
            pos_hint={"x": 0.1, "top": 0.725},
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
            halign="center",
        )
        self.input.bind(size=self._update_text_padding1)
        self.window.add_widget(self.input)
        with self.input.canvas.before:
            Color(0, 0, 0, 1)  
            self.border = Line(rectangle=(self.input.x, self.input.y, self.input.width, self.input.height), width=1.0)
        self.input.bind(size=self._update_border, pos=self._update_border)

        self.search_button = Button(
            background_normal = "search.png",
            font_size = 40,
            background_color = (1, 1, 1, 1),
            size_hint=(0.1, 0.06),
            pos_hint={"x": 0.4, "top": 0.725},
            on_press = self.perform_search
        )
        self.window.add_widget(self.search_button)

        self.amount_input = TextInput(
            multiline = False,
            font_size = 40,
            hint_text = "Amount in grams",
            size_hint=(0.3, 0.06),
            pos_hint={"x": 0.5, "top": 0.725},
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
            input_filter="float",
            halign="center",
        )
        self.amount_input.bind(size=self._update_text_padding2)
        self.window.add_widget(self.amount_input)
        with self.amount_input.canvas.before:
            Color(0, 0, 0, 1) 
            self.border2 = Line(rectangle=(self.amount_input.x, self.amount_input.y, self.amount_input.width, self.amount_input.height), width=1.0)
        self.amount_input.bind(size=self._update_border2, pos=self._update_border2)

        self.add_button = Button(
            background_normal = "plus.png",
            font_size = 40,
            background_color = (1, 1, 1, 1),
            size_hint=(0.1, 0.06),
            pos_hint={"x": 0.8, "top": 0.725},
            on_press = self._add_food
        )
        self.window.add_widget(self.add_button)

        self.labels = []
        for i in range(10):
            label = ColoredLabel(
                text="",
                font_size=35,
                size_hint=(0.8, 0.06),
                pos_hint={"x": 0.1, "top": 0.665 - i * (0.06 + 5/900)},
                color=(1, 1, 1, 1),
                text_color=(0, 0, 0, 1),
                halign="left",
                markup=True
            )
            label.bind(size=lambda instance, value: setattr(instance, 'text_size', (instance.width - 15, None)))
            self.labels.append(label)
            self.window.add_widget(label)

        self.remove_buttons = []
        for i in range(10):
            button = Button(
                background_normal = "remove.png",
                font_size = 40,
                background_color = (1, 1, 1, 1),
                size_hint=(0.1, 0.06),
                pos_hint={"x": 0.9, "top": 0.665 - i * (0.06 + 5/900)},
                opacity=0,
                disabled=True,
                on_press = self.remove
            )
            self.remove_buttons.append(button)
            self.window.add_widget(button)

        self.result_buttons = []
        for i in range(10):
            button = Button(
                text="",
                font_size=35,
                size_hint=(0.8, 0.06),
                pos_hint={"x": 0.1, "top": 0.665 - i * (0.06 + 5/900)},
                on_press=self.word_clicked,
                halign="left",
                opacity=0,
                disabled=True,
                background_normal="",
                background_color=(0.85, 0.85, 0.85, 1), 
                color=(0.376, 0.376, 0.376, 1)
            )
            button.bind(size=lambda instance, vlaue: setattr(instance, 'text_size', (instance.width - 15, None)))
            self.result_buttons.append(button)
            self.window.add_widget(button)
        
        self.toastLabel = RoundedStencilView(
            size_hint=(0.6, 0.06),
            pos_hint={"x": 0.2, "top": 1},
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.toastLabel)
        self.toast = ColoredLabel(
            text="",
            font_size=40,
            size_hint=(0.6, 0.06),
            pos_hint={"x": 0.2, "top": 1},
            color=(1, 1, 1, 0),
            text_color=(1, 1, 1, 1),
            halign="center",
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.toast)

        ###

        self.add_widget(self.window)

    def _update_text_padding1(self, instance, value):
        instance.padding_y = [(instance.height - instance.line_height) / 2, 0]

    def _update_text_padding2(self, instance, value):
        instance.padding_y = [(instance.height - instance.line_height) / 2, 0]

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def go_home(self, instance):
        self.manager.current = "main"

    def on_enter(self):
        Window.bind(on_keyboard=self.on_keyboard)
        day = datetime.now().strftime("%A")
        self.dayInput.text = day
        hour = int(datetime.now().strftime("%H"))
        
        if hour < 11:
            self.mealInput.text = "Breakfast"
        elif 11 <= hour < 17:
            self.mealInput.text = "Lunch"
        else:
            self.mealInput.text = "Dinner"

        for button in self.result_buttons:
            button.opacity = 0
            button.disabled = True

    def on_leave(self):
        Window.unbind(on_keyboard=self.on_keyboard)
        self.input.text = ""
        self.amount_input.text = ""
        self.temp_food = ""
        for button in self.result_buttons:
            button.opacity = 0
            button.disabled = True

    def on_keyboard(self, window, key, *args):
        if key == 27:
            if self.manager.current == "foodTracker":
                self.go_home(self)
                return True
        return False
        
    def _update_meal(self, instance, value):
        self.temp_food = ""
        self.input.text = ""
        self.amount_input.text = ""
        day = self.dayInput.text.lower()
        meal = self.mealInput.text.lower()

        for label, button in zip(self.labels, self.remove_buttons):
            button.opacity = 0
            button.disabled = True
            label.text = ""

        if day != "" and meal != "":
            global current_username
            foods = users_data[current_username]["self_menu"][day][meal]
            for food, amount in foods.items():
                for label, button in zip(self.labels, self.remove_buttons):
                    button.opacity = 1
                    button.disabled = False
                    if label.text == "":
                        label.text = f"[b]{food}[/b] ({amount}g)"
                        break

    def _add_food(self, instance):
        amount = self.amount_input.text
        food = self.temp_food

        if (amount == "" or food == ""):
            self.input.text = ""
            self.amount_input.text = ""
            self.temp_food = ""
            self.show_toast("Please fill in all fields")
            Clock.schedule_once(lambda dt: self.hide_toast(), 2)
            return
        
        amount = float(amount)

        if(amount <= 0):
            self.input.text = ""
            self.amount_input.text = ""
            self.temp_food = ""
            self.show_toast("Please enter a valid amount")
            Clock.schedule_once(lambda dt: self.hide_toast(), 2)
            return

        day = self.dayInput.text.lower()
        meal = self.mealInput.text.lower()

        global current_username
        foods = users_data[current_username]["self_menu"][day][meal]
        if len(foods) >= 10:
            self.input.text = ""
            self.amount_input.text = ""
            self.temp_food = ""
            self.show_toast("You can only add 10 foods per meal")
            Clock.schedule_once(lambda dt: self.hide_toast(), 2)
            return
        
        if food in foods:
            self.input.text = ""
            self.amount_input.text = ""
            self.temp_food = ""
            self.show_toast("Food already exists in the meal")
            Clock.schedule_once(lambda dt: self.hide_toast(), 2)
            return

        add_food(day, meal, food, amount)

        self._update_meal(self.dayInput, self.dayInput.text)
        self.input.text = ""
        self.amount_input.text = ""
        self.temp_food = ""

    def remove(self, instance):
        index = self.remove_buttons.index(instance)
        day = self.dayInput.text.lower()
        meal = self.mealInput.text.lower()

        global current_username
        foods = users_data[current_username]["self_menu"][day][meal]
        food = list(foods.keys())[index]
        
        self.labels[index].text = ""

        remove_food(day, meal, food)

        self._update_meal(self.dayInput, self.dayInput.text)
        
    def hide_toast(self):
        self.toastLabel.pos_hint = {"x": 0.2, "top": 1}
        self.toast.pos_hint = {"x": 0.2, "top": 1}
        self.toastLabel.opacity = 0
        self.toastLabel.disabled = True
        self.toast.opacity = 0
        self.toast.disabled = True

    def show_toast(self, message):
        self.toastLabel.pos_hint = {"x": 0.2, "top": 0.2}
        self.toast.pos_hint = {"x": 0.2, "top": 0.2}
        self.toastLabel.opacity = 1
        self.toastLabel.disabled = False
        self.toast.opacity = 1
        self.toast.disabled = False
        self.toast.text = message

    def word_clicked(self, instance):
        for button in self.result_buttons:
            button.opacity = 0
            button.disabled = True

        self.temp_food = instance.text   
        self.input.text = instance.text   

    def on_touch_down(self, touch):
        if (self.input.collide_point(*touch.pos) or 
            self.search_button.collide_point(*touch.pos) or 
            any(button.collide_point(*touch.pos) for button in self.result_buttons)):
            return super(FoodTrackerWindow, self).on_touch_down(touch)

        for button in self.result_buttons:
            button.opacity = 0
            button.disabled = True

        return super(FoodTrackerWindow, self).on_touch_down(touch)

    def _update_border(self, instance, value):
        self.border.rectangle = (instance.x, instance.y, instance.width, instance.height)

    def _update_border2(self, instance, value):
        self.border2.rectangle = (instance.x, instance.y, instance.width, instance.height)

    def perform_search(self, instance):    
        if not self.input.text.strip():
            for button in self.result_buttons:
                button.opacity = 0
                button.disabled = True
            return
        
        self.search_button.background_normal = "hourglass.png"
        self.search_button.background_down = "hourglass.png"

        def cont(dt):
            words = self.get_words(self.input.text)

            self.search_button.background_normal = "search.png"

            if words: 
                for i, button in enumerate(self.result_buttons):
                    if i < len(words):
                        button.text = words[i]
                        button.opacity = 1
                        button.disabled = False
                    else:
                        button.opacity = 0
                        button.disabled = True

        Clock.schedule_once(cont, 0)

    def get_words(self, query):
        try:
            server_url = "https://cs-project-m5hy.onrender.com/"

            requests.get(server_url + "wakeup")

            query = {
            "query": query
            }

            response = requests.get(server_url + "search", json=query)

            if response.status_code == 200:
                return response.json().get("results", [])
            else:
                print("Error:", response.json())

        except Exception as e:
            self.show_toast("No internet connection. Retrying...")
            Clock.schedule_once(lambda _: self.hide_toast(), 3)

################################

class DictionaryWindow(Screen):
    def __init__(self, **kw):
        super(DictionaryWindow, self).__init__(**kw)
        Window.bind(on_keyboard=self.on_keyboard)
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

        self.logo = Image(
            source = "logo.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.45, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.title = ColoredLabel(
            text = "[b]Dictionary[/b]",
            font_size = 100,
            size_hint = (0.8, 0.05),
            pos_hint = {"x": 0.1, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup = True
        )
        self.window.add_widget(self.title)

        self.input = TextInput(
            multiline = False,
            font_size = 40,
            hint_text = "Search",
            size_hint=(0.675, 0.05),
            pos_hint={"x": 0.1, "top": 0.775},
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
            halign="center"
        )
        self.input.bind(size=self._update_text_padding1)
        self.window.add_widget(self.input)
        with self.input.canvas.before:
            Color(0, 0, 0, 1) 
            self.border = Line(rectangle=(self.input.x, self.input.y, self.input.width, self.input.height), width=1.0)
        self.input.bind(size=self._update_border, pos=self._update_border)

        self.search_button = Button(
            background_normal = "search.png",
            font_size = 40,
            background_color = (1, 1, 1, 1),
            size_hint=(0.1, 0.05),
            pos_hint={"x": 0.8, "top": 0.775},
            on_press = self.perform_search
        )
        self.window.add_widget(self.search_button)

        self.labels = []
        self.title1 = ColoredLabel1(
            text = "Macronutrients",
            font_size = 35,
            size_hint = (0.8, 0.04),
            pos_hint = {"x": 0.1, "top": 0.7},
            color=(0.86, 0.94, 0.98, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.title1)
        self.labels.append(self.title1)

        self.Calories = ColoredLabel1(
            text = "Calories",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.1, "top": 0.66},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Calories)
        self.labels.append(self.Calories)

        self.Caloriesdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.1, "top": 0.62},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Caloriesdata)
        self.labels.append(self.Caloriesdata)

        self.Protein = ColoredLabel1(
            text = "Protein",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.3, "top": 0.66},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Protein)
        self.labels.append(self.Protein)

        self.Proteindata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.3, "top": 0.62},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Proteindata)
        self.labels.append(self.Proteindata)

        self.Carbohydrate = ColoredLabel1(
            text = "Carbohydrate",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.5, "top": 0.66},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Carbohydrate)
        self.labels.append(self.Carbohydrate)

        self.Carbohydratedata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.5, "top": 0.62},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Carbohydratedata)
        self.labels.append(self.Carbohydratedata)

        self.Fat = ColoredLabel1(
            text = "Fat",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.7, "top": 0.66},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Fat)
        self.labels.append(self.Fat)

        self.Fatdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.7, "top": 0.62},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Fatdata)
        self.labels.append(self.Fatdata)

        self.Water = ColoredLabel1(
            text = "Water",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.1, "top": 0.58},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Water)
        self.labels.append(self.Water)

        self.Waterdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.1, "top": 0.54},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Waterdata)
        self.labels.append(self.Waterdata)

        self.Fiber = ColoredLabel1(
            text = "Fiber",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.3, "top": 0.58},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Fiber)
        self.labels.append(self.Fiber)

        self.Fiberdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.3, "top": 0.54},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Fiberdata)
        self.labels.append(self.Fiberdata)

        self.Sugars = ColoredLabel1(
            text = "Sugars",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.5, "top": 0.58},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Sugars)
        self.labels.append(self.Sugars)

        self.Sugarsdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.5, "top": 0.54},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Sugarsdata)
        self.labels.append(self.Sugarsdata)

        self.Saturatedfat = ColoredLabel1(
            text = "Saturated fat",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.7, "top": 0.58},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Saturatedfat)
        self.labels.append(self.Saturatedfat)

        self.Saturatedfatdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.2, 0.04),
            pos_hint = {"x": 0.7, "top": 0.54},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Saturatedfatdata)
        self.labels.append(self.Saturatedfatdata)

        self.title2 = ColoredLabel1(
            text = "Micronutrients",
            font_size = 35,
            size_hint = (0.8, 0.04),
            pos_hint = {"x": 0.1, "top": 0.5},
            color=(0.86, 0.94, 0.98, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.title2)
        self.labels.append(self.title2)

        self.Cholesterol = ColoredLabel1(
            text = "Cholesterol",
            font_size = 30,
            size_hint = (0.8 / 3, 0.04),
            pos_hint = {"x": 0.1, "top": 0.46},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Cholesterol)
        self.labels.append(self.Cholesterol)

        self.Cholesteroldata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.8 / 3, 0.04),
            pos_hint = {"x": 0.1, "top": 0.42},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Cholesteroldata)
        self.labels.append(self.Cholesteroldata)

        self.Calcium = ColoredLabel1(
            text = "Calcium",
            font_size = 30,
            size_hint = (0.8 / 3, 0.04),
            pos_hint = {"x": 0.1 + 0.8 / 3, "top": 0.46},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Calcium)
        self.labels.append(self.Calcium)

        self.Calciumdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.8 / 3, 0.04),
            pos_hint = {"x": 0.1 + 0.8 / 3, "top": 0.42},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Calciumdata)
        self.labels.append(self.Calciumdata)

        self.Iron = ColoredLabel1(
            text = "Iron",
            font_size = 30,
            size_hint = (0.8/3, 0.04),
            pos_hint = {"x": 0.1 + 2 * 0.8 / 3, "top": 0.46},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Iron)
        self.labels.append(self.Iron)

        self.Irondata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.8/3, 0.04),
            pos_hint = {"x": 0.1 + 2 * 0.8 / 3, "top": 0.42},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Irondata)
        self.labels.append(self.Irondata)

        self.Sodium = ColoredLabel1(
            text = "Sodium",
            font_size = 30,
            size_hint = (0.4, 0.04),
            pos_hint = {"x": 0.1, "top": 0.38},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Sodium)
        self.labels.append(self.Sodium)

        self.Sodiumdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.4, 0.04),
            pos_hint = {"x": 0.1, "top": 0.34},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Sodiumdata)
        self.labels.append(self.Sodiumdata)

        self.Magnesium = ColoredLabel1(
            text = "Magnesium",
            font_size = 30,
            size_hint = (0.4, 0.04),
            pos_hint = {"x": 0.5, "top": 0.38},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Magnesium)
        self.labels.append(self.Magnesium)

        self.Magnesiumdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.4, 0.04),
            pos_hint = {"x": 0.5, "top": 0.34},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.Magnesiumdata)
        self.labels.append(self.Magnesiumdata)

        self.title3 = ColoredLabel1(
            text = "Vitamins",
            font_size = 35,
            size_hint = (0.8, 0.04),
            pos_hint = {"x": 0.1, "top": 0.3},
            color=(0.86, 0.94, 0.98, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.title3)
        self.labels.append(self.title3)

        self.VitaminA = ColoredLabel1(
            text = "A",
            font_size = 30,
            size_hint = (0.8 / 3, 0.04),
            pos_hint = {"x": 0.1, "top": 0.26},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.VitaminA)
        self.labels.append(self.VitaminA)

        self.VitaminAdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.8 / 3, 0.04),
            pos_hint = {"x": 0.1, "top": 0.22},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.VitaminAdata)
        self.labels.append(self.VitaminAdata)

        self.VitaminB12 = ColoredLabel1(
            text = "B12",
            font_size = 30,
            size_hint = (0.8 / 3, 0.04),
            pos_hint = {"x": 0.1 + 0.8 / 3, "top": 0.26},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.VitaminB12)
        self.labels.append(self.VitaminB12)

        self.VitaminB12data = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.8 / 3, 0.04),
            pos_hint = {"x": 0.1 + 0.8 / 3, "top": 0.22},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.VitaminB12data)
        self.labels.append(self.VitaminB12data)

        self.VitaminC = ColoredLabel1(
            text = "C",
            font_size = 30,
            size_hint = (0.8/3, 0.04),
            pos_hint = {"x": 0.1 + 2 * 0.8 / 3, "top": 0.26},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.VitaminC)
        self.labels.append(self.VitaminC)

        self.VitaminCdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.8/3, 0.04),
            pos_hint = {"x": 0.1 + 2 * 0.8 / 3, "top": 0.22},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.VitaminCdata)
        self.labels.append(self.VitaminCdata)

        self.VitaminD = ColoredLabel1(
            text = "D",
            font_size = 30,
            size_hint = (0.8 / 3, 0.04),
            pos_hint = {"x": 0.1, "top": 0.18},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.VitaminD)
        self.labels.append(self.VitaminD)

        self.VitaminDdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.8 / 3, 0.04),
            pos_hint = {"x": 0.1, "top": 0.14},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.VitaminDdata)
        self.labels.append(self.VitaminDdata)

        self.VitaminE = ColoredLabel1(
            text = "E",
            font_size = 30,
            size_hint = (0.8 / 3, 0.04),
            pos_hint = {"x": 0.1 + 0.8 / 3, "top": 0.18},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.VitaminE)
        self.labels.append(self.VitaminE)

        self.VitaminEdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.8 / 3, 0.04),
            pos_hint = {"x": 0.1 + 0.8 / 3, "top": 0.14},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.VitaminEdata)
        self.labels.append(self.VitaminEdata)

        self.VitaminK = ColoredLabel1(
            text = "K",
            font_size = 30,
            size_hint = (0.8/3, 0.04),
            pos_hint = {"x": 0.1 + 2 * 0.8 / 3, "top": 0.18},
            color=(0.93, 0.97, 0.99, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.VitaminK)
        self.labels.append(self.VitaminK)

        self.VitaminKdata = ColoredLabel1(
            text = "",
            font_size = 30,
            size_hint = (0.8/3, 0.04),
            pos_hint = {"x": 0.1 + 2 * 0.8 / 3, "top": 0.14},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            border_color=(0, 0, 0, 1),
            border_width=1,
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.VitaminKdata)
        self.labels.append(self.VitaminKdata)

        self.result_buttons = []
        for i in range(10):
            button = Button(
                text="",
                font_size=35,
                size_hint=(0.675, 0.065),
                pos_hint={"x": 0.1, "top": 0.725 - i * (0.065 + 5/900)},
                on_press=self.word_clicked,
                halign="left",
                opacity=0,
                disabled=True,
                background_normal="",
                background_color=(0.85, 0.85, 0.85, 1), 
                color=(0.376, 0.376, 0.376, 1)
            )
            button.bind(size=lambda instance, vlaue: setattr(instance, 'text_size', (instance.width - 15, None)))
            self.result_buttons.append(button)
            self.window.add_widget(button)

        self.toastLabel = RoundedStencilView(
            size_hint=(0.6, 0.06),
            pos_hint={"x": 0.2, "top": 1},
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.toastLabel)
        self.toast = ColoredLabel(
            text="",
            font_size=40,
            size_hint=(0.6, 0.06),
            pos_hint={"x": 0.2, "top": 1},
            color=(1, 1, 1, 0),
            text_color=(1, 1, 1, 1),
            halign="center",
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.toast)

        ###

        self.add_widget(self.window)

    def hide_toast(self):
        self.toastLabel.pos_hint = {"x": 0.2, "top": 1}
        self.toast.pos_hint = {"x": 0.2, "top": 1}
        self.toastLabel.opacity = 0
        self.toastLabel.disabled = True
        self.toast.opacity = 0
        self.toast.disabled = True

    def show_toast(self, message):
        self.toastLabel.pos_hint = {"x": 0.2, "top": 0.2}
        self.toast.pos_hint = {"x": 0.2, "top": 0.2}
        self.toastLabel.opacity = 1
        self.toastLabel.disabled = False
        self.toast.opacity = 1
        self.toast.disabled = False
        self.toast.text = message

    def _update_text_padding1(self, instance, value):
        instance.padding_y = [(instance.height - instance.line_height) / 2, 0]

    def _update_border(self, instance, value):
        self.border.rectangle = (instance.x, instance.y, instance.width, instance.height)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def go_home(self, instance):
        self.manager.current = "main"

    def on_enter(self):
        Window.bind(on_keyboard=self.on_keyboard)
        for button in self.result_buttons:
            button.opacity = 0
            button.disabled = True

        for label in self.labels:
            label.opacity = 0
            label.disabled = True

    def on_leave(self):
        Window.unbind(on_keyboard=self.on_keyboard)
        self.input.text = ""
        for button in self.result_buttons:
            button.opacity = 0
            button.disabled = True
            
        for label in self.labels:
            label.opacity = 0
            label.disabled = True

    def on_keyboard(self, window, key, *args):
        if key == 27:
            if self.manager.current == "dictionary":
                self.go_home(self)
                return True
        return False

    def perform_search(self, instance):    
        if not self.input.text.strip():
            for button in self.result_buttons:
                button.opacity = 0
                button.disabled = True
            return
        
        self.search_button.background_normal = "hourglass.png"
        self.search_button.background_down = "hourglass.png"

        def cont(dt):
            for label in self.labels:
                label.opacity = 0
                label.disabled = True

            words = self.get_words(self.input.text)

            self.search_button.background_normal = "search.png"

            if words:
                for i, button in enumerate(self.result_buttons):
                    if i < len(words):
                        button.text = words[i]
                        button.opacity = 1
                        button.disabled = False
                    else:
                        button.opacity = 0
                        button.disabled = True

        Clock.schedule_once(cont, 0)

    def word_clicked(self, instance):
        self.input.text = instance.text

        for button in self.result_buttons:
            button.opacity = 0
            button.disabled = True

        food = instance.text
        food = foods_dict[food]
        
        self.Proteindata.text = str(food["Protein"]) + " " + units["Protein"]
        self.Caloriesdata.text = str(food["Calories"]) + " " + units["Calories"]
        self.Carbohydratedata.text = str(food["Carbohydrate"]) + " " + units["Carbohydrate"]
        self.Fatdata.text = str(food["Fat"]) + " " + units["Fat"]
        self.Sugarsdata.text = str(food["Sugars"]) + " " + units["Sugars"] + f' ({(food["Sugars"] / 4):.1f} tbsp)'
        self.Waterdata.text = str(food["Water"]) + " " + units["Water"]
        self.Fiberdata.text = str(food["Fiber"]) + " " + units["Fiber"]
        self.Saturatedfatdata.text = str(food["Saturated fat"]) + " " + units["Saturated fat"]
        self.Cholesteroldata.text = str(food["Cholesterol"]) + " " + units["Cholesterol"]
        self.Calciumdata.text = str(food["Calcium"]) + " " + units["Calcium"]
        self.Irondata.text = str(food["Iron"]) + " " + units["Iron"]
        self.Sodiumdata.text = str(food["Sodium"]) + " " + units["Sodium"]
        self.Magnesiumdata.text = str(food["Magnesium"]) + " " + units["Magnesium"]
        self.VitaminAdata.text = str(food["Vitamin A"]) + " " + units["Vitamin A"]
        self.VitaminB12data.text = str(food["Vitamin B-12"]) + " " + units["Vitamin B-12"]
        self.VitaminCdata.text = str(food["Vitamin C"]) + " " + units["Vitamin C"]
        self.VitaminDdata.text = str(food["Vitamin D"]) + " " + units["Vitamin D"]
        self.VitaminEdata.text = str(food["Vitamin E"]) + " " + units["Vitamin E"]
        self.VitaminKdata.text = str(food["Vitamin K"]) + " " + units["Vitamin K"]

        for label in self.labels:
            label.opacity = 1
            label.disabled = False
        
    def get_words(self, query):
        try:
            server_url = "https://cs-project-m5hy.onrender.com/"

            requests.get(server_url + "wakeup")

            query = {
            "query": query
            }

            response = requests.get(server_url + "search", json=query)

            if response.status_code == 200:
                return response.json().get("results", [])
            else:
                print("Error:", response.json())

        except Exception as e:
            self.show_toast("No internet connection. Retrying...")
            Clock.schedule_once(lambda _: self.hide_toast(), 3)

    def on_touch_down(self, touch):
        if (self.input.collide_point(*touch.pos) or 
            self.search_button.collide_point(*touch.pos) or 
            any(button.collide_point(*touch.pos) for button in self.result_buttons)):
            return super(DictionaryWindow, self).on_touch_down(touch)

        for button in self.result_buttons:
            button.opacity = 0
            button.disabled = True

        return super(DictionaryWindow, self).on_touch_down(touch)
        
################################

class CreateAccountWindow(Screen):
    def __init__(self, **kw):
        super(CreateAccountWindow, self).__init__(**kw)
        Window.bind(on_keyboard=self.on_keyboard)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.logo = Image(
            source = "logo.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.45, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.title = ColoredLabel(
            text = "[b]Create an Account[/b]",
            font_size = 100,
            size_hint = (0.8, 0.2),
            pos_hint = {"x": 0.1, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup=True,
        )
        self.window.add_widget(self.title)

        self.userName = TextInput(
            multiline = False,
            font_size = 70,
            hint_text = "Username",
            size_hint=(0.8, 0.1),
            pos_hint={"x": 0.1, "top": 0.68},
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
            halign="center"
        )
        self.userName.bind(size=self._update_text_padding1)
        self.window.add_widget(self.userName)
        with self.userName.canvas.before:
            Color(0, 0, 0, 1)
            self.border = Line(rectangle=(self.userName.x, self.userName.y, self.userName.width, self.userName.height), width=1.0)
        self.userName.bind(size=self._update_border, pos=self._update_border)

        self.password = TextInput(
            multiline = False,
            font_size = 70,
            hint_text = "Password",
            size_hint=(0.8, 0.1),
            pos_hint={"x": 0.1, "top": 0.56},
            password=True,
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
            halign="center"
        )
        self.window.add_widget(self.password)
        self.password.bind(size=self._update_text_padding2)
        with self.password.canvas.before:
            Color(0, 0, 0, 1) 
            self.border2 = Line(rectangle=(self.password.x, self.password.y, self.password.width, self.password.height), width=1.0)
        self.password.bind(size=self._update_border2, pos=self._update_border2)

        self.showpassword = ColoredLabel(
            text = "[b]Show Password[/b]",
            font_size = 40,
            size_hint = (0.2, 0.05),
            pos_hint = {"x": 0.3, "top": 0.45},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup=True,
        )
        self.window.add_widget(self.showpassword)

        self.showpasswordInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.475},
            color= LABEL_BG,
            on_press = self.show_password
        )
        self.window.add_widget(self.showpasswordInput)

        self.errorMessage = ColoredLabel(
            text = "",
            font_size = 50,
            size_hint = (0.8, 0.05),
            pos_hint = {"x": 0.1, "top": 0.27},
            color=(1, 1, 1, 1),
            text_color=(0.718, 0.11, 0.11, 1),
            markup=True,
        )
        self.window.add_widget(self.errorMessage)

        self.submit = Button(
            text = "[b]Submit[/b]",
            font_size = 90,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.2},
            on_press = self.registration,
            markup=True,
        )
        self.window.add_widget(self.submit)

        ###

        self.add_widget(self.window)

    def _update_text_padding1(self, instance, value):
        instance.padding_y = [(instance.height - instance.line_height) / 2, 0]

    def _update_text_padding2(self, instance, value):
        instance.padding_y = [(instance.height - instance.line_height) / 2, 0]

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def log_in(self, instance):
        self.manager.current = "login"
        self.errorMessage.text = ""
        self.showpasswordInput.active = False

    def on_enter(self):
        Window.bind(on_keyboard=self.on_keyboard)

    def on_leave(self):
        Window.unbind(on_keyboard=self.on_keyboard)

    def registration(self, instance):
        global info
        info = Info()

        username = self.userName.text
        password = self.password.text

        if(username == "" or password == ""):
            self.errorMessage.text = "[b]Cannot leave fields empty[/b]"
        elif username in users_data:
            self.errorMessage.text = "[b]Username already exists[/b]"
        else:
            self.userName.text = ""
            self.password.text = ""
            
            global current_username
            current_username = username
            info.password = password

            self.manager.current = "registration1"
            self.errorMessage.text = ""

    def show_password(self, instance):
        self.password.password = not self.password.password

    def on_keyboard(self, window, key, *args):
        if key == 27:
            if self.manager.current == "createAccount":
                self.log_in(self)
                return True
        return False
    
    def _update_border(self, instance, value):
        self.border.rectangle = (instance.x, instance.y, instance.width, instance.height)

    def _update_border2(self, instance, value):
        self.border2.rectangle = (instance.x, instance.y, instance.width, instance.height)

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

        self.logo = Image(
            source = "logo.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.45, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.title = ColoredLabel(
            text = "[b]Registration[/b]",
            font_size = 150,
            size_hint = (0.775, 0.2),
            pos_hint = {"x": 0.1125, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup=True,
        )
        self.window.add_widget(self.title)

        self.weightLabel = ColoredLabel(
            text = "[b]Weight:[/b]",
            font_size = 60,
            size_hint = (0.44, 0.1),
            pos_hint = {"x": 0.05, "top": 0.7},
            color = LABEL_BG,
            text_color =LABEL_TEXT,
            markup=True
        )
        self.window.add_widget(self.weightLabel)

        self.weightInput = TextInput(
            multiline = False,
            font_size = 50,
            hint_text = "Kg",
            size_hint=(0.44, 0.1),
            pos_hint={"x": 0.51, "top": 0.7},
            input_filter="int",
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
            halign="center"
        )
        self.weightInput.bind(size=self._update_text_padding1)
        self.window.add_widget(self.weightInput)
        with self.weightInput.canvas.before:
            Color(0, 0, 0, 1) 
            self.border = Line(rectangle=(self.weightInput.x, self.weightInput.y, self.weightInput.width, self.weightInput.height), width=1.0)
        self.weightInput.bind(size=self._update_border, pos=self._update_border)

        self.heightLabel = ColoredLabel(
            text = "[b]Height:[/b]",
            font_size = 60,
            size_hint = (0.44, 0.1),
            pos_hint = {"x": 0.05, "top": 0.58},
            color = LABEL_BG,
            text_color =LABEL_TEXT,
            markup=True
        )
        self.window.add_widget(self.heightLabel)

        self.heightInput = TextInput(
            multiline = False,
            font_size = 50,
            hint_text = "cm",
            size_hint=(0.44, 0.1),
            pos_hint={"x": 0.51, "top": 0.58},
            input_filter="int",
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
            halign="center"
        )
        self.heightInput.bind(size=self._update_text_padding2)
        self.window.add_widget(self.heightInput)
        with self.heightInput.canvas.before:
            Color(0, 0, 0, 1) 
            self.border2 = Line(rectangle=(self.heightInput.x, self.heightInput.y, self.heightInput.width, self.heightInput.height), width=1.0)
        self.heightInput.bind(size=self._update_border2, pos=self._update_border2)

        self.AgeLabel = ColoredLabel(
            text = "[b]Age:[/b]",
            font_size = 60,
            size_hint = (0.44, 0.1),
            pos_hint = {"x": 0.05, "top": 0.46},
            color = LABEL_BG,
            text_color =LABEL_TEXT,
            markup=True
        )
        self.window.add_widget(self.AgeLabel)

        self.AgeInput = TextInput(
            multiline = False,
            font_size = 50,
            hint_text = "Years",
            size_hint=(0.44, 0.1),
            pos_hint={"x": 0.51, "top": 0.46},
            input_filter="int",
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
            halign="center",
        )
        self.AgeInput.bind(size=self._update_text_padding3)
        self.window.add_widget(self.AgeInput)
        with self.AgeInput.canvas.before:
            Color(0, 0, 0, 1) 
            self.border3 = Line(rectangle=(self.AgeInput.x, self.AgeInput.y, self.AgeInput.width, self.AgeInput.height), width=1.0)
        self.AgeInput.bind(size=self._update_border3, pos=self._update_border3)

        self.genderLabel = ColoredLabel(
            text = "[b]Gender:[/b]",
            font_size = 60,
            size_hint = (0.44, 0.1),
            pos_hint = {"x": 0.05, "top": 0.34},
            color = LABEL_BG,
            text_color =LABEL_TEXT,
            markup=True
        )
        self.window.add_widget(self.genderLabel)

        self.genderInput = Spinner(
            text="Select a Gender",
            values=("Male", "Female"),
            size_hint=(0.44, 0.1),
            pos_hint={"x": 0.51, "top": 0.34},
            font_size=50,
            background_color = SPINNER_BG,
            color = SPINNER_TEXT,
            option_cls=CustomSpinnerOption
        )
        self.window.add_widget(self.genderInput)

        self.errorMessage = ColoredLabel(
            text = "",
            font_size = 50,
            size_hint = (0.8, 0.06),
            pos_hint = {"x": 0.1, "top": 0.22},
            color=(1, 1, 1, 1),
            text_color=(0.718, 0.11, 0.11, 1),
            markup=True,
        )
        self.window.add_widget(self.errorMessage)

        self.nextPage = Button(
            text = "[b]Next Page[/b]",
            font_size = 50,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.14},
            on_press = self.next,
            markup=True,
        )
        self.window.add_widget(self.nextPage)

        ###

        self.add_widget(self.window)

    def _update_text_padding1(self, instance, value):
        instance.padding_y = [(instance.height - instance.line_height) / 2, 0]

    def _update_text_padding2(self, instance, value):
        instance.padding_y = [(instance.height - instance.line_height) / 2, 0]

    def _update_text_padding3(self, instance, value):
        instance.padding_y = [(instance.height - instance.line_height) / 2, 0]

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self, instance):
        global info

        weight_input = self.weightInput.text
        height_input = self.heightInput.text
        age_input = self.AgeInput.text
        gender_input = self.genderInput.text
        if(weight_input == "" or height_input == "" or age_input == "" or gender_input == "Select a Gender"):
            self.errorMessage.text = "[b]Please fill in all fields[/b]"
        elif(int(weight_input) < 40):
            self.errorMessage.text = "[b]Weight must be greater than 40 kg[/b]"
        elif(int(height_input) < 140 or int(height_input) > 250):
            self.errorMessage.text = "[b]Height must be between 140 and 250 cm[/b]"
        elif(int(age_input) < 18 or int(age_input) > 100):
            self.errorMessage.text = "[b]Age must be between 18 and 100 years[/b]"
        else:
            info.weight = weight_input
            info.height = height_input
            info.age = age_input
            info.gender = gender_input

            self.manager.current = "registration2"
            self.errorMessage.text = ""

    def _update_border(self, instance, value):
        self.border.rectangle = (instance.x, instance.y, instance.width, instance.height)

    def _update_border2(self, instance, value):
        self.border2.rectangle = (instance.x, instance.y, instance.width, instance.height)

    def _update_border3(self, instance, value):
        self.border3.rectangle = (instance.x, instance.y, instance.width, instance.height)

################################

class Registration2Window(Screen):
    def __init__(self, **kw):
        super(Registration2Window, self).__init__(**kw)
        Window.bind(on_keyboard=self.on_keyboard)
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

        self.logo = Image(
            source = "logo.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.45, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.title = ColoredLabel(
            text = "[b]Registration[/b]",
            font_size = 150,
            size_hint = (0.775, 0.2),
            pos_hint = {"x": 0.1125, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup=True,
        )
        self.window.add_widget(self.title)

        self.activityLabel = ColoredLabel(
            text = "[b]Level of Activity:[/b]",
            font_size = 60,
            size_hint = (0.44, 0.1),
            pos_hint = {"x": 0.05, "top": 0.7},
            color = LABEL_BG,
            text_color = LABEL_TEXT,
            markup=True
        )
        self.window.add_widget(self.activityLabel)

        self.activityInput = Spinner(
            text="Level of Activity",
            values=("Sedentary",
                    "Lightly active",
                    "Moderately active",
                    "Active",
                    "Extremely active"),
            size_hint=(0.44, 0.1),
            pos_hint={"x": 0.51, "top": 0.7},
            font_size = 40,
            background_color = SPINNER_BG,
            color = SPINNER_TEXT,
            option_cls=CustomSpinnerOption
        )
        self.window.add_widget(self.activityInput)

        self.activityTypeLabel = ColoredLabel(
            text = "[b]Types of Activity:[/b]",
            font_size = 60,
            size_hint = (0.6, 0.1),
            pos_hint = {"x": 0.2, "top": 0.58},
            color = LABEL_BG,
            text_color = LABEL_TEXT,
            markup=True
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
            color= LABEL_BG
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
            color= LABEL_BG
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
            color= LABEL_BG
        )
        self.window.add_widget(self.muscleInput)

        self.errorMessage = ColoredLabel(
            text = "",
            font_size = 50,
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.255},
            color=(1, 1, 1, 1),
            text_color=(0.718, 0.11, 0.11, 1),
            markup=True,
        )
        self.window.add_widget(self.errorMessage)

        self.nextPage = Button(
            text = "[b]Next Page[/b]",
            font_size = 50,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.14},
            on_press = self.next,
            markup=True,
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
        if(activity == "Level of Activity"):
            self.errorMessage.text = "[b]Please select a level of activity[/b]"
        else:
            global info
            info.activity = activity
            info.cardio = "1" if cardio else "0"
            info.strength = "1" if strength else "0"
            info.muscle = "1" if muscle else "0"

            self.manager.current = "registration3"
            self.errorMessage.text = ""

    def previous(self, instance):
        self.manager.current = "registration1"
        self.errorMessage.text = ""

    def on_enter(self):
        Window.bind(on_keyboard=self.on_keyboard)
    
    def on_leave(self):
        Window.unbind(on_keyboard=self.on_keyboard)
    
    def on_keyboard(self, window, key, *args):
        if key == 27:
            if self.manager.current == "registration2":
                self.previous(self)
                return True
        return False

################################

class Registration3Window(Screen):
    def __init__(self, **kw):
        super(Registration3Window, self).__init__(**kw)
        Window.bind(on_keyboard=self.on_keyboard)
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

        self.logo = Image(
            source = "logo.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.45, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.title = ColoredLabel(
            text = "[b]Registration[/b]",
            font_size = 150,
            size_hint = (0.775, 0.2),
            pos_hint = {"x": 0.1125, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup=True,
        )
        self.window.add_widget(self.title)

        self.dietLabel = ColoredLabel(
            text = "[b]Diet:[/b]",
            font_size = 60,
            size_hint = (0.9, 0.1),
            pos_hint = {"x": 0.05, "top": 0.6},
            color = LABEL_BG,
            text_color = LABEL_TEXT,
            markup=True
        )
        self.window.add_widget(self.dietLabel)

        self.dietInput = Spinner(
            text="Diet",
            values=("Vegetarian", "Vegan", "Regular"),
            size_hint=(0.9, 0.1),
            pos_hint={"x": 0.05, "top": 0.46},
            font_size = 40,
            background_color = SPINNER_BG,
            color = SPINNER_TEXT,
            option_cls=CustomSpinnerOption
        )
        self.window.add_widget(self.dietInput)

        self.errorMessage = ColoredLabel(
            text = "",
            font_size = 50,
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.3},
            color=(1, 1, 1, 1),
            text_color=(0.718, 0.11, 0.11, 1),
            markup=True,
        )
        self.window.add_widget(self.errorMessage)

        self.nextPage = Button(
            text = "[b]Next Page[/b]",
            font_size = 50,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.14},
            on_press = self.next,
            markup=True,
        )
        self.window.add_widget(self.nextPage)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self, instance):
        diet_input = self.dietInput.text
        if(diet_input == "Diet"):
            self.errorMessage.text = "[b]Please select a diet[/b]"
        else:
            global info
            if(diet_input == "Vegetarian"):
                info.vegetarian = "1"
                info.vegan = "0"
            elif(diet_input == "Vegan"):
                info.vegetarian = "1"
                info.vegan = "1"
            else:
                info.vegetarian = "0"
                info.vegan = "0"

            self.manager.current = "registration4"
            self.errorMessage.text = ""

    def previous(self, instance):
        self.manager.current = "registration2"
        self.errorMessage.text = ""

    def on_enter(self):
        Window.bind(on_keyboard=self.on_keyboard)

    def on_leave(self):
        Window.unbind(on_keyboard=self.on_keyboard)

    def on_keyboard(self, window, key, *args):
        if key == 27:
            if self.manager.current == "registration3":
                self.previous(self)
                return True
        return False

################################

class Registration4Window(Screen):
    def __init__(self, **kw):
        super(Registration4Window, self).__init__(**kw)
        Window.bind(on_keyboard=self.on_keyboard)
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

        self.logo = Image(
            source = "logo.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.45, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.title = ColoredLabel(
            text = "[b]Registration[/b]",
            font_size = 150,
            size_hint = (0.775, 0.2),
            pos_hint = {"x": 0.1125, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup=True,
        )
        self.window.add_widget(self.title)

        self.title2 = ColoredLabel(
            text = "[b]Target weight[/b]",
            font_size = 80,
            size_hint = (0.6, 0.1),
            pos_hint = {"x": 0.2, "top": 0.7},
            color = LABEL_BG,
            text_color = LABEL_TEXT,
            markup=True
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
            text = "[b]Weight:[/b]",
            font_size = 60,
            size_hint = (0.44, 0.1),
            pos_hint = {"x": 0.05, "top": 0.44},
            color = LABEL_BG,
            text_color = LABEL_TEXT,
            markup=True
        )
        self.window.add_widget(self.goalweightLabel)

        self.goalweightInput = TextInput(
            multiline = False,
            font_size = 70,
            hint_text = "Kg",
            size_hint=(0.44, 0.1),
            pos_hint={"x": 0.51, "top": 0.44},
            input_filter="int",
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
            halign="center"
        )
        self.goalweightInput.bind(size=self._update_text_padding)
        self.window.add_widget(self.goalweightInput)
        with self.goalweightInput.canvas.before:
            Color(0, 0, 0, 1)  
            self.border = Line(rectangle=(self.goalweightInput.x, self.goalweightInput.y, self.goalweightInput.width, self.goalweightInput.height), width=1.0)
        self.goalweightInput.bind(size=self._update_border, pos=self._update_border)

        self.errorMessage = ColoredLabel(
            text = "",
            font_size = 50,
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.29},
            color=(1, 1, 1, 1),
            text_color=(0.718, 0.11, 0.11, 1),
            markup=True
        )
        self.window.add_widget(self.errorMessage)

        self.nextPage = Button(
            text = "[b]Next Page[/b]",
            font_size = 50,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.14},
            on_press = self.next,
            markup=True,
        )
        self.window.add_widget(self.nextPage)

        ###

        self.add_widget(self.window)

    def _update_text_padding(self, instance, value):
        instance.padding_y = [(instance.height - instance.line_height) / 2, 0]

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self, instance):
        if(self.goalweightInput.text == ""):
            self.errorMessage.text = "[b]Please fill in the field[/b]"
        elif(int(self.goalweightInput.text) < 40):
            self.errorMessage.text = "[b]Weight must be greater than 40 kg[/b]"
        else:
            self.errorMessage.text = ""
            global info
            info.goal_weight = self.goalweightInput.text
            if(info.goal_weight > info.weight):
                info.goal = "Gain Weight"
            elif(info.goal_weight < info.weight):
                info.goal = "Lose Weight"
            else:
                info.goal = "Maintain Weight"
            t = time_of_change(int(info.weight), int(info.goal_weight))
            if(t == 0):
                info.goal_time = 0
                self.manager.current = "loading"
            else:   
                self.manager.current = "registration5"

    def previous(self, instance):
        self.manager.current = "registration3"
        self.errorMessage.text = ""

    def on_enter(self):
        Window.bind(on_keyboard=self.on_keyboard)
        global info
        idealBodyWeight = ideal_body_weight(int(info.height), info.gender)
        self.suggestedWeight.text = "Suggested weight: " + str(idealBodyWeight) + " kg"

    def on_leave(self):
        Window.unbind(on_keyboard=self.on_keyboard)

    def on_keyboard(self, window, key, *args):
        if key == 27:
            if self.manager.current == "registration4":
                self.previous(self)
                return True
        return False

    def _update_border(self, instance, value):
        self.border.rectangle = (instance.x, instance.y, instance.width, instance.height)

################################

class Registration5Window(Screen):
    def __init__(self, **kw):
        super(Registration5Window, self).__init__(**kw)
        Window.bind(on_keyboard=self.on_keyboard)
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

        self.logo = Image(
            source = "logo.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.45, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.title = ColoredLabel(
            text = "[b]Registration[/b]",
            font_size = 150,
            size_hint = (0.775, 0.2),
            pos_hint = {"x": 0.1125, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            markup=True,
        )
        self.window.add_widget(self.title)

        self.title2 = ColoredLabel(
            text = "[b]Time of the process[/b]",
            font_size = 80,
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.7},
            color = LABEL_BG,
            text_color = LABEL_TEXT,
            markup=True
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
            color = LABEL_BG,
            text_color = LABEL_TEXT,
            markup=True
        )
        self.window.add_widget(self.timeLabel)

        self.timeInput = TextInput(
            multiline = False,
            font_size = 70,
            hint_text = "Weeks",
            size_hint=(0.44, 0.1),
            pos_hint={"x": 0.51, "top": 0.44},
            input_filter="int",
            background_normal="",
            background_color=(0.95, 0.95, 0.95, 1),
            halign="center",
        )
        self.timeInput.bind(size=self._update_text_padding)
        self.window.add_widget(self.timeInput)
        with self.timeInput.canvas.before:
            Color(0, 0, 0, 1)
            self.border = Line(rectangle=(self.timeInput.x, self.timeInput.y, self.timeInput.width, self.timeInput.height), width=1.0)
        self.timeInput.bind(size=self._update_border, pos=self._update_border)

        self.errorMessage = ColoredLabel(
            text = "",
            font_size = 50,
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.29},
            color=(1, 1, 1, 1),
            text_color=(0.718, 0.11, 0.11, 1),
            markup=True
        )
        self.window.add_widget(self.errorMessage)

        self.nextPage = Button(
            text = "Finish",
            font_size = 50,
            background_color = BUTTON_BG,
            color = BUTTON_TEXT,
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.14},
            on_press = self.next,
            markup=True,
        )
        self.window.add_widget(self.nextPage)

        ###

        self.add_widget(self.window)

    def _update_text_padding(self, instance, value):
        instance.padding_y = [(instance.height - instance.line_height) / 2, 0]

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self, instance):
        if(self.timeInput.text == ""):
            self.errorMessage.text = "[b]Please fill in the field[/b]"
        elif(int(self.timeInput.text) <= 0):
            self.errorMessage.text = "[b]Time must be greater than 0 weeks[/b]"
        else:
            self.errorMessage.text = ""
            global info
            info.goal_time = self.timeInput.text
            self.manager.current = "loading"

    def previous(self, instance):
        self.manager.current = "registration4"
        self.errorMessage.text = ""

    def on_enter(self):
        Window.bind(on_keyboard=self.on_keyboard)
        global info
        self.time = time_of_change(int(info.weight), int(info.goal_weight))
        self.suggestedTime.text = "Suggested time: " + str(self.time) + " weeks (" + f'{(self.time * 7 /30):.1f}' + " months)"

    def on_leave(self):
        Window.unbind(on_keyboard=self.on_keyboard)

    def on_keyboard(self, window, key, *args):
        if key == 27:
            if self.manager.current == "registration5":
                self.previous(self)
                return True
        return False 

    def _update_border(self, instance, value):
        self.border.rectangle = (instance.x, instance.y, instance.width, instance.height)

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

        self.logo = Image(
            source = "logo.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.45, "top": 1}
        )
        self.window.add_widget(self.logo)

        self.loading = ColoredLabel(
            text = "Loading...",
            font_size = 150,
            size_hint = (0.6, 0.6),
            pos_hint = {"x": 0.2, "top": 0.8},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.loading)

        self.toastLabel = RoundedStencilView(
            size_hint=(0.6, 0.06),
            pos_hint={"x": 0.2, "top": 1},
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.toastLabel)
        self.toast = ColoredLabel(
            text="",
            font_size=40,
            size_hint=(0.6, 0.06),
            pos_hint={"x": 0.2, "top": 1},
            color=(1, 1, 1, 0),
            text_color=(1, 1, 1, 1),
            halign="center",
            opacity=0,
            disabled=True
        )
        self.window.add_widget(self.toast)

        ###

        self.add_widget(self.window)

    def hide_toast(self):
        self.toastLabel.pos_hint = {"x": 0.2, "top": 1}
        self.toast.pos_hint = {"x": 0.2, "top": 1}
        self.toastLabel.opacity = 0
        self.toastLabel.disabled = True
        self.toast.opacity = 0
        self.toast.disabled = True

    def show_toast(self, message):
        self.toastLabel.pos_hint = {"x": 0.2, "top": 0.2}
        self.toast.pos_hint = {"x": 0.2, "top": 0.2}
        self.toastLabel.opacity = 1
        self.toastLabel.disabled = False
        self.toast.opacity = 1
        self.toast.disabled = False
        self.toast.text = message

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self):
        if(menu_request_window == "main"):
            self.resetData()
            self.manager.current = "main"
        else:
            self.manager.current = "menu"

    def on_enter(self):
        global info
        if(menu_request_window == "main"):
            current_weight_temp = int(info.weight)
            goal_weight_temp = int(info.goal_weight)
            goal_time_temp = int(info.goal_time)
            height_temp = int(info.height)
            age_temp = int(info.age)
            gender_temp = info.gender
            goal_temp = info.goal
            cardio_temp = info.cardio
            strength_temp = info.strength
            muscle_temp = info.muscle
            activity_temp = info.activity
            vegetarian_temp = info.vegetarian
            vegan_temp = info.vegan
        else:
            global current_username
            user_data = users_data[current_username]

            current_weight_temp = int(user_data["weight"])
            goal_weight_temp = int(user_data["goal weight"])
            goal_time_temp = int(user_data["goal time"])
            height_temp = int(user_data["height"])
            age_temp = int(user_data["age"])
            gender_temp = user_data["gender"]
            goal_temp = user_data["goal"]
            cardio_temp = user_data["cardio"]
            strength_temp = user_data["strength"]
            muscle_temp = user_data["muscle"]
            activity_temp = user_data["activity"]
            vegetarian_temp = user_data["vegetarian"]
            vegan_temp = user_data["vegan"]

        self.vector = get_vector(current_weight_temp, goal_weight_temp, goal_time_temp, height_temp, age_temp,
                                            gender_temp, goal_temp, cardio_temp, strength_temp, muscle_temp, activity_temp,
                                            vegetarian_temp, vegan_temp)

        self.build_menu()

    def build_menu(self):
        try:
            server_url = "https://cs-project-m5hy.onrender.com/"

            requests.get(server_url + "wakeup")
            response = requests.post(server_url + "predict", json=self.vector)

            if response.status_code == 200:
                result = response.json()
                result = convert_to_dict(result)

                global current_username
                if current_username not in users_data:
                    users_data[current_username] = {}
                users_data[current_username]["menu"] = result
                with open(USERS_DATA_PATH, "w") as file:
                    json.dump(users_data, file)
                self.next()
            else:
                print("Error: " + str(response.status_code))

        except Exception as e:
            self.show_toast("No internet connection. Retrying...")
            Clock.schedule_once(lambda _: self.hide_toast(), 3)
            Clock.schedule_once(lambda dt: self.build_menu(), 4)

    def resetData(self):
        global info
        global current_username
        users_data[current_username]["password"] = info.password
        users_data[current_username]["weight"] = info.weight
        users_data[current_username]["height"] = info.height
        users_data[current_username]["age"] = info.age
        users_data[current_username]["gender"] = info.gender
        users_data[current_username]["activity"] = info.activity
        users_data[current_username]["cardio"] = info.cardio
        users_data[current_username]["strength"] = info.strength
        users_data[current_username]["muscle"] = info.muscle
        users_data[current_username]["goal"] = info.goal
        users_data[current_username]["vegetarian"] = info.vegetarian
        users_data[current_username]["vegan"] = info.vegan
        users_data[current_username]["goal weight"] = info.goal_weight
        users_data[current_username]["goal time"] = info.goal_time
        users_data[current_username]["calories"] = self.vector["calories"]
        users_data[current_username]["carbohydrate"] = self.vector["carbohydrates"]
        users_data[current_username]["sugar"] = self.vector["sugar"]
        users_data[current_username]["fat"] = self.vector["fat"]
        users_data[current_username]["protein"] = self.vector["protein"]
        users_data[current_username]["calories today"] = 0
        users_data[current_username]["carbohydrates today"] = 0
        users_data[current_username]["sugar today"] = 0
        users_data[current_username]["fat today"] = 0
        users_data[current_username]["protein today"] = 0
        users_data[current_username]["self_menu"] = {
            "sunday": {
                "breakfast": {},
                "lunch": {},
                "dinner": {}
            },
            "monday": {
                "breakfast": {},
                "lunch": {},
                "dinner": {}
            },
            "tuesday": {
                "breakfast": {},
                "lunch": {},
                "dinner": {}
            },
            "wednesday": {
                "breakfast": {},
                "lunch": {},
                "dinner": {}
            },
            "thursday": {
                "breakfast": {},
                "lunch": {},
                "dinner": {}
            },
            "friday": {
                "breakfast": {},
                "lunch": {},
                "dinner": {}
            },
            "saturday": {
                "breakfast": {},
                "lunch": {},
                "dinner": {}
            }
        }
        users_data[current_username]["history_weight"] = []
        users_data[current_username]["history_bmi"] = []
        users_data[current_username]["history_times"] = []
        users_data[current_username]["last_visit_time"] = datetime.now().isoformat(timespec='minutes')

        today = datetime.now().date().isoformat()
        bmi_temp = bmi(int(info.weight), int(info.height))

        if users_data[current_username]["history_times"] and today in users_data[current_username]["history_times"]:
            users_data[current_username]["history_weight"] = users_data[current_username]["history_weight"][:-1] + [float(int(info.weight))]
            users_data[current_username]["history_bmi"] = users_data[current_username]["history_bmi"][:-1] + [bmi_temp] 
        else:
            users_data[current_username]["history_weight"] = users_data[current_username]["history_weight"] + [float(int(info.weight))]
            users_data[current_username]["history_bmi"] = users_data[current_username]["history_bmi"] + [bmi_temp]
            users_data[current_username]["history_times"] = users_data[current_username]["history_times"] + [today] 

        with open(USERS_DATA_PATH, "w") as file:
            json.dump(users_data, file)

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
        self.add_widget(FoodTrackerWindow(name = "foodTracker"))
        self.add_widget(DictionaryWindow(name = "dictionary"))
        self.add_widget(CreateAccountWindow(name = "createAccount"))
        self.add_widget(Registration1Window(name = "registration1"))
        self.add_widget(Registration2Window(name = "registration2"))
        self.add_widget(Registration3Window(name = "registration3"))
        self.add_widget(Registration4Window(name = "registration4"))
        self.add_widget(Registration5Window(name = "registration5"))

class MainApp(App):
    def build(self):
        self.end_of_week()
        self.reset_day()
        return WindowManager()
    
    def end_of_week(self):
        for username in users_data:
            if users_data[username]["last_visit_time"] != "":
                current_time = datetime.fromisoformat(datetime.now().isoformat(timespec='minutes'))
                last_visit_time = datetime.fromisoformat(users_data[username]["last_visit_time"])
                while last_visit_time <= current_time:
                    if last_visit_time.weekday() == 5 and last_visit_time.hour == 23 and last_visit_time.minute == 59:
                        clean_self_menu(username)
                        break
                    last_visit_time += timedelta(minutes=1)

    def reset_day(self):
        entry_time = datetime.now().date().isoformat()

        for username in users_data:
            last_visit_time = users_data[username]["last_visit_time"].split("T")[0]
            
            if last_visit_time != entry_time:
                users_data[username]["calories today"] = 0
                users_data[username]["carbohydrates today"] = 0
                users_data[username]["sugar today"] = 0
                users_data[username]["fat today"] = 0
                users_data[username]["protein today"] = 0
                with open(USERS_DATA_PATH, "w") as file:
                    json.dump(users_data, file)

            users_data[username]["last_visit_time"] = datetime.now().isoformat(timespec='minutes')
            with open(USERS_DATA_PATH, "w") as file:
                    json.dump(users_data, file)

if __name__ == "__main__":
    MainApp().run()