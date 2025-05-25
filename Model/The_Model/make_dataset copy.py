import json
import torch
import menu_output_transform as mot
from torch.utils.data import Dataset
from enum import Enum

MENUS_INPUT = "../../Data/layouts/MenusInput.json"
MENUS_BY_ID = "../../Data/layouts/MenusById.json"

FOODS_DATA_PATH = "../../Data/layouts/FoodsByID.json"

class FoodProperties(Enum):
    CALORIES = 0
    CARBOHYDRATE = 1
    SUGARS = 2
    FAT = 3
    PROTEIN = 4
    VEGETARIAN = 5
    VEGAN = 6
    CONTAINS_EGGS = 7
    CONTAINS_MILK = 8
    CONTAINS_PEANUTS_OR_NUTS = 9
    CONTAINS_FISH = 10
    CONTAINS_SESAME = 11
    CONTAINS_SOY = 12
    CONTAINS_GLUTEN = 13
    FRUIT = 14
    VEGETABLE = 15
    CHEESE = 16
    MEAT = 17
    CEREAL = 18

def read_foods_tensor():
    foods = open(FOODS_DATA_PATH, "r")
    data = json.load(foods)
    foods.close()

    data_tensor = torch.zeros(len(data) + 1, len(data["1"]) - 1, dtype=torch.float32)

    for food_id in data:
        index = int(food_id)

        if index == 0:
            continue

        data_tensor[index][0] = data[food_id]["Calories"]
        data_tensor[index][1] = data[food_id]["Carbohydrate"]
        data_tensor[index][2] = data[food_id]["Sugars"]
        data_tensor[index][3] = data[food_id]["Fat"]
        data_tensor[index][4] = data[food_id]["Protein"]
        data_tensor[index][5] = data[food_id]["Vegetarian"]
        data_tensor[index][6] = data[food_id]["Vegan"]
        data_tensor[index][7] = data[food_id]["Contains eggs"]
        data_tensor[index][8] = data[food_id]["Contains milk"]
        data_tensor[index][9] = data[food_id]["Contains peanuts or nuts"]
        data_tensor[index][10] = data[food_id]["Contains fish"]
        data_tensor[index][11] = data[food_id]["Contains sesame"]
        data_tensor[index][12] = data[food_id]["Contains soy"]
        data_tensor[index][13] = data[food_id]["Contains gluten"]
        data_tensor[index][14] = data[food_id]["Fruit"]
        data_tensor[index][15] = data[food_id]["Vegetable"]
        data_tensor[index][16] = data[food_id]["Cheese"]
        data_tensor[index][17] = data[food_id]["Meat"]
        data_tensor[index][18] = data[food_id]["Cereal"]

    return data_tensor

def make_xs(split="train"):
    xs = []

    if split not in ["train", "val", "test"]:
        raise ValueError("Invalid split value. Choose from 'train', 'val', or 'test'.")

    with open(MENUS_INPUT, "r") as dataset_file:
        dataset = json.load(dataset_file)

        total_menus = len(dataset)
        train_split = int(total_menus * 0.8)  # 80% for training
        val_split = int(total_menus * 0.9)    # 10% for validation

        for i, menu_id in enumerate(dataset):
            x = [dataset[menu_id]["Initial"][entry] for entry in dataset[menu_id]["Initial"]]

            # if split == "train" and i < train_split:
            #     xs.append(x)
            # elif split == "val" and train_split <= i < val_split:
            #     xs.append(x)
            # elif split == "test" and val_split <= i:
            #     xs.append(x)

            if split == "train":
                xs.append(x)

    return torch.tensor(xs)

def make_ys(split="train"):
    ys = []

    if split not in ["train", "val", "test"]:
        raise ValueError("Invalid split value. Choose from 'train', 'val', or 'test'.")

    with open(MENUS_BY_ID, "r") as dataset_file:
        max_foods_in_meal = 0

        dataset = json.load(dataset_file)

        total_menus = len(dataset)
        train_split = int(total_menus * 0.8)  # 80% for training
        val_split = int(total_menus * 0.9)    # 10% for validation

        max_foods_in_meal = 10

        # Check what is the maximum number of foods in a meal
        for i, menu_id in enumerate(dataset):
            y = dataset[menu_id]
            y = mot.menu_dict_to_tensor(y)

            # if split == "train" and i < train_split:
            #     ys.append(y)
            # elif split == "val" and train_split <= i < val_split:
            #     ys.append(y)
            # elif split == "test" and val_split <= i:
            #     ys.append(y)

            if split == "train":
                ys.append(y)

        for i in range(len(ys)):
            y = torch.zeros(7, 3, max_foods_in_meal, 2)
            y[:, :, :ys[i].shape[2], :] = ys[i]
            ys[i] = y
        
    return torch.stack(ys)
    
# The DataSet

class MenusDataset(Dataset):
    def __init__(self, split="train"):
        self.xs = make_xs(split)
        self.ys = make_ys(split)

    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, index):
        # return self.xs[index], self.mids[index], self.ys[index]
        foods_id = self.ys[index][..., 0].long()
        foods_amount = self.ys[index][..., 1].float()
        return self.xs[index], foods_id, foods_amount
    
    @staticmethod
    def merge_ids_and_amounts(ids, amounts):
        return torch.stack((ids, amounts), dim=-1)

if __name__ == "__main__":
    xs = make_xs("test")
    ys = make_ys("test")

    print(xs.shape)
    print(ys.shape)
