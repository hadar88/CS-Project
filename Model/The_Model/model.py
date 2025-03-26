import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from make_dataset import MenusDataset, read_foods_tensor, FoodProperties as FP
from menu_output_transform import transform2
import argparse

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

SPLIT = ["train", "val", "test"][0]

MODEL_VERSION = 1.0
BATCH_SIZE = 256

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, help="The split to use (train, val, test)", choices=["train", "val", "test"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Load the dataset ###

    split = SPLIT if args.split is None else args.split

    print(f"Loading {split} set...")
    menus = MenusDataset(split=SPLIT)
    # menus = Subset(menus, range(10))
    dataloader = DataLoader(menus, batch_size=BATCH_SIZE, shuffle=(SPLIT == "train"))
    
    model = MenuGenerator()

    if split == "train":
        ### Define the model, loss function and optimizer ###

        # myLoss = MenuLoss(device).to(device)                                       
        # myLoss = ZeroLoss()                                       
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion_food_id = nn.CrossEntropyLoss()
        criterion_amount = nn.MSELoss()

        train_transformer_model(dataloader, model, criterion_food_id, criterion_amount, optimizer, 20, device, True)

        # criterions_and_epochs = [
        #     (nn.MSELoss(), 5000),
        #     (AllergensLoss(device), 50),
        #     #(PreferenceLoss(device), 50),
        #     #(IngredientsLoss(device), 10)
        # ]

        # used_loss_functions = []

        # for i, (criterion, epochs) in enumerate(criterions_and_epochs):
        #     print(f"Training the model with {criterion.__class__.__name__}")
        #     used_loss_functions.append(criterion)
        #     train_model(dataloader, model, used_loss_functions, optimizer, epochs, device, True)
        #     torch.save(model.state_dict(), f"saved_models/model_v{MODEL_VERSION}_{criterion.__class__.__name__[0]}.pth")

        torch.save(model.state_dict(), f"saved_models/model_v{MODEL_VERSION}.pth")
        print(f"Model saved as saved_models/model_v{MODEL_VERSION}.pth")

        evaluate_transformer_on_random_sample(dataloader, model, device)

    elif split == "val" or split == "test":
        ### Load the model and evaluate it ###

        model.load_state_dict(torch.load(f"saved_models/model_v{MODEL_VERSION}.pth", weights_only=True))
        model.eval()

        evaluate_on_random_sample(dataloader, model, device)

    # loss = evaluate_model(dataloader, model, [nn.MSELoss(), AllergensLoss(device)], device)
    # print(f"Loss on the {split} set: {loss:.4f}")

class MenuGenerator(nn.Module):
    def __init__(self):
        super(MenuGenerator, self).__init__()

        self.emb_dim = 16

        self.fc1 = nn.Linear(14, 128)
        self.fc2 = nn.Linear(128, 256)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
            num_layers=2
        )

        self.food_fc = nn.Linear(256, 7 * 3 * 10 * 223)
        self.amount_fc = nn.Linear(256, 7 * 3 * 10)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        x = x.unsqueeze(0) 
        x = self.transformer(x)
        x = x.squeeze(0)

        food_logits = self.food_fc(x)
        food_logits = food_logits.view(-1, 7, 3, 10, 223)

        amount = self.amount_fc(x)
        amount = amount.view(-1, 7, 3, 10, 1)

        return food_logits, amount
    
class ZeroLoss(nn.Module):
    def __init__(self):
        super(ZeroLoss, self).__init__()
        self.ZERO_NONZERO_PENALTY = 3
        self.l1loss = nn.L1Loss()

    def forward(self, y_pred, y):
        pred_ids = y_pred[..., 0]       # Shape: (batch_size, 7, 3, M)
        pred_amounts = y_pred[..., 1]

        ### Penalize the model for giving rows with no id but with an amount or vice versa ###
        
        zero_id = zero_mask(pred_ids)
        zero_amount = zero_mask(pred_amounts)

        return self.l1loss(zero_id, zero_amount) * self.ZERO_NONZERO_PENALTY

class RangeLoss(nn.Module):
    def __init__(self):
        super(RangeLoss, self).__init__()
        self.OUT_OF_RANGE_PENALTY = 2

    def forward(self, y_pred, y):
        pred_ids = y_pred[..., 0]

        # id_out_of_range = torch.relu(pred_ids - 222)
        id_out_of_range = (torch.tanh(20 * (pred_ids - 222.5)) + 1) / 2
        # id_out_of_range = 1 - zero_mask(id_out_of_range)
        id_range_penalty =  id_out_of_range.sum(dim=(1, 2, 3)).mean()

        return id_range_penalty * self.OUT_OF_RANGE_PENALTY
    
class NutritionLoss(nn.Module):
    def __init__(self, device):
        super(NutritionLoss, self).__init__()
        self.NUTRITION_PENALTY = 5
        self.DENOMINATOR = 1
        self.l1loss = nn.L1Loss()
        self.device = device
        self.data = read_foods_tensor().to(device)

    def forward(self, y_pred, y):
        pred_ids = y_pred[..., 0]       # Shape: (batch_size, 7, 3, M)
        pred_amounts = y_pred[..., 1]

        true_ids = y[..., 0]
        true_amounts = y[..., 1]

        nutrition_diff = 0.0    # Calories, Carbs, Sugars, Fat, Protein

        for fp in [FP.CALORIES, FP.CARBOHYDRATE, FP.SUGARS, FP.FAT, FP.PROTEIN]:
            gold = (get_continuous_value(true_ids, self.data, fp) * true_amounts / 100).sum(dim=(1, 2, 3)) / 7
            pred = (get_continuous_value(round_and_bound(pred_ids), self.data, fp) * pred_amounts / 100).sum(dim=(1, 2, 3)) / 7
            nutrition_diff += self.l1loss(pred, gold) / self.DENOMINATOR

        return self.NUTRITION_PENALTY * nutrition_diff
    
class PreferenceLoss(nn.Module):
    def __init__(self, device):
        super(PreferenceLoss, self).__init__()

        self.PREF_PENALTY = 7
        self.device = device

        self.data = read_foods_tensor().to(device)

    def forward(self, y_pred, y):
        pred_ids = y_pred[..., 0]       # Shape: (batch_size, 7, 3, M)

        true_ids = y[..., 0]

        preferences_diff = 0.0

        for fp in [FP.VEGETARIAN, FP.VEGAN]:
            gold = (1 - get_binary_value(true_ids, self.data, fp)).sum(dim=(1, 2, 3))
            pred = (1 - get_binary_value(round_and_bound(pred_ids), self.data, fp)).sum(dim=(1, 2, 3))
            preferences_diff += (torch.exp(-10 * gold) * pred.pow(2)).mean()

        return preferences_diff * self.PREF_PENALTY
    
class AllergensLoss(nn.Module):
    def __init__(self, device):
        super(AllergensLoss, self).__init__()

        self.ALERGENS_PENALTY = 5
        self.device = device

        self.data = read_foods_tensor().to(device)

    def forward(self, y_pred, y):
        pred_ids = y_pred[..., 0]       # Shape: (batch_size, 7, 3, M)

        true_ids = y[..., 0]

        alergens_diff = 0.0     # Contains Eggs, Gluten, Milk, Peanuts, Soy, Fish, Sesame

        for fp in [FP.CONTAINS_EGGS, FP.CONTAINS_GLUTEN, FP.CONTAINS_MILK, FP.CONTAINS_PEANUTS_OR_NUTS, FP.CONTAINS_SOY, FP.CONTAINS_FISH, FP.CONTAINS_SESAME]:
            gold = get_binary_value(true_ids, self.data, fp).sum(dim=(1, 2, 3))
            pred = get_binary_value(round_and_bound(pred_ids), self.data, fp).sum(dim=(1, 2, 3))
            alergens_diff += (torch.exp(-10 * gold) * pred.pow(2)).mean()

        return alergens_diff * self.ALERGENS_PENALTY
    
class IngredientsLoss(nn.Module):
    def __init__(self, device):
        super(IngredientsLoss, self).__init__()

        self.INGREDIENTS_PENALTY = 3
        self.device = device

        self.data = read_foods_tensor().to(device)
        self.DENOMINATOR = 1

        self.l1loss = nn.L1Loss()
    
    def forward(self, y_pred, y):
        pred_ids = y_pred[..., 0]       # Shape: (batch_size, 7, 3, M)

        true_ids = y[..., 0]

        ingredients_diff = 0.0

        for fp in [FP.FRUIT, FP.VEGETABLE, FP.CHEESE, FP.MEAT, FP.CEREAL]:
            gold = get_binary_value(true_ids, self.data, fp).sum(dim=(1, 2, 3))
            pred = get_binary_value(round_and_bound(pred_ids), self.data, fp).sum(dim=(1, 2, 3))
            ingredients_diff += self.l1loss(pred, gold) / self.DENOMINATOR

        return ingredients_diff * self.INGREDIENTS_PENALTY

class CaloriesMSELoss(nn.Module):
    def __init__(self, device):
        super(CaloriesMSELoss, self).__init__()
        self.MSE_PENALTY = 10
        self.l1loss = nn.L1Loss()
        self.DENOMINATOR = 1

        self.data = read_foods_tensor().to(device)

    def forward(self, y_pred, y):
        pred_ids = y_pred[..., 0]       # Shape: (batch_size, 7, 3, M)
        pred_amounts = y_pred[..., 1]

        true_ids = y[..., 0]
        true_amounts = y[..., 1]

        ### Compute differences in calories in breakfast, lunch and dinner ###

        meals_diff = 0.0

        for i in range(3):
            gold = (get_continuous_value(true_ids[:, :, i], self.data, FP.CALORIES) * true_amounts[:, :, i] / 100).sum(dim=(1, 2)) / 7
            pred = (get_continuous_value(round_and_bound(pred_ids[:, :, i]), self.data, FP.CALORIES) * pred_amounts[:, :, i] / 100).sum(dim=(1, 2)) / 7
            meals_diff += self.l1loss(pred, gold) / self.DENOMINATOR

        ### Compute the MSE ###

        pred_calorie_value = get_continuous_value(round_and_bound(pred_ids), self.data, FP.CALORIES) / 100
        pred_calories_per_day = (pred_amounts * pred_calorie_value).sum(dim=(2, 3))
        pred_mses = ((pred_calories_per_day - pred_calories_per_day.mean(dim=1, keepdim=True)).pow(2)).mean(dim=1)

        return self.MSE_PENALTY * pred_mses.mean() + meals_diff    
    
def entropy_penalty(self, pred_ids):
    # Apply softmax to the predicted logits (IDs)
    softmax_probs = torch.nn.functional.softmax(pred_ids, dim=-1)
    
    # Compute entropy (the higher the entropy, the more uniform the distribution)
    entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-8), dim=-1).mean()
    
    return entropy

def get_binary_value(x, data, category: FP):
    return torch.sum(
        torch.stack([
                v * torch.exp(-((x - i * v).pow(2)) / 0.01)
                for i, v in enumerate(data[:, category.value])],
            dim=0,
        ),
        dim=0,
    )

def get_continuous_value(x, data, category: FP):
    return torch.sum(
        torch.stack(
            [
                v * torch.exp(-((x - i).pow(2)) / 0.01)
                for i, v in enumerate(data[:, category.value])
            ],
            dim=0,
        ),
        dim=0,
    )

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
def round_ste(input):
    return RoundSTE.apply(input)

def bound(x):
    """approx. 0 for any id > 222 and the id itself for any id <= 222."""
    return x * torch.sigmoid(50 * (222.5 - x))

def round_and_bound(x):
    return bound(round_ste(x))
    
def zero_mask(x):
    return torch.exp(-4 * x)

def train_transformer_model(dataloader, model, criterion_food_id, criterion_amount, optimizer, epochs, device, plot_loss=True):
    model.to(device)
    model.train()

    bar = tqdm(range(epochs))

    loss_history = []

    for _ in bar:
        epoch_loss = 0.0

        for x, ids, amounts in dataloader:
            x, ids, amounts = x.to(device), ids.to(device), amounts.to(device)

            optimizer.zero_grad()

            # forward
            food_logits, pred_amounts = model(x)

            # reshape for loss computation
            food_logits = food_logits.view(-1, 223)
            ids = ids.view(-1)

            pred_amounts = pred_amounts.view(-1, 1)
            amounts = amounts.view(-1, 1)

            # compute losses
            loss_id = criterion_food_id(food_logits, ids)
            loss_amount = criterion_amount(pred_amounts, amounts)

            # joint loss weighted importance
            loss = loss_id + loss_amount
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        bar.set_postfix_str(f"Loss = {epoch_loss:.4f}")
        loss_history.append(epoch_loss)

    if plot_loss:
        plt.plot(loss_history)
        plt.savefig("loss_plot.png")
        plt.show()

def train_model(dataloader, model, criterions: list, optimizer, epochs, device, plot_loss=True):
    model.train()

    epochs_bar = tqdm(range(epochs))
    loss_history = []

    for _ in epochs_bar:
        total_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)

            loss = 0

            for criterion in criterions:
                loss += criterion(y_pred, y)
            
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_loss = total_loss / num_batches
        epochs_bar.set_postfix_str(f"Loss = {epoch_loss:.4f}")

        loss_history.append(epoch_loss)

    if plot_loss:
        plt.plot(loss_history)
        plt.savefig("loss_plot.png")
        plt.show()
        
def evaluate_model(dataloader, model, criterions, device):
    """
    Evaluate the model on the given dataset.

    Returns:
        float: the average loss on the dataset.
    """

    model.eval()

    total_loss = 0.0
    num_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        y_pred = model(x)

        loss = 0

        for criterion in criterions:
            loss += criterion(y_pred, y)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

def evaluate_on_random_sample(dataloader, model, device):
    model.eval()

    print("Here is a random prediction:")

    print("Reading the foods data...\n")
    FOODS_DATA_PATH = "../../Data/layouts/FoodsByID.json"
    foods = open(FOODS_DATA_PATH, "r")
    data = json.load(foods)

    random_index = torch.randint(0, len(dataloader.dataset), (1,)).item()
    x, y = dataloader.dataset[random_index]
    y_pred = model(x.unsqueeze(0).to(device))
        
    # print("For the following input:")
    # print(x)
    print()
    print("The model predicted:")
    print(y_pred.squeeze())
    print("The ground truth was:")
    print(y)
    print()

    print("Here's a comparison between the ground truth and the model's prediction:")
    print("Model's prediction:")
    print(transform2(y_pred.squeeze(), data, device, bound))
    print("Ground truth:")
    print(transform2(y, data, device))
    

def evaluate_transformer_on_random_sample(dataloader, model, device):
    model.eval()
    model.to(device)

    print("Here is a random prediction:")

    # print("Reading the foods data...\n")
    # FOODS_DATA_PATH = "../../Data/layouts/FoodsByID.json"
    # foods = open(FOODS_DATA_PATH, "r")
    # data = json.load(foods)

    random_index = torch.randint(0, len(dataloader.dataset), (1,)).item()
    x, y_id, y_amount = dataloader.dataset[random_index]
    x, y_id, y_amount = x.to(device), y_id.to(device), y_amount.to(device)

    pred_id, pred_amount = model(x.unsqueeze(0).to(device))
        
    print("For the following input:")
    print(x)
    print()

    print("The model predicted:")
    print("IDs:")
    print(pred_id.squeeze())
    print("Amounts:")
    print(pred_amount.squeeze())
    print()

    # print("The ground truth was:")
    # print("IDs:")
    # print(y_id.squeeze())
    # print("Amounts:")
    # print(y_amount.squeeze())
    # print()

    # print("Here's a comparison between the ground truth and the model's prediction:")
    # print("Model's prediction:")
    # print(transform2(y_pred.squeeze(), data, device, bound))
    # print("Ground truth:")
    # print(transform2(y, data, device))

if __name__ == "__main__":
    main()
