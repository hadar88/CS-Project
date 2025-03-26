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
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion_food_id = nn.CrossEntropyLoss()
        criterion_amount = nn.MSELoss()
        other_criterions = []

        train_transformer_model(dataloader, model, criterion_food_id, criterion_amount, other_criterions, optimizer, 2, device, False)

        other_criterions.append(XORLoss())

        train_transformer_model(dataloader, model, criterion_food_id, criterion_amount, other_criterions, optimizer, 2, device, False)

        other_criterions.append(NutritionLoss(device))

        train_transformer_model(dataloader, model, criterion_food_id, criterion_amount, other_criterions, optimizer, 2, device, True)

        torch.save(model.state_dict(), f"saved_models/model_v{MODEL_VERSION}.pth")
        print(f"Model saved as saved_models/model_v{MODEL_VERSION}.pth")

        evaluate_transformer_on_random_sample(dataloader, model, device)

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
        amount = self.activation(amount)

        return food_logits, amount
    
class XORLoss(nn.Module):
    def __init__(self):
        super(XORLoss, self).__init__()
        self.ZERO_NONZERO_PENALTY = 25
        self.l1loss = nn.L1Loss()

    def forward(self, pred_ids, pred_amounts, ids, amounts):
        ### Penalize the model for giving rows with no id but with an amount or vice versa ###
        
        zero_id = zero_mask(pred_ids)
        zero_amount = zero_mask(pred_amounts)

        return self.l1loss(zero_id, zero_amount) * self.ZERO_NONZERO_PENALTY

class NutritionLoss(nn.Module):
    def __init__(self, device):
        super(NutritionLoss, self).__init__()
        self.NUTRITION_PENALTY = 1
        self.DENOMINATOR = 1
        self.l1loss = nn.L1Loss()
        self.device = device
        self.data = read_foods_tensor().to(device)

    def forward(self, pred_ids, pred_amounts, ids, amounts):
        nutrition_diff = 0.0    # Calories, Carbs, Sugars, Fat, Protein

        pred_ids = pred_ids.view(-1, 7, 3, 10)
        pred_amounts = pred_amounts.view(-1, 7, 3, 10)
        ids = ids.view(-1, 7, 3, 10)
        amounts = amounts.view(-1, 7, 3, 10)

        for fp in [FP.CALORIES, FP.CARBOHYDRATE, FP.SUGARS, FP.FAT, FP.PROTEIN]:
            gold = (get_continuous_value(ids, self.data, fp) * amounts / 100).sum(dim=(1,2,3)) / 7
            pred = (get_continuous_value(round_and_bound(pred_ids), self.data, fp) * pred_amounts / 100).sum(dim=(1,2,3)) / 7
            nutrition_diff += self.l1loss(pred, gold) / self.DENOMINATOR

        return self.NUTRITION_PENALTY * nutrition_diff

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

def train_transformer_model(dataloader, model, criterion_food_id, criterion_amount, other_criterions, optimizer, epochs, device, plot_loss=True):
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

            pred_ids = torch.argmax(food_logits, dim=-1)
            pred_amounts = pred_amounts.view(-1)

            ids = ids.view(-1)
            amounts = amounts.view(-1)

            loss_id = criterion_food_id(food_logits, ids)
            loss_amount = criterion_amount(pred_amounts, amounts)

            # joint loss weighted importance
            loss = loss_id + loss_amount

            for criterion in other_criterions:
                loss += criterion(pred_ids, pred_amounts, ids, amounts)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        bar.set_postfix_str(f"Loss = {epoch_loss:.4f}")
        loss_history.append(epoch_loss)

    if plot_loss:
        plt.plot(loss_history)
        plt.savefig("loss_plot.png")
        plt.show()

def evaluate_transformer_on_random_sample(dataloader, model, device):
    model.eval()
    model.to(device)

    print("Here is a random prediction:")

    print("Reading the foods data...\n")
    FOODS_DATA_PATH = "../../Data/layouts/FoodsByID.json"
    foods = open(FOODS_DATA_PATH, "r")
    data = json.load(foods)

    random_index = torch.randint(0, len(dataloader.dataset), (1,)).item()
    x, y_id, y_amount = dataloader.dataset[random_index]
    x, y_id, y_amount = x.to(device), y_id.to(device), y_amount.to(device)

    pred_id, pred_amount = model(x.unsqueeze(0).to(device))

    pred_id, pred_amount = pred_id[0], pred_amount[0]

    pred_id = torch.argmax(pred_id, dim=-1)

    pred_amount = pred_amount.squeeze(-1)

    print("The model predicted:")
    merged_pred = MenusDataset.merge_ids_and_amounts(pred_id, pred_amount)
    print(merged_pred)
    print()

    # print("The ground truth was:")
    merged_y = MenusDataset.merge_ids_and_amounts(y_id, y_amount)
    # print(merged_y)
    # print()

    print("Here's a comparison between the ground truth and the model's prediction:")
    print("Model's prediction:")
    print(transform2(merged_pred, data, device))
    print("Ground truth:")
    print(transform2(merged_y, data, device))

if __name__ == "__main__":
    main()
