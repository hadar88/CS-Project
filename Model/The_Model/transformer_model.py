import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from make_dataset import MenusDataset, read_foods_tensor, FoodProperties as FP
from menu_output_transform import transform2, check_menu
import argparse

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

SPLIT = ["train", "val", "test"][1]

MODEL_VERSION = 1.0
BATCH_SIZE = 512

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
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion_food_id = nn.CrossEntropyLoss()
        other_criterions = []

        other_criterions.append(nn.MSELoss())

        train_transformer_model(dataloader, model, criterion_food_id, other_criterions, optimizer, 5000, device, True) 

        # other_criterions.append(XORLoss())

        # train_transformer_model(dataloader, model, criterion_food_id, other_criterions, optimizer, 1000, device, True)

        # other_criterions.append(NutritionLoss(device))

        # train_transformer_model(dataloader, model, criterion_food_id, other_criterions, optimizer, 300, device, True)

        torch.save(model.state_dict(), f"saved_models/model_v{MODEL_VERSION}.pth")
        print(f"Model saved as saved_models/model_v{MODEL_VERSION}.pth")

    model.load_state_dict(torch.load(f"saved_models/model_v{MODEL_VERSION}.pth"))

    evaluate_on_random_sample(dataloader, model, device)

class MenuGenerator(nn.Module):
    def __init__(self, food_vocab_size=223, hidden_dim=256):
        super(MenuGenerator, self).__init__()

        self.input_encoder = nn.Sequential(
            nn.Linear(14, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

        self.slot_proj = nn.Linear(hidden_dim, 210 * hidden_dim)

        self.slot_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.food_id_head = nn.Linear(64, food_vocab_size)
        self.amount_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.size(0)

        latent = self.input_encoder(x)  # (batch, hidden_dim)

        slot_input = self.slot_proj(latent).view(batch_size, 210, -1)  # (batch, 210, hidden_dim)

        decoded = self.slot_decoder(slot_input)  # (batch, 210, 64)

        food_logits = self.food_id_head(decoded)  # (batch, 210, 223)
        amounts = self.amount_head(decoded).squeeze(-1)  # (batch, 210)

        food_logits = food_logits.view(batch_size, 7, 3, 10, 223)
        amounts = amounts.view(batch_size, 7, 3, 10)

        return food_logits, amounts
    
class XORLoss(nn.Module):
    def __init__(self):
        super(XORLoss, self).__init__()
        self.ZERO_NONZERO_PENALTY = 20
        self.l1loss = nn.L1Loss()

    def forward(self, pred_ids, pred_amounts, ids, amounts):
        zero_id = zero_mask(pred_ids)
        zero_amount = zero_mask(pred_amounts)

        return self.l1loss(zero_id, zero_amount) * self.ZERO_NONZERO_PENALTY

class NutritionLoss(nn.Module):
    def __init__(self, device):
        super(NutritionLoss, self).__init__()
        self.NUTRITION_PENALTY = 2
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

def train_transformer_model(dataloader, model, criterion_food_id, other_criterions, optimizer, epochs, device, plot_loss=True):
    model.to(device)
    model.train()

    loss_history = []

    bar = tqdm(range(epochs))

    for _ in bar:
        epoch_loss = 0.0

        for x, ids, amounts in dataloader:
            x, ids, amounts = x.to(device), ids.to(device), amounts.to(device)

            optimizer.zero_grad()

            food_logits, pred_amounts = model(x)

            # Flatten
            pred_ids_flat = food_logits.view(-1, 223)
            pred_amounts_flat = pred_amounts.view(-1)
            ids_flat = ids.view(-1)
            amounts_flat = amounts.view(-1)

            # ID loss with ignore_index for padding
            loss_id = criterion_food_id(pred_ids_flat, ids_flat)

            # Amount loss with mask
            mask = (amounts_flat > 0).float()
            loss_amount = ((pred_amounts_flat - amounts_flat) ** 2 * mask).sum() / (mask.sum() + 1e-8)

            loss = loss_id + loss_amount

            pred_ids = torch.argmax(pred_ids_flat, dim=-1)

            for criterion in other_criterions:
                if isinstance(criterion, nn.MSELoss):
                    # Apply MSELoss only to the predicted amounts and ground truth amounts
                    loss += criterion(pred_amounts_flat, amounts_flat)
                else:
                    # Apply custom loss functions with additional arguments
                    loss += criterion(pred_ids, pred_amounts_flat, ids_flat, amounts_flat)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        bar.set_postfix_str(f"Loss = {epoch_loss:.4f}")
        loss_history.append(epoch_loss)

    if plot_loss:
        loss_history = loss_history[200:]
        plt.plot(loss_history)
        plt.savefig("loss_plot.png")
        # plt.show()

def evaluate_on_random_sample(dataloader, model, device):
    model.eval()
    model.to(device)

    # print("Here is a random prediction:")

    #print("Reading the foods data...\n")
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

    print("\nThe model predicted:")
    merged_pred = MenusDataset.merge_ids_and_amounts(pred_id, pred_amount)
    print(merged_pred)
    check_menu(merged_pred)
    print("\nThe menu is in 'check_menu.json'")

    merged_y = MenusDataset.merge_ids_and_amounts(y_id, y_amount)

    values = ["Calories", "Calories1", "Calories2", "Calories3", "Calories MSE", "Carbohydrate", 
                "Sugars", "Fat", "Protein", "Fruit", "Vegetable", "Cheese", "Meat", "Cereal", 
                "Vegetarian", "Vegan", "Contains eggs", "Contains milk", "Contains peanuts or nuts", 
                "Contains fish", "Contains sesame", "Contains soy", "Contains gluten"]
    print("\nComparison between the ground truth and the model's prediction:")
    menu1 = transform2(merged_y, data, device)
    menu2 = transform2(merged_pred, data, device)
    print("\nGround truth vs Model's prediction")    
    print("" + "-" * 30)
    for i, (m1, m2) in enumerate(zip(menu1, menu2)):
        print(f"{values[i]}: {m1.item():.0f} vs {m2.item():.0f}")

if __name__ == "__main__":
    main()
