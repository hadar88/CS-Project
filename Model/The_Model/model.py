import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from Model.The_Model.make_dataset import MenusDataset, FoodProperties as FP
from Model.The_Model.menu_output_transform import transform2, check_menu
import argparse

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

SPLIT = ["train", "val", "test"][0]

MODEL_VERSION = 19
BATCH_SIZE = 512

# ------ Main --------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, help="The split to use (train, val, test)", choices=["train", "val", "test"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Load the dataset ###

    split = SPLIT if args.split is None else args.split

    print(f"Loading {split} set...")
    menus = MenusDataset(split)
    dataloader = DataLoader(menus, batch_size=BATCH_SIZE, shuffle=(SPLIT == "train"))

    model = MenuGenerator()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    foods_criterions = [nn.CrossEntropyLoss()]
    amounts_criterions = [nn.MSELoss()]
    other_criterions = []

    if split == "train":
        train_model(dataloader, model, foods_criterions, amounts_criterions, other_criterions, optimizer, 10000, device)

        # model.load_state_dict(torch.load(f"model_v{MODEL_VERSION}.pth", weights_only=True))
        # evaluate_on_random_sample(dataloader, model, device)
    elif split == "val" or split == "test":
        # Load the model and evaluate
        model.load_state_dict(torch.load(f"model_v{MODEL_VERSION}.pth", weights_only=True))
        evaluate_on_random_sample(dataloader, model, device)

# ------ Model --------- #

class MenuGenerator(nn.Module):
    def __init__(self, food_vocab_size=223, hidden_dim=256):
        super(MenuGenerator, self).__init__()

        self.input_encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

        self.slot_proj = nn.Linear(hidden_dim, 210 * hidden_dim)

        self.slot_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
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

# ----- Loss Components --------- #

class AmountsPerMealLoss(nn.Module):
    def __init__(self):
        super(AmountsPerMealLoss, self).__init__()

    def forward(self, pred_amounts, gold_amounts):
        # Both pred_amounts and gold_amounts are of shape (batch_size, 7, 3, 10)

        pred_sums = pred_amounts.sum(dim=-1) # Shape: (batch_size, 7, 3)
        gold_sums = gold_amounts.sum(dim=-1) # Shape: (batch_size, 7, 3)

        loss = torch.mean((pred_sums - gold_sums) ** 2)

        return loss

# ---- Model Training --------- #

def train_model(dataloader, model, foods_criterions: list, amounts_criterions: list, other_criterions: list, optimizer, epochs, device):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f"runs/model_v{MODEL_VERSION}")

    model.to(device)
    model.train()

    loss_history = []

    bar = tqdm(range(epochs))
    min_loss = float("inf")
    best_model = None
    best_epoch = -1

    for e in bar:
        epoch_loss = 0.0

        for x, ids, amounts in dataloader:
            # x: (batch_size, 7)
            # ids: (batch_size, 7, 3, 10)
            # amounts: (batch_size, 7, 3, 10)
            x, gold_ids, gold_amounts = x.to(device), ids.to(device), amounts.to(device)

            optimizer.zero_grad()

            # food_logits: (batch_size, 7, 3, 10, 223)
            # pred_amounts: (batch_size, 7, 3, 10)
            food_logits, pred_amounts = model(x)

            # Flatten
            food_logits_flat = food_logits.view(-1, 223)    # Shape: (batch_size * 7 * 3 * 10, 223)
            pred_amounts_flat = pred_amounts.view(-1)       # Shape: (batch_size * 7 * 3 * 10)
            gold_ids_flat = gold_ids.view(-1)                         # Shape: (batch_size * 7 * 3 * 10)
            gold_amounts_flat = gold_amounts.view(-1)                 # Shape: (batch_size * 7 * 3 * 10)

            # Choose the most probable food ID
            pred_ids = torch.argmax(food_logits, dim=-1)    # Shape: (batch_size, 7, 3, 10)

            # Losses than only consider food IDs
            loss_id = 0.0

            for criterion in foods_criterions:
                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss_id += criterion(food_logits_flat, gold_ids_flat)
                else:
                    loss_id += criterion(pred_ids, gold_ids)

            # Mask zero amounts
            mask = (gold_amounts_flat > 0).float()

            # Losses that only consider food amounts
            loss_amount = ((pred_amounts_flat - gold_amounts_flat) ** 2 * mask).sum() / (mask.sum() + 1e-8)

            for criterion in amounts_criterions:
                loss_amount += criterion(pred_amounts, gold_amounts)

            # Comvine the losses
            loss = loss_id + loss_amount

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        bar.set_postfix_str(f"Loss = {epoch_loss:.0f}")
        loss_history.append(epoch_loss)


        writer.add_scalar("Loss/train", epoch_loss, e)

        if epoch_loss < min_loss and (min_loss - epoch_loss) > 10:
            min_loss = epoch_loss
            best_model = model.state_dict()
            best_epoch = e

    print(f"Best model at epoch {best_epoch} with loss {min_loss:.4f}")
    torch.save(best_model, f"model_v{MODEL_VERSION}.pth")
    print(f"Model saved as model_v{MODEL_VERSION}.pth")

    loss_history = loss_history[200:]
    plt.plot(loss_history)
    plt.savefig(f'models_plots/loss_plot_{int(MODEL_VERSION)}.png')

# ----- Model Evaluation --------- #

def evaluate_on_random_sample(dataloader, model, device):
    model.eval()
    model.to(device)

    FOODS_DATA_PATH = "../../Data/layouts/FoodsByID_copy.json"
    foods = open(FOODS_DATA_PATH, "r")
    data = json.load(foods)

    random_index = torch.randint(0, len(dataloader.dataset), (1,)).item()
    x, y_id, y_amount = dataloader.dataset[random_index]
    x, y_id, y_amount = x.to(device), y_id.to(device), y_amount.to(device)

    my_sample = [2826, 326, 27, 72, 190, 0, 0]
    my_sample = torch.tensor([my_sample], dtype=torch.float32)
    pred_id, pred_amount = model(my_sample.to(device))

    pred_id, pred_amount = model(x.unsqueeze(0).to(device))

    pred_id, pred_amount = pred_id[0], pred_amount[0]

    pred_id = torch.argmax(pred_id, dim=-1)

    pred_amount = pred_amount.squeeze(-1)

    merged_pred = MenusDataset.merge_ids_and_amounts(pred_id, pred_amount)

    check_menu(merged_pred)

    merged_y = MenusDataset.merge_ids_and_amounts(y_id, y_amount)

    values = ["Calories", "Carbohydrate", "Sugars", "Fat", "Protein", "Vegetarian", "Vegan"]
    menu1 = transform2(merged_y, data, device)
    menu2 = transform2(merged_pred, data, device)
    print("\nGround truth vs Model's prediction")
    print("" + "-" * 30)
    for i, (m1, m2) in enumerate(zip(menu1, menu2)):
        print(f"{values[i]}: {m1.item():.0f} vs {m2.item():.0f}")

if __name__ == "__main__":
    main()
