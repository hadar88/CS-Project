import torch
import torch.nn as nn
from make_dataset import read_foods_tensor, FoodProperties as FP
    
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
            pred = (get_continuous_value(round(pred_ids), self.data, fp) * pred_amounts / 100).sum(dim=(1, 2, 3)) / 7
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
            pred = (1 - get_binary_value(round(pred_ids), self.data, fp)).sum(dim=(1, 2, 3))
            preferences_diff += (torch.exp(-10 * gold) * pred.pow(2)).mean()

        return preferences_diff * self.PREF_PENALTY
    
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
            pred = get_binary_value(round(pred_ids), self.data, fp).sum(dim=(1, 2, 3))
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
            pred = (get_continuous_value(round(pred_ids[:, :, i]), self.data, FP.CALORIES) * pred_amounts[:, :, i] / 100).sum(dim=(1, 2)) / 7
            meals_diff += self.l1loss(pred, gold) / self.DENOMINATOR

        ### Compute the MSE ###

        pred_calorie_value = get_continuous_value(round(pred_ids), self.data, FP.CALORIES) / 100
        pred_calories_per_day = (pred_amounts * pred_calorie_value).sum(dim=(2, 3))
        pred_mses = ((pred_calories_per_day - pred_calories_per_day.mean(dim=1, keepdim=True)).pow(2)).mean(dim=1)

        return self.MSE_PENALTY * pred_mses.mean() + meals_diff    
    
def entropy_penalty(self, pred_ids):
    # Apply softmax to the predicted logits (IDs)
    softmax_probs = torch.nn.functional.softmax(pred_ids, dim=-1)
    
    # Compute entropy (the higher the entropy, the more uniform the distribution)
    entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-8), dim=-1).mean()
    
    return entropy

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
    
def zero_mask(x):
    return torch.exp(-4 * x)

def get_binary_value(x, data, category: FP):
    return torch.sum(
        torch.stack([
                v * torch.exp(-((x - i * v).pow(2)) / 0.01)
                for i, v in enumerate(data[:, category.value])],
            dim=0,
        ),
        dim=0,
    )