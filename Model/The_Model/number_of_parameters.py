import torch
import torch.nn as nn

class MenuGenerator(nn.Module):
    def __init__(self, food_vocab_size=223, hidden_dim=256):
        super(MenuGenerator, self).__init__()

        self.input_encoder = nn.Sequential(
            nn.Linear(14, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

        # Project to a 3D tensor for convs: (batch, channels, height, width)
        self.proj_to_conv = nn.Linear(hidden_dim, 64 * 7 * 3)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
        )

        # Flatten and project to slots (10 per meal)
        self.slot_proj = nn.Linear(64 * 7 * 3, 7 * 3 * 10 * hidden_dim)
        self.slot_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
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
        conv_input = self.proj_to_conv(latent).view(batch_size, 64, 7, 3)
        conv_out = self.conv_layers(conv_input)  # (batch, 64, 7, 3)
        flat = conv_out.view(batch_size, -1)  # (batch, 64*7*3)
        slot_input = self.slot_proj(flat).view(batch_size, 7, 3, 10, -1)  # (batch, 7, 3, 10, hidden_dim)
        decoded = self.slot_decoder(slot_input)  # (batch, 7, 3, 10, 64)

        food_logits = self.food_id_head(decoded)  # (batch, 7, 3, 10, 223)
        amounts = self.amount_head(decoded).squeeze(-1)  # (batch, 7, 3, 10)

        return food_logits, amounts
    
model = MenuGenerator()

MODEL_VERSION = 16

model.load_state_dict(torch.load(f"saved_models/model_v{MODEL_VERSION}.pth"))

total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")