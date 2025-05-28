import torch.nn as nn

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