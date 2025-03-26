import torch
import torch.nn as nn
from flask import Flask, request, jsonify

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
    
####################

MODEL_PATH = "model.pth"
model = MenuGenerator()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    vec = []
    for key in data:
        vec.append(data[key])
    vec = torch.tensor([vec], dtype=torch.float32)
    output = model(vec)
    print(output)
    return jsonify({"output": output})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
