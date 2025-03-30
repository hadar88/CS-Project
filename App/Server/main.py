from server import Server
from MenuGenerator import MenuGenerator
import torch
import os

MODEL_PATH = "model_weights.pth"

if __name__ == "__main__":
    # Load the model
    print("Loading model...")
    model = MenuGenerator()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    # Create the server instance with the model
    server = Server(model)

    # Run the server
    print("Starting server...")
    port = int(os.environ.get('PORT', 5000))
    server.run(port=port)
