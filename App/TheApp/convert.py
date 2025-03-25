import torch
from model import MenuGenerator
# convert the model to onnx in order to use numpy.
# the vector is of size 14

MODEL_PATH = "model.pth"
ONNX_PATH = "model.onnx"

model = MenuGenerator()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

dummy_input = torch.randn(1, 14)

torch.onnx.export(model, dummy_input, ONNX_PATH)

print(f"Model has been converted to ONNX and saved at {ONNX_PATH}")