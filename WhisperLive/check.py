import torch

# Specify the path to the model file
model_path = '/root/.cache/huggingface/hub/BELLE-2--Belle-whisper-large-v3-zh-punct/model.bin'

# Load the model with map_location to CPU or the available GPU
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Inspect the keys in the checkpoint
print(checkpoint.keys())
