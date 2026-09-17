"""One-batch DiT smoke test; run from diffc_image_classification/ on a GPU node."""
import torch
from constants import model_base_dict
from models import ImageClassifier_DiT

model_id = "facebook/DiT-XL-2-512"
assert model_base_dict.get(model_id) == "dit", "Add the DiT entry to constants.py first."
assert torch.cuda.is_available(), "Run this smoke test on a GPU node."

device = torch.device("cuda")
config = {
    "model_name": model_id, "feature_model": "dit",
    "diffusion_step_type": "onestep", "diffusion_layer": "14",
    "diffusion_timestep": 25, "input_channels": 1152,
    "num_classes": 37, "dropout_rate": 0.0, "device": device,
}
model = ImageClassifier_DiT(config).to(device)
model.train()
images = torch.randn(1, 3, 512, 512, device=device)
logits = model(images, None, 25)
assert logits.shape == (1, 37), f"Unexpected logits shape: {tuple(logits.shape)}"
loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0], device=device))
loss.backward()
assert model.classifier.fc.weight.grad is not None
assert all(param.grad is None for param in model.feature_model.parameters())
print("DiT block 14 one-batch forward and classifier backward: PASS")
print("Logits:", tuple(logits.shape))
