import torch
from datetime import datetime
import subprocess
import onnx
from app.model import LightningBirdsClassifier

MODEL_PATH = "./app/birds_model.pt"
ONNX_PATH = "./app/birds_model.onnx"

model = LightningBirdsClassifier.load_from_checkpoint(
    MODEL_PATH,
    map_location=torch.device('cpu'),
    lr=1e-4,
    transfer=False)
model.eval()


torch.onnx.export(
    model,
    torch.randn(1, 3, 250, 240),
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
)

# get useful info
date_saved = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
exp_name = "birds experiment 2"

try:
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
except Exception:
    git_hash = "_"

onnx_model = onnx.load(ONNX_PATH)
meta = onnx_model.metadata_props.add()
meta.key = "commit_hash"
meta.value = git_hash
meta = onnx_model.metadata_props.add()
meta.key = "save_date"
meta.value = date_saved
meta = onnx_model.metadata_props.add()
meta.key = "exp_name"
meta.value = exp_name

onnx.save(onnx_model, ONNX_PATH)