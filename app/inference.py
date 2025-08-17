import torch
import numpy as np
import PIL

from .utils import val_transformations


def classify(model_res, image_predict: PIL.Image.Image):
    model_res.eval()
    with torch.no_grad():
        image = np.asarray(image_predict)
        if len(image.shape) < 3:
            image = np.stack([image] * 3, axis=-1)
        transformed_img = val_transformations(image=image)['image']
        transformed_img = torch.from_numpy(transformed_img).permute(2, 0, 1)[None, :]
        
        pred = model_res(transformed_img).numpy()
        pred_lable = pred[0, :].argmax()

    return int(pred_lable)