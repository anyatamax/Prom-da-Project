import numpy as np
import PIL.Image
import torch
import lightning as L

from torch.utils.data import Dataset, DataLoader

import os
from sklearn.model_selection import train_test_split

from app.utils import train_transformations, val_transformations, get_imgs_names, convert_lables_to_dict
from app.model import LightningBirdsClassifier


class ImageBirdsDataset(Dataset):
    def __init__(self, imgs_path, imgs_list, lables_gt, test_fraction=0.1, train=True):
        super(ImageBirdsDataset).__init__()
        
        self.imgs_path = imgs_path
        self.imgs_list = imgs_list
        self.lables_gt = lables_gt
        self.test_fraction = test_fraction
        self.train = train
        
        self.train_transformations = train_transformations
        self.val_transformations = val_transformations
        
        imgs_train, imgs_val = train_test_split(self.imgs_list, test_size=self.test_fraction, stratify=list(self.lables_gt.values()), random_state=42)
        
        if self.train == True:
            self.imgs_list = imgs_train
        else:
            self.imgs_list = imgs_val
        
    def __getitem__(self, index):
        image = np.asarray(PIL.Image.open(os.path.join(self.imgs_path, self.imgs_list[index])))
        if len(image.shape) < 3:
            image = np.stack([image] * 3, axis=-1)
        lable_gt = self.lables_gt[self.imgs_list[index]]
            
        if self.train == True:
            transformed_img = self.train_transformations(image=image)['image']
        else:
            transformed_img = self.val_transformations(image=image)['image']
        
        return torch.from_numpy(transformed_img).permute(2, 0, 1), lable_gt
    
    def __len__(self):
        return len(self.imgs_list)
    
    
def get_dataloaders(img_folder, imgs_names, gt_lables, test_fraction, batch_size, num_workers):
    ds_train = ImageBirdsDataset(
        img_folder,
        imgs_names,
        gt_lables,
        train=True,
        test_fraction=test_fraction,
    )

    ds_val = ImageBirdsDataset(
        img_folder,
        imgs_names,
        gt_lables,
        train=False,
        test_fraction=test_fraction,
    )
    
    dataloader_train = DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    dataloader_val = DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    
    return dataloader_train, dataloader_val

    
    
def train_model(
    model,
    experiment_path,
    dl_train,
    dl_valid,
    max_epochs=100,
    fast_train=False,
    ckpt_path=None,
    **trainer_kwargs,
):
    callbacks = [
        # L.pytorch.callbacks.TQDMProgressBar(),
        L.pytorch.callbacks.LearningRateMonitor(),
        L.pytorch.callbacks.ModelCheckpoint(
            filename="{epoch}-{val_accs:.3f}",
            monitor="val_accs",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor="val_accs",
            mode="max",
            patience=10,
            verbose=True,
        )
    ]
    trainer = L.Trainer(
        callbacks=callbacks,
        max_epochs=max_epochs,
        default_root_dir=experiment_path,
        **trainer_kwargs,
    )
    if fast_train == True:
        trainer = L.Trainer(
            callbacks=None,
            max_epochs=1,
            default_root_dir=experiment_path,
            max_steps=2,
            num_sanity_val_steps=1,
            log_every_n_steps=50,
            enable_checkpointing=False,
            accelerator="cpu",
            logger=False,
        )
    trainer.fit(model, dl_train, dl_valid, ckpt_path=ckpt_path)
    
    
def train_classifier(train_gt, train_img_dir, fast_train=False):
    img_names = get_imgs_names(train_img_dir)
    
    batch_size = 64
    if fast_train == True:
        batch_size = 2
    dataloader_train, dataloader_val = get_dataloaders(train_img_dir, img_names, train_gt, test_fraction=0.15, batch_size=batch_size, num_workers=0)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if fast_train == True:
        device = 'cpu'
        
    transfer = True
    if fast_train == True:
        transfer = False
    efficientnet_b2 = LightningBirdsClassifier(
                        transfer=transfer,
                        lr=1e-4,
                        steps_per_epoch=len(dataloader_train))
    train_model(
        efficientnet_b2,
        "./birds_logs",
        dataloader_train,
        dataloader_val,
        accelerator=device,
        max_epochs=25,
        fast_train=fast_train,
    )
    
    return efficientnet_b2

if __name__ == "__main__":
    train_img_dir = "./data/train/images"
    train_gt = "./data/train/gt.csv"
    
    gt_lables = convert_lables_to_dict(train_gt)
    model = train_classifier(gt_lables, train_img_dir, fast_train=False)
    torch.save(model.state_dict(), "./app/birds_model.pt")
    