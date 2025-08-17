import torchvision
from torch import nn
from collections import OrderedDict
import torch
import torchmetrics
import lightning as L



def get_frozen_mobilenet_v2(num_classes, transfer=True):
    weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if transfer else None
    model = torchvision.models.mobilenet_v2(weights=weights)

    model.classifier = nn.Sequential(OrderedDict([
        (f'dropout1', nn.Dropout(p=0.3, inplace=False)),
        (f'batchnorm1', nn.BatchNorm1d(1280)),
        (f'lin1', nn.Linear(in_features=1280, out_features=512, bias=True)),
        (f'batchnorm2', nn.BatchNorm1d(512)),
        (f'lin2', nn.Linear(in_features=512, out_features=num_classes, bias=True)),
    ]))
    
    model_features = model.features

    for child in list(model_features.children()):
        for param in child.parameters():
            param.requires_grad = False

    return model

def get_frozen_efficientnet_b2(num_classes, transfer=True):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT if transfer else None
    model = torchvision.models.efficientnet_b2(weights=weights)

    model.classifier = nn.Sequential(OrderedDict([
        (f'dropout1', nn.Dropout(p=0.4, inplace=False)),
        (f'batchnorm1', nn.BatchNorm1d(1408)),
        (f'lin1', nn.Linear(in_features=1408, out_features=512, bias=True)),
        (f'batchnorm2', nn.BatchNorm1d(512)),
        (f'dropout2', nn.Dropout(p=0.3, inplace=False)),
        (f'lin2', nn.Linear(in_features=512, out_features=num_classes, bias=True)),
    ]))
    
    model_features = model.features

    for child in list(model_features.children()):
        for param in child.parameters():
            param.requires_grad = False

    return model

def get_frozen_last_efficientnet_b2(num_classes, transfer=True):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT if transfer else None
    model = torchvision.models.efficientnet_b2(weights=weights)

    model.classifier = nn.Sequential(OrderedDict([
        (f'dropout1', nn.Dropout(p=0.4, inplace=False)),
        (f'batchnorm1', nn.BatchNorm1d(1408)),
        (f'lin1', nn.Linear(in_features=1408, out_features=512, bias=True)),
        (f'act', nn.SiLU(inplace=True)),
        (f'batchnorm2', nn.BatchNorm1d(512)),
        (f'dropout2', nn.Dropout(p=0.3, inplace=False)),
        (f'lin2', nn.Linear(in_features=512, out_features=num_classes, bias=True)),
    ]))
    
    model_features = model.features
    model_avgpool = model.avgpool

    for child in list(model.features.children())[:-1]:
        for param in child.parameters():
            param.requires_grad = False

    return model

def warmup_then_cosine_annealing_lr(
    optimizer,
    start_factor,
    T_max,
    warmup_duration,
):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_duration,
    )
    cos_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=0.00001,
    )
    warmup_then_cos_anneal = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup, cos_annealing],
        milestones=[warmup_duration],
    )
    return warmup_then_cos_anneal


class LightningBirdsClassifier(L.LightningModule):
    num_classes = 50

    def __init__(self, *, transfer=True, lr=1e-4, steps_per_epoch=35, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.transfer = transfer
        self.steps_per_epoch = steps_per_epoch
        self.model = self.get_model()
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
        )

    def get_model(self):
        return get_frozen_last_efficientnet_b2(
            self.num_classes,
            self.transfer,
        )

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))

        steps_per_epoch = self.steps_per_epoch
        warmup_duration = 0.4 * steps_per_epoch

        scheduler = warmup_then_cosine_annealing_lr(
            optimizer,
            start_factor=0.05,
            T_max=30,
            warmup_duration=warmup_duration,
        )

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "val")

    def _step(self, batch, kind):
        imgs, target = batch
        pred = self.model(imgs)

        loss = self.loss_fn(pred, target)
        accs = self.accuracy(pred.argmax(axis=-1), target)

        return self._log_metrics(loss, accs, kind)

    def _log_metrics(self, loss, accs, kind):
        metrics = {}
        if loss is not None:
            metrics[f"{kind}_loss"] = loss
        if accs is not None:
            metrics[f"{kind}_accs"] = accs
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss
    
    def forward(self, imgs):
        return self.model(imgs)