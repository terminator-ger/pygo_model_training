# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any, Dict

from lightning import LightningModule
import torch
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import F1Score, AUROC

from src.models.stone_loss import StoneLoss
import timm


class PyGoNetModule(LightningModule):
    """Example of LightningModule for MNIST classification.
    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.net = net

        # loss function
        self.criterion_stones = StoneLoss()
        #self.criterion_board = torch.nn.CrossEntropyLoss()
        #self.criterion_occluded = torch.nn.BCEWithLogitsLoss()
        self.metrics = torch.nn.ModuleDict()
        # metric objects for calculating and averaging accuracy across batches
        for split in ["metrics_train", "metrics_val", "metrics_test"]:
            self.metrics[split] = torch.nn.ModuleDict({
                "stones": MetricCollection([
                    Accuracy(task="multiclass", num_classes=3),
                    F1Score(task="multiclass", num_classes=3),
                    AUROC(task="multiclass", num_classes=3),
                ],
                prefix=f"{split}/stones/"),
                #"board_size": MetricCollection([
                #    Accuracy(task="multiclass", num_classes=3),
                #    F1Score(task="multiclass", num_classes=3),
                #    AUROC(task="multiclass", num_classes=3),
                #],
                #prefix=f"{split}/board_size/"),
                #"occluded": MetricCollection([
                #    Accuracy(task="binary"),
                #    F1Score(task="binary"),
                #    AUROC(task="binary"),
                #],
                #prefix=f"{split}/occluded/"),
                }
            )
        
       
        #for step, dict_ in self.metrics.items():
        #    for key, target in metrics.items():
        #        dict_[key] = target.copy()
        #        #for metric_type, metric in metrics[step][target].items():
        #            #self.metrics[step][target][metric_type] = metric.clone()
        
        # for averaging loss across batches
        loss_metric = MeanMetric()
        self.train_loss = loss_metric.clone()
        self.val_loss = loss_metric.clone()
        self.test_loss = loss_metric.clone()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
    
    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        #x, (y_stones, y_occluded, y_board) = batch
        x, y_stones = batch
        logits = self.forward(x)
        loss_stones = self.criterion_stones(logits["stones"], y_stones)
        #loss_board = self.criterion_board(logits["board_size"], y_board.reshape(-1).to(torch.long))
        #loss_occluded = self.criterion_occluded(logits["occluded"], y_occluded.float())
        
        preds_stones = torch.softmax(logits['stones'], dim=-1)   
        #preds_board = torch.softmax(logits["board_size"], dim=-1)
        #preds_occluded = logits["occluded"]
        #loss = torch.mean(torch.tensor([loss_stones, loss_board, loss_occluded]))
        #loss = torch.mean(torch.tensor([loss_stones, loss_occluded]))
        return loss_stones, {"stones": preds_stones}, {"stones": y_stones}
                    #"board_size": preds_board, 
                    #"occluded": preds_occluded}, 
                #{"stones": y_stones} 
                    #"board_size": y_board, 
                    #"occluded": y_occluded}

    def log_metrics(self, split: str, preds: Any, targets: Any):
        for name in self.metrics[split].keys():
            if name in ["stones", "board_size"]:
                m = self.metrics[split][name](preds[name].reshape(-1, 3), targets[name].reshape(-1))
            else:
                m = self.metrics[split][name](preds[name].sigmoid().reshape(-1), targets[name].reshape(-1))
        
            self.log_dict(m, on_step=True, on_epoch=True)
 
    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log_metrics("metrics_train", preds, targets)
        
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log_metrics("metrics_val", preds, targets)

        return {"loss": loss, "preds": preds, "targets": targets}


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log_metrics("metrics_test", preds, targets)

        return {"loss": loss, "preds": preds, "targets": targets}


    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
           
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    m = timm.create_model("mobilenetv3mh_large_150d", pretrained=True, num_classes=1)
    print(m)
