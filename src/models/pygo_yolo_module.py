from typing import Dict, Any
import torch
from lightning import LightningModule
from torch.cuda import amp
from torchvision.ops import batched_nms, nms

from .yolo.core.yolo_trainer import YOLOTrainer
from .yolo.configs.my_config import MyConfig

from .yolo.utils.utils import xywh_to_xyxy
from .yolo.utils import sampler_set_epoch
from .yolo.models.yolo import YOLO
from .yolo.core.loss import get_loss_fn
from .yolo.utils.metrics import get_det_metrics

class PyGoYoloModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ):
        super().__init__()
        config = MyConfig()
        config.init_dependent_config()
        self.config = config
        #self.trainer = YOLOTrainer(config)
        self.model = YOLO(config.num_class, backbone_type=config.backbone_type, 
                            label_assignment_method=config.label_assignment_method, anchor_boxes=config.anchor_boxes,
                            channel_sparsity=config.channel_sparsity)

        self.loss_fn = get_loss_fn(config)
        self.mAP = get_det_metrics().to(self.device)
 
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

 
    #def on_train_epoch_start(self, trainer, module):
        #self.trainer.model.train()
        #sampler_set_epoch(self.config, self.trainer.train_loader, self.trainer.cur_epoch)
    
     
    def model_step(self, pixel_values, is_training=True):
        #pixel_values = pixel_values.to(self.device, dtype=torch.float32)
        #self.trainer.optimizer.zero_grad()

        # Forward path
        with torch.amp.autocast(enabled=self.config.amp_training, device_type='cuda'):
            preds = self.model(pixel_values, is_training)
        
        return preds


    def training_step(self, batch, idx):
        
        bboxes = batch['bbox'].to(self.device, dtype=torch.float32)    
        classes = batch['cls'].to(self.device, dtype=torch.float32)    

        # Forward path
        with torch.amp.autocast(enabled=self.config.amp_training, device_type='cuda'):
            preds = self.model_step(batch['pixel_values'], is_training=True)
            loss, (conf_loss, iou_loss, class_loss) = self.loss_fn(preds, bboxes, classes)

        if self.config.use_tb and self.main_rank:
            self.log_metrics('train/loss', loss.detach(), self.train_itrs)
            self.log_metrics('train/conf_loss', conf_loss, self.train_itrs)
            self.log_metrics('train/iou_loss', iou_loss, self.train_itrs)
            self.log_metrics('train/class_loss', class_loss, self.train_itrs)


        return loss
    

    @torch.no_grad()
    def validation_step(self, batch, idx):
        
        empty_tensor = torch.tensor([]).to(self.device)
        preds = self.model_step(batch['pixel_values'], is_training=False)
        bboxes = batch['bbox']
        classes = batch['cls']
        #preds = self.trainer.ema_model.ema(images, is_training=False)

        _, _, height, width = batch['pixel_values'].shape
        outputs, targets = [], []
        for i, pred in enumerate(preds):
            pred_conf = pred[:, 0]

            pred_boxes = xywh_to_xyxy(pred[:, 1:5])
            pred_boxes[:, 0::2].clamp_(0, width)
            pred_boxes[:, 1::2].clamp_(0, height)

            cls_logits, pred_cls = preds[i][:, 5:].max(dim=1)
            pred_conf *= cls_logits
            pred = torch.cat([pred_conf.unsqueeze(1), pred_boxes, pred_cls.unsqueeze(1)], dim=1)
            output = pred[pred_conf > self.config.conf_thrs]

            # NMS per image
            kept_indices = self.nms(output, max_nms_num=self.config.max_nms_num, nms_iou=self.config.val_iou)

            if kept_indices is None:
                outputs.append(dict(boxes=empty_tensor, scores=empty_tensor, labels=empty_tensor.long()))
            else:
                outputs.append(dict(boxes=output[kept_indices][:, 1:5], scores=output[kept_indices][:, 0], 
                                    labels=output[kept_indices][:, 5].long()))

            if bboxes.shape[0]:
                targets.append(dict(boxes=bboxes[bboxes[:, 0]==i][:, 1:], 
                                    labels=classes[classes[:, 0]==i][:, 1].long()))
            else:
                targets.append(dict(boxes=empty_tensor, labels=empty_tensor.long()))

        self.mAP.update(outputs, targets)

        val_results = self.mAP.compute()

        map50 = val_results['map_50']
        map50_95 = val_results['map']
        
        ret = {'map50': map50, 'map50_95': map50_95}
        self.log_dict(ret, on_step=True)
        return ret


    @classmethod
    def nms(cls, output, class_agnostic=False, max_nms_num=100, nms_iou=0.6):
        if class_agnostic:
            kept_indices = nms(boxes=output[:, 1:5], scores=output[:, 0], iou_threshold=nms_iou)
        else:
            kept_indices = batched_nms(boxes=output[:, 1:5], scores=output[:, 0], idxs=output[:, 5], iou_threshold=nms_iou)

        if not kept_indices.shape[0]:
            return None

        if kept_indices.shape[0] > max_nms_num:
            kept_indices = kept_indices[:max_nms_num]

        return kept_indices
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.model.parameters())
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