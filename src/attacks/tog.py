from torchattacks.attack import Attack
from . import TOGAttacks
from ..utils import get_device_to_use, clean_env_after_one_run
import copy
import torch
from ultralytics.yolo.engine.trainer import BaseTrainer
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
import torchvision.transforms as T
from ultralytics.yolo.utils.ops import xyxy2xywh
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from ultralytics.yolo.utils import DEFAULT_CFG, IterableSimpleNamespace
from ultralytics.yolo.cfg import get_cfg, cfg2dict
from ultralytics.yolo.utils.torch_utils import de_parallel
from ultralytics.yolo.v8.detect.train import Loss

transform_to_image = T.ToPILImage()


def transform_to_prediction_tensor(img):
    return torch.tensor(np.array(img)).unsqueeze_(0).permute(0, 3, 1, 2)


def create_config_for_attack(model_args):
    a = get_cfg(DEFAULT_CFG, {})
    a = cfg2dict(a)
    if isinstance(model_args, IterableSimpleNamespace):
        model_args = cfg2dict(model_args)
    for k in a.keys():
        if k not in model_args.keys():
            model_args[k] = a[k]
    model_args['conf'] = 0.001  # or maybe 0.25
    return IterableSimpleNamespace(**model_args)


def predict_and_show_tensor(model, img_tensor, img_name=''):
    """
    model -- YOLO wrapper model
    """
    if isinstance(img_name, str):
        model.predictor.batch = [[img_name for i in range(img_tensor.shape[0])]]
    elif isinstance(img_name, Iterable):
        model.predictor.batch = [img_name]
    r_raw = model.model(img_tensor)
    result = model.predictor.postprocess(r_raw, img_tensor, img_tensor)
    for each in result:
        pr = each.plot(img=np.ascontiguousarray(transform_to_image(each.orig_img[0])))
        plt.imshow(pr)


def recalculate_loss_based_on_mode(loss, mode):
    if TOGAttacks.fabrication == mode:
        return -1 * loss
    return loss


def change_groundtruth_for_vanishing(batch):
    batch['bboxes'] = torch.zeros(0, 4)
    batch['batch_idx'] = torch.tensor([])
    batch['cls'] = torch.zeros(0, 1)
    return batch


def change_groundtruth_for_fabrication(batch, result, imgsz=640):
    all_xyxy = []
    all_cls = []
    all_batch_idx = []
    for i in range(len(result)):
        xyxy_bboxes = xyxy2xywh(result[i].boxes.xyxy) / imgsz
        batch_cls = result[i].boxes.cls.unsqueeze(1)
        batch_idx = torch.tensor([i for j in range(result[i].boxes.xyxy.shape[0])])
        all_xyxy.append(xyxy_bboxes)
        all_cls.append(batch_cls)
        all_batch_idx.append(batch_idx)
    concat_xyxy = torch.concat(all_xyxy)
    concat_cls = torch.concat(all_cls)
    concat_batch_idx = torch.concat(all_batch_idx)
    batch['bboxes'] = concat_xyxy
    batch['batch_idx'] = concat_batch_idx
    batch['cls'] = concat_cls
    return batch


def create_targets_based_on_mode(inf_model, batch, batch_size=0, mode=TOGAttacks.vanishing):
    """please make sure that batch has been deep copied"""
    if TOGAttacks.vanishing == mode:
        batch = change_groundtruth_for_vanishing(batch)
    elif TOGAttacks.fabrication == mode:
        inf_model.predictor.batch = [batch['im_file']]

        # result = inf_model.predict(transform_to_image(batch['img']))
        result = inf_model.model(batch['img'])
        result = inf_model.predictor.postprocess(result,
                                                 batch['img'],
                                                 batch['img'])
        batch = change_groundtruth_for_fabrication(batch, result)
    else:
        raise NotImplementedError("Other Attack Not Implemented.")
    return batch


class TOG(Attack):
    def __init__(self,
                 model,
                 inf_model,
                 compute_loss,
                 steps=10,
                 eps=8 / 255.,
                 alpha=2 / 255.,
                 default_mode=TOGAttacks.fabrication):
        # eps_iter = alpha
        super().__init__("TOG", model)
        self.eps = eps
        self.alpha = alpha
        # TODO: check default steps for 0
        if steps == 0:
            self.steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        else:
            self.steps = steps
        self.supported_mode = TOGAttacks
        self.detection_model = model
        self.inf_model = inf_model
        self.compute_loss = compute_loss
        self.mode = default_mode
        self.device = get_device_to_use()
        self.model = self.model.to(self.device)
        self.inf_model.model = self.inf_model.model.to(self.device)

    def forward(self,
                batch,
                labels,
                mode=None,
                show_changes=False):
        """
        batch - data loaded with YOLODataset whose img key has been preprocessed, this reasoning is for target creation
        labels - not needed for now
        mode - TOGAttacks object
        """
        clean_env_after_one_run()
        mode = self.mode if mode is None else mode
        batch = copy.deepcopy(batch)
        self.inf_model.predictor.batch = [batch['im_file']]
        # batch['img'] = self.inf_model.predictor.preprocess(batch['img'])
        batch['img'] = batch['img'].float() / 255.
        x_tensor = batch['img'].clone()
        x_tensor = x_tensor.detach().to(self.device)

        bz = x_tensor.size()[0]
        eta = torch.FloatTensor((x_tensor.size())).uniform_(-self.eps, self.eps).to(self.device)
        x_adv = torch.clamp(x_tensor + eta, min=0.0, max=1.0).detach()
        optimizer = BaseTrainer.build_optimizer(model=self.detection_model)
        self.detection_model.train()
        for _ in range(self.steps):
            optimizer.zero_grad()
            x_adv.requires_grad_()
            x_adv.retain_grad()
            detections = self.detection_model(x_adv)
            batch['img'] = x_adv.clone().detach().to(self.device)
            batch = create_targets_based_on_mode(self.inf_model,
                                                 batch,
                                                 batch_size=bz,
                                                 mode=mode)
            batch['bboxes'] = batch['bboxes'].to(self.device)
            batch['batch_idx'] = batch['batch_idx'].to(self.device)
            batch['cls'] = batch['cls'].to(self.device)
            loss, loss_items = self.compute_loss(detections, batch)
            loss = recalculate_loss_based_on_mode(loss=loss,
                                                  mode=mode)
            grad = torch.autograd.grad(loss, x_adv,
                                       retain_graph=False,
                                       create_graph=False,
                                       allow_unused=True)[0]
            x_adv = x_adv.detach()
            x_adv -= self.alpha * grad.sign()
            eta = torch.clamp(x_adv - x_tensor, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x_tensor + eta, min=0.0, max=1.0)
            if show_changes:
                plt.figure()
                predict_and_show_tensor(self.inf_model, x_adv, batch['im_file'])
        return x_adv.detach()

    @classmethod
    def from_weights_file_yolo(cls,
                               weight_file_path,
                               nc=1,  # number of classes
                               steps=10, eps=8 / 255., alpha=2 / 255.,
                               default_mode=TOGAttacks.vanishing):
        device = get_device_to_use()
        weights, ckpt = attempt_load_one_weight(weight=weight_file_path,
                                                device=get_device_to_use())
        yolo_model = DetectionModel(cfg=ckpt['model'].yaml, nc=nc, verbose=False)
        yolo_model.load(weights)
        inf_model = YOLO(model=weight_file_path)
        inf_model.model.args = create_config_for_attack(inf_model.model.args)
        yolo_model.args = create_config_for_attack(inf_model.model.args)
        yolo_model = yolo_model.to(device)
        compute_loss = Loss(de_parallel(yolo_model))
        inf_model.model = inf_model.model.to(device)
        inf_model.predict()
        return TOG(
            model=yolo_model,
            inf_model=inf_model,
            compute_loss=compute_loss,
            steps=steps, eps=eps, alpha=alpha,
            default_mode=default_mode
        )
