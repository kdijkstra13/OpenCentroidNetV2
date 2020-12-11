import torch
import torch.nn
import torch.autograd
import torch.nn.modules.loss
import torch.nn.functional as functional
from centroidnet.backbones import UNet
import numpy as np
from typing import List
from skimage.draw import ellipse
from skimage.feature import peak_local_max

from centroidnet.postprocess import calc_centroid_and_border_vectors, calc_vote_image, calc_centroids, calc_borders


class CentroidNetV2(torch.nn.Module):
    def __init__(self, num_classes, num_channels):
        torch.nn.Module.__init__(self)
        self.backbone = UNet(num_classes=num_classes + 4, in_channels=num_channels, depth=5, start_filts=64)
        self.num_classes = num_classes
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def __str__(self):
        return f"CentroidNet: {self.backbone}"


class CentroidLossV2(torch.nn.Module):
    @staticmethod
    def vector_loss(result: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ret = torch.mean(torch.sqrt(torch.sum((target - result) ** 2, dim=1)))
        return ret

    @staticmethod
    def ce_loss(result: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = target.shape[1]
        result_logits = result.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target_ids = torch.argmax(target, dim=1).long().view(-1, 1).squeeze()
        ret = functional.cross_entropy(result_logits, target_ids)
        return ret

    def forward(self, result, target):
        centroid_vectors_result, centroid_vectors_target = result[:, 0:2], target[:, 0:2]
        border_vectors_result, border_vectors_target = result[:, 2:4], target[:, 2:4]
        logits_result, logits_target = result[:, 4:], target[:, 4:]

        loss = self.vector_loss(centroid_vectors_result, centroid_vectors_target) + \
            self.vector_loss(border_vectors_result, border_vectors_target) + \
            self.ce_loss(logits_result, logits_target)

        return loss


def encode(boxes: List[np.ndarray], height, width, max_dist, num_classes):

    # Encode logits and instances
    logits = np.zeros([num_classes, height, width])
    logits[0] = 1
    instance_ids = np.zeros([height, width])

    for i, (y, x, ymin, ymax, xmin, xmax, class_id) in enumerate(boxes):
        ymin, ymax, xmin, xmax = min(ymin, ymax), max(ymin, ymax), min(xmin, xmax), max(xmin, xmax)
        rr, cc = ellipse(y, x, (ymax - ymin) // 2, (xmax - xmin) // 2, shape=(height, width))
        logits[class_id + 1][rr, cc] = 1
        logits[0][rr, cc] = 0
        instance_ids[rr, cc] = i + 1

    # Encode vectors
    centroid_vectors, border_vectors = calc_centroid_and_border_vectors(instance_ids=instance_ids, max_dist=max_dist)

    centroid_vectors = centroid_vectors / max_dist
    border_vectors = border_vectors / max_dist

    # Create target
    target = np.concatenate([centroid_vectors, border_vectors, logits], axis=0)
    return target


def decode(input: np.ndarray, max_dist: int, nm_window: int, centroid_threshold: int, border_threshold: int):
    _, image_height, image_width = input.shape
    centroid_vectors = input[0:2] * max_dist
    border_vectors = input[2:4] * max_dist
    logits = input[4:]

    # Calculate class ids and class probabilities
    class_ids = np.expand_dims(np.argmax(logits, axis=0), axis=0)
    sum_logits = np.expand_dims(np.sum(logits, axis=0), axis=0)
    class_probs = np.expand_dims(np.max((logits / sum_logits), axis=0), axis=0)
    class_probs = np.clip(class_probs, 0, 1)

    # Start decoding instances
    centroid_votes = calc_vote_image(centroid_vectors=centroid_vectors)

    centroid_list, votes_list = calc_centroids(centroid_votes=centroid_votes, centroid_threshold=centroid_threshold,
                                               nm_window=nm_window)

    border_list, border_votes = calc_borders(centroid_list, centroid_vectors, border_vectors, border_threshold)

    return centroid_list, border_list, class_ids, class_probs, \
        centroid_vectors, border_vectors, \
        centroid_votes, border_votes, votes_list


def fit_circle(centroid, border):
    ctr_y, ctr_x = centroid
    y, x = border
    yy = y - ctr_y
    xx = x - ctr_x
    r = int(np.mean(np.sqrt(np.power(yy, 2) + np.power(xx, 2))))
    return ctr_x, ctr_y, r
