import torch
import numpy as np
from tqdm import tqdm
from crepes import ConformalClassifier
from crepes.extras import hinge

from raps import raps_scores
from saps import saps_scores
from enum import Enum
from torchmetrics.classification import MulticlassCalibrationError, MulticlassStatScores


class ConformalCategory(Enum):
    MARGINAL = 0
    CLASS_CONDITIONAL = 1
    GROUP_BALANCED = 2


def get_logits_targets_groups(dataset_class, data_loader, model, device):
    """
    Compute logits, targets, groups, and input identifiers for each record
    """
    logits_list = []
    target_list = []
    group_list = []
    input_identifiers = []
    with torch.no_grad():
        # switch to evaluate mode
        model.eval()
        for data in tqdm(data_loader):
            x, target, group, input_data = dataset_class.prepare_model_inputs(data, device)
            # compute output
            output = model(x)
            logits_list.append(output)
            target_list.append(target)
            group_list.append(group)
            input_identifiers.extend(input_data)
        logits = torch.cat(logits_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        groups = torch.cat(group_list, dim=0)

    return logits.detach().cpu(), targets.detach().cpu(), groups.detach().cpu(), input_identifiers


def compute_nonconformity_score(calib_logits, calib_targets, test_logits, h_params, cfg):
    """
    Computes non-conformity scores for conformal classifiers
    """
    score_fn = cfg["score_fn"]
    if score_fn is None:
        raise ValueError("score_fn must be specified")

    # Temperature scaling before softmax
    calib_probs = torch.softmax(calib_logits / h_params["T"], dim=1)
    test_probs = torch.softmax(test_logits / h_params["T"], dim=1)

    if score_fn == "hinge":
        # For calib data, consider targets and classes
        # Class labels are remapped from 0 to n_classes in `get_loader` method
        classes = torch.tensor([x for x in range(calib_logits.shape[1])])
        calib_non_conf_scores = hinge(calib_probs, classes, calib_targets).numpy()
        test_non_conf_scores = hinge(test_probs).numpy()
    elif score_fn == "raps":
        calib_non_conf_scores = raps_scores(calib_probs, h_params, targets=calib_targets)
        test_non_conf_scores = raps_scores(test_probs, h_params)
    elif score_fn == "saps":
        calib_non_conf_scores = saps_scores(calib_probs, h_params, targets=calib_targets)
        test_non_conf_scores = saps_scores(test_probs, h_params)
    else:
        raise ValueError(f"score_fn {score_fn} not implemented.")

    return calib_non_conf_scores, test_non_conf_scores


def get_conformal_set(
    non_conformity_scores_calib,
    non_conformity_scores_test,
    labels,
    conformal_category,
    confidence=0.95,
    bins_cal=None,
    bins_test=None,
):
    """
    Compute the conformal set based on conformal category - marginal, class-conditional, or group-balanced
    """

    if conformal_category == ConformalCategory.MARGINAL:
        if bins_cal is not None:
            raise ValueError("Bins must be None for marginal conformal category")

        cc_marginal = ConformalClassifier()
        cc_marginal.fit(non_conformity_scores_calib)

        # predict conformal set
        prediction_set = cc_marginal.predict_set(
            non_conformity_scores_test, confidence=confidence
        )

    elif conformal_category == ConformalCategory.CLASS_CONDITIONAL:
        if bins_cal is None:
            raise ValueError(
                "Bins must be provided for class-conditional conformal category"
            )

        cc_class_cond = ConformalClassifier()
        cc_class_cond.fit(non_conformity_scores_calib, bins_cal)

        # Class labels are remapped from 0 to n_classes in `get_loader` method
        class_labels = torch.tensor([x for x in range(len(labels))])

        # predict conformal set
        prediction_set = np.array(
            [
                cc_class_cond.predict_set(
                    non_conformity_scores_test,
                    np.full(len(non_conformity_scores_test), class_labels[c]),
                    confidence=confidence,
                )[:, c]
                for c in range(len(class_labels))
            ]
        ).T

    elif conformal_category == ConformalCategory.GROUP_BALANCED:
        if bins_cal is None or bins_test is None:
            raise ValueError(
                "Both calib and test bins must be provided for group-balanced conformal category"
            )

        cc_group_cond = ConformalClassifier()
        cc_group_cond.fit(non_conformity_scores_calib, bins_cal)

        prediction_set = cc_group_cond.predict_set(
            non_conformity_scores_test, bins_test, confidence=confidence
        )

    # convert the prediction set to labels
    prediction_labels = [
        np.array([idx for idx, val in enumerate(row) if val == 1])
        for row in prediction_set
    ]

    # avoid zero set sizes. Choose argmin from test_non_conformity_scores if no elements in the prediction_labels
    for i in range(len(prediction_labels)):
        if len(prediction_labels[i]) == 0:
            prediction_labels[i] = np.array([np.argmin(non_conformity_scores_test[i])])

    return prediction_labels


def calculate_metrics(logits, targets, prediction_sets, k=3, group=None, 
                      compute_detailed_accs=False, label_map=None, group_map=None):
    """
    Compute the metrics
    """
    prec_1, prec_k = accuracy(logits, targets, topk=(1, k))
    cvg, sz = coverage_size(prediction_sets, targets)
    ece = calibration_error(logits, targets)
    tp, fp, tn, fn, _ = classification_scores(logits, targets)

    metrics = {
        "top1": round(prec_1.item() / 100.0, 4),
        "topk": round(prec_k.item() / 100.0, 4),
        "tpr": round(tp.item() / (tp.item() + fn.item()), 4),
        "fpr": round(fp.item() / (fp.item() + tn.item()), 4),
        "ece": round(ece, 4),
        "coverage": cvg,
        "size": sz,
    }

    if label_map:
        unique_labels = torch.unique(targets)
        if compute_detailed_accs:
            label_accs = {}
            for label in unique_labels:
                label_mask = targets == label
                label_prec_1 = accuracy(logits[label_mask], targets[label_mask], topk=(1,))[0]
                label_accs[label_map[label.item()]] = round(label_prec_1.item() / 100.0, 4)
            metrics["top1_acc_per_label"] = label_accs
        
        label_covs = {}
        label_sizes = {}
        for label in unique_labels:
            label_mask = targets == label
            filtered_prediction_sets = [pred_set for pred_set, mask_value in zip(prediction_sets, label_mask) if mask_value]
            cvg, sz = coverage_size(filtered_prediction_sets, targets[label_mask])
            label_covs[label_map[label.item()]] = cvg
            label_sizes[label_map[label.item()]] = sz
        metrics["cov_per_label"] = label_covs
        metrics["size_per_label"] = label_sizes

    if (group is not None) and group_map:
        unique_groups = torch.unique(group)
        if compute_detailed_accs:
            group_accs = {}
            for grp in unique_groups:
                group_mask = group == grp
                group_prec_1 = accuracy(logits[group_mask], targets[group_mask], topk=(1,))[0]
                group_accs[group_map[grp.item()]] = round(group_prec_1.item() / 100.0, 4)
            metrics["top1_acc_per_group"] = group_accs
            metrics["disparate_impact_acc"] = max(group_accs.values()) - min(group_accs.values())
        
        group_covs = {}
        group_sizes = {}
        for grp in unique_groups:
            group_mask = group == grp
            filtered_prediction_sets = [pred_set for pred_set, mask_value in zip(prediction_sets, group_mask) if mask_value]
            cvg, sz = coverage_size(filtered_prediction_sets, targets[group_mask])
            group_covs[group_map[grp.item()]] = cvg
            group_sizes[group_map[grp.item()]] = sz
        metrics["cov_per_group"] = group_covs
        metrics["size_per_group"] = group_sizes
        metrics["disparate_impact_cov"] = max(group_covs.values()) - min(group_covs.values())
        metrics["disparate_impact_size"] = max(group_sizes.values()) - min(group_sizes.values())

    return metrics


# ================================================================================================
# Below code has been borrowed from https://github.com/aangelopoulos/conformal_classification associated with the paper
# Angelopoulos et al. "Uncertainty Sets for Image Classifiers using Conformal Prediction", ICLR 2021
# Published under the MIT License


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def coverage_size(S, targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if targets[i].item() in S[i]:
            covered += 1
        size = size + S[i].shape[0]
    return float(covered) / targets.shape[0], size / targets.shape[0]


def calibration_error(logits_calib, targets_calib, n_bins=10, norm='l1'):
    """
    Computes the top-label multiclass expected calibration error for the specified number of bins `n_bins`
    logits_calib (Tensor): A float tensor of shape (N, C, ...) containing logits for each observation.
    targets_calib (Tensor): An int tensor of shape (N, ...) containing ground truth labels, and therefore only contain values
    in the [0, n_classes-1] range.

    https://lightning.ai/docs/torchmetrics/stable/classification/calibration_error.html#calibration-error
    """

    num_classes = logits_calib.size()[1]

    compute_calib_error = MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins, norm=norm)
    calib_probs = torch.softmax(logits_calib, dim=1)

    ece = compute_calib_error(calib_probs, targets_calib.reshape([-1]))

    return ece.item()


def classification_scores(logits, targets, top_k=1, average='micro'):
    '''Computes a tensor of shape (..., 5), where the last dimension corresponds to [tp, fp, tn, fn, sup]
    (sup stands for support and equals tp + fn).
    N.B: specify average='micro'/'macro' for overall metrics and average=None for per class metrics
    https://lightning.ai/docs/torchmetrics/stable/classification/stat_scores.html#torchmetrics.classification.MulticlassStatScores
    '''

    num_classes = logits.size()[1]

    metric = MulticlassStatScores(num_classes=num_classes, top_k=top_k, average=average)
    mcss = metric(logits, targets.reshape([-1]))

    return mcss
