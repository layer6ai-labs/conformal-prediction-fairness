import random
import argparse
import pprint
import torch
import datasets
import time
import numpy as np

from config import get_config, parse_config_arg
from writer import get_writer
from data_sets import *
from model_factory import get_model
from utils import (
    compute_nonconformity_score,
    get_logits_targets_groups,
    get_conformal_set,
    calculate_metrics,
    accuracy,
    coverage_size,
    ConformalCategory,
)
from average_k import get_average_k_sets
from hpo import run_hpo_conformal

datasets.logging.set_verbosity_error()

def main():
    start_time = time.time()
    
    # Parse args and create config dict
    parser = argparse.ArgumentParser(description="Generate conformal prediction sets.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use.")
    parser.add_argument(
        "--config",
        default=[],
        action="append",
        help="Override config entries. Specify as `key=value`.",
    )

    args = parser.parse_args()

    cfg = get_config(dataset=args.dataset)
    cfg = {**cfg, **dict(parse_config_arg(kv) for kv in args.config)}

    # set hyperparams for chosen non conformity score
    cfg["h_params_conformal"] = cfg.get(f"h_params_{cfg['score_fn']}", None)
    if cfg["h_params_conformal"] is None:
        raise ValueError(f"Unknown score function {cfg['score_fn']}")

    pprint.sorted = lambda x, key=None: x
    pp = pprint.PrettyPrinter(indent=4)
    print(10 * "-" + "cfg" + 10 * "-")
    pp.pprint(cfg)

    writer = get_writer(args, cfg=cfg)

    # Set random seeds for reproducibility
    np.random.seed(seed=cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get specified dataset in the form of loaders
    dataset_class, loader_dict = get_loaders(cfg)

    # Get model specific for each dataset, trained from scratch or loaded from saved weights
    used_labels = (
        loader_dict["top_m_labels"]
        if dataset_class.uses_top_m_labels
        else [i for i in range(cfg["m"])]
    )

    model = get_model(
        cfg, device, dataset_class, loader_dict["train"], loader_dict["val"], used_labels
    )
    if cfg['save_model_ckpt']:
        writer.write_checkpoint("model", model.state_dict())

    logits_calib, targets_calib, groups_calib, _ = get_logits_targets_groups(
        dataset_class, loader_dict["calib"], model, device
    )
    check_dataset_balance(writer, "calib", targets_calib, groups_calib, label_map=dataset_class.get_id2label(return_dict=True),
        group_map=dataset_class.get_id2group(return_dict=True))
    logits_test, targets_test, groups_test, input_identifiers_test = get_logits_targets_groups(
        dataset_class, loader_dict["test"], model, device
    )
    check_dataset_balance(writer, "test", targets_test, groups_test, label_map=dataset_class.get_id2label(return_dict=True),
        group_map=dataset_class.get_id2group(return_dict=True))

    if 'calib_val' not in loader_dict:
        logits_val = logits_test
        targets_val = targets_test
        groups_val = groups_test
    else:
        logits_val, targets_val, groups_val, _ = get_logits_targets_groups(
            dataset_class, loader_dict["calib_val"], model, device
        )
        check_dataset_balance(writer, "val", targets_val, groups_val, label_map=dataset_class.get_id2label(return_dict=True),
        group_map=dataset_class.get_id2group(return_dict=True))

    k=cfg["k"]

    ### Top-K
    # cvg_topk_val = accuracy(logits_val, targets_val, topk=(k,))[0].item() / 100.0
    # print(f"Empirical coverage of top {k} prediction sets on the validation set: {cvg_topk_val: .4f}")
    cvg_topk_calib = accuracy(logits_calib, targets_calib, topk=(k,))[0].item() / 100.0
    print(f"Empirical coverage of top-{k} prediction sets on the calibration set: {cvg_topk_calib: .4f}")
    cvg_topk_test = accuracy(logits_test, targets_test, topk=(k,))[0].item() / 100.0
    print(f"Empirical coverage of top-{k} prediction sets on the test set: {cvg_topk_test: .4f}")

    if cfg["alpha"] is None:
        print("Setting alpha according to top-k coverage on calibration set")
        cfg["alpha"] = 1 - cvg_topk_calib
    print(f"Using alpha {cfg['alpha']:.4f}")

    ### Marginal Conformal
    h_params_marg = cfg["h_params_conformal"]
    if cfg["hpo_iterations"] > 0:
        h_params_marg = run_hpo_conformal(
            logits_calib,
            targets_calib,
            logits_val,
            targets_val,
            used_labels,
            h_params_marg,
            cfg,
            conformal_category = ConformalCategory.MARGINAL,
            )
    print(f"Best hyperparams for Marginal: {h_params_marg}")
    # Compute non conformity score for each class
    non_conf_scores_marg_calib, non_conf_scores_marg_test = compute_nonconformity_score(
        logits_calib, targets_calib, logits_test, h_params_marg, cfg
    )
    print(f'non_conf_scores_marg_calib: {non_conf_scores_marg_calib.shape}')
    print(f'non_conf_scores_marg_test: {non_conf_scores_marg_test.shape}')
    # Get marginal conformal sets for test set
    conformal_preds_marg_test = get_conformal_set(
        non_conf_scores_marg_calib,
        non_conf_scores_marg_test,
        labels=used_labels,
        confidence=1-cfg["alpha"],
        conformal_category = ConformalCategory.MARGINAL,
    )

    metrics_marg = calculate_metrics(
        logits_test, targets_test, conformal_preds_marg_test, k=k, group=groups_test,
        compute_detailed_accs=True,
        label_map=dataset_class.get_id2label(return_dict=True),
        group_map=dataset_class.get_id2group(return_dict=True)
    )

    metrics_per_label_group = {k: v for k, v in metrics_marg.items() if k in ["top1_acc_per_label", "top1_acc_per_group", "disparate_impact_acc"]}
    metrics_marg = {k: v for k, v in metrics_marg.items() if k not in ["top1_acc_per_label", "top1_acc_per_group", "disparate_impact_acc"]}
    
    metrics_marg["alpha"] = round(cfg["alpha"], 3)
    writer.write_json("metrics_marginal", metrics_marg)

    print("Marginal metrics:")
    print(f"alpha = {metrics_marg['alpha']}")
    print(f"Cvg@1 = {round(metrics_marg['top1'], 3)}")
    print(f"Cvg@k = {round(metrics_marg['topk'], 3)}")
    print(f"Cvg   = {round(metrics_marg['coverage'], 3)}")
    print(f"Size  = {round(metrics_marg['size'], 3)}")
    print(f"ECE@1  = {round(metrics_marg['ece'], 3)}")
    print(f"TPR@1  = {round(metrics_marg['tpr'], 3)}")
    print(f"FPR@1  = {round(metrics_marg['fpr'], 3)}")


    if metrics_per_label_group:
        writer.write_json("metrics_per_label_group", metrics_per_label_group)

        if "top1_acc_per_label" in metrics_per_label_group:
            print("\nCvg@1 per label:")
            for label, acc in metrics_per_label_group["top1_acc_per_label"].items():
                print(f"{label}: {round(acc,3)}")
        if "top1_acc_per_group" in metrics_per_label_group:
            print("\nCvg@1 per group:")
            for group, acc in metrics_per_label_group["top1_acc_per_group"].items():
                print(f"{group}: {round(acc,3)}")
        if "disparate_impact" in metrics_per_label_group:
            print(f"\nDisparate Impact: {metrics_per_label_group['disparate_impact']}")

    ### Conditional Conformal
    h_params_cond = cfg["h_params_conformal"]
    if cfg["hpo_iterations"] > 0:
        h_params_cond = run_hpo_conformal(
            logits_calib,
            targets_calib,
            logits_val,
            targets_val,
            used_labels,
            h_params_cond,
            cfg,
            conformal_category = dataset_class.group_conformal_category,
            bins_calib=groups_calib.numpy(),
            bins_test=groups_val.numpy()
            )
    print(f"Best hyperparams for Conditional: {h_params_cond}")

    # Compute non conformity score for each class
    non_conf_scores_cond_calib, non_conf_scores_cond_test = compute_nonconformity_score(
        logits_calib, targets_calib, logits_test, h_params_cond, cfg
    )

    # Get conditional conformal sets for test set
    conformal_preds_cond_test = get_conformal_set(
        non_conf_scores_cond_calib,
        non_conf_scores_cond_test,
        labels=used_labels,
        confidence=1-cfg["alpha"],
        conformal_category = dataset_class.group_conformal_category,
        bins_cal=groups_calib.numpy(),
        bins_test=groups_test.numpy()
    )

    metrics_cond = calculate_metrics(
        logits_test, targets_test, conformal_preds_cond_test, k=k, group=groups_test,
        label_map=dataset_class.get_id2label(return_dict=True),
        group_map=dataset_class.get_id2group(return_dict=True)
    )
    metrics_cond["alpha"] = round(cfg["alpha"], 3)
    writer.write_json("metrics_conditional", metrics_cond)
    print("Conditional metrics:")
    print(f"alpha = {metrics_cond['alpha']}")
    print(f"Cvg@1 = {round(metrics_cond['top1'], 3)}")
    print(f"Cvg@k = {round(metrics_cond['topk'], 3)}")
    print(f"Cvg   = {round(metrics_cond['coverage'], 3)}")
    print(f"Size  = {round(metrics_cond['size'], 3)}")
    print(f"ECE@1  = {round(metrics_cond['ece'], 3)}")
    print(f"TPR@1  = {round(metrics_cond['tpr'], 3)}")
    print(f"FPR@1  = {round(metrics_cond['fpr'], 3)}")

    ### Average-K    
    target_coverage = round(1 - cfg["alpha"], 3)
    print(f"Performing binary search to find k that matches target coverage of {target_coverage} on validation set")
    # binary search to find the k that matches the target coverage, start with k_low = 0 and k_high = conformal marginal set size
    k_low = 0
    k_high = int(metrics_marg["size"])
    k_avgk = int(metrics_marg["size"])
    while k_low < k_high:
        # No tunable parameters for avg-k
        preds_avgk_val = get_average_k_sets(logits_calib, logits_val, k_avgk)
        cvg_avgk_val, size_avgk_val = coverage_size(preds_avgk_val, targets_val)
        coverage = round(cvg_avgk_val, 3)
        print(f"target k = {k_avgk}, val actual size = {size_avgk_val}, target coverage = {target_coverage},  val actual coverage = {coverage}")
        if coverage < target_coverage:
            if k_high == k_avgk:
                k_avgk += 0.5
                k_high += 0.5
            else:
                k_low = k_avgk
                k_avgk = (k_high+k_low)/2
            print(f"Increasing k_avgk to {k_avgk}, searching beween {k_low} and {k_high}")
        if coverage > target_coverage:
            k_high = k_avgk
            k_avgk = (k_high+k_low)/2
            print(f"Decreasing k_avgk to {k_avgk}, searching beween {k_low} and {k_high}")

        if coverage == target_coverage or round(k_low, 5) == round(k_high, 5):
            # termination conditions:
            # when coverage is within 0.0001 of target coverage, we use the current k_avgk as the final k
            # when k_low and k_high are within 0.00001 of each other, we use the current k_avgk as the final k
            print(f"Found k={k_avgk} that matches target coverage of {target_coverage} on validation set")
            preds_avgk_calib = get_average_k_sets(logits_calib, logits_calib, k_avgk)
            cvg_avgk_calib, size_avgk_calib = coverage_size(preds_avgk_calib, targets_calib)
            preds_avgk_test = get_average_k_sets(logits_calib, logits_test, k_avgk)
            metrics_avgk = calculate_metrics(
                logits_test, targets_test, preds_avgk_test, k=k, group=groups_test,
                label_map=dataset_class.get_id2label(return_dict=True),
                group_map=dataset_class.get_id2group(return_dict=True)
            )
            print(f"Empirical coverage of average-k prediction sets on the validation set: {cvg_avgk_val: .4f}")
            print(f"Size of average-k prediction sets on the validation set: {size_avgk_val: .4f}")

            print(f"Empirical coverage of average-k prediction sets on the calibration set: {cvg_avgk_calib: .4f}")
            print(f"Size of average-k prediction sets on the calibration set: {size_avgk_calib: .4f}")
            break
    metrics_avgk["alpha"] = round(cfg["alpha"], 3)
    writer.write_json("metrics_avgk", metrics_avgk)
    print("Average-K metrics:")
    print(f"k_avgk = {k_avgk}")
    print(f"Cvg@1 = {round(metrics_avgk['top1'], 3)}")
    print(f"Cvg@k = {round(metrics_avgk['topk'], 3)}")
    print(f"Cvg   = {round(metrics_avgk['coverage'], 3)}")
    print(f"Size  = {round(metrics_avgk['size'], 3)}")
    print(f"ECE@1  = {round(metrics_avgk['ece'], 3)}")
    print(f"TPR@1  = {round(metrics_avgk['tpr'], 3)}")
    print(f"FPR@1  = {round(metrics_avgk['fpr'], 3)}")
    
    ### Format data for csv
    print("Formatting data for CSV output")
    df = data_prep_to_generate_csv(
        logits_test,
        preds_avgk_test,
        conformal_preds_marg_test,
        conformal_preds_cond_test,
        k=k,
        input_identifiers=input_identifiers_test,
        group_label=groups_test.numpy(),
        y=targets_test
        )

    df = dataset_class.process_dataframe(df, loader_dict, k=k)
    
    print("Preparing to save data to csv")
    format_and_write_to_csv(df, writer, args, cfg)

    print(
        f"total time to run {cfg['dataset']} dataset : {time.time() - start_time:.2f} s"
    )

if __name__ == "__main__":
    main()
