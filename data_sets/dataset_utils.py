import torch
import pandas as pd
from typing import Any, Tuple


PUNCTUATION_TOKENS = [":", ",", ".", ";", "'", "``", "`", '"', "!", ")", "(", "%"]


### General Tools
class GroupDataset(torch.utils.data.Dataset):
    '''
    Implementation of torch Dataset that returns a tuple of data needed for inference 'x',
    classification labels 'y', group labels 'z'
    '''

    def __init__(self, role, x, y=None, z=None):
        if y is None:
            y = torch.zeros(x.shape[0]).long()

        if z is None:
            z = torch.zeros(x.shape[0]).long()

        assert x.shape[0] == y.shape[0] and x.shape[0] == z.shape[0]
        assert role in ["train", "val", "calib", "test"]

        self.role = role

        self.x = x
        self.y = y
        self.z = z

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        return self.x[index], self.y[index], self.z[index]

    def to(self, device):
        return GroupDataset(
            self.role,
            self.x.to(device),
            self.y.to(device),
            self.z.to(device),
        )


def get_loader(dset, batch_size, shuffle=True, drop_last=False, **loader_kwargs):
    return torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **loader_kwargs,
    )


def stratified_sample_df(df, col, n_samples, return_remaining=False):
    n = min(n_samples, df[col].value_counts().min())
    df_2 = df.groupby(col).apply(lambda x: x.sample(n))
    df_2.index = df_2.index.droplevel(0)
    df_3 = df.drop(index=df_2.index)
    if return_remaining:
        return df_2, df_3
    return df_2


def check_dataset_balance(writer, split, targets, groups, label_map, group_map):
    target_counts = torch.unique(targets, return_counts=True)
    target_names = [label_map[x] for x in target_counts[0].tolist()]
    target_counts = target_counts[1].tolist()
    group_counts = torch.unique(groups, return_counts=True)
    group_names = [group_map[x] for x in group_counts[0].tolist()]
    group_counts = group_counts[1].tolist()
    counts_dict = {"classes": {}, "groups": {}}
    for name, count in zip(target_names, target_counts):
        counts_dict["classes"][name] = count
    for name, count in zip(group_names, group_counts):
        counts_dict["groups"][name] = count
    writer.write_json(f"dataset_counts_{split}", counts_dict)


def bring_examples_to_top(df, m, min_num, repeat=3):
    # Sort dataframe, then put example instances from each class at the top
    df = df.sort_values("label", axis=0)
    sorted_order = df.index.to_list()
    temps = [[0] * m for i in range(repeat)]
    for i in range(m):
        for temp in temps:
            index_to_pop = i * min_num - i
            if index_to_pop < len(sorted_order):
                temp[i] = sorted_order.pop(index_to_pop)
            else:
                break
    for j, temp in enumerate(temps):
        for i, x in enumerate(temp):
            sorted_order.insert(i + m*j, x)
    df = df.reindex(sorted_order)
    return df


def bring_ravdess_examples_to_top(df, repeat=3):
    df = df.sort_values("label", axis=0)
    label_indices = {label: df[df["label"] == label].index.tolist() for label in df["label"].unique()}
    
    new_order = []
    for _ in range(repeat):
        for label, indices in label_indices.items():
            count = min(1, len(indices))
            new_order.extend(indices[:count])
            label_indices[label] = indices[count:]  # Remove the collected index from the list
    
    remaining_indices = [index for indices in label_indices.values() for index in indices]
    new_order.extend(remaining_indices)
    
    df = df.reindex(new_order)
    return df


def prediction_set_text_fn(prediction_set, idlabels):
    out = ""
    for el in prediction_set.split():
        if el != " ":
            text = idlabels[int(el)].title()
            out += el + f". {text}  "
    return out[:-2]  # remove final spaces


def relabel_set_obj(prediction_set, label_reordering):
    out = ""
    prediction_set = str(prediction_set)  # For single element int sets
    prediction_set = prediction_set.split()
    prediction_set = [label_reordering[int(el)] for el in prediction_set]
    prediction_set.sort()
    for el in prediction_set:
        out += str(el) + " "
    return out[:-1]  # remove final space


def corr_ans_text_fn(label_text, reindexed_label):
    out = f"The best answer is {reindexed_label}. {label_text.title()}."
    return out


def prep_conf_set(conformal_set):
    conf_set = [arr.tolist() for arr in conformal_set]
    for lst in conf_set:
        lst.sort()
        if (len(lst) > 0) and lst[0] == 0:
            lst.append(lst.pop(0))  # Move 0 to end for display purposes
    
    return conf_set


def set_to_string(conf_set, i):
    cp_set_str = ""
    for lab in conf_set[i]:
        cp_set_str += f"{lab} "
    
    return cp_set_str


def data_prep_to_generate_csv(
    logits,
    set_avgk,
    conformal_set_marg,
    conformal_set_cond,
    k,
    input_identifiers,
    group_label,
    y
    ):
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    columns = ["prompt", "label", "group", "top1", "topk_set", "avgk_set",
     "conformal_marginal_set", "conformal_conditional_set"]

    _, indices = logits.topk(k, 1, largest=True, sorted=True)
    indices = indices.cpu().numpy()
    top1 = indices[:, :1]
    # We are not providing information on logit values or ordering
    topk_set = indices.tolist()
    for lst in topk_set:
        lst.sort()
        if lst[0] == 0:
            lst.append(lst.pop(0))  # Move 0 to end for display purposes
    set_avgk = prep_conf_set(set_avgk)
    conformal_set_marg = prep_conf_set(conformal_set_marg)
    conformal_set_cond = prep_conf_set(conformal_set_cond)

    batch_dict = {}

    for i, y_i in enumerate(y):
        topk_str = ""
        for idx in topk_set[i]:
            topk_str += f"{idx} "
        avgk_set_str = set_to_string(set_avgk, i)
        cp_set_str_marg = set_to_string(conformal_set_marg, i)
        cp_set_str_cond = set_to_string(conformal_set_cond, i)
        batch_dict[i] = [input_identifiers[i], y_i, group_label[i], top1[i][0], topk_str, avgk_set_str, cp_set_str_marg,
                         cp_set_str_cond]

    df_formatted = pd.DataFrame.from_dict(batch_dict, orient="index", columns=columns)

    return df_formatted


def format_and_write_to_csv(df, writer, args, cfg):
    cols = [
        "prompt",
        "label_text",
        "label",
        "original_label",
        "group",
        "group_text",
        "top1",
        "topk_set",
        "avgk_set",
        "conformal_marginal_set",
        "conformal_conditional_set",
        "corr_ans_text",
        "topk_text",
        "avgk_text",
        "conformal_marginal_text",
        "conformal_conditional_text",
    ]

    if args.dataset == "fashion-mnist":
        cols.pop(0)
    if args.dataset == "few-nerd":
        cols.extend(["original_label_text", "prompt_with_original_fine_ner_tags"])
    if args.dataset == "bios":
        cols.extend(["prompt_original"]) # Unshortened prompt

    df = df[cols]
    df.index.name = "idx"

    dataset = cfg["dataset"]
    writer.write_pandas(f"{dataset}", df)

    return None
