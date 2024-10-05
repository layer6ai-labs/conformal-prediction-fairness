import pickle
import random
import re
import torch
import datasets
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from collections import Counter
from datasets import Dataset
from unidecode import unidecode
from nltk.tokenize import sent_tokenize 
from data_sets.dataset_utils import *
from data_sets.data_sets.abstract_dataset import CustomDataset
from networks import BiosBiasModel
from utils import ConformalCategory


datasets.utils.logging.disable_progress_bar()


class BiosBias(CustomDataset):

    def __init__(self):
        self.uses_top_m_labels = True
        self.group_conformal_category = ConformalCategory.GROUP_BALANCED

    def get_data(
        self,
        data_root,
        train_batch_size=256,
        test_batch_size=256,
        calib_batch_size=256,
        n_calib=5000,
        n_test=2000,
        n_calib_val=5000,
        n_val=5000,
        n_train=50000,
        m=10,
        **kwargs
    ):
        print("Loading data")

        with open(f"{data_root}train_all.pickle", "rb") as f:
            train_data = pickle.load(f)

        with open(f"{data_root}dev_all.pickle", "rb") as f:
            val_data = pickle.load(f)

        with open(f"{data_root}test_all.pickle", "rb") as f:
            test_data = pickle.load(f)

        dataset_dict = {"train": train_data, "validation": val_data, "test": test_data}

        # filter on single label and most popular m:
        dataset_dict, top_m_labels = self.filter_bios(dataset_dict, m=m)

        print(
            (f"Dataset sizes after filtering: Train {len(dataset_dict['train'])}, "
             f"Validation {len(dataset_dict['validation'])}, Test {len(dataset_dict['test'])}.")
        )

        d_conf_df = dataset_dict["test"]
        n_conf = len(d_conf_df)
        if n_calib + n_test + n_calib_val > n_conf:
            raise ValueError(
                (f"Not enough data for requested calibration size. n_calib: {n_calib}, "
                 f"n_calib_val: {n_calib_val}, validation_size: {n_conf}")
            )
        print(f'd_conf: {len(d_conf_df)}')
        calib_subset_df, d_conf_remain = stratified_sample_df(
            d_conf_df, col="label", n_samples=n_calib//m,
            return_remaining=True
        )
        print(f'd_conf_remain: {len(d_conf_remain)}')
        test_subset_df, d_conf_remain = stratified_sample_df(
            d_conf_remain, col="label", n_samples=n_test//m,
            return_remaining=True
        )
        print(f'd_conf_remain: {len(d_conf_remain)}')
        calib_val_subset_df, _ = stratified_sample_df(
            d_conf_remain, col="label", n_samples=n_calib_val//m,
            return_remaining=True
        )
        
        calib_subset = Dataset.from_pandas(calib_subset_df)
        test_subset = Dataset.from_pandas(test_subset_df)
        calib_val_subset = Dataset.from_pandas(calib_val_subset_df)

        train_subset = Dataset.from_pandas(stratified_sample_df(
                dataset_dict["train"], col="label", n_samples=n_train//m
            ))
        
        val_subset = Dataset.from_pandas(stratified_sample_df(
                dataset_dict["validation"], col="label", n_samples=n_val//m
            ))


        for dataset in [train_subset, val_subset, calib_subset, test_subset, calib_val_subset]:
            dataset.classes = [i for i in range(m)]

        id2label_original = self.__get_original_id2label()
        self.label_map = {x: id2label_original[top_m_labels[x]] for x in range(m)}
        
        calib_loader = get_loader(
            calib_subset, calib_batch_size, shuffle=False, drop_last=False
        )
        test_loader = get_loader(
            test_subset, test_batch_size, shuffle=False, drop_last=False
        )
        calib_val_loader = get_loader(
            calib_val_subset, calib_batch_size, shuffle=False, drop_last=False
        )

        train_loader = get_loader(
            train_subset, train_batch_size, shuffle=True, drop_last=False
        )

        val_loader = get_loader(
            val_subset, test_batch_size, shuffle=False, drop_last=False
        )

        print(
            f"Dataset sizes: Train {len(train_loader.dataset)}, Validation {len(val_loader.dataset)}, "
            + f"Calib {len(calib_loader.dataset)}, Test {len(test_loader.dataset)}, "
            + f"Calib_val {len(calib_val_loader.dataset)}."
        )

        return {
            "calib": calib_loader,
            "test": test_loader,
            "calib_val": calib_val_loader,
            "train": train_loader,
            "val": val_loader,
            "top_m_labels": top_m_labels,
        }
    
    def __get_original_id2label(self):
        id2label = {
            0: "Accountant",
            1: "Architect",
            2: "Attorney",
            3: "Chiropractor",
            4: "Comedian",
            5: "Composer",
            6: "Dentist",
            7: "Dietitian",
            8: "DJ",
            9: "Filmmaker",
            10: "Interior Designer",
            11: "Journalist",
            12: "Model",
            13: "Nurse",
            14: "Painter",
            15: "Paralegal",
            16: "Pastor",
            17: "Personal Trainer",
            18: "Photographer",
            19: "Physician",
            20: "Poet",
            21: "Professor",
            22: "Psychologist",
            23: "Rapper",
            24: "Software Engineer",
            25: "Surgeon",
            26: "Teacher",
            27: "Yoga Teacher",
        }
        return id2label

    def get_id2label(self, id=None, return_dict=False):
        if return_dict:
            return self.label_map
        return self.label_map.get(id, None)
    
    def get_id2group(self, id=None, return_dict=False):
        group_map = {0:'Female', 1:"Male"}
        if return_dict:
            return group_map
        return group_map.get(id, None)
    
    def subsample_bios(self, dataset, n, m):
        num_samples_per_group = int(n / m)
        random_ids = []
        for i in range(m):
            unique_ids = set(
                dataset.filter(lambda x: x["label"] == i)['__index_level_0__']
            )
            random_ids.extend(random.sample(unique_ids, num_samples_per_group))
        random_indices = [
            index
            for index, element in enumerate(dataset["__index_level_0__"])
            if element in random_ids
        ]
        return dataset.select(random_indices), random_indices

    def filter_bios(self, datasets, m=10):
        # remove architect, attorney classes
        # remove nurse class because it has hundreds of near-duplicate bios
        for split in datasets:
            datasets[split] = list(filter(lambda x: x["label"] not in [1, 2, 13], datasets[split]))
        # only keeps samples with the top m popular labels
        label_counts = Counter([d["label"] for d in datasets["train"]])
        top_m_labels = sorted(label_counts, key=label_counts.get, reverse=True)[:m]
        print(label_counts)
        print(sorted(label_counts, key=label_counts.get, reverse=True))
        print(f"Top {m} labels: {top_m_labels}")

        for split in datasets:
            datasets[split] = self.filter_and_remap_labels(
                datasets[split], top_m_labels
            )

        # keep an equal number of each of the classes using stratified sampling
        for split in datasets:
            # get the number of examples in the least common class
            split_label_counts = Counter([d["label"] for d in datasets[split]])
            min_num = split_label_counts[
                min(split_label_counts, key=split_label_counts.get)
            ]
            datasets[split] = pd.DataFrame(datasets[split])
            pd_dataset = datasets[split][:]
            pd_dataset = stratified_sample_df(
                pd_dataset, col="label", n_samples=min_num
            )
            pd_dataset['groups'] = pd_dataset['g'].apply(lambda x: {"f": 0, "m": 1}[x])
            datasets[split] = pd_dataset

        return datasets, top_m_labels

    def filter_and_remap_labels(self, dataset, top_m_labels):
        dataset = list(filter(lambda x: x["label"] in top_m_labels, dataset))

        def replace_labels(example):
            example["label"] = top_m_labels.index(example["label"])
            return example

        dataset = list(map(replace_labels, dataset))
        return dataset

    def get_model(self, device, train_loader, val_loader, **kwargs):
        used_labels = kwargs["used_labels"]
        model = BiosBiasModel(num_classes=len(used_labels)).to(device)
        if kwargs['model_checkpoint'] is not None:
            print(f"Loading model from {kwargs['model_checkpoint']}")
            model.load_state_dict(torch.load(kwargs['model_checkpoint']))
            return model
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = kwargs["optimizer"].lower()
        lr = kwargs["lr"]
        if optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer {optimizer} not implemented.")

        for epoch in range(kwargs["epochs"]):
            running_loss = 0.0
            correct = 0
            total = 0
            for idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = torch.stack(data["avg_enc"], dim=1).to(torch.float32)
                labels = data["label"]
                outputs = model(inputs.to(device))
                total += labels.size(0)
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                pred = outputs.data.cpu().argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                running_loss += loss.item()
            if epoch % 1 == 0:
                print(
                    f"Epoch {epoch + 1},  loss: {running_loss / len(train_loader):.3f}, accuracy: {100 * correct // total} %"
                )

        print("Finished Training")

        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs = torch.stack(data["avg_enc"], dim=1).to(torch.float32)
                labels = data["label"]
                outputs = model(inputs.to(device))
                _, predicted = torch.max(outputs.data.cpu(), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the network on the validation bios: {100 * correct // total} %"
        )

        return model

    def prepare_model_inputs(self, data, device):
        target = data["label"].to(device)
        group = data["groups"].to(device)
        input_data = data["hard_text"]
        input = torch.stack(data["avg_enc"], dim=1).to(torch.float32).to(device)

        return input, target, group, input_data

    def clean_prompt(self, input_text):
        # remove \x00 characters for csv
        result = input_text.replace('\x00', '').strip()
        # remove escape characters
        result = result.replace(' / ', ' ')
        result = result.replace(' - ', '-')
        # convert non-ascii characters
        result = unidecode(result)
        # remove spaces after parenthesis and quotes
        pattern_quotes_parentheses = r'([`"\'\'(])\s*'
        result = re.sub(pattern_quotes_parentheses, r"\1", result)

        # remove phone numbers
        pattern_phone = r'\(?\d{3}\)?[-\s\.]+\d{3}[-\s\.]+\d{4}'
        result = re.sub(pattern_phone, '[Number]', result)

        # remove emails
        pattern_emails = r'\S+@\S+\.\S+'
        result = re.sub(pattern_emails, '[Email]', result)

        # removeu urls
        pattern_urls = r'((http?:|www.)\S+\.\S+)'# remove urls with http(s): or www.
        result = re.sub(pattern_urls, '[URL]', result)

        # remove spaces before punctuation etc
        # input_text = input_text.replace("``",'')
        pattern = r"\s*([`.,;()\'\[\]])"
        result = re.sub(pattern, r"\1", result)

        return result

    def shorten_prompt(self, input_text, max_characters=400):
        if len(input_text) < max_characters:
            return input_text
        sentences = sent_tokenize(input_text)
        res = ''
        for sentence in sentences:
            if len(res) + len(sentence) < max_characters:
                res += sentence
            else:
                break
        if len(res) == 0:
            # If the first sentence is too long,  cut it off and add ...
            res = sentences[0][:max_characters]
            res = res + '...'
        return res

    def remove_brackets(self, prompt):
        if "[[" in prompt and "]]" in prompt:
            # Remove double brackets from the string
            prompt = prompt.replace("[[", "").replace("]]", "")
        if "[" in prompt and "]" in prompt:
            # Remove brackets from the string
            prompt = prompt.replace("[", "").replace("]", "")
        return prompt

    def process_dataframe(self, df, loader_dict, k):
        # modify the input data to display on psychopy

        id2label = self.__get_original_id2label()
        df['prompt_original'] = df['prompt'].apply(lambda x: self.clean_prompt(x))
        df['prompt'] = df['prompt_original'].apply(lambda x: self.shorten_prompt(x))
        df['prompt'] = df['prompt'].apply(lambda x: self.remove_brackets(x))
        top_m_labels = loader_dict["top_m_labels"]
        m = len(top_m_labels)
        id2label_ge = {x: id2label[top_m_labels[x]] for x in range(m)}

        df["label_text"] = df["label"].apply(lambda label: id2label_ge[label])
        df["original_label"] = df["label"].apply(lambda label: top_m_labels[label])

        group_to_text = {0:'Female', 1:"Male"}
        df["group_text"] = df["group"].apply(lambda group: group_to_text[group])

        df["corr_ans_text"] = df.apply(
            lambda x: corr_ans_text_fn(x["label_text"], x["label"]), axis=1
        )
        df["topk_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["topk_set"], id2label_ge), axis=1
        )
        df["avgk_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["avgk_set"], id2label_ge), axis=1
        )
        df["conformal_marginal_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["conformal_marginal_set"], id2label_ge),
            axis=1,
        )
        df["conformal_conditional_text"] = df.apply(
            lambda x: prediction_set_text_fn(
                x["conformal_conditional_set"], id2label_ge
            ),
            axis=1,
        )
        # Sort dataframe, then put example instances from each class at the top
        min_num = df["label"].value_counts().min()
        df = bring_examples_to_top(df, m, min_num)

        return df
