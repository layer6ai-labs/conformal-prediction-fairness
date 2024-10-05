import os
import torchaudio
from collections import OrderedDict
from transformers import Wav2Vec2FeatureExtractor # Wav2Vec2Processor, AutoProcessor
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from datasets import Dataset
from networks import RAVDESSModel
from utils import ConformalCategory
from data_sets.dataset_utils import *
from data_sets.data_sets.abstract_dataset import CustomDataset


class RAVDESS(CustomDataset):
    
    # Inner class is huggingface Dataset
    class RAVDESS_Dataset(Dataset):
        def __init__(self, audio_processor, classnames, classes, annotation):
            self.group_conformal_category = ConformalCategory.GROUP_BALANCED
            self.annotation = annotation
            self.classnames = classnames
            self.classes = classes
            self.audio_processor = audio_processor

        def __len__(self):
            return len(self.annotation)

        def __getitem__(self, index):
            ann = self.annotation[index]

            audio_path = ann["audio_path"]
            waveform, sample_rate = torchaudio.load(audio_path)

            # If the audio has more than one channel, convert it to mono
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            input_values = self.audio_processor(waveform, sample_rate)

            return {"input_values": input_values,
                    "label": ann["label"],
                    "group": ann["group"],
                    "audio_path": audio_path,
            }
        
        def custom_collate_fn(batch):
            input_values = [item['input_values'].squeeze() for item in batch]
            labels = [item['label'] for item in batch]
            groups = [item['group'] for item in batch]
            input_data = [item['audio_path'] for item in batch]

            # Pad sequences to the maximum length in the batch
            input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=0,)

            return {
                'input_values': input_values_padded,
                'labels': labels,
                'groups': groups,
                'audio_paths': input_data
            }

 
    def __init__(self):
        self.uses_top_m_labels = False
        self.group_conformal_category = ConformalCategory.GROUP_BALANCED

    def _fetch_sample_name_to_label_map(self):
        #In the RAVDESS dataset, the two digits of the file name represent the emotion label
        return OrderedDict([
            ("01", "Neutral"),
            ("02", "Calm"),
            ("03", "Happy"),
            ("04", "Sad"),
            ("05", "Angry"),
            ("06", "Fearful"),
            ("07", "Disgust"),
            ("08", "Surprised")
        ])


    def _fetch_classnames(self):
        # below classnames are in the order of the labels model was trained on
        return ['Angry', 'Calm', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    
    def get_id2label(self, id=None, return_dict=False, start_index_one=False):
        id2label = {
            0: "Neutral",
            1: "Calm",
            2: "Happy",
            3: "Sad",
            4: "Angry",
            5: "Fearful",
            6: "Disgust",
            7: "Surprised"
        }
        if start_index_one:
            id2label = {k+1: v for k, v in id2label.items()}
        if return_dict:
            return id2label
        return id2label.get(id, "unknown")
    
    def get_label2id(self, label=None, return_dict=False):
        label2id = {
            "Angry": 4,
            "Calm": 1,
            "Disgust": 6,
            "Fearful": 5,
            "Happy": 2,
            "Neutral": 0,
            "Sad": 3,
            "Surprised": 7
        }
        if return_dict:
            return label2id
        return label2id.get(label, "unknown")
    
    def get_reordered_label(self, label=None, return_dict=False):
        reindexed_label = {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 7,
            7: 8
        }
        if return_dict:
            return reindexed_label
        return reindexed_label.get(label, "unknown")
    
    def get_id2group(self, id=None, return_dict=False):
        id2group = {
            0: "Male",
            1: "Female"
        }
        if return_dict:
            return id2group
        return id2group.get(id, "unknown")
    
    def get_group2id(self, group=None, return_dict=False):
        group2id = {
            "Male": 0,
            "Female": 1
        }
        if return_dict:
            return group2id
        return group2id.get(group, "unknown")

    def _preprocess_audio(self, waveform, sample_rate):
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("Wiam/wav2vec2-large-xlsr-53-english-finetuned-ravdess-v5")
        
        # Resample the audio to 16kHz since Wave2Vec is trained on that
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        input_values = feature_extractor(waveform, sampling_rate=16000, padding=True, return_tensors="pt").input_values.float()
        return input_values


    def get_data(
        self,
        data_root,
        test_batch_size=256,
        calib_val_batch_size=256,
        calib_batch_size=256,
        n_calib=2000,
        n_calib_val=2000,
        n_test=-1,
        *args,
        **kwargs
    ):
        
        train_loader, val_loader = None, None  # Training not required

        RAVDESS_MAX = 1440
        if n_calib + n_test + n_calib_val > RAVDESS_MAX:
            raise ValueError(
                (f"Not enough data for requested calibration size. n_calib: {n_calib}, "
                 f"n_calib_val: {n_calib_val}, validation_size: {RAVDESS_MAX}")
            )
    
        calib_subset, calib_val_subset, test_subset, classnames = self.load_and_split_ravdess(
            data_root, n_calib, n_test, n_calib_val
        )

        for dataset in [calib_subset, test_subset, calib_val_subset]:
            dataset.classes = [i for i in range(len(classnames))]
        
        calib_loader = get_loader(
            calib_subset, calib_batch_size, shuffle=False, drop_last=False, collate_fn=self.RAVDESS_Dataset.custom_collate_fn
        )
        calib_val_loader = get_loader(
            calib_val_subset, calib_val_batch_size, shuffle=False, drop_last=False, collate_fn=self.RAVDESS_Dataset.custom_collate_fn
        )
        test_loader = get_loader(
            test_subset, test_batch_size, shuffle=False, drop_last=False, collate_fn=self.RAVDESS_Dataset.custom_collate_fn
        )
        print(
            f"Dataset sizes: Calib {len(calib_loader.dataset)}, Calib Val {len(calib_val_loader.dataset)}, Test {len(test_loader.dataset)}."
        )

        if n_calib_val > 0:
            return {
                "calib": calib_loader,
                "test": test_loader,
                "calib_val": calib_val_loader,
                "train": train_loader,
                "val": val_loader,
                "labels": classnames,
            }
        else:
            return {
                "calib": calib_loader,
                "test": test_loader,
                "train": train_loader,
                "val": val_loader,
                "labels": classnames,
            }


    def load_and_split_ravdess(self, audio_root, n_calib, n_test, n_calib_val):
        
        label_mapper = self._fetch_sample_name_to_label_map()
        group_mapper = {str(i).zfill(2): "Male" if i % 2 != 0 else "Female" for i in range(1, 25)}

        pd_list = []

        for root, _, files in os.walk(audio_root):
            for file in files:
                label = label_mapper[file.split("-")[2]]
                group = group_mapper[file.split("-")[-1][:2]]
                pd_list.append({
                    "audio_path": os.path.join(root, file),
                    "label": label,
                    "group": group,
                    "label_group": label + '_' + group,
                })
        df = pd.DataFrame(pd_list)

        classnames = self._fetch_classnames()
        m = len(classnames)

        calib_subset_df, d_conf_remain = train_test_split(df, test_size=(n_calib_val+n_test), stratify=df['label_group'])
        new_ratio = n_calib_val / (n_calib_val+n_test)
        test_subset_df, calib_val_subset_df = train_test_split(d_conf_remain, test_size=(new_ratio), stratify=d_conf_remain['label_group'])

        calib_annotation = []
        test_annotation = []
        calib_val_annotation = []

        for index, row in calib_subset_df.iterrows():
            calib_annotation.append(
            {"audio_path": row["audio_path"], "label": self.get_label2id(row["label"]), "group": self.get_group2id(row["group"])}
            )
        for index, row in test_subset_df.iterrows():
            test_annotation.append(
            {"audio_path": row["audio_path"], "label": self.get_label2id(row["label"]), "group": self.get_group2id(row["group"])}
            )
        for index, row in calib_val_subset_df.iterrows():
            calib_val_annotation.append(
            {"audio_path": row["audio_path"], "label": self.get_label2id(row["label"]), "group": self.get_group2id(row["group"])}
            )
        classes = [i for i in range(m)]
        audio_processor = self._preprocess_audio
        calib_dataset = self.RAVDESS_Dataset(audio_processor, classnames, classes, calib_annotation)
        calib_val_dataset = self.RAVDESS_Dataset(audio_processor, classnames, classes, calib_val_annotation)
        test_dataset = self.RAVDESS_Dataset(audio_processor, classnames, classes, test_annotation)

        # Downstream processing expects a Subset object
        calib_subset = torch.utils.data.Subset(
            calib_dataset, torch.randperm(len(calib_dataset)).tolist()
        )
        calib_val_subset = torch.utils.data.Subset(
            calib_val_dataset, torch.randperm(len(calib_val_dataset)).tolist()
        )
        test_subset = torch.utils.data.Subset(
            test_dataset, torch.randperm(len(test_dataset)).tolist()
        )

        return calib_subset, calib_val_subset, test_subset, classnames


    def get_model(self, device, train_loader, val_loader, **kwargs):
        model = RAVDESSModel(kwargs["used_labels"]).to(device)

        return model


    def prepare_model_inputs(self, data, device):
        x = data["input_values"].to(device)
        target = torch.tensor(data["labels"]).to(device)
        group = torch.tensor(data["groups"]).to(device)
        input_data = data["audio_paths"]

        return x, target, group, input_data


    def process_dataframe(self, df, loader_dict, k):
        
        id2label_reordered = self.get_id2label(return_dict=True, start_index_one=True)
        id2group = self.get_id2group(return_dict=True)
        reordered_labels = self.get_reordered_label(return_dict=True)
        
        m = len(loader_dict['labels'])

        df["original_label"] = df["label"] # Keeping original label for reference
        df["label"] = df["label"].apply(lambda label: reordered_labels[label])

        # Reindex labels to start from 1
        df["conformal_marginal_set"] = df["conformal_marginal_set"].apply(relabel_set_obj, label_reordering=reordered_labels)
        df["conformal_conditional_set"] = df["conformal_conditional_set"].apply(relabel_set_obj, label_reordering=reordered_labels)
        df["topk_set"] = df["topk_set"].apply(relabel_set_obj, label_reordering=reordered_labels)
        df["avgk_set"] = df["avgk_set"].apply(relabel_set_obj, label_reordering=reordered_labels)
        df["top1"] = df["top1"].apply(relabel_set_obj, label_reordering=reordered_labels)
        

        df["label_text"] = df["label"].apply(lambda label: id2label_reordered[label])
        df['group_text'] = df['group'].apply(lambda group: id2group[group])
        df["corr_ans_text"] = df.apply(
            lambda x: corr_ans_text_fn(x["label_text"], x["label"]), axis=1
        )
        df["topk_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["topk_set"], id2label_reordered), axis=1
        )
        df["avgk_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["avgk_set"], id2label_reordered), axis=1
        )
        df["conformal_marginal_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["conformal_marginal_set"], id2label_reordered),
            axis=1,
        )
        df["conformal_conditional_text"] = df.apply(
            lambda x: prediction_set_text_fn(
                x["conformal_conditional_set"], id2label_reordered
            ),
            axis=1,
        )
        # Sort dataframe, then put example instances from each class at the top
        df = bring_ravdess_examples_to_top(df, repeat=3)

        return df
