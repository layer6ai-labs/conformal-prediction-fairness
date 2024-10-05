import os
import torchvision
from PIL import Image

from datasets import Dataset
from data_sets.dataset_utils import *
from data_sets.data_sets.abstract_dataset import CustomDataset
from networks import FACETModel
from utils import ConformalCategory


class FACET(CustomDataset):
    # Inner class is huggingface Dataset
    class FACET(Dataset):
        def __init__(self, vis_processor, classnames, classes, annotation):
            self.vis_processor = vis_processor
            self.classnames = classnames
            self.classes = classes
            self.annotation = annotation
            self.group_conformal_category = ConformalCategory.GROUP_BALANCED

        def __len__(self):
            return len(self.annotation)

        def __getitem__(self, index):
            ann = self.annotation[index]

            image_path = ann["image"]
            image = Image.open(image_path).convert("RGB")

            image = self.vis_processor(image)

            return {"image": image,
                    "label": ann["label"],
                    "groups": ann["group"],
                    "image_id": image_path,
                    }

    def __init__(self):
        self.uses_top_m_labels = True
        self.group_conformal_category = ConformalCategory.GROUP_BALANCED

    def get_data(
        self,
        data_root,
        calib_batch_size=256,
        calib_val_batch_size=256,
        test_batch_size=256,
        n_calib=1000,
        n_calib_val=-1,
        n_test=-1,
        m=20,
        **kwargs
    ):

        print("Loading data")

        train_loader, val_loader = None, None  # Training not required

        calib_subset, calib_val_subset, test_subset, classnames = self.create_and_split_facet(
            data_root, n_calib, m
        )

        calib_loader = get_loader(
            calib_subset, calib_batch_size, shuffle=False, drop_last=False
        )
        calib_val_loader = get_loader(
            calib_val_subset, calib_val_batch_size, shuffle=False, drop_last=False
        )
        test_loader = get_loader(
            test_subset, test_batch_size, shuffle=False, drop_last=False
        )
        print(
            f"Dataset sizes: Calib {len(calib_loader.dataset)}, Calib Val {len(calib_val_loader.dataset)}, Test {len(test_loader.dataset)}."
        )

        return {
            "train": train_loader,
            "val": val_loader,
            "calib": calib_loader,
            "calib_val": calib_val_loader,
            "test": test_loader,
            "top_m_labels": classnames,
        }

    def create_and_split_facet(self, vis_root, n_calib, m=20):

        vis_processor = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.PILToTensor(),
            ]
        )
        inner_dataset = torchvision.datasets.ImageFolder(os.path.join(vis_root, "images"))
        annotations_df = pd.read_csv(os.path.join(vis_root, "annotations/annotations.csv"))

        # Only keep images with single person
        nb_person_per_image = annotations_df.groupby(['filename'])['person_id'].count().reset_index()
        nb_person_per_image = nb_person_per_image.rename(columns={'person_id': 'nb_person'})
        annotations_df_nb_person = annotations_df.merge(nb_person_per_image, left_on='filename', right_on='filename')
        single_person_images = annotations_df_nb_person.loc[annotations_df_nb_person.nb_person == 1]

        # Only keep images with single label
        filtered_annotations_df = single_person_images.loc[single_person_images['class2'].isnull()].reset_index()

        # Concatenate groups into a single feature
        age_info = filtered_annotations_df.loc[:, filtered_annotations_df.columns.str.startswith('age_')]
        filtered_annotations_df['age_groups'] = age_info.idxmax(axis=1)
        groupnames = ['age_presentation_young', 'age_presentation_middle', 'age_presentation_older',
                     'age_presentation_na']
        self.group_map = {group: group_text for group, group_text in enumerate(groupnames)}

        # Only keep top m most common classes
        label_counts = dict(filtered_annotations_df['class1'].value_counts())
        top_m_labels = list(label_counts.keys())[:m]

        min_num = label_counts[top_m_labels[-1]]
        size = m * min_num

        classnames = top_m_labels
        classnames.sort()  # alphabetical order
        classes = [i for i in range(len(classnames))]
        self.label_map = {label:label_text for label, label_text in enumerate(classnames)}

        if n_calib <= 0:
            raise ValueError(
                f"Invalid choice of n_calib: {n_calib}. Can't use the entire dataset for FACET calibration."
            )
        elif n_calib > size:
            raise ValueError(
                f"Not enough data for requested calibration size. n_calib: {n_calib}, dataset_size: {size}"
            )
        elif 0 < n_calib < 1:
            # Take fraction of the dataset
            calib_frac = n_calib
            n_calib = int(n_calib * size)
            n_calib = n_calib - (n_calib % m)  # ensure that dataset will be class balanced
            print(
                f"Using {calib_frac * 100}% of the dataset for calibration, {n_calib} datapoints."
            )
        else:
            n_calib = n_calib - (n_calib % m)  # ensure that dataset will be class balanced

        min_num_calib = int(n_calib / m)  # Number of each class for calib
        min_num_calib_val = (min_num - min_num_calib) // 2
        min_num_test = (min_num - min_num_calib) // 2

        # Ensure that calibration, val and test sets are stratified the same way (exchangable)
        num_examples_per_class_calib = [0 for i in range(len(classnames))]
        num_examples_per_class_calib_val = [0 for i in range(len(classnames))]
        num_examples_per_class_test = [0 for i in range(len(classnames))]
        calib_annotation = []
        calib_val_annotation = []
        test_annotation = []

        # stratified sampling, e.g. load only min_num_calib examples per class for calib
        for path, _ in inner_dataset.imgs:
            # Check if image is part of filtered data and get its label
            label = filtered_annotations_df.loc[filtered_annotations_df.filename == path[25:], 'class1'].values  #TODO: remove hardcoded [25:]
            age_group = filtered_annotations_df.loc[filtered_annotations_df.filename == path[25:], 'age_groups'].values

            if label.size > 0:
                if label in classnames:
                    # Get classname and group indices
                    class_idx = classnames.index(label)
                    group_idx = groupnames.index(age_group)

                    if num_examples_per_class_calib[class_idx] < min_num_calib:
                        num_examples_per_class_calib[class_idx] += 1
                        calib_annotation.append(
                            {"image": path,
                             "label": class_idx,
                             "group": group_idx,
                             }
                        )
                    elif num_examples_per_class_calib_val[class_idx] < min_num_calib_val:
                        num_examples_per_class_calib_val[class_idx] += 1
                        calib_val_annotation.append(
                            {"image": path,
                             "label": class_idx,
                             "group": group_idx,
                             }
                        )
                    elif num_examples_per_class_test[class_idx] < min_num_test:
                        num_examples_per_class_test[class_idx] += 1
                        test_annotation.append(
                            {"image": path,
                             "label": class_idx,
                             "group": group_idx,
                             }
                        )

        calib_dataset = self.FACET(vis_processor, classnames, classes, calib_annotation)
        calib_val_dataset = self.FACET(vis_processor, classnames, classes, calib_val_annotation)
        test_dataset = self.FACET(vis_processor, classnames, classes, test_annotation)

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

    def get_id2label(self, id=None, return_dict=False):
        if return_dict:
            return self.label_map
        return self.label_map.get(id, None)
    
    def get_id2group(self, id=None, return_dict=False):
        if return_dict:
            return self.group_map
        return self.group_map.get(id, None)

    def get_model(self, device, train_loader, val_loader, **kwargs):
        model = FACETModel(device, kwargs["used_labels"], kwargs['model_size'])

        return model


    def prepare_model_inputs(self, data, device):
        x = data["image"].to(device)
        target = torch.as_tensor(data["label"]).to(device)
        group = torch.as_tensor(data["groups"]).to(device)
        input_data = data["image_id"]

        return x, target, group, input_data


    def process_dataframe(self, df, loader_dict, k):
        classnames = loader_dict["top_m_labels"]
        id2label_fa = {i: classnames[i] for i in range(len(classnames))}
        m = len(classnames)
        df["original_label"] = df["label"]
        if m == 20:  # Original label -> shift by one
            label_reordering = {i: i+1 for i in range(m)}
            df["label"] = df["label"].apply(lambda label: label_reordering[label])
        df["conformal_marginal_set"] = df["conformal_marginal_set"].apply(relabel_set_obj, label_reordering=label_reordering)
        df["conformal_conditional_set"] = df["conformal_conditional_set"].apply(relabel_set_obj, label_reordering=label_reordering)
        df["topk_set"] = df["topk_set"].apply(relabel_set_obj, label_reordering=label_reordering)
        df["avgk_set"] = df["avgk_set"].apply(relabel_set_obj, label_reordering=label_reordering)
        df["top1"] = df["top1"].apply(relabel_set_obj, label_reordering=label_reordering)
        id2label_fa = {
                1: "Backpacker",
                2: "Boatman",
                3: "Computer User",
                4: "Craftsman",
                5: "Farmer",
                6: "Guard",
                7: "Guitarist",
                8: "Gymnast",
                9: "Hairdresser",
                10: "Horse Rider",
                11: "Laborer",
                12: "Officer",
                13: "Motorcyclist",
                14: "Painter",
                15: "Repairman",
                16: "Salesperson",
                17: "Singer",
                18: "Skateboarder",
                19: "Speaker",
                20: "Tennis Player",
            }
        df["label_text"] = df["label"].apply(lambda label: id2label_fa[label])

        group_to_text = {0: 'Younger',
                         1: 'Middle',
                         2: 'Older',
                         3: 'Unknown',
                         }
        df["group_text"] = df["group"].apply(lambda group: group_to_text[group])

        df["corr_ans_text"] = df.apply(
            lambda x: corr_ans_text_fn(x["label_text"], x["label"]), axis=1
        )
        df["topk_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["topk_set"], id2label_fa), axis=1
        )
        df["avgk_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["avgk_set"], id2label_fa), axis=1
        )
        df["conformal_marginal_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["conformal_marginal_set"], id2label_fa),
            axis=1,
        )
        df["conformal_conditional_text"] = df.apply(
            lambda x: prediction_set_text_fn(
                x["conformal_conditional_set"], id2label_fa
            ),
            axis=1,
        )

        # Sort dataframe, then put example instances from each class at the top
        min_num = df["label"].value_counts().min()
        df = bring_examples_to_top(df, m, min_num)

        return df
