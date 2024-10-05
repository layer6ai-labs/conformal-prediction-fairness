import torch
import torch.nn as nn
import torchvision

from data_sets.dataset_utils import *
from data_sets.data_sets.abstract_dataset import CustomDataset
from networks import ConvNet

from utils import ConformalCategory

class Fashion_MNIST(CustomDataset):

    def __init__(self):
        self.uses_top_m_labels = False
        self.group_conformal_category = ConformalCategory.CLASS_CONDITIONAL

    def get_data(
            self,
            data_root,
            train_batch_size=256,
            valid_batch_size=256,
            test_batch_size=256,
            calib_batch_size=256,
            n_calib=2000,
            n_test=8000,
            m=10,
            **kwargs
            ):
        train_data = torchvision.datasets.FashionMNIST(
            root=data_root,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        test_data = torchvision.datasets.FashionMNIST(
            root=data_root,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        n_val = int(0.1 * len(train_data)) # FMNIST is pre-shuffled
        val_targets = train_data.targets[:n_val] # Groups are defined as class labels
        val_dset = GroupDataset("val", self.transform(train_data.data[:n_val]), val_targets, val_targets)
        train_targets = train_data.targets[n_val:]
        train_dset = GroupDataset("train", self.transform(train_data.data[n_val:]), train_targets, train_targets)

        calib_targets = test_data.targets[:n_calib]
        calib_dset = GroupDataset("calib", self.transform(test_data.data[:n_calib]), calib_targets, calib_targets)
        if n_test > len(test_data) - n_calib:
            raise ValueError(
                f"Not enough data for requested calibration size. n_calib: {n_calib}, n_test: {n_test}"
            )
        elif n_test <= 0: # Use all remaining test data
            test_x = test_data.data[n_calib:]
            test_targets = test_data.targets[n_calib:]
        else: # Use exactly n_test datapoints
            test_x = test_data.data[n_calib:n_test+n_calib]
            test_targets = test_data.targets[n_calib:n_test+n_calib]
        test_dset = GroupDataset("test", self.transform(test_x), test_targets, test_targets)

        train_loader = get_loader(
            train_dset, train_batch_size, shuffle=True, drop_last=True
        )
        val_loader = get_loader(
            val_dset, valid_batch_size, shuffle=False, drop_last=False
        )
        calib_loader = get_loader(
            calib_dset, calib_batch_size, shuffle=False, drop_last=False
        )
        test_loader = get_loader(
            test_dset, test_batch_size, shuffle=False, drop_last=False
        )

        print(
            f"Dataset sizes: Train {len(train_loader.dataset)}, Val {len(val_loader.dataset)}, Calib {len(calib_loader.dataset)}, Test {len(test_loader.dataset)}."
        )

        return {
            "calib": calib_loader,
            "test": test_loader,
            "train": train_loader,
            "val": val_loader,
        }


    def get_id2label(self, id=None, return_dict=False):
        label_map = {
            0: "T-Shirt",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot",
        }
        if return_dict:
            return label_map
        return label_map.get(id, None)
    
    def get_id2group(self, id=None, return_dict=False):
        if self.group_conformal_category == ConformalCategory.CLASS_CONDITIONAL:
            return self.get_id2label(id, return_dict)
        else:
            raise ValueError("Not Implemented")
    
    def transform(self, x):
        return x[:, None, :, :] / 255


    def get_model(
        self, device, train_loader, val_loader, optimizer="adam", lr=0.001, epochs=2, **kwargs
    ):
        model = ConvNet().to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optimizer.lower()
        if optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer {optimizer} not implemented.")

        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels, _) in enumerate(train_loader, 0):
                optimizer.zero_grad()
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 20 == 19:
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}")
                    running_loss = 0.0

        print("Finished Training")

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data.cpu(), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the network on the validation images: {100 * correct // total} %"
        )

        return model


    def process_dataframe(self, df, loader_dict, k):
        
        id2label_fm = self.get_id2label(return_dict=True)
        min_num = min(203, df["label"].value_counts().min())
        m = len(id2label_fm)
        df = stratified_sample_df(df, "label", min_num * m)

        df["label_text"] = df["label"].apply(lambda label: id2label_fm[label])
        df["group_text"] = df["label_text"]
        df["original_label"] = df["label"]

        def corr_ans_text_fn(label_text, label):
            out = f"The best answer is {label}. {label_text}."
            return out

        df["corr_ans_text"] = df.apply(
            lambda x: corr_ans_text_fn(x["label_text"], x["label"]), axis=1
        )
        df["topk_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["topk_set"], id2label_fm), axis=1
        )
        df["avgk_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["avgk_set"], id2label_fm), axis=1
        )
        df["conformal_marginal_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["conformal_marginal_set"], id2label_fm), axis=1
        )
        df["conformal_conditional_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["conformal_conditional_set"], id2label_fm), axis=1
        )   

        df = bring_examples_to_top(df, m, min_num)
        return df
