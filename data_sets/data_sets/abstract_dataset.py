from abc import ABC, abstractmethod


class CustomDataset(ABC):

    @abstractmethod
    def get_data(self, data_root, **loader_kwargs):
        pass


    @abstractmethod
    def get_model(self, device, train_loader, val_loader, **kwargs):
        pass


    def prepare_model_inputs(self, data, device):
        assert len(data) == 3
        x, target, group = data
        x = x.to(device)
        target = target.to(device)
        group = group.to(device)
        input_data = x.cpu().numpy().tolist()

        return x, target, group, input_data


    @abstractmethod
    def process_dataframe(self, df, loader_dict, k):
        pass


    @abstractmethod
    def get_id2label(self, id, return_dict):
        pass


    @abstractmethod
    def get_id2group(self, id, return_dict):
        pass