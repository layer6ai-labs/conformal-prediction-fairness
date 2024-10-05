import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForSequenceClassification, CLIPModel, CLIPProcessor


# CNN for Fashion-MNIST
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5, 1, 2)
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FACETModel(nn.Module):
    def __init__(self, device, used_labels, size="ViT-L/14"):
        super(FACETModel, self).__init__()
        self.device = device
        if size == "ViT-B/32":
            clip_link = "openai/clip-vit-base-patch32"
        elif size == "ViT-B/16":
            clip_link = "openai/clip-vit-base-patch16"
        elif size == "ViT-L/14":
            clip_link = "openai/clip-vit-large-patch14"
        else:
            raise ValueError(f"Invalid CLIP model size {size}")
        print(f"Loading CLIP model {clip_link}")
        self.model = CLIPModel.from_pretrained(clip_link).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_link)

        self.used_labels = used_labels
        self.text = ["A photo of a " + k for k in used_labels]

    def forward(self, x):
        inputs = self.clip_processor(text=self.text, images=x, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image

        return logits_per_image

class BiosBiasModel(nn.Module):
    def __init__(self, num_classes):
        super(BiosBiasModel, self).__init__()
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.fc(x)
        # output = F.log_softmax(x, dim=1)
        return x


class RAVDESSModel(nn.Module):
    def __init__(self, used_labels):
        super(RAVDESSModel, self).__init__()
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "Wiam/wav2vec2-large-xlsr-53-english-finetuned-ravdess-v5"
        )
        self.used_labels = used_labels
        

    def forward(self, x):   
        logits = self.model(x).logits
        return logits[:, self.used_labels]