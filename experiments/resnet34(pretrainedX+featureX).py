import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torchvision.models import ResNet34_Weights, VGG16_Weights
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score

# Setting
data_dir = '../PokemonData'
MODEL_NAME = "resnet34"
FEATURE_EXTRACT = False
BATCH_SIZE = 32
NUM_EPOCHS = 10
USE_PRETRAINED = False


# 데이터 전처리
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}


# 데이터 로드
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Path nod found: {data_dir}")

full_dataset = datasets.ImageFolder(data_dir)
num_classes = len(full_dataset.classes)

# 데이터의 80%는 Training set 나머지 20%는 Validation set으로 사용
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_db, val_db = random_split(full_dataset, [train_size, val_size])

train_db.dataset = copy.deepcopy(full_dataset)
train_db.dataset.transform = data_transforms['train']
val_db.dataset.transform = data_transforms['val']

# Dataloader (batch 단위로 데이터 공급)
dataloaders = {
    'train': DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True),
    'val': DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False)
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 모델 정의
def initialize_model(model_name, num_classes, feature_extract):
    if model_name == "resnet34":
        weights = ResNet34_Weights.DEFAULT if USE_PRETRAINED else None
        model_ft = models.resnet34(weights=weights)

        if feature_extract:
            for param in model_ft.parameters():
                param.requires_grad = False

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg16":
        weights = VGG16_Weights.DEFAULT if USE_PRETRAINED else None
        model_ft = models.vgg16(weights=weights)

        if feature_extract:
            for param in model_ft.parameters():
                param.requires_grad = False

        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError("Unsupported model")

    return model_ft


model = initialize_model(MODEL_NAME, num_classes, FEATURE_EXTRACT).to(device)


# 학습 설정
params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params_to_update, lr=0.001)
criterion = nn.CrossEntropyLoss()


print("Training in progress...")


# 학습 루프
for epoch in range(NUM_EPOCHS):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        all_labels = []
        all_preds = []

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        # epoch 단위 성능 계산
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f'Epoch {epoch}/{NUM_EPOCHS-1} | {phase} Loss: {epoch_loss:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f}')


# 모델 저장
model_filename = f"pokemon_{MODEL_NAME}_fixed_{FEATURE_EXTRACT}_pre_{USE_PRETRAINED}.pth"
torch.save(model.state_dict(), model_filename)

print("Model saved successfully")