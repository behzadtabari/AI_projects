from pathlib import Path

import numpy as np
import torch
from utils.generator import generate_toy_data


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        if split == 'train':
            train_data, train_labels = generate_toy_data(4096)
            # Uncomment to see the shapes
            # print(f"The shape of train data is: {train_data.shape}")
            # print(f"The shape of train labels is: {train_labels.shape}")
            self.dataset = train_data
            self.labels = train_labels
        elif split == 'val':
            val_data, val_labels = generate_toy_data(1024)
            self.dataset = val_data
            self.labels = val_labels
        else:
            raise ValueError("Invalid split; please choose 'train' or 'val'")

    def __getitem__(self, idx):
        # Uncomment to see the shape
        # print(f"The shape of this item is: {self.dataset[idx].shape}")
        return self.dataset[idx][np.newaxis, :], self.labels[idx]

    def __len__(self):
        return len(self.dataset)


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=4,
                                     kernel_size=4, stride=3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(4)

        self.conv2 = torch.nn.Conv3d(in_channels=4, out_channels=8,
                                     kernel_size=4, stride=3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(8)

        self.conv3 = torch.nn.Conv3d(in_channels=8, out_channels=16,
                                     kernel_size=4, stride=3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(16)

        self.fc = torch.nn.Linear(16, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 16)
        x = self.fc(x)
        return x


def train(model, train_dataloader, val_dataloader, device, config):
    loss_criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    model.train()
    best_accuracy = 0.0

    for epoch in range(config['max_epochs']):
        train_loss_running = 0.0

        for i, batch in enumerate(train_dataloader):
            input_data, target_labels = batch
            input_data = input_data.to(device)
            target_labels = target_labels.to(device)

            optimizer.zero_grad()
            prediction = model(input_data)
            loss = loss_criterion(prediction, target_labels)
            loss.backward()
            optimizer.step()

            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + i
            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                train_loss_running = 0.0

            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                model.eval()
                total, correct = 0, 0
                loss_val = 0.0
                for batch_val in val_dataloader:
                    input_data, target_labels = batch_val
                    input_data = input_data.to(device)
                    target_labels = target_labels.to(device)
                    with torch.no_grad():
                        prediction = model(input_data)
                    _, predicted_labels = torch.max(prediction, dim=1)
                    total += predicted_labels.shape[0]
                    correct += (predicted_labels == target_labels).sum().item()
                    loss_val += loss_criterion(prediction, target_labels).item()

                accuracy = 100 * correct / total
                print(f'[{epoch:03d}/{i:05d}] val_loss: '
                      f'{loss_val / len(val_dataloader):.3f},'
                      f' val_accuracy: {accuracy:.3f}%')

                if accuracy > best_accuracy:
                    torch.save(model.state_dict(), f'exercise_2/runs/{config["experiment_name"]}/model_best.ckpt')
                    best_accuracy = accuracy
                model.train()


def main(config):
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    train_dataset = SimpleDataset('train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_dataset = SimpleDataset('val')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = SimpleModel().to(device)
    Path(f'exercise_2/runs/{config["experiment_name"]}').mkdir(exist_ok=True,
                                                               parents=True)
    train(model, train_dataloader, val_dataloader, device, config)


if __name__ == '__main__':
    main(config={
        'experiment_name': 'simple_nn',
        'device': 'cuda:0',
        'batch_size': 32,
        'resume_ckpt': None,
        'learning_rate': 0.001,
        'max_epochs': 5,
        'print_every_n': 10,
        'validate_every_n': 100
    })
