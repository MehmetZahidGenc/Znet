import torch
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import torch.nn as nn
from Znet_model import ZNet
from Znet_utils import train_model, visualize_model

def main():
    Znet_model = ZNet(n_channels=3, n_classes=5)

    """
    Dataset Format;
    
    Dataset
      --> train
            ->ClassName1
                  - imageX.jpg(or png)
            ->ClassName2
                  - imageY.jpg(or png)
      --> val
            same architecture as train
    
      --> classes.txt
      --> test(optional)
            same architecture as train
    """

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    """
    Defining Data
    """

    data_dir = 'Chicken'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                  shuffle=True, num_workers=2)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    print(f'Dataset sizes: {dataset_sizes} \nClass names: {class_names}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Znet_model.classifier_part.parameters())

    model_trained = train_model(Znet_model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10)


if __name__ == '__main__':
    main()
