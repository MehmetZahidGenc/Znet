import torch
from matplotlib import pyplot as plt
import numpy as np
from torchmetrics import Accuracy, Recall, Precision, F1Score
from tabulate import tabulate

def train_model(model, criterion, optimizer, dataloaders, num_classes, num_epochs=5):
    accuracy = Accuracy(average='macro', num_classes=num_classes)
    precision = Precision(average='macro', num_classes=num_classes)
    recall = Recall(average='macro', num_classes=num_classes)
    f1 = F1Score(average='macro', num_classes=num_classes)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        Tacc, Tprec, Trec, Tf1 = 0.0, 0.0, 0.0, 0.0
        Vacc, Vprec, Vrec, Vf1 = 0.0, 0.0, 0.0, 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()

                acc_list = []
                prec_list = []
                rec_list = []
                f1_score_list = []

                for inputs, labels in dataloaders[phase]:

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)

                    acc = accuracy(preds, labels.data)
                    prec = precision(preds, labels.data)
                    rec = recall(preds, labels.data)
                    f1_score = f1(preds, labels.data)

                    acc_list.append(acc)
                    prec_list.append(prec)
                    rec_list.append(rec)
                    f1_score_list.append(f1_score)

                Tacc = sum(acc_list) / len(acc_list)
                Tprec = sum(prec_list) / len(prec_list)
                Trec = sum(rec_list) / len(rec_list)
                Tf1 = sum(f1_score_list) / len(f1_score_list)

            else:
                model.eval()

                Vacc_list = []
                Vprec_list = []
                Vrec_list = []
                Vf1_score_list = []

                for inputs, labels in dataloaders[phase]:

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)

                    Vacc = accuracy(preds, labels.data)
                    Vprec = precision(preds, labels.data)
                    Vrec = recall(preds, labels.data)
                    Vf1_score = f1(preds, labels.data)

                    Vacc_list.append(Vacc)
                    Vprec_list.append(Vprec)
                    Vrec_list.append(Vrec)
                    Vf1_score_list.append(Vf1_score)

                #
                Vacc = sum(Vacc_list) / len(Vacc_list)
                Vprec = sum(Vprec_list) / len(Vprec_list)
                Vrec = sum(Vrec_list) / len(Vrec_list)
                Vf1 = sum(Vf1_score_list) / len(Vf1_score_list)

        data = [["Train", Tacc, Tprec, Trec, Tf1],
                ["Validation", Vacc, Vprec, Vrec, Vf1]]

        headers = ["Type", 'Accuracy', 'Precision', 'Recall', 'F1 Score']

        print(tabulate(data, headers=headers))
        print("\n")

    return model



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, dataloaders, class_names, num_images=6, device='cpu'):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
