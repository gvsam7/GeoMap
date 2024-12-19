"""
Author: Georgios Voulgaris
Date:   01/03/2022
Description:    Apply Deep Learning techniques, to map cement factories in China and monitor the pollution. Classify
                cement factories using satellite images. In more detail, LandSat band 10 (B10) Thermal infrared (TIRS) 1
                (10.6-11.19 micrometers wavelength) and band 11 Thermal infrared (TIRS) 2 (11.50-12.51 micrometers)
                images were extracted from the Satellites and used to train various Deep Learning architectures to
                classify the cement plants and the surrounding land cover.
"""

# imports
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from Utilities.Save import save_checkpoint, load_checkpoint
from Utilities.Data import DataRetrieve
from Utilities.config import train_transforms, val_transforms, test_transforms
from Utilities.Networks import networks
from Utilities.Hyperparameters import arguments
from plots.ModelExam import parameters, get_predictions, plot_confusion_matrix, plot_most_incorrect, \
    get_representations, get_pca, plot_representations, get_tsne
from pandas import DataFrame
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
import wandb


def step(data, targets, model, optimizer, criterion, train):
    with torch.set_grad_enabled(train):
        outputs = model(data)
        acc = outputs.argmax(dim=1).eq(targets).sum().item()
        loss = criterion(outputs, targets)

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return acc, loss

@torch.no_grad()
def get_all_preds(model, loader, device):
    all_preds = []
    for x, _ in loader:
        x = x.to(device)
        preds = model(x)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).cpu()
    return all_preds


def database(data):
    if data == 'b10':
        dataset = ImageFolder("Sat_Data_b10")
    elif data == 'b11':
        dataset = ImageFolder("Sat_Data_b11")
    elif data == 'b6':
        dataset = ImageFolder("Sat_Data_b6")
    elif data == 'b7':
        dataset = ImageFolder("Sat_Data_b7")
    elif data == 'b76':
        dataset = ImageFolder("Sat_Data_b76")
    elif data == 'fusion':
        # Load each dataset separately; fusion happens at the model level
        dataset = {
            'b10': ImageFolder("Sat_Data_b10"),
            'b11': ImageFolder("Sat_Data_b11"),
            'b6': ImageFolder("Sat_Data_b6"),
            'b7': ImageFolder("Sat_Data_b7"),
            'b76': ImageFolder("Sat_Data_b76")
        }
    else:
        raise ValueError(f"Unknown dataset type: {data}")
    return dataset


def loss_fun(class_weight):
    if class_weight == 'True':
        # use class weighting for unbalanced dataset
        weights = [3, 1]
        class_weights = torch.FloatTensor(weights).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    print(f"Class weight: {class_weight}")
    return criterion


def main():
    args = arguments()
    wandb.init(project="PlantPollution", config=args)

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")  # Can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # Load data
    dataset = database(args.dataset)
    print(f"Dataset is {args.dataset}")

    if args.dataset == 'fusion':
        # Use the 'b10' dataset as a reference to extract class labels
        labels = dataset['b10'].classes
    else:
        labels = dataset.classes
        
    num_classes = len(labels)
    y = dataset.targets
    dataset_len = len(dataset)

    # Stratify split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(np.arange(dataset_len), y, test_size=0.2, stratify=y,
                                                              random_state=args.random_state, shuffle=True)
    X2 = X_trainval
    y2 = y_trainval
    X_train, X_val, y_train, y_val = train_test_split(X2, y2, test_size=0.2, stratify=y2,
                                                      random_state=args.random_state, shuffle=True)
    train_ds = Subset(dataset, X_train)
    val_ds = Subset(dataset, X_val)
    test_ds = Subset(dataset, X_test)
    filepaths = np.array(tuple(zip(*dataset.imgs))[0])
    train_filepaths = filepaths[X_train]
    val_filepaths = filepaths[X_val]
    test_filepaths = filepaths[X_test]

    # Create train, validation and test datasets
    train_dataset = DataRetrieve(
        train_ds,
        transforms=train_transforms(args.width, args.height, args.Augmentation),
        augmentations=args.augmentation
    )

    val_dataset = DataRetrieve(
        val_ds,
        transforms=val_transforms(args.width, args.height)
    )

    test_dataset = DataRetrieve(
        test_ds,
        transforms=test_transforms(args.width, args.height)
    )
    # Create train, validation and test dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    prediction_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Network
    model = networks(architecture=args.architecture, in_channels=args.in_channels, num_classes=num_classes,
                     pretrained=args.pretrained, requires_grad=args.requires_grad,
                     global_pooling=args.global_pooling, version=args.version).to(device)
    print(model)
    n_parameters = parameters(model)
    print(f"The model has {n_parameters:,} trainable parameters")

    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = loss_fun(args.class_weighting)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Define Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

    # Load model
    if args.load_model == 'True':
        print(f"Load model is {args.load_model}")
        if device == torch.device("cpu"):
            load_checkpoint(torch.load("my_checkpoint.pth.tar", map_location=torch.device('cpu')), model, optimizer)
        else:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    # Train Network
    for epoch in range(args.epochs):
        model.train()
        sum_acc = 0
        for data, targets in train_loader:
            data = data.to(device=device)
            targets = targets.to(device=device)

            if args.augmentation == "cutmix":
                None
            elif args.augmentation == "mixup":
                None
            else:
                acc, loss = step(data, targets, model=model, optimizer=optimizer, criterion=criterion, train=True)
                sum_acc += acc
        train_avg_acc = sum_acc / len(train_loader)
        # After each epoch perform optimizer.step. Note in the optimizer, it is required to send in loss for that epoch!
        # optimizer.step()
        # After each epoch perform scheduler.step. Note in the scheduler, it is required to send in loss for that epoch!
        # scheduler.step(train_avg_acc)

        # Saving the model
        if args.save_model == 'True':
            if epoch % 10 == 0:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint)

        # Evaluate Network
        model.eval()
        sum_acc = 0
        for data, targets in val_loader:
            data = data.to(device=device)
            targets = targets.to(device=device)
            val_acc, val_loss = step(data, targets, model=model, optimizer=optimizer, criterion=criterion, train=False)
            sum_acc += val_acc
        val_avg_acc = sum_acc / len(val_loader)
        # After each epoch perform scheduler.step, note in this scheduler we need to send in loss for that epoch!
        # scheduler.step(val_avg_acc)

        print(f"Epoch: {epoch + 1} \tTraining accuracy: {train_avg_acc:.2f} \n\t\tValidation accuracy: {val_avg_acc:.2f}")

        train_steps = len(train_loader) * (epoch + 1)
        wandb.log({"Train Accuracy": train_avg_acc, "Validation Accuracy": val_avg_acc}, step=train_steps)

    train_preds = get_all_preds(model, loader=prediction_loader, device=device)
    print(f"Train predictions shape: {train_preds.shape}")
    print(f"The label the network predicts strongly: {train_preds.argmax(dim=1)}")
    predictions = train_preds.argmax(dim=1)

    # Most Confident Incorrect Predictions
    images, labels, probs = get_predictions(model, prediction_loader, device)

    pred_labels = torch.argmax(probs, 1)
    print(f"pred_labels: {pred_labels}")

    corrects = torch.eq(labels, pred_labels)

    incorrect_examples = []

    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if not correct:
            incorrect_examples.append((image, label, prob))
    incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)

    n_images = 48
    classes = ['Cement', 'Landcover']
    plot_most_incorrect(incorrect_examples, classes, n_images)
    wandb.save('Most_Conf_Incorrect_Pred.png')

    # Principle Components Analysis (PCA)
    outputs, intermediates, labels = get_representations(model, train_loader, device)

    output_pca_data = get_pca(outputs)
    plot_representations(output_pca_data, labels, classes, "PCA")
    wandb.save('PCA.png')

    # t-Distributed Stochastic Neighbor Embedding (t-SNE)
    n_images = 10_000

    output_tsne_data = get_tsne(outputs, n_images=n_images)
    plot_representations(output_tsne_data, labels, classes, "TSNE", n_images=n_images)
    wandb.save('TSNE.png')

    plot_confusion_matrix(y_test, train_preds.argmax(dim=1), classes)
    wandb.save('Confusion_Matrix.png')

    # Confusion Matrix
    wandb.sklearn.plot_confusion_matrix(y_test, train_preds.argmax(dim=1), labels)
    # Class proportions
    wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
    precision, recall, f1_score, support = score(y_test, train_preds.argmax(dim=1))
    test_acc = accuracy_score(y_test, train_preds.argmax(dim=1))
    wandb.log({"Test Accuracy": test_acc})

    print(f"Test Accuracy: {test_acc}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1_score: {f1_score}")
    print(f"support: {support}")

    # Test data saved in Excel document
    df = DataFrame({'Test Accuracy': test_acc, 'precision': precision, 'recall': recall, 'f1_score': f1_score,
                    'support': support})
    df.to_excel('test.xlsx', sheet_name='sheet1', index=False)
    df.to_csv('test.csv', index=False)
    compression_opts = dict(method='zip', archive_name='out.csv')
    df.to_csv('out.zip', index=False, compression=compression_opts)

    wandb.save('test.csv')
    wandb.save('my_checkpoint.pth.tar')
    wandb.save('Predictions.csv')


if __name__ == "__main__":
    main()
