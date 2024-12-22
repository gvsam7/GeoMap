"""
Author: Georgios Voulgaris
Date:   20/12/2024
Description:    Apply Deep Learning techniques, to map cement factories in China and monitor the pollution. Classify
                cement factories using satellite images. In more detail, LandSat band 10 (B10) Thermal infrared (TIRS) 1
                (10.6-11.19 micrometers wavelength), band 11 Thermal infrared (TIRS) 2 (11.50-12.51 micrometers), band 7
                Short Wave Infrared (SWI) 1 (2.11-2.29 micrometers), band 6 Short Wave Infrared (SWI) 2
                (1.57-1.65 micrometers), and a ration of bands7:6 Short Wave Infrared (SWI) (1.34 and 1.39 micrometers)
                images were extracted from the Satellites and used to train various Deep Learning architectures to
                classify the cement plants and the surrounding land cover. The addition of this code allows to train
                data fusion models.
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
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
from plots.ModelExam import get_fusion_predictions, plot_fusion_confusion_matrix
import pandas as pd
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
        datasets = {
            'b10': ImageFolder("Sat_Data_b10"),
            'b11': ImageFolder("Sat_Data_b11"),
            'b6': ImageFolder("Sat_Data_b6"),
            'b7': ImageFolder("Sat_Data_b7"),
            'b76': ImageFolder("Sat_Data_b76")
        }
        return datasets
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


def custom_collate_fn(batch):
    resize = transforms.Resize((256, 256))

    # Process each item in the batch individually
    images, labels = zip(*batch)

    # Check if images are already tensors
    if isinstance(images[0], torch.Tensor):
        # If tensors, stack them directly
        images = torch.stack(images, dim=0)
    else:
        # If not tensors, resize and convert to tensors
        images = torch.stack([transforms.ToTensor()(resize(image)) for image in images], dim=0)

    # Convert labels to a tensor
    labels = torch.tensor(labels)

    return images, labels


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
    datasets = database(args.dataset)  # Load datasets separately
    print(f"Datasets loaded for {args.dataset}: {datasets.keys()}")

    classes = datasets['b10'].classes  # or dataset['b11'], it's the same
    print(f"Classes: {classes}")
    num_classes = len(classes)

    # Get labels for the individual datasets (example with 'b10' and 'b11')
    labels_b10 = datasets['b10'].classes
    labels_b11 = datasets['b11'].classes
    labels_b7 = datasets['b7'].classes
    labels_b6 = datasets['b6'].classes
    labels_b76 = datasets['b76'].classes

    # Perform a stratified split for each dataset
    def stratified_split(dataset):
        y = dataset.targets
        dataset_len = len(dataset)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            np.arange(dataset_len), y, test_size=0.2, stratify=y,
            random_state=args.random_state, shuffle=True)
        X2 = X_trainval
        y2 = y_trainval
        X_train, X_val, y_train, y_val = train_test_split(
            X2, y2, test_size=0.2, stratify=y2,
            random_state=args.random_state, shuffle=True)
        return X_train, X_val, X_test, y_train, y_val, y_test

    # Split each dataset separately
    X_train_b10, X_val_b10, X_test_b10, y_train_b10, y_val_b10, y_test_b10 = stratified_split(datasets['b10'])
    X_train_b11, X_val_b11, X_test_b11, y_train_b11, y_val_b11, y_test_b11 = stratified_split(datasets['b11'])
    X_train_b7, X_val_b7, X_test_b7, y_train_b7, y_val_b7, y_test_b7 = stratified_split(datasets['b7'])
    X_train_b6, X_val_b6, X_test_b6, y_train_b6, y_val_b6, y_test_b6 = stratified_split(datasets['b6'])
    X_train_b76, X_val_b76, X_test_b76, y_train_b76, y_val_b76, y_test_b76 = stratified_split(datasets['b76'])

    # Create subsets for each dataset split
    train_ds_b10 = Subset(datasets['b10'], X_train_b10)
    train_ds_b11 = Subset(datasets['b11'], X_train_b11)
    train_ds_b7 = Subset(datasets['b7'], X_train_b7)
    train_ds_b6 = Subset(datasets['b6'], X_train_b6)
    train_ds_b76 = Subset(datasets['b76'], X_train_b76)
    val_ds_b10 = Subset(datasets['b10'], X_val_b10)
    val_ds_b11 = Subset(datasets['b11'], X_val_b11)
    val_ds_b7 = Subset(datasets['b7'], X_val_b7)
    val_ds_b6 = Subset(datasets['b6'], X_val_b6)
    val_ds_b76 = Subset(datasets['b76'], X_val_b76)
    test_ds_b10 = Subset(datasets['b10'], X_test_b10)
    test_ds_b11 = Subset(datasets['b11'], X_test_b11)
    test_ds_b7 = Subset(datasets['b7'], X_test_b7)
    test_ds_b6 = Subset(datasets['b6'], X_test_b6)
    test_ds_b76 = Subset(datasets['b76'], X_test_b76)

    # Create data retrieval objects for training, validation, and testing
    train_dataset = DataRetrieve(
        train_ds_b10 + train_ds_b11 + train_ds_b7 + train_ds_b6 + train_ds_b76,
        transforms=train_transforms(args.width, args.height, args.Augmentation),
        augmentations=args.augmentation
    )

    val_dataset = DataRetrieve(
        val_ds_b10 + val_ds_b11 + val_ds_b7 + val_ds_b6 + val_ds_b76,
        transforms=val_transforms(args.width, args.height)
    )

    test_dataset = DataRetrieve(
        test_ds_b10 + test_ds_b11 + test_ds_b7 + test_ds_b6 + test_ds_b76,
        transforms=test_transforms(args.width, args.height)
    )

    # Create train, validation, and test dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    prediction_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)

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
        total_batches = 0
        for dataset_key, dataset in datasets.items():
            # Create DataLoader for each dataset
            train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

            for data, targets in train_loader:
                # Split the data according to the branches, assuming you have 5 different inputs
                b10_data = data
                b11_data = data
                b7_data = data
                b6_data = data
                b76_data = data

                # Assuming `data` needs to be split into branches
                inputs = {
                    'b10': b10_data,
                    'b11': b11_data,
                    'b7': b7_data,
                    'b6': b6_data,
                    'b76': b76_data,
                }

                # Send data to the device (GPU/CPU)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                targets = targets.to(device)

                if args.augmentation == "cutmix":
                    # Implement cutmix augmentation
                    pass
                elif args.augmentation == "mixup":
                    # Implement mixup augmentation
                    pass
                else:
                    acc, loss = step(inputs, targets, model=model, optimizer=optimizer, criterion=criterion, train=True)
                    sum_acc += acc
                    total_batches += 1

        train_avg_acc = sum_acc / total_batches if total_batches > 0 else 0.0
        # optimizer.step()
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
        model.eval()  # Set the model to evaluation mode
        sum_acc = 0
        total_batches = 0  # To correctly average over all validation batches

        with torch.no_grad():  # Disable gradient computation for validation
            for dataset_key, dataset in datasets.items():
                # Create DataLoader for each dataset
                val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

                for data, targets in val_loader:
                    # Extract individual branches from the data
                    b10_data = data
                    b11_data = data
                    b7_data = data
                    b6_data = data
                    b76_data = data

                    # Create the `inputs` dictionary
                    inputs = {
                        'b10': b10_data,
                        'b11': b11_data,
                        'b7': b7_data,
                        'b6': b6_data,
                        'b76': b76_data,
                    }

                    # Send data to the device (GPU/CPU)
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    targets = targets.to(device)

                    # Calculate validation metrics (e.g. accuracy, loss)
                    acc, loss = step(inputs, targets, model=model, optimizer=optimizer, criterion=criterion, train=False)

                    # Accumulate accuracy across batches
                    sum_acc += acc
                    total_batches += 1

        # Calculate average validation accuracy across all datasets
        val_avg_acc = sum_acc / total_batches if total_batches > 0 else 0.0

        print(
            f"Epoch: {epoch + 1} \tTraining accuracy: {train_avg_acc:.2f} \n\t\tValidation accuracy: {val_avg_acc:.2f}")

        train_steps = len(train_loader) * (epoch + 1)
        wandb.log({"Train Accuracy": train_avg_acc, "Validation Accuracy": val_avg_acc}, step=train_steps)

    # Predictions
    predictions = {}
    for dataset_key, dataset in datasets.items():
        iterator = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        images, labels, probs = get_fusion_predictions(model, iterator, device)
        predictions[dataset_key] = (images, labels, probs)

    # Generate and plot confusion matrices for each dataset
    for dataset_key, (images, labels, probs) in predictions.items():
        pred_labels = torch.argmax(probs, 1)
        corrects = torch.eq(labels, pred_labels)

        incorrect_examples = []

        for image, label, prob, correct in zip(images, labels, probs, corrects):
            if not correct:
                incorrect_examples.append((image, label, prob))

        incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)

        n_images = 48
        classes = ['Cement', 'Landcover']
        plot_most_incorrect(incorrect_examples, classes, n_images)
        wandb.save(f'Most_Conf_Incorrect_Pred_{dataset_key}.png')

        # Confusion Matrix
        plot_fusion_confusion_matrix(labels, pred_labels, classes, dataset_key)
        wandb.save(f'Confusion_Matrix_{dataset_key}.png')

        # Log confusion matrix with WandB
        wandb.sklearn.plot_confusion_matrix(labels, pred_labels, classes)
        wandb.sklearn.plot_class_proportions(labels, labels, classes)

        # precision, recall, f1_score, support = precision_recall_fscore_support(labels, pred_labels, average='weighted')
        # Calculate precision, recall, F1 score, and support
        report = classification_report(labels, pred_labels, target_names=classes, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']
        support = report['weighted avg']['support']

        test_acc = accuracy_score(labels, pred_labels)
        wandb.log({f"Test Accuracy {dataset_key}": test_acc})

        print(f"Dataset: {dataset_key}")
        print(f"Test Accuracy: {test_acc}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1_score: {f1_score}")
        print(f"support: {support}")

        # Print detailed classification report
        print(
            f"Classification Report for {dataset_key}:\n{classification_report(labels, pred_labels, target_names=classes)}")

        # Save test data in Excel document
        df = DataFrame(
            {'Test Accuracy': [test_acc], 'precision': [precision], 'recall': [recall], 'f1_score': [f1_score],
             'support': [support]})
        df.to_excel(f'test_{dataset_key}.xlsx', sheet_name='sheet1', index=False)
        df.to_csv(f'test_{dataset_key}.csv', index=False)
        compression_opts = dict(method='zip', archive_name=f'out_{dataset_key}.csv')
        df.to_csv(f'out_{dataset_key}.zip', index=False, compression=compression_opts)

        wandb.save(f'test_{dataset_key}.csv')
        wandb.save(f'my_checkpoint_{dataset_key}.pth.tar')
        wandb.save(f'Predictions_{dataset_key}.csv')


    """
    preds_list = []
    targets_list = []

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations during inference
    with torch.no_grad():
        for data, targets in prediction_loader:
            # Assuming `data` needs to be split into 5 branches
            b10_data = data.clone()
            b11_data = data.clone()
            b7_data = data.clone()
            b6_data = data.clone()
            b76_data = data.clone()

            # Create the inputs dictionary for the model
            inputs = {
                'b10': b10_data,
                'b11': b11_data,
                'b7': b7_data,
                'b6': b6_data,
                'b76': b76_data,
            }

            # Send data to the appropriate device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            targets = targets.to(device)

            # Model inference
            preds = model(inputs)

            # Print the entire preds tensor for debugging
            print("Preds tensor:", preds)

            # Store predictions and targets
            preds_list.append(preds)
            targets_list.append(targets)

        # Concatenate all predictions and targets
        preds_tensor = torch.cat(preds_list, dim=0).argmax(dim=1)
        targets_tensor = torch.cat(targets_list, dim=0)

        # Calculate metrics
        precision, recall, f1_score, support = precision_recall_fscore_support(
            targets_tensor.cpu(), preds_tensor.cpu(), average=None
        )
        accuracy = accuracy_score(targets_tensor.cpu(), preds_tensor.cpu())

        # Store metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1_score": f1_score.tolist(),
            "support": support.tolist(),
        }

        # Generate confusion matrix
        cm = confusion_matrix(targets_tensor.cpu(), preds_tensor.cpu())
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("confusion_matrix.png")
        plt.close()

        # Save classification report
        report = classification_report(
            targets_tensor.cpu(), preds_tensor.cpu(), target_names=classes
        )
        with open("classification_report.txt", "w") as f:
            f.write(report)

        # Log metrics to WandB and save outputs
        wandb.log({
            "Test Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"],
            "Support": metrics["support"],
        })

        # Save metrics to CSV
        df = pd.DataFrame(metrics)
        df.to_csv("metrics.csv", index=False)

        # Save confusion matrix
        wandb.save("confusion_matrix.png")

        # Save classification report
        wandb.save("classification_report.txt")"""

    """
    # This part of the code is working, but lucks individual dataset precision, recall and confusion matrix 
    # functionality. 
    # Initialise lists for predictions and targets
    all_preds = []
    all_targets = []

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations during inference
    with torch.no_grad():
        for data, targets in prediction_loader:
            # Assuming `data` needs to be split into 5 branches
            b10_data = data.clone()  # Modify if specific transformations are needed
            b11_data = data.clone()
            b7_data = data.clone()
            b6_data = data.clone()
            b76_data = data.clone()

            # Create the inputs dictionary for the model
            inputs = {
                'b10': b10_data,
                'b11': b11_data,
                'b7': b7_data,
                'b6': b6_data,
                'b76': b76_data,
            }

            # Send data to the appropriate device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            targets = targets.to(device)

            # Model inference
            preds = model(inputs)

            # Store predictions and targets
            all_preds.append(preds)
            all_targets.append(targets)

    # Convert predictions and targets to tensors
    train_preds = torch.cat(all_preds, dim=0)
    train_targets = torch.cat(all_targets, dim=0)

    # Output shapes and top predictions
    print(f"Train predictions shape: {train_preds.shape}")
    print(f"Predicted labels: {train_preds.argmax(dim=1)}")

    # Final predicted class labels
    predictions = train_preds.argmax(dim=1)

    # Calculate metrics
    precision, recall, f1_score, support = precision_recall_fscore_support(
        train_targets.cpu(), predictions.cpu(), average='weighted'
    )
    accuracy = accuracy_score(train_targets.cpu(), predictions.cpu())

    # Log metrics to wandb
    wandb.log({
        "Test Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    })

    # Output metrics
    print(f"Test Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"Support: {support}")

    # Save metrics and predictions
    df = pd.DataFrame({
        "Test Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1_score],
        "Support": [support]
    })
    df.to_csv('metrics.csv', index=False)

    compression_opts = dict(method='zip', archive_name='out.csv')
    df.to_csv('out.zip', index=False, compression=compression_opts)

    wandb.save('test.csv')
    wandb.save('my_checkpoint.pth.tar')
    wandb.save('Predictions.csv')"""


if __name__ == "__main__":
    main()


