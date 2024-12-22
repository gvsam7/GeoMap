import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition, manifold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
import seaborn as sns


def parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_fusion_predictions(model, iterator, device):
    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for data, targets in iterator:
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

            y_pred = model(inputs)

            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)

            images.append(data.cpu())
            labels.append(targets.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


def get_predictions(model, iterator, device):

    model.eval()

    images = []
    labels = []
    probs = []

    # Define a transform to convert data into a tensor if needed
    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for data, targets in iterator:
            data = data.to(device=device)
            y_pred = model(data)

            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)

            images.append(data.cpu())
            labels.append(targets.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)

    return image


# Function to plot fusion confusion matrix
def plot_fusion_confusion_matrix(y_true, y_pred, classes, dataset_key):
    cm = confusion_matrix(y_true.cpu(), y_pred.cpu())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix for {dataset_key}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'Confusion_Matrix_{dataset_key}.png')
    plt.close()


def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize=(10, 10));
    ax = fig.add_subplot(1, 1, 1);
    cm = confusion_matrix(labels, pred_labels);
    cm = ConfusionMatrixDisplay(cm, display_labels=classes);
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    plt.xticks(rotation=45)
    fig.savefig("Confusion_Matrix", bbox_inches='tight')


def plot_most_incorrect(incorrect, classes, n_images, normalize=True):
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(25, 20))

    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)

        image, true_label, probs = incorrect[i]
        image = image.permute(1, 2, 0)
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim=0)
        true_class = classes[true_label]
        incorrect_class = classes[incorrect_label]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.cpu().numpy())
        ax.set_title(f"True label: {true_class} ({true_prob:.3f})\n"
                     f"Pred label: {incorrect_class} ({incorrect_prob:.3f})")
        ax.axis('off')

    fig.subplots_adjust(hspace=0.4)
    fig.savefig("Most_Conf_Incorrect_Pred", bbox_inches='tight')


def get_representations(model, iterator, device):

    model.eval()

    outputs = []
    intermediates = []
    labels = []

    with torch.no_grad():
        for x, y in iterator:
            x = x.to(device)
            y_pred = model(x)
            h = model(x)

            outputs.append(y_pred.cpu())
            intermediates.append(h.cpu())
            labels.append(y)

    outputs = torch.cat(outputs, dim=0)
    intermediates = torch.cat(intermediates, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs, intermediates, labels


def get_pca(data, n_components=2):
    pca = decomposition.PCA()
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    return pca_data


def plot_representations(data, labels, classes, type, n_images=None):
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
    handles, labels = scatter.legend_elements()
    legend = ax.legend(handles=handles, labels=classes)
    if type == "PCA":
        fig.savefig("PCA", bbox_inches='tight')
    else:
        fig.savefig("TSNE", bbox_inches='tight')


def get_tsne(data, n_components=2, n_images=None):
    if n_images is not None:
        data = data[:n_images]

    tsne = manifold.TSNE(n_components=n_components, random_state=0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data