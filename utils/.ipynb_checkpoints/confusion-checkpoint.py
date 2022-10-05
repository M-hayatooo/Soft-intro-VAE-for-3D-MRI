import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

matplotlib.use("Agg")


def testing(net, dataloader, device):
    correct = 0
    total = 0
    net.to(device)
    y_pred = []
    y_true = []
    net.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            voxels = inputs.to(device)
            labels = labels.to(device)
            outputs = net(voxels)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(labels)):
                y_true.append(int(labels[i]))
                y_pred.append(int(predicted[i]))
    print("Test accuracy on the", total, "images:", 100 * correct / total)
    return y_pred, y_true


def make_confusion_matrix(net, dataloader, class_map, device, path="./"):
    class_list = [key for key in class_map.keys()]
    y_pred, y_true = testing(net, dataloader, device)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    plt.rcParams["font.size"] = 18
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm)
    ax.set_xticklabels(class_map.keys(), rotation=30)
    ax.set_yticklabels(class_map.keys())
    ax.set_title("confusion_matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.savefig(path + "/img/confusion_matrix.png")