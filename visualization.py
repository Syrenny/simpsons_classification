import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap

# hello i.syrennyj
def my_display_confusion_matrix(cm, name_classes):
    fig, ax = plt.subplots()
    cm = pd.DataFrame(cm, index=name_classes, columns=name_classes)
    heatmap(cm, ax=ax)


def plot_loss_history(loss_history):
    fig, ax = plt.subplots()
    train_loss = loss_history[0]
    test_loss = loss_history[1]
    ax.plot(train_loss, color="blue")
    ax.plot(test_loss, color="red")
    ax.set_title('Loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')


def plot_accuracy_history(accuracy_history):
    fig, ax = plt.subplots()
    train_accuracy = accuracy_history[0]
    test_accuracy = accuracy_history[1]
    ax.plot(train_accuracy, color="blue")
    ax.plot(test_accuracy, color="red")
    ax.set_title('Accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
