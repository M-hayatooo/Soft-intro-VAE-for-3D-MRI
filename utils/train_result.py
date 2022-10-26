import argparse
import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def result(train_acc, train_loss, val_acc, val_loss, path="."):
    os.makedirs(path + "/img", exist_ok=True)
    epoch = len(train_acc)
    plt.rcParams["font.size"] = 18
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.plot(range(1, epoch + 1), train_loss, label="train_loss")
    ax1.plot(range(1, epoch + 1), val_loss, label="val_loss")
    ax1.set_title("loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    fig1.savefig(path + "/img/loss.png")

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.plot(range(1, epoch + 1), train_acc, label="train_acc")
    ax2.plot(range(1, epoch + 1), val_acc, label="val_acc")
    ax2.set_title("accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    fig2.savefig(path + "/img/acc.png")

def result_ae(train_loss, val_loss, path="."):
    os.makedirs(path + "/img", exist_ok=True)
    epoch = len(train_loss)
    plt.rcParams["font.size"] = 18
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.plot(range(1, epoch + 1), train_loss, label="train_loss")
    ax1.plot(range(1, epoch + 1), val_loss, label="val_loss")
    ax1.set_title("loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    fig1.savefig(path + "/img/loss.png")



def read_csv(path):
    df = pd.read_csv(path)
    train_loss = df["train_loss"].values
    train_acc = df["train_acc"].values
    val_loss = df["val_loss"].values
    val_acc = df["val_acc"].values
    return train_loss, train_acc, val_loss, val_acc


# 動作確認用
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="ADNI2-2")
    arg = parser.parse_args()
    path = "./logs/" + arg.path + "/"
    train_loss, train_acc, val_loss, val_acc = read_csv(path + "train_result.csv")
    train_result(train_acc, train_loss, val_acc, val_loss, path)


if __name__ == "__main__":
    main()
