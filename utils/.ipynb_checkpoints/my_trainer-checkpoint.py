import csv
import os
import time
from asyncore import loop

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


def write_csv(epoch, train_loss, train_acc, val_loss, val_acc, path):
    with open(path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])


        
        
# trainer for ResNet mackysan VAE
def train_ResNetVAE(
    net,
    train_loader,
    val_loader,
    epochs=1,
    lr=0.001,
    device=torch.device("cpu"),
    path="./output_vae/",
):
    path = path + "train_result.csv"
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    optimizer = optim.Adam(net.parameters(), lr)
    print(optimizer)

    net.to(device)
    train_loss_list, val_loss_list = [], []
    start_time = time.time()
    for epoch in range(epochs):
        net.train()
        train_loss, val_loss = 0.0, 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            x_re, mu, logvar = net.forward(inputs)
            loss = net.loss(x_re, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)


        net.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                #labels = labels.to(device)
                x_re, mu, logvar = net.forward(inputs)
                loss = net.loss(x_re, inputs, mu, logvar)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        elapsed_time = time.time()
        print(
            "Epoch [%3d/%3d], loss: %.3f, val_loss: %.3f, total time %d秒"
            % (
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                elapsed_time - start_time,
            )
        )
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        #write_csv(epoch, train_loss, val_loss, path)

    print("Finished Traininig")
    return train_loss_list, val_loss_list

# --------------------------------------------------------------------------------#
        


        








        






# trainer for CNN
def train(
    net,
    train_loader,
    val_loader,
    epochs=1,
    lr=0.001,
    device=torch.device("cpu"),
    path="./output/",
):
    path = path + "train_result.csv"
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    #criterion = nn.MSELoss()
    #criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(params=net.parameters(), lr=0.0005, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr)
    print(optimizer)

    net.to(device)
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    start_time = time.time()
    for epoch in range(epochs):
        net.train()
        train_loss, train_acc, val_loss, val_acc = 0.0, 0.0, 0.0, 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            #labels = nn.functional.one_hot(labels, num_classes=2).to(torch.float32)         # sigmoid用
            loss = criterion(outputs, labels)
            #loss = criterion(outputs, labels.unsqueeze(1).float())          # bce用
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            #predicted = (outputs > 0.5).float()                    # bce用
            #predicted = predicted.squeeze(1)                       # bce用
            #labels = torch.argmax(labels, dim=1)                   # sigmoid用 
            train_acc += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)


        net.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                #labels = nn.functional.one_hot(labels, num_classes=2).to(torch.float32)         # sigmoid用
                loss = criterion(outputs, labels)
                #loss = criterion(outputs, labels.unsqueeze(1).float())          # bce用
                val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                #predicted = (outputs > 0.5).float()                  # bce用
                #predicted = predicted.squeeze(1)
                #labels = torch.argmax(labels, dim=1)                # sigmoid用 
                val_acc += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)
        elapsed_time = time.time()
        print(
            "Epoch [%3d/%3d], loss: %.3f acc: %.3f, val_loss: %.3f val_acc: %.3f total time %d秒"
            % (
                epoch + 1,
                epochs,
                train_loss,
                train_acc * 100,
                val_loss,
                val_acc * 100,
                elapsed_time - start_time,
            )
        )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        write_csv(epoch, train_loss, train_acc, val_loss, val_acc, path)

    print("Finished Traininig")
    return train_loss_list, train_acc_list, val_loss_list, val_acc_list

# ----------------------------------------------------------------------------------------------------------- #

# trainer for CAE
def train_cae(
    net,
    train_loader,
    val_loader,
    epochs=1,
    lr=0.001,
    device=torch.device("cpu"),
    path="./output/",
):
    path = path + "train_result.csv"
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr)
    print(optimizer)
    net = torch.nn.DataParallel(net, device_ids=[0,1,2,3])
    net.to(device)
    train_loss_list, val_loss_list = [], []
    start_time = time.time()
    for epoch in range(epochs):
        net.train()
        train_loss, val_loss = 0.0, 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            #labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        net.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                #labels = labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, inputs)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        elapsed_time = time.time()
        print(
            "Epoch [%3d/%3d], loss: %.3f, val_loss: %.3f, total time %d秒"
            % (
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                elapsed_time - start_time,
            )
        )
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        #write_csv(epoch, train_loss, val_loss, path)

    print("Finished Traininig")
    return train_loss_list, val_loss_list

    # ----------------------------------------------------------------------------------------------------------- #

# trainer for Caee
def train_caee(
    net,
    train_loader,
    val_loader,
    epochs=1,
    lr=0.001,
    device=torch.device("cpu"),
    path="./output/",
):
    path = path + "train_result.csv"
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr)
    print(optimizer)

    net.to(device)
    train_loss_list, val_loss_list = [], []
    start_time = time.time()
    for epoch in range(epochs):
        loop_start_time = time.time()
        net.train()
        train_loss, val_loss = 0.0, 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            #labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        net.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                #labels = labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, inputs)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        now_time = time.time()
        print(
            "Epoch [%3d/%3d], loss: %.3f, val_loss: %.3f, %d秒/epoch, total time %d秒"
            % (
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                loop_start_time - now_time,
                now_time - start_time,
            )
        )
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        #write_csv(epoch, train_loss, val_loss, path)

    print("Finished Traininig")
    return train_loss_list, val_loss_list

# ----------------------------------------------------------------------------------------------------------- #

# trainer for VAE
def train_vae(
    net,
    train_loader,
    val_loader,
    epochs=1,
    lr=0.001,
    device=torch.device("cpu"),
    path="./output_vae/",
):
    path = path + "train_result.csv"
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    optimizer = optim.Adam(net.parameters(), lr)
    print(optimizer)

    net.to(device)
    train_loss_list, val_loss_list = [], []
    start_time = time.time()
    for epoch in range(epochs):
        net.train()
        train_loss, val_loss = 0.0, 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            x_re, mu, logvar = net.forward(inputs)
            loss = net.loss(x_re, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)


        net.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                #labels = labels.to(device)
                x_re, mu, logvar = net.forward(inputs)
                loss = net.loss(x_re, inputs, mu, logvar)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        elapsed_time = time.time()
        print(
            "Epoch [%3d/%3d], loss: %.3f, val_loss: %.3f, total time %d秒"
            % (
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                elapsed_time - start_time,
            )
        )
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        #write_csv(epoch, train_loss, val_loss, path)

    print("Finished Traininig")
    return train_loss_list, val_loss_list

# --------------------------------------------------------------------------------#
