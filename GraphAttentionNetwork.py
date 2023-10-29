import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from DatasetCreation5 import GenerateDataset
import yaml
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torchsummary import summary
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.nn import GATConv, TopKPooling
import pandas as pd
from sklearn.model_selection import StratifiedKFold



import csv

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU Available")
else:
    print("No GPU Available")


def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)


num_heads = 4


#
class GATN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GATN, self).__init__()
        torch.manual_seed(42)

        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads ,dropout=0.2)
        self.transform1 = Linear(hidden_dim * num_heads, hidden_dim)
        self.pool1 = TopKPooling(hidden_dim, ratio=0.8)

        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=num_heads ,dropout=0.2)
        self.transform2 = Linear(hidden_dim * num_heads, hidden_dim)
        self.pool2 = TopKPooling(hidden_dim, ratio=0.8)

        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=num_heads,dropout=0.2)
        self.transform3 = Linear(hidden_dim * num_heads, hidden_dim)
        self.pool3 = TopKPooling(hidden_dim, ratio=0.8)

        self.linear1 = Linear(hidden_dim * 2, 256)
        self.linear2 = Linear(256, 1024)
        self.linear3 = Linear(1024, 1)

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.conv1(x, edge_index, edge_attr).relu()

        x = self.transform1(x)


        x, edge_index, edge_attr, batch, _, _ = self.pool1(
            x,
            edge_index,
            edge_attr,
            batch
        )

        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        # x1 = global_mean_pool(x, batch)

        # Second Block

        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.transform2(x)

        x, edge_index, edge_attr, batch, _, _ = self.pool2(
            x,
            edge_index,
            edge_attr,
            batch
        )

        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # x2 = global_mean_pool(x, batch)

        # Third Block

        # x = self.conv3(x, edge_index, edge_attr)
        # x = self.head_transform3(x)
        #
        # x, edge_index, edge_attr, batch, _, _ = self.pool3(
        #     x,
        #     edge_index,
        #     edge_attr,
        #     batch
        # )
        #
        # x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = x1 + x2
        # print(x.size())

        # Output

        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.linear2(x).relu()
        x = self.linear3(x)

        return x


#
# class GATN(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(GATN, self).__init__()
#         torch.manual_seed(42)
#
#         # GCN layers
#         self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=False)
#         self.conv2 = GATConv(hidden_dim, hidden_dim, heads=int(num_heads), concat=False)
#         self.conv3 = GATConv(hidden_dim, hidden_dim, heads=int(num_heads), concat=False)
#         self.conv4 = GATConv(hidden_dim, hidden_dim, heads=int(num_heads), concat=False)
#         # self.conv3 = GATConv(hidden_dim, hidden_dim)
#         # self.conv4 = GATConv(hidden_dim, hidden_dim)
#         self.fc1 = nn.Linear(hidden_dim*2, 512)
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(512, 1024)
#         self.fc3 = nn.Linear(512, 1024)
#         self.fc4 = nn.Linear(1024, 1)
#
#     def forward(self, x, edge_index, edge_attr, batch):
#         x = F.relu(self.conv1(x, edge_index, edge_attr))
#
#         x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
#
#         x = F.relu(self.conv3(x, edge_index, edge_attr=edge_attr))
#
#
#
#
#         # x = global_mean_pool(x, batch)
#
#         x = torch.cat([global_mean_pool(x, batch),
#                             global_max_pool(x, batch)], dim=1)
#
#         x = self.fc1(x)
#         x= self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         # x = self.fc3(x)
#         # x = self.leaky_relu(x)
#
#
#
#         x= self.fc4(x)
#
#         return x


with open("variablesConfig.yaml", "r") as file:
    variablesConfig = yaml.safe_load(file)
genData = GenerateDataset(variablesConfig['folders']['sdfPath'], variablesConfig['folders']['datasetPath'])

listOfGenData = []
listOfGenData_Test = []
targetVariable = []

for i in range(1, len(genData)):
    # if(genData[i].y.item() < -180000):

    targetVariable.append(genData[i].y.item())
    if i % 100 != 0:
        listOfGenData.append(genData[i])
    else:
        listOfGenData_Test.append(genData[i])


#

targetDataFrame = pd.DataFrame(targetVariable, columns=['freeEnergy'])
bins = [-310000, -280000, -270000, -260000, -250000, -240000, -230000, -220000, -210000, -200000, -190000, -180000,
        -170000, -160000, -150000, 140000]
labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
# "10","11","12","13","14","15"
# targetDataFrame['binned'] = pd.cut(targetDataFrame['freeEnergy'], bins,labels=labels)

targetDataFrame['binned'] = pd.qcut(targetDataFrame['freeEnergy'], 10, labels=labels)
# print(targetDataFrame['binned'])
# for i in range(len(genData) - 99, len(genData)):
#     listOfGenData_Test.append(genData[i])

# for i in range(1,len(genData)-1,2):
#     listOfGenData.append(genData[i])
#     listOfGenData_Test.append(genData[i+1])


input_dim = listOfGenData[0].x.size(1)
hidden_dim = 32
output_dim = 1
model = GATN(input_dim, hidden_dim)
model.to(device)
criterion = nn.MSELoss()


from torch.nn import L1Loss


def reset_weights(m):

    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def train(num_epochs):
    c = 0
    mae = L1Loss()


    iqrlist = []
    batch_size = 64

    kfold = KFold(n_splits=5, shuffle=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # , targetDataFrame['binned']
    for fold, (train_indices, val_indices) in enumerate(kfold.split(listOfGenData)):
        # for fold, (train_indices, val_indices) in enumerate(kfold.split(listOfGenData)):

        training_loss = []

        validation_loss = []

        model_new = GATN(input_dim, hidden_dim)
        model_new.to(device)
        model_new.apply(reset_weights)
        criterion2 = nn.MSELoss()
        optimizer = optim.Adam(model_new.parameters(), lr=0.0001)
        foldTrainDataset = []
        foldValDataset = []
        foldTestDataset =[]
        for i in listOfGenData_Test:
            foldTestDataset.append(i)
        for i in train_indices:
            foldTrainDataset.append(listOfGenData[i])
        for j in val_indices:
            foldValDataset.append(listOfGenData[j])

        trainDataloader = DataLoader(foldTrainDataset, batch_size=64)
        valDataLoader = DataLoader(foldValDataset, batch_size=64)
        testDataLoader = DataLoader(foldTestDataset,batch_size=64)
        model_new.train()
        for epoch in range(num_epochs):
            runningLoss = 0
            count = 0
            for batch in trainDataloader:
                optimizer.zero_grad()

                output = model_new(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device),
                                   batch.batch.to(device))
                loss = criterion2(output, batch.y.view(-1, output_dim).float().to(device))
                loss.backward()
                optimizer.step()

                loss2 = torch.abs(output - batch.y.view(-1, output_dim).float().to(device))

                runningLoss = runningLoss + loss2.sum().item()

                count=count + 64

            runningLoss = runningLoss/count


            if (epoch + 1) % 2 == 0:



                print(str(fold + 1) + " iteration")

                print(f'Epoch {epoch + 1}/{num_epochs}, Training  Loss: {runningLoss}')



                print("*****validation loss****")
                model_new.eval()
                with torch.no_grad():
                    runningValidationLoss =0
                    runningTestLoss =0
                    count =0
                    countTest=0
                    for batch in valDataLoader:
                        # Extract batch data

                        val_output = model_new(batch.x.to(device), batch.edge_index.to(device),
                                               batch.edge_attr.to(device), batch.batch.to(device))


                        absDiff = torch.abs(val_output - batch.y.view(-1, output_dim).float().to(device))
                        runningValidationLoss = runningValidationLoss + absDiff.sum().item()

                        count = count + 64

                    for batch in testDataLoader:
                        # Extract batch data

                        test_output = model_new(batch.x.to(device), batch.edge_index.to(device),
                                               batch.edge_attr.to(device), batch.batch.to(device))


                        absDiff = torch.abs(test_output - batch.y.view(-1, output_dim).float().to(device))
                        runningTestLoss = runningTestLoss + absDiff.sum().item()

                        countTest = countTest + 64


                training_loss.append(runningLoss)
                validation_loss.append(runningValidationLoss/count)



                print("Validation MAE")
                print(runningValidationLoss/count)
                print("Test MAE")
                print(runningTestLoss / countTest)

                if (runningValidationLoss/count) < 50 and runningLoss <90 :
                    print("target  reached")
                    break

        model_new.eval()
        predictions = []
        actual_values = []

        test_predictions = []
        test_actual_values = []
        with torch.no_grad():
            for batch in valDataLoader:
                # Extract batch data

                val_output = model_new(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device),
                                       batch.batch.to(device))

                predictions.append(val_output)
                actual_values.append(batch.y.to(device))

            for batch in testDataLoader:
                # Extract batch data

                test_output = model_new(batch.x.to(device), batch.edge_index.to(device),
                                        batch.edge_attr.to(device), batch.batch.to(device))

                test_predictions.append(test_output)

                test_actual_values.append(batch.y.to(device))

        test_predictions = torch.cat(test_predictions)
        # actual_values = -1 * np.square(actual_values)
        test_actual_values = torch.cat(test_actual_values)

        test_predictions = torch.flatten(test_predictions)

        test_actual_values = torch.flatten(test_actual_values)

        test_predictions_path = "gatn_test_predictions" + str(fold + 1) + ".csv"
        test_truevalues_path = "gatn_test_truevalues" + str(fold + 1) + ".csv"

        test_moleculeIndexPath = "gatn_test_mol_index" + str(fold + 1) + ".csv"



        with open(test_predictions_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['predictions'])  # Write header
            for item in test_predictions:
                csv_writer.writerow([item.item()])

        with open(test_truevalues_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['true_values'])  # Write header
            for item in test_actual_values:
                csv_writer.writerow([item.item()])

        with open(test_moleculeIndexPath, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['mol_index'])  # Write header
            for item in range(100, len(genData)):
                if item%100==0:
                    csv_writer.writerow([item])




        # predictions = -1 * np.square(predictions)
        predictions = torch.cat(predictions)
        # actual_values = -1 * np.square(actual_values)
        actual_values = torch.cat(actual_values)

        predictions = torch.flatten(predictions)

        actual_values = torch.flatten(actual_values)

        predictions_path = "gatn_predictions" + str(fold + 1) + ".csv"
        truevalues_path = "gatn_truevalues" + str(fold + 1) + ".csv"

        moleculeIndexPath = "gatn_mol_index" + str(fold + 1) + ".csv"

        # Write the list to the CSV file as a column
        with open(predictions_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['predictions'])  # Write header
            for item in predictions:
                csv_writer.writerow([item.item()])

        with open(truevalues_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['true_values'])  # Write header
            for item in actual_values:
                csv_writer.writerow([item.item()])

        with open(moleculeIndexPath, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['mol_index'])  # Write header
            for item in val_indices:
                csv_writer.writerow([item])

        testLossPath = "gatn_testLossFold" + str(fold+1) + ".csv"
        validationLossPath = "gatn_validationLossFold" + str(fold + 1) + ".csv"


        with open(testLossPath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["MAE Loss"])  # Write header
            writer.writerows([[value] for value in training_loss])
        with open(validationLossPath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["MAE Loss"])  # Write header
            writer.writerows([[value] for value in validation_loss])


        print(str(fold + 1) + " iteration" + " Statistics")
        print("********PRINTING  MEAN ABSOLUTE ERROR for Validation *********")
        mae = L1Loss()
        print(mae(predictions, actual_values))

        absolute_error = abs(predictions - actual_values)
        absolute_error_np = absolute_error
        iqrlist.append(absolute_error_np)



num_epochs = 60000

train(num_epochs)

# model = torch.load("model_3.pt")
# model.eval()
#
# predictions = []
# actual_values =[]
# with torch.no_grad():
#     for batch in dataloader_Test:
#         # Extract batch data
#
#         output = model(batch.x, batch.edge_index,batch.batch)
#
#
#         predictions.append(output)
#         actual_values.append(output)
#
# # Concatenate the predictions from all batches
# predictions = torch.cat(predictions)
#
# actual_values = torch.cat(actual_values)

# Print the predictions
# print("********PRINTING MEAN ABSOLUTE ERROR*********")
# from torch.nn import L1Loss
# mae = L1Loss()
# print(mae(torch.tensor(predictions, dtype=torch.float), torch.tensor(predictions, dtype=torch.float)))
# print("**********************************************")
