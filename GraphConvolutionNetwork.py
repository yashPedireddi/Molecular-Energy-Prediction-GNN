import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from DatasetCreation2 import GenerateDataset
import yaml
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torchsummary import summary
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np



import csv


def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)




if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU Available")
else:
    print("No GPU Available")


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv2(x, edge_index, edge_attr))

        # x = F.relu(self.conv4(x, edge_index))

        # x = global_mean_pool(x, batch)

        x = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


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
# for i in range(1,len(genData)-1,2):
#     listOfGenData.append(genData[i])
#     listOfGenData_Test.append(genData[i+1])


input_dim = listOfGenData[0].x.size(1)
hidden_dim = 128
output_dim = 1
model = GCN(input_dim, hidden_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

batch_size = 1
dataloader = DataLoader(listOfGenData, batch_size=64)

dataloader_Test = DataLoader(listOfGenData_Test, batch_size=32)

# data = next(iter(dataloader))
# print(data)
# print(data.batch)

from torch.nn import L1Loss


def reset_weights(m):

    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def train(model, num_epochs):
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

        model_new = GCN(input_dim, hidden_dim)
        model_new.to(device)
        model_new.apply(reset_weights)
        criterion2 = nn.MSELoss()
        optimizer = optim.Adam(model_new.parameters(), lr=0.01,weight_decay=5e-4)
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

                if (runningValidationLoss/count) < 9000 and runningLoss <9000 :
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

        test_predictions_path = "gcn_test_predictions" + str(fold + 1) + ".csv"
        test_truevalues_path = "gcn_test_truevalues" + str(fold + 1) + ".csv"

        test_moleculeIndexPath = "gcn_test_mol_index" + str(fold + 1) + ".csv"



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

        predictions_path = "gcn_predictions" + str(fold + 1) + ".csv"
        truevalues_path = "gcn_truevalues" + str(fold + 1) + ".csv"

        moleculeIndexPath = "gcn_mol_index" + str(fold + 1) + ".csv"

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

        testLossPath = "gcn_testLossFold" + str(fold+1) + ".csv"
        validationLossPath = "gcn_validationLossFold" + str(fold + 1) + ".csv"


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


num_epochs = 50000

train(model, num_epochs)

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
