from molmagic.parser import read_sdf_archive
from pathlib import Path
from openbabel import openbabel as ob
import re
import torch
from torch_geometric.data import Data
import os
from torch_geometric.loader import DataLoader
import yaml
import numpy as np
from typing import Any
import pandas as pd


class GenerateDataset:

    def __init__(self, sdfPath, datasetPath):
        self.sdfPath = sdfPath
        self.datasetPath = datasetPath
        # Load bin data
        self.binData = pd.read_csv("binData.csv")
        self.angleBinData = pd.read_csv("angleBinData.csv")

    def _should_reverse(self, arr: list[Any]) -> bool:

        middle = len(arr) // 2

        for left, right in zip(arr[:middle], reversed(arr[middle:])):

            if left == right:
                continue
            elif left < right:
                return True
            else:
                return False

        return False

    def swap_sort(self, a, b, c):

        if a > c:
            (a, c) = (c, a)

        return str(a) + str(b) + str(c)

    def _sort_atoms(self, atoms: tuple[ob.OBAtom]) -> tuple[ob.OBAtom]:

        atom_nums = [i.GetAtomicNum() for i in atoms]
        return atoms[::-1] if self._should_reverse(atom_nums) else atoms

    def CreateNodeFeatures(self, molecule):
        listOfAllNodeFeatures = []

        count_C = 0
        count_H = 0
        count_O = 0
        count_N = 0

        for j in ob.OBMolAtomIter(molecule.OBMol):

            if j.GetAtomicNum() == 1:
                count_H = count_H + 1
                continue
            elif j.GetAtomicNum() == 6:
                count_C = count_C + 1
            elif j.GetAtomicNum() == 7:
                count_N = count_N + 1
            elif j.GetAtomicNum() == 8:
                count_O = count_O + 1

        for j in ob.OBMolAtomIter(molecule.OBMol):
            final_features = []

            final_features.append(count_C*100)
            final_features.append(count_H*100)
            final_features.append(count_O*100)
            final_features.append(count_N*100)
            if j.GetAtomicNum() == 1:
                final_features.append(1*1000)
                final_features.append(0)
                final_features.append(0)
                final_features.append(0)
            elif j.GetAtomicNum() == 6:
                final_features.append(0)
                final_features.append(1*1000)
                final_features.append(0)
                final_features.append(0)

            elif j.GetAtomicNum() == 7:
                final_features.append(0)
                final_features.append(0)
                final_features.append(1*1000)
                final_features.append(0)

            elif j.GetAtomicNum() == 8:
                final_features.append(0)
                final_features.append(0)
                final_features.append(0)
                final_features.append(1*1000)

            final_features.append(j.GetHyb())
            final_features.append(j.IsAromatic())
            final_features.append(j.IsInRing())
            # final_features.append(j.HasAlphaBetaUnsat())
            final_features.append(j.HasAromaticBond())
            # final_features.append(j.IsHeteroatom())
            final_features.append(j.IsHbondDonor())
            final_features.append(j.IsHbondAcceptor())

            ringCount = [0, 0, 0, 0, 0]
            ringSize = [0, 0, 0, 0, 0, 0, 0, 0, 0]

            atomRingSize = j.MemberOfRingSize()
            atomRingCount = j.MemberOfRingCount()

            ringCount[atomRingCount] = 1

            if (atomRingCount != 0):
                ringSize[atomRingCount - 3] = 1

            final_features.append(j.CountBondsOfOrder(1))
            final_features.append(j.CountBondsOfOrder(2))
            final_features.append(j.CountBondsOfOrder(3))
            for p in ringCount:
                final_features.append(p)
            for q in ringSize:
                final_features.append(q)
            atom_index = j.GetIdx()
            """
            plot all the lengths in the dataset

            assign a value closest to the minima peak in the kde plot of bond lengths
            classify the bond based on the distance between them

            c-c bonds four significant peaks

            plot the distances in the dataset using histogram.

            remove 3 carbon atoms or less

            atomic number  convert to one hot encoding

            First step analyze the clusters and find a differentiating factor
            ***** Loop through the clusters ******


            """
            atom_index = j.GetIdx()
            bondDict = {

                '16': [0, 0, 0, 0, 0, 0],
                '17': [0, 0, 0, 0, 0, 0],
                '18': [0, 0, 0, 0, 0, 0],
                '66': [0, 0, 0, 0, 0, 0],
                '67': [0, 0, 0, 0, 0, 0],
                '68': [0, 0, 0, 0, 0, 0],
                '77': [0, 0, 0, 0, 0, 0],
                '78': [0, 0, 0, 0, 0, 0],
            }
            for ob_bond in ob.OBMolBondIter(molecule.OBMol):

                if (ob_bond.GetBeginAtom().GetIdx() == atom_index or ob_bond.GetEndAtom().GetIdx() == atom_index):
                    b = ob_bond.GetBeginAtom().GetAtomicNum()
                    e = ob_bond.GetEndAtom().GetAtomicNum()

                    be = (str(b) + str(e)) if b <= e else str(e) + str(b)

                    binIndex = np.digitize([ob_bond.GetLength()], self.binData[be])

                    bondDict[be][binIndex[0] - 1] = bondDict[be][binIndex[0] - 1] + 1

            for key in bondDict:
                for i in bondDict[key]:
                    final_features.append(i*100)

            angleDict = {'161': [0,0,0,0,0,0],
                         '166': [0,0,0,0,0,0],
                         '167': [0,0,0,0,0,0],
                         '168': [0,0,0,0,0,0],
                         '171': [0,0,0,0,0,0],
                         '176': [0,0,0,0,0,0],
                         '177': [0,0,0,0,0,0],
                         '178': [0,0,0,0,0,0],
                         '181': [0,0,0,0,0,0],
                         '186': [0,0,0,0,0,0],
                         '187': [0,0,0,0,0,0],
                         '666': [0,0,0,0,0,0],
                         '667': [0,0,0,0,0,0],
                         '668': [0,0,0,0,0,0],
                         '676': [0,0,0,0,0,0],
                         '677': [0,0,0,0,0,0],
                         '678': [0,0,0,0,0,0],
                         '686': [0,0,0,0,0,0],
                         '687': [0,0,0,0,0,0],
                         '767': [0,0,0,0,0,0],
                         '768': [0,0,0,0,0,0],
                         '777': [0,0,0,0,0,0],
                         '778': [0,0,0,0,0,0],
                         '787': [0,0,0,0,0,0],
                         '868': [0,0,0,0,0,0],
                         '878': [0,0,0,0,0,0]}

            for ob_angle in ob.OBMolAngleIter(molecule.OBMol):
                if (molecule.OBMol.GetAtom(ob_angle[0] + 1).GetIdx() == atom_index):
                    a = molecule.OBMol.GetAtom(ob_angle[1] + 1).GetAtomicNum()
                    b = molecule.OBMol.GetAtom(ob_angle[0] + 1).GetAtomicNum()
                    c = molecule.OBMol.GetAtom(ob_angle[2] + 1).GetAtomicNum()
                    aaa = self.swap_sort(a, b, c)

                    atoms = (molecule.OBMol.GetAtom(ob_angle[1] + 1), molecule.OBMol.GetAtom(ob_angle[0] + 1),
                             molecule.OBMol.GetAtom(ob_angle[2] + 1))

                    angle = molecule.OBMol.GetAngle(*atoms)

                    angleBinIndex = np.digitize([angle], self.angleBinData[aaa])

                    angleDict[aaa][angleBinIndex[0] - 1] = angleDict[aaa][angleBinIndex[0] - 1] + 1
            for key in angleDict:
                for i in angleDict[key]:
                    final_features.append(i)

            listOfAllNodeFeatures.append(final_features)

        return torch.tensor(listOfAllNodeFeatures, dtype=torch.float)

    def CreateEdgeWeights(self, molecule):
        listOfEdgeWeights = []

        for ob_bond in ob.OBMolBondIter(molecule.OBMol):
            bondOrderList = [0, 0, 0]
            templist = []

            if (ob_bond.GetBondOrder() == 1):
                bondOrderList[0] = 1
            elif (ob_bond.GetBondOrder() == 2):
                bondOrderList[1] = 1
            else:
                bondOrderList[2] = 1

            for i in bondOrderList:
                templist.append(i)

            templist.append(ob_bond.IsInRing())

            bondDict = {

                '16': [0, 0, 0, 0, 0, 0],
                '17': [0, 0, 0, 0, 0, 0],
                '18': [0, 0, 0, 0, 0, 0],
                '66': [0, 0, 0, 0, 0, 0],
                '67': [0, 0, 0, 0, 0, 0],
                '68': [0, 0, 0, 0, 0, 0],
                '77': [0, 0, 0, 0, 0, 0],
                '78': [0, 0, 0, 0, 0, 0],
            }

            b = ob_bond.GetBeginAtom().GetAtomicNum()
            e = ob_bond.GetEndAtom().GetAtomicNum()

            be = (str(b) + str(e)) if b <= e else str(e) + str(b)

            binIndex = np.digitize([ob_bond.GetLength()], self.binData[be])

            bondDict[be][binIndex[0] - 1] = bondDict[be][binIndex[0] - 1] + 1

            for key in bondDict:
                for i in bondDict[key]:
                    templist.append(i*100)

            listOfEdgeWeights.append(templist)

        return torch.tensor(listOfEdgeWeights, dtype=torch.float)

    def CreateEdgeIndices(self, molecule):
        listOfIndices = []
        startIndices = []
        endIndices = []
        for bond in ob.OBMolBondIter(molecule.OBMol):
            startIndices.append(bond.GetBeginAtomIdx() - 1)
            endIndices.append(bond.GetEndAtomIdx() - 1)

        listOfIndices.append(startIndices)
        listOfIndices.append(endIndices)

        return torch.tensor(listOfIndices, dtype=torch.long)

    def CreateDataset(self, atomString):

        sdfPath = Path(self.sdfPath)
        listOfMolecules = [i for i in read_sdf_archive(sdfPath)]
        count = 1

        for molecule in listOfMolecules:

            moleculeFormula = re.sub(r"[^a-zA-Z]", "", molecule.OBMol.GetFormula())
            setOfAtoms = set(moleculeFormula)
            setOfAtomsString = ""

            for i in setOfAtoms:
                setOfAtomsString = setOfAtomsString + i

            setOfAtomsString = ''.join(sorted(setOfAtomsString))
            #or atomString == "ALL"
            # uncomment to generate CH dataset
            #if setOfAtomsString == atomString:

            # Comment the below line to generate CH dataset
            if setOfAtomsString == atomString or atomString == "ALL" :
                # print(molecule.OBMol.GetFormula())

                nodeData = self.CreateNodeFeatures(molecule)
                edgeIndicesData = self.CreateEdgeIndices(molecule)
                edgeWeightsData = self.CreateEdgeWeights(molecule)

                targetData = torch.tensor(np.float64(molecule.data['free_energy']))

                if not os.path.exists(self.datasetPath):
                    # Create the folder
                    os.makedirs(self.datasetPath)

                torch.save(Data(x=nodeData,
                                edge_index=edgeIndicesData,

                                edge_attr=edgeWeightsData,
                                y=targetData), os.path.join(self.datasetPath, f'molecule_{count}.pt'))

                count = count + 1

    def GenerateDataloader(self):
        folder_path = os.path.join(os.getcwd(), self.datasetPath)
        listOfDataObject = []

        for i in range(1, len(folder_path)):
            data = torch.load(os.path.join(self.datasetPath, f'molecule_{i}.pt'))
            listOfDataObject.append(data)

        return DataLoader(listOfDataObject, batch_size=32)

    def __getitem__(self, item):
        data = torch.load(os.path.join(self.datasetPath, f'molecule_{item}.pt'))

        return data

    def __len__(self):
        folder_path = os.path.join(os.getcwd(), self.datasetPath)
        folderContent = os.listdir(folder_path)

        return len(folderContent)


with open("variablesConfig.yaml", "r") as file:
    variablesConfig = yaml.safe_load(file)

genData = GenerateDataset(variablesConfig['folders']['sdfPath'], variablesConfig['folders']['datasetPath'])
# genData.CreateDataset("CH")

genData.CreateDataset("ALL")
#
# print(genData[1].edge_attr.size())
# print(genData[2].x[5])
# print(len(genData))
