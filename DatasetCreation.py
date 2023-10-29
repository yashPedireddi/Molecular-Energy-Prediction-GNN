from molmagic.parser import read_sdf_archive
from pathlib import Path
from openbabel import openbabel as ob
from openbabel import pybel as pb
import re
import torch
from torch_geometric.data import Data
import os
from torch_geometric.loader import DataLoader
import yaml
import math
import numpy as np
from typing import Any, DefaultDict, Generator, Iterator, TypeVar
from numpy import sign
from decimal import Decimal


class GenerateDataset:

    def __init__(self, sdfPath, datasetPath):
        self.sdfPath = sdfPath
        self.datasetPath = datasetPath

    def _should_reverse(self,arr: list[Any]) -> bool:

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

    def _sort_atoms(self,atoms: tuple[ob.OBAtom]) -> tuple[ob.OBAtom]:

        atom_nums = [i.GetAtomicNum() for i in atoms]
        return atoms[::-1] if self._should_reverse(atom_nums) else atoms

    def CreateNodeFeatures(self, molecule):
        listOfAllNodeFeatures = []

        # for j in ob.OBMolAtomIter(molecule.OBMol):
        #     nodeFeature = [j.GetAtomicNum()*100, j.AverageBondAngle(), j.GetHyb()*100, j.IsChiral()*100, j.IsAromatic()*100,
        #                    j.IsInRing()*100,
        #                    j.HighestBondOrder()*100, j.GetX()*100, j.GetY()*100, j.GetZ()*100]
        #
        #     listOfAllNodeFeatures.append(nodeFeature)

        for j in ob.OBMolAtomIter(molecule.OBMol):
            final_features = []

            # final_features.append(j.GetX())
            # final_features.append(j.GetY())
            # final_features.append(j.GetZ() *100)
            # final_features.append(j.GetAtomicNum()*1000)
            if j.GetAtomicNum()==1:
                final_features.append(1*100)
                final_features.append(0)
            else:
                final_features.append(0)
                final_features.append(1*100)




            final_features.append(j.GetHyb() )
            final_features.append(j.IsAromatic() )
            final_features.append(j.IsInRing() )
            final_features.append(j.HasAlphaBetaUnsat() )
            final_features.append(j.HasAromaticBond() )


            ringCount =[0,0,0,0,0]
            ringSize = [0,0,0,0,0,0,0,0,0,0]

            atomRingSize = j.MemberOfRingSize()
            atomRingCount = j.MemberOfRingCount()

            if(atomRingCount==0):
                ringCount[0]=1
            elif(atomRingCount==1):
                ringCount[1] = 1
            elif (atomRingCount == 2):
                ringCount[2] = 1
            elif (atomRingCount == 3):
                ringCount[3] = 1
            elif (atomRingCount == 4):
                ringCount[4] = 1

            if(atomRingSize==0):
                ringSize[0]=1
            elif(atomRingCount==3):
                ringSize[1] = 1
            elif (atomRingCount == 4):
                ringSize[2] = 1
            elif (atomRingCount == 5):
                ringSize[3] = 1
            elif (atomRingCount == 6):
                ringSize[4] = 1
            elif (atomRingCount == 7):
                ringSize[5] = 1
            elif (atomRingCount == 8):
                ringSize[6] = 1
            elif (atomRingCount == 9):
                ringSize[7] = 1
            elif (atomRingCount == 10):
                ringSize[8] = 1
            elif (atomRingCount == 11):
                ringSize[9] = 1







            final_features.append(j.CountBondsOfOrder(1) )
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

            bond_dict = {"11": 0, "16": 0, "66": 0}
            angle_dict = {"666": 0, "166": 0, "161": 0}
            kde_ch = [0, 0, 0]
            kde_cc = [0, 0, 0, 0]
            for ob_bond in ob.OBMolBondIter(molecule.OBMol):


                if (ob_bond.GetBeginAtom().GetIdx() == atom_index or ob_bond.GetEndAtom().GetIdx() == atom_index):
                    b = ob_bond.GetBeginAtom().GetAtomicNum()
                    e = ob_bond.GetEndAtom().GetAtomicNum()

                    be = (str(b) + str(e)) if b <= e else str(e) + str(b)

                    # bond_dict[be] = bond_dict[be] + ob_bond.GetLength()
                    #
                    bond_dict[be] = bond_dict[be] + 1
                    if be == "16" or be == "61":
                        if  ob_bond.GetLength()<1.07:
                            kde_ch[0] = kde_ch[0] + 1

                        elif ob_bond.GetLength() >= 1.07 and ob_bond.GetLength()<=1.09:
                            kde_ch[1] = kde_ch[1] + 1

                        elif ob_bond.GetLength() >= 1.09 and ob_bond.GetLength()<=1.2:
                            kde_ch[2] = kde_ch[2] + 1


                    if be == "66":

                        if  ob_bond.GetLength()<1.25:
                            kde_cc[0] = kde_cc[0] + 1

                        elif ob_bond.GetLength() >= 1.3 and ob_bond.GetLength()<=1.4:
                            kde_cc[1] = kde_cc[1] + 1

                        elif ob_bond.GetLength() >= 1.4 and ob_bond.GetLength()<=1.48:
                            kde_cc[2] = kde_cc[2] + 1

                        elif ob_bond.GetLength() > 1.48 and ob_bond.GetLength()<=2:
                            kde_cc[3] = kde_cc[3] + 1

            for i in kde_ch:
                final_features.append(i)
            for j in kde_cc:
                final_features.append(j)

            kde_ccc = [0,0,0,0,0,0]
            kde_hcc = [0,0,0,0]
            kde_hch = [0,0,0,0,0]

            for ob_angle in ob.OBMolAngleIter(molecule.OBMol):

                if (molecule.OBMol.GetAtom(ob_angle[0] + 1).GetIdx() == atom_index):
                    a = molecule.OBMol.GetAtom(ob_angle[1] + 1).GetAtomicNum()
                    b = molecule.OBMol.GetAtom(ob_angle[0] + 1).GetAtomicNum()
                    c = molecule.OBMol.GetAtom(ob_angle[2] + 1).GetAtomicNum()
                    aaa = self.swap_sort(a, b, c)

                    atoms = (molecule.OBMol.GetAtom(ob_angle[1] + 1), molecule.OBMol.GetAtom(ob_angle[0] + 1),
                             molecule.OBMol.GetAtom(ob_angle[2] + 1))

                    # angle_dict[aaa] = angle_dict[aaa] + molecule.OBMol.GetAngle(*atoms)
                    angle_dict[aaa] = angle_dict[aaa] + 1
                    angle = molecule.OBMol.GetAngle(*atoms)
                    if (aaa == "666"):
                        if angle>0 and angle <65:
                            kde_ccc[0]=kde_ccc[0]+1
                        elif angle >65 and angle <=80:
                            kde_ccc[1] = kde_ccc[1] + 1

                        elif angle>80 and angle <=100:

                            kde_ccc[2] = kde_ccc[2] + 1

                        elif angle > 100 and angle<=120:

                            kde_ccc[3] = kde_ccc[3] + 1

                        elif angle>120 and angle <=140:

                            kde_ccc[4] = kde_ccc[4] + 1

                        elif angle >140:

                            kde_ccc[5] = kde_ccc[5] + 1


                    elif(aaa == "166"):

                        if angle>0 and angle <=116:
                            kde_hcc[0]=kde_hcc[0]+1


                        elif angle>116 and angle <=120:

                            kde_hcc[1]=kde_hcc[1]+1

                        elif angle > 120 and angle<=140:

                            kde_hcc[2]=kde_hcc[2]+1

                        elif angle>140 :
                            kde_hcc[3] = kde_hcc[3] + 1


                    elif (aaa == "161"):
                        if angle>0 and angle <=80:
                            kde_hch[0]=kde_hch[0]+1


                        elif angle>80 and angle <=100:

                            kde_hch[1]=kde_hch[1]+1

                        elif angle > 100 and angle<=120:

                            kde_hch[1]=kde_hch[1]+1
                        elif angle > 120 and angle<=100:

                            kde_hch[1]=kde_hch[1]+1

                        elif angle>140 :
                            kde_hch[1]=kde_hch[1]+1

            for i in kde_ccc:
                final_features.append(i)
            for j in kde_hcc:
                final_features.append(j)

            for k in kde_hch:
                final_features.append(k)






            orderedBondLength = [0,0,0]
            orderedAngle = [0,0,0]
            for key in bond_dict:
                if key == "11":
                    orderedBondLength[0] = bond_dict[key]
                elif key == "16":
                    orderedBondLength[1] = bond_dict[key]
                else:
                    orderedBondLength[2] = bond_dict[key]
            for key in angle_dict:
                if key == "666":
                    orderedAngle[0] = angle_dict[key]
                elif key == "166":
                    orderedAngle[1] = angle_dict[key]
                else:
                    orderedAngle[2] = angle_dict[key]

            for key in orderedBondLength:
                final_features.append(key)

            for key in orderedAngle:
                final_features.append(key)

            listOfAllNodeFeatures.append(final_features)

        return torch.tensor(listOfAllNodeFeatures, dtype=torch.float)

    def CreateEdgeWeights(self, molecule):
        listOfEdgeWeights = []


        for ob_bond in ob.OBMolBondIter(molecule.OBMol):

            listOfEdgeWeights.append(ob_bond.GetBondOrder())



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
            if setOfAtomsString == atomString:

                nodeData = self.CreateNodeFeatures(molecule)
                edgeIndicesData = self.CreateEdgeIndices(molecule)
                edgeWeightsData = self.CreateEdgeWeights(molecule)

                targetData = torch.tensor(float(molecule.data['free_energy']))

                # targetData = torch.Tensor([targetData.item()])

                if not os.path.exists(self.datasetPath):
                    # Create the folder
                    os.makedirs(self.datasetPath)

                torch.save(Data(x=nodeData,
                                edge_index=edgeIndicesData,

                                edge_attr= edgeWeightsData,
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
#
# genData.CreateDataset("CH")
#
# print(genData[1].y)
# print(len(genData))
