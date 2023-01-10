from rdkit.Chem import AllChem
import numpy  as np
from rdkit.DataStructs import cDataStructs
from rdkit import Chem
from rdkit.Chem import Draw
from PropertyMol import PropertyMol
import pandas as pd


class FeaVisual():
    def __init__(self,smiles,bit,fea) -> None:
        self.smiles = smiles
        self.bit = bit
        self.fea = fea
        self.mol = Chem.MolFromSmiles(self.smiles)

    #
    def get_bit(self):
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(self.mol,radius = 2,bitInfo = bit_info, nBits=self.bit)
        ecfp_arr = np.zeros((0,),dtype=np.int8)
        cDataStructs.ConvertToNumpyArray(fp,ecfp_arr)
        fea_dict = {k:v for k,v in bit_info.items() if k in self.fea.keys()}
        print(fea_dict)
        self.fea_dict = fea_dict
        
    def get_fragment(self):
        submols  = []
        importances = []
        for k,v in self.fea_dict.items():
            for atom_info in v:
                atom,radius= atom_info
                amap = {}
                env = Chem.FindAtomEnvironmentOfRadiusN(self.mol, radius, atom)
                submol=Chem.PathToSubmol(self.mol, env, atomMap=amap)
                pm = PropertyMol(submol)
                Importance = self.fea[k]
                pm.SetProp('Importance',Importance)
                submols.append(pm)
                importances.append(Importance)
        
        pm = PropertyMol(self.mol)
        pm.SetProp('Importance',round(sum(importances),4))
        submols.insert(0,pm)
        img = Draw.MolsToGridImage(
            submols,
            molsPerRow = 3,

            subImgSize=(200,200),

            legends=[PropertyMol(x).GetProp('Importance') for x in submols if PropertyMol(x).HasProp('Importance')]
        )
        img.save(f'Result/{self.smiles}.jpg')
        return True


