'''
Author: ZMJ majian2020@hotmail.com
Date: 2023-01-05 18:09:38
LastEditors: ZMJ majian2020@hotmail.com
LastEditTime: 2023-01-05 18:18:27
FilePath: \Feature_Importance_Analysis\main.py
Description: 

Copyright (c) 2023 by ZMJ majian2020@hotmail.com, All Rights Reserved. 
'''

from fea_rank import FeaImportance
from fea_visual import FeaVisual

if __name__ == "__main__":
    chems = ['CC(N(C(C1=C2C=CC=C1)=O)C2=O)C(O)=O',
             'CC1=CC=C(S(N2CCC(C(O)=O)CC2)(=O)=O)C=C1'
             ]
    data_path = 'MAF.xlsx'
    seed = 0
    test_size = 0.2
    top_k = 30
    bits = 128

    fea_importance = FeaImportance(data_path=data_path,seed=seed,test_size = test_size,top_k = top_k,fea_nums=bits)
    fea_importance.data_pre()
    fea_importance.rf_fit()
    features_rank = fea_importance.get_fea()


    for smiles in chems:
        fea_visual =  FeaVisual(smiles=smiles,bit = bits,fea = features_rank)
        fea_visual.get_bit()
        fea_visual.get_fragment()


