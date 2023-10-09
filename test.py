#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import scanpy as sc
import pandas as pd
import anndata
import torch
import scipy
import time
from STEM.model import *
from STEM.utils import *

def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map

def main():
    # parser.add_argument('--device', default='npu', type=str, help='npu or gpu')                        
    # parser.add_argument('--addr', default='127.0.0.1', type=str, help='master addr')                       
    # parser.add_argument('--device_list', default='4,5,6,7', type=str, help='device id list')                  
    # parser.add_argument('--dist_backend', default='hccl', type=str,
    #                     help='distributed backend')

    print('loadings')
    scdata = anndata.read_h5ad('./data/fetal/scRNA_merge_Annoed.h5ad')
    #scdata = scdata.T
    stdata = anndata.read_h5ad('./data/fetal/BL_D5_lasso_cellbin.h5ad')                      #stdata = stdata.T

    sc.pp.calculate_qc_metrics(scdata,percent_top=None, log1p=False, inplace=True)
    scdata.obs['n_genes_by_counts'].median()

    sc.pp.calculate_qc_metrics(stdata,percent_top=None, log1p=False, inplace=True)
    stdata.obs['n_genes_by_counts'].median()



    # 定义条件：提取批次为 'batch_1' 的细胞
    condition = scdata.obs['batch'] == 'w15_2/5'

    # 使用布尔索引提取满足条件的细胞
    adata = scdata[condition, :]

    spcoor=np.array(stdata.obs[['x','y']])
    print(spcoor)



    st_neighbor = scipy.spatial.distance.cdist(spcoor,spcoor)
    sigma = 3
    st_neighbor = np.exp(-st_neighbor**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

    # 获取两个 AnnData 对象的基因名称
    genes1=adata.var_names
    genes2=stdata.var_names
    intersection = list(set(genes1) & set(genes2))
                            
    intscdata = adata[:,intersection].copy()
    intstdata = stdata[:,intersection].copy()
                            
    sc.pp.calculate_qc_metrics(intscdata,percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(intstdata,percent_top=None, log1p=False, inplace=True)
    dp = 1-intscdata.obs['n_genes_by_counts'].median()/intstdata.obs['n_genes_by_counts'].median()
    print(dp)
                            
    dp=0
                            
    sc.pp.normalize_total(intstdata)
    sc.pp.log1p(intstdata)
    print(intstdata.X.shape)
    sc.pp.highly_variable_genes(intstdata, n_top_genes=2000)
    # 获取高变异基因的掩码
    highly_variable_mask = intstdata.var['highly_variable']
    print(len(highly_variable_mask))
    intstdata_top=intstdata[:,highly_variable_mask]
    print(intstdata_top.X.shape)
    print(len(intstdata_top.var_names))
    sc.pp.normalize_total(intscdata)
    sc.pp.log1p(intscdata)
    intscdata_top=intscdata[:,highly_variable_mask]

    sc_adata_df=pd.DataFrame(intscdata_top.X.toarray(),index=intscdata_top.obs_names,columns=intscdata_top.var_names)
    st_adata_df=pd.DataFrame(intstdata_top.X.toarray(),index=intstdata_top.obs_names,columns=intstdata_top.var_names)                           

    class setting( object ):
        pass
    seed_all(2022)
    opt= setting()
    setattr(opt, 'device', 'cuda:0')
    setattr(opt, 'outf', 'log/fetal_brain_hms')
    setattr(opt, 'n_genes', sc_adata_df.shape[1])
    setattr(opt, 'no_bn', False)
    setattr(opt, 'lr', 0.002)
    setattr(opt, 'sigma', 0.5)
    setattr(opt, 'alpha', 0.8)
    setattr(opt, 'verbose', True)
    setattr(opt, 'mmdbatch', 128)
    setattr(opt, 'dp', 0)

    testmodel = SOmodel(opt)
    testmodel.togpu()
    loss_curve = testmodel.train_wholedata(500,torch.tensor(sc_adata_df.values).float(),torch.tensor(st_adata_df.values).float(),torch.tensor(spcoor).float())

if __name__ == '__main__':
    main()
