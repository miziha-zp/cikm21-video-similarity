#!/usr/bin/env python
# coding: utf-8

# In[1]:
ALLIN = False
EPOCH = 10
START_INFERTEN_EPOCH = 6
BATCHSIZE = 2048
BATCHSIZETEST = 2048
MODELSZ = 1536
import os
import sys
import joblib
import numpy as np 
# from tqdm import tqdm_notebook as tqdm
sys.path.append('../')
annotation_file = '/data03/yrqUni/Workspace/QQB/Data/pairwise/label.tsv'

import json
from zipfile import ZipFile, ZIP_DEFLATED
# from util import test_spearmanr

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from torch.utils.data import DataLoader

import os

import scipy
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def test_spearmanr(vid_embedding, annotation_file):
    relevances, similarities = [], []
    with open(annotation_file, 'r') as f:
        for line in tqdm(f):
            query, candidate, relevance = line.split()
            if query not in vid_embedding:
                print(f'ERROR: {query} NOT found')
                continue
                # raise Exception(f'ERROR: {query} NOT found')
            if candidate not in vid_embedding:
                print(f'ERROR: {candidate} NOT found')
                continue
                # raise Exception(f'ERROR: {candidate} NOT found')
            # print('pass')
            query_embedding = vid_embedding.get(query)
            candidate_embedding = vid_embedding.get(candidate)
            similarity = cosine_similarity([query_embedding], [candidate_embedding])[0][0]
            similarities.append(similarity)
            relevances.append(float(relevance))

    spearmanr = scipy.stats.spearmanr(similarities, relevances).correlation
    return spearmanr


# yrq_valid
import pickle
with open('../embedding/yrq_VAL_1.pickle', 'rb') as file:
    yrq_bert_valid = pickle.load(file)
    
with open('../embedding/yrq_TEST_1.pickle', 'rb') as file:
    yrq_bert_test = pickle.load(file)
    
with open('../embedding/yrq_VAL_2.pickle', 'rb') as file:
    yrq_roberta_valid = pickle.load(file)
    
with open('../embedding/yrq_TEST_2.pickle', 'rb') as file:
    yrq_roberta_test = pickle.load(file)
    


valid_two_stage_file = '../embedding/test_bitm_vision_transdp01_bertlast_valid.pkl'
test_two_stage_file = '../embedding/test_bitm_vision_transdp01_bertlast_test.pkl'

baseline_valid = joblib.load(valid_two_stage_file)
baseline_test = joblib.load(test_two_stage_file)

valid_two_stage_file2 = '../embedding/test_robitm_vision_transdp01_bertlast_valid.pkl'
test_two_stage_file2 = '../embedding/test_robitm_vision_transdp01_bertlast_test.pkl'

baseline_valid2 = joblib.load(valid_two_stage_file2)
baseline_test2 = joblib.load(test_two_stage_file2)
zlh_trans_valid = joblib.load('../embedding/valid_two_stagedict_input_zlh1002.pkl')
zlh_trans_test = joblib.load('../embedding/test_two_stagedict_input_zlh1002.pkl')

roberta_valid = joblib.load('../embedding/roberta_valid_seq.pkl')
roberta_test = joblib.load('../embedding/roberta_test_seq.pkl')

valid_knowledgePool_t = baseline_valid + roberta_valid + zlh_trans_valid + baseline_valid2 + yrq_bert_valid[:3]+yrq_roberta_valid[:3]
test_knowledgePool_t = baseline_test + roberta_test + zlh_trans_test + baseline_test2 + yrq_bert_test[:3] + yrq_roberta_test[:3]
 
# valid_knowledgePool_t = ich_valid
# test_knowledgePool_t = ich_test
# knowledgePool is list of dict 
'''
every dict like:
{
    "vid0":np.array([1,2,3])...
}
'''


# In[10]:



valid_knowledgePool = []
test_knowledgePool = []
print(len(valid_knowledgePool_t))
for k1,k2 in zip(valid_knowledgePool_t, test_knowledgePool_t):
    if len(k1['2345203561710400875'].shape) == 1:
#         score = test_spearmanr(k1, annotation_file)
#         print(k1['2345203561710400875'].shape, score)
        valid_knowledgePool.append(k1)
        test_knowledgePool.append(k2)

for k1, k2 in zip(valid_knowledgePool_t, test_knowledgePool_t):
    if len(k1['2345203561710400875'].shape) == 2:
        print(k1['2345203561710400875'].shape)
        valid_knowledgePool.append(k1)
        test_knowledgePool.append(k2)
del valid_knowledgePool_t, test_knowledgePool_t
import gc
gc.collect()


# In[11]:


for i in valid_knowledgePool[0]:
    print(i)
    break


# In[12]:


inputszlist = [know['2345203561710400875'].shape for know in valid_knowledgePool]
print(inputszlist)


# In[13]:


import os
import pandas as pd
import torch
from torch.utils.data import Dataset
# from torchvision import datasets
from torchvision.transforms import ToTensor

class ValidDataSet(Dataset):
    def __init__(self, knowledgePool, df, valid=True):
        self.df = df
        self.knowledgePool = knowledgePool

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        line = self.df.iloc[idx]
        query, candidate, relevance = line['query'], line["candidate"], line["relevance"]

        queryemblist = []
        for knowledge in self.knowledgePool:
            queryemblist.append(torch.from_numpy(knowledge[query].astype(np.float32)))

        candidateemblist = []
        for knowledge in self.knowledgePool:
            candidateemblist.append(torch.from_numpy(knowledge[candidate].astype(np.float32)))

        relevance = float(relevance)
        return  tuple(queryemblist), tuple(candidateemblist), relevance
            

class testDataSet(Dataset):
    def __init__(self, knowledgePool, valid=True):
        self.knowledgePool = knowledgePool
        self.id = [id for id in self.knowledgePool[0]]

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        id = self.id[idx]
        emblist = []
        for knowledge in self.knowledgePool:
            emblist.append(torch.from_numpy(knowledge[id].astype(np.float32)))
            
        return emblist, id
            
        


# In[14]:


query_list = []
candidate_list = []
relevance_list = []

with open(annotation_file, 'r') as f:
    for line in f:
        query, candidate, relevance = line.split()
        query_list.append(query)
        candidate_list.append(candidate)
        relevance_list.append(relevance)


valid_df = pd.DataFrame({
    "query":query_list,
    "candidate":candidate_list,
    "relevance":relevance_list
})

valid_df['relevance'] =  valid_df['relevance'].apply(float)
print(valid_df['relevance'].value_counts(normalize=True))
valid_df.head()


# In[15]:



testdataset = testDataSet(test_knowledgePool)
test_dataloader = DataLoader(testdataset, batch_size=BATCHSIZETEST, shuffle=False, drop_last=False)


# In[ ]:


from sklearn.model_selection import train_test_split
# ?train_test_split

from KN_TEXTCNN import Network

# testing




def inference_one_epoch(test_dataloader, NET):
    vid_embedding = {}
    NET.eval()
    for data1, ids in test_dataloader:
        data1 = tuple([data.to(DEVICE) for data in data1])
        with torch.no_grad():
            out1 = NET(data1).detach().cpu().numpy().astype(np.float16)
        for vid, embedding in zip(ids, out1):
            vid_embedding[vid] = embedding.tolist()
    return vid_embedding


def merge_embedding(vid_embedding_list):
    info = f'total {len(vid_embedding_list)} embeddings'
    print(colored(info, 'blue'))
    embedding_number = len(vid_embedding_list)
    final_embedding = {}
    all_vids = [key for key in vid_embedding_list[0]] 
    from tqdm import tqdm
    for vid in tqdm(all_vids):
        ans = 0
        for embedding in vid_embedding_list:
            ans += np.array(embedding[vid]) #/ embedding_number
        final_embedding[vid] = ans.astype(np.float16).tolist()
    return final_embedding



import scipy
from torch.nn.functional import normalize, dropout

from customloss import *
from termcolor import colored
from sklearn.model_selection import train_test_split
TEMP = 0.2
OUTSZ = 256
LOSS = SpearmanCorrelationLoss(TEMP) 
# LOSS = WhiteMSE()
# LOSS = wiseMSE()
# LOSS = SentenceBertLoss(OUTSZ).cuda()
print(LOSS)
DEVICE = 'cuda:0'
print(len(inputszlist), inputszlist)

NET = Network(len(inputszlist), inputszlist, hiddensz=MODELSZ, outsz=OUTSZ)#.cuda()
if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        NET = nn.DataParallel(NET)
NET = NET.cuda()
# print(NET)
optimizer = torch.optim.Adam(NET.parameters(), lr=1e-4, weight_decay=1e-5)

from sklearn.model_selection import GroupKFold
np.random.seed(42)
group_kfold = GroupKFold(n_splits=10)
# train_part, valid_part = train_test_split(valid_df, test_size=0.01)

y = valid_df['relevance']
groups = valid_df['candidate']
train_part, valid_part = 0, 0 

if not ALLIN:
    for train_index, test_index in group_kfold.split(valid_df, y, groups):
        # print("TRAIN:", train_index, "TEST:", test_index)
        train_part, valid_part = valid_df.iloc[train_index], valid_df.iloc[test_index]
        print(train_part.shape)
        print(valid_part.shape)
        
        break
else:
    train_part, valid_part = valid_df, valid_df

training_data = ValidDataSet(valid_knowledgePool, train_part)
validing_data = ValidDataSet(valid_knowledgePool, valid_part)


train_dataloader = DataLoader(training_data, batch_size=BATCHSIZE, shuffle=True, drop_last=False, num_workers=4)
valid_dataloader = DataLoader(validing_data, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=4)


def inference_one_epoch(test_dataloader, NET):
    vid_embedding = {}
    for data1, ids in test_dataloader:
        data1 = tuple([data.cuda() for data in data1])
        with torch.no_grad():
            out1 = NET(data1).detach().cpu().numpy().astype(np.float16)
        for vid, embedding in zip(ids, out1):
            vid_embedding[vid] = embedding.tolist()
    return vid_embedding


def merge_embedding(vid_embedding_list):
    info = f'total {len(vid_embedding_list)} embeddings'
    print(colored(info, 'blue'))
    
    final_embedding = {}
    all_vids = [key for key in vid_embedding_list[0]] 
    from tqdm import tqdm
    for vid in tqdm(all_vids):
        ans = 0
        for embedding in vid_embedding_list:
            ans += np.array(embedding[vid])
        final_embedding[vid] = ans.astype(np.float16).tolist()
    return final_embedding


# NET = Network(len(inputszlist), inputszlist, hiddensz=1536, outsz=OUTSZ).to(DEVICE)
# print(NET)
vid_embedding_list = []
EPOCH = 8
START_INFERTEN_EPOCH = 5

for train_index, test_index in group_kfold.split(valid_df, y, groups):
    # print("TRAIN:", train_index, "TEST:", test_index)
    train_part, valid_part = valid_df.iloc[train_index], valid_df.iloc[test_index]
#     train_part = query_expansion(train_part)
    print(train_part.shape)
    print(valid_part.shape)

    training_data = ValidDataSet(valid_knowledgePool, train_part)
    validing_data = ValidDataSet(valid_knowledgePool, valid_part)


    train_dataloader = DataLoader(training_data, batch_size=BATCHSIZE, shuffle=True, drop_last=False, num_workers=4)
    valid_dataloader = DataLoader(validing_data, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=4)
    
    NET = Network(len(inputszlist), inputszlist, hiddensz=1536, outsz=OUTSZ).to(DEVICE)
    # print(NET)
    optimizer = torch.optim.Adam(NET.parameters(), lr=1e-4, weight_decay=0)


    for epoch in range(EPOCH):
        data_enumerator = enumerate(train_dataloader)
        valid_enumerator = enumerate(valid_dataloader)
        loss_epoch = 0
        train_ans = []
        y_list = []
        NET.train()
        for step, (data1, data2, y) in data_enumerator:
            optimizer.zero_grad()
            data1 = tuple([data.to(DEVICE) for data in data1])
            data2 = tuple([data.to(DEVICE) for data in data2])

            y = y.float().to(DEVICE)
            out1 = NET(data1)
            out2 = NET(data2)
            loss, preds = LOSS(out1, out2, y, return_sim=True)
            # print(preds.shape)
            # print(y.shape)
            # preds = cosine(out1, out2)

            train_ans.extend(list(preds.detach().cpu().numpy()))
            y_list.extend(list(y.detach().cpu().numpy()))
            loss_epoch += loss.item()
            if step % 50 == 0:
                print(y)
                print(f"epoch:{epoch} \t , step:{step}\t loss:", loss.item())
            loss.backward()
            optimizer.step()

        if epoch >= START_INFERTEN_EPOCH:
            vid_emb = inference_one_epoch(test_dataloader, NET)
            vid_embedding_list.append(vid_emb)

        if not ALLIN:
            spearmanr = scipy.stats.spearmanr(train_ans, y_list).correlation
            info = f"epoch:{epoch}\t spearmanr:{spearmanr} \tloss_epoch:{loss_epoch / len(train_dataloader)}"
            print(colored(info, 'blue'))

        loss_epoch = 0
        val_ans = []
        y_list = []
        if not ALLIN:
            NET.eval()
            for step, (data1, data2, y) in valid_enumerator:
                optimizer.zero_grad()
                data1 = tuple([data.to(DEVICE) for data in data1])
                data2 = tuple([data.to(DEVICE) for data in data2])

                y = y.float().to(DEVICE)
                with torch.no_grad():
                    out1 = NET(data1)
                    out2 = NET(data2)
                    # print(final_emb1)
                    loss, preds = LOSS(out1, out2, y, return_sim=True)
                    val_ans.extend(list(preds.detach().cpu().numpy()))
                    y_list.extend(list(y.detach().cpu().numpy()))
                    loss_epoch += loss.item()
                if step % 10 == 0:
                    print(y)
                    print(f"valid epoch:{epoch} \t , step:{step}\t loss:", loss.item())

            spearmanr = scipy.stats.spearmanr(val_ans, y_list).correlation
            info = f"valid ::: epoch:{epoch}\t spearmanr:{spearmanr} \tloss_epoch:{loss_epoch / len(valid_dataloader)}"
            print(colored(info, 'green'))


vid_embedding = merge_embedding(vid_embedding_list)


path = 'two_stage/TTA22'
import os

if not os.path.exists(path):
    os.makedirs(path)
output_json = path + '/result.json'
output_zip = path + '/result.zip'

import json
import joblib
from zipfile import ZIP_DEFLATED, ZipFile

with open(output_json, 'w') as f:
    json.dump(vid_embedding, f)
with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
    zip_file.write(output_json)



# ##### 
