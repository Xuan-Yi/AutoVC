import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--num_uttrs", dest="num_uttrs", default=10, type=int, help="Number of different contents")
parser.add_argument("--rootDir", dest="rootDir", default='./spmel', help="spmel")
parser.add_argument("--model", dest="model_path", default='3000000-BL.ckpt', help="Path of 3000000-BL.ckpt")

args = parser.parse_args()

class D_VECTOR(nn.Module):
    """d vector speaker embedding."""
    def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64):
        super(D_VECTOR, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell, 
                            num_layers=num_layers, batch_first=True)  
        self.embedding = nn.Linear(dim_cell, dim_emb)


    def forward(self, x):
        self.lstm.flatten_parameters()            
        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:,-1,:])
        norm = embeds.norm(p=2, dim=-1, keepdim=True) 
        embeds_normalized = embeds.div(norm)
        return embeds_normalized
    
import os
import pickle
from model_bl import D_VECTOR # type: ignore
from collections import OrderedDict
import numpy as np
import torch

C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load(args.model_path, map_location=torch.device('cuda'))
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)

# 指的是說一個語者說了幾種不同內容的話，讓資料的數量盡量一樣，內容可以不一樣。
num_uttrs = args.num_uttrs
len_crop = 176

# Directory containing mel-spectrograms
rootDir = args.rootDir
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size) # type: ignore
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    fileList = fileList[:num_uttrs]
    # make speaker embedding
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in range(num_uttrs):
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        # pad if the current one is too short   
        if tmp.shape[0] <= len_crop:
            tmp = pad_along_axis(tmp,len_crop)
            melsp = torch.from_numpy(tmp[np.newaxis,:, :]).cuda()
        else:            
            left = np.random.randint(0, tmp.shape[0]-len_crop)
            melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())    
        
    utterances.append(np.mean(embs, axis=0))
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)

with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)