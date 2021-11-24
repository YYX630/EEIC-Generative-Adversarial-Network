import pathlib
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pickle
import os
import csv
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("./transformers/")
p_temp =  pathlib.Path('./confirmed_fronts/').glob('**/*.jpg')

model.eval()
cnt = 0
for p in p_temp:
    attr = p.name.split('$$')
    sentence_embeddings = model.encode(attr[3].lower())
    path = str(p)
    vec_path = path.replace('.jpg', '.txt')
    if os.path.exists(os.path.dirname(vec_path)) != True:
        os.makedirs(os.path.dirname(vec_path))
    with open(vec_path, 'wb') as fp:
        pickle.dump(sentence_embeddings.tolist(), fp)
    print(cnt)
    cnt += 1
