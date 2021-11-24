import json
import numpy as np
import os
import pickle
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
import logging
import matplotlib.pyplot as plt

annotation_list = ["QA_Action.json", "QA_Combined_Action_Reason.json",
                    "QA_Reason.json", "Sentiments.json",
                    "Slogans.json", "Strategies.json",
                    "Symbols.json", "Topics.json"
                    ]

jsons = []
for i in range (len(annotation_list)):
    jsons.append(json.load(open(os.path.join("./annotation", annotation_list[i]), 'r')))
f = open('./image_name_list/all_images.txt', 'r')
cnt = 0
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
while True:
    img_path = f.readline()
    img_path = img_path.replace("\n", "")
    if img_path:
        if img_path in jsons[0]:
            text = ""
            for i in jsons[0][img_path]:
                text += i
            for i in jsons[2][img_path]:
                text += i
            sentence_embedding = model.encode(text)
            img_path = img_path.replace('.jpg', '.txt')
            img_path = img_path.replace('.png', '.txt')
            with open('./image/' + img_path, 'wb') as fp:
                pickle.dump(sentence_embedding.tolist(), fp)
            print(cnt)
            cnt += 1
    else:
        break