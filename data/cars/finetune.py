import csv
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

csv_file = open("./color/colorhexa_com.csv", "r", encoding="ms932", errors="", newline="")
csvf = csv.reader(csv_file, delimiter = ",", doublequote = True, lineterminator = "\r\n", quotechar = '"', skipinitialspace=True)
header = next(csvf)

train_examples = []
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
for row in csvf:
    train_examples.append(InputExample(texts=[row[0], 'red'], label=int(row[2]) / 255))
    train_examples.append(InputExample(texts=[row[0], 'green'], label=int(row[3]) / 255))
    train_examples.append(InputExample(texts=[row[0], 'blue'], label=int(row[4]) / 255))

print(train_examples[0])

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.CosineSimilarityLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, output_path="./transformers")

