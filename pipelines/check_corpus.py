import gzip
import json
import pickle

input_file = 'ner_wikiner/corpus/dev.spacy'

# with open(input_file, 'rb') as fp:
#     TRAIN_DATA = json.load(fp)

# for example in TRAIN_DATA:
#     print(example)
#     exit()

with open(input_file, 'rb') as f:
    TRAIN_DATA = json.load(f)
