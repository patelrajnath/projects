import os
import pickle

import spacy
from spacy.gold import biluo_tags_from_offsets
with open ('json/train-ubuntu', 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)

with open ('json/test-ubuntu', 'rb') as fp:
    TEST_DATA = pickle.load(fp)

nlp_blank = spacy.blank('en')
out_dir = 'iob/ubuntu'
try:
    os.makedirs(out_dir)
except:
    pass

with open(os.path.join(out_dir, 'train.iob'), 'w', encoding='utf-8') as fout:
    for example in TRAIN_DATA:
        text, entities = example
        doc = nlp_blank(text)
        tags = biluo_tags_from_offsets(doc, entities['entities'])
        tokens = [token.text for token in doc]
        ner_training = [tok + '|' + tag for tok, tag in zip(tokens, tags)]
        fout.write(' '.join(ner_training) + '\n')

with open(os.path.join(out_dir, 'test.iob'), 'w', encoding='utf-8') as fout:
    for example in TEST_DATA:
        text, entities = example
        doc = nlp_blank(text)
        tags = biluo_tags_from_offsets(doc, entities)
        tokens = [token.text for token in doc]
        ner_training = [tok + '|' + tag for tok, tag in zip(tokens, tags)]
        fout.write(' '.join(ner_training) + '\n')


# with open(os.path.join(out_dir, 'ubuntu_train_text.txt'), 'w', encoding='utf-8') as fout_text, \
#         open(os.path.join(out_dir, 'ubuntu_train_labels.txt'), 'w', encoding='utf-8') as fout_labels:
#     for example in TRAIN_DATA:
#         text, entities = example
#         doc = nlp_blank(text)
#         tags = biluo_tags_from_offsets(doc, entities['entities'])
#         tokens = [token.text for token in doc]
#         fout_text.write(' '.join(tokens) + '\n')
#         fout_labels.write(' '.join(tags) + '\n')
#
# with open(os.path.join(out_dir, 'ubuntu_test_text.txt'), 'w', encoding='utf-8') as fout_text, \
#         open(os.path.join(out_dir, 'ubuntu_test_labels.txt'), 'w', encoding='utf-8') as fout_labels:
#     for example in TEST_DATA:
#         text, entities = example
#         doc = nlp_blank(text)
#         tags = biluo_tags_from_offsets(doc, entities)
#         tokens = [token.text for token in doc]
#         fout_text.write(' '.join(tokens) + '\n')
#         fout_labels.write(' '.join(tags) + '\n')
