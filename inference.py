import re

import pandas
import spacy
from spacy.tokens.doc import Doc
from spacy.training.iob_utils import biluo_tags_from_offsets
from seqeval import metrics


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)


nlp = spacy.load("models/nlu/model-best")
# nlp = spacy.load("models/accounts/model-best")
# nlp = spacy.load("models/kaggle/model-best")

# Disable the default tokenizer
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

df = pandas.read_csv('csv/nlu/bio/nlu_test.csv', encoding='utf8', sep='\t')
# df = pandas.read_csv('csv/accounts/bio/accounts_test.csv', encoding='utf8', sep='\t')
# df = pandas.read_csv('csv/kaggle/test_combined_3.csv', encoding='utf8', sep='\t')

true_labels = []
predicted = []
ignored = 0
for row_id, row in df.iterrows():
    content = row.text
    print(row_id, content)
    doc = nlp(content)
    entities = []
    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))
    pred = biluo_tags_from_offsets(doc, entities)
    true = row.labels.split()

    pred = [p.replace('_', '-').replace('U-', 'B-').replace('L-', 'I-') for p in pred]
    true = [t.replace('_', '-').replace('U-', 'B-').replace('L-', 'I-') for t in true]

    if len(pred) != len(true):
        print(row_id, content)
        print(pred, len(pred))
        print(true, len(true))
        ignored += 1
        continue
    predicted.append(pred)
    true_labels.append(true)

print(f'Samples ignored:{ignored}')
results = dict(
    f1=metrics.f1_score(true_labels, predicted),
    precision=metrics.precision_score(true_labels, predicted),
    recall=metrics.recall_score(true_labels, predicted),
    # f1_span=f1_score_span(true_labels_final, predicted_labels_final),
    # precision_span=precision_score_span(true_labels_final, predicted_labels_final),
    # recall_span=recall_score_span(true_labels_final, predicted_labels_final),
)
print(results)
