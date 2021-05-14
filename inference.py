import pandas
import spacy
from spacy.training.iob_utils import biluo_tags_from_offsets
from seqeval import metrics

nlp = spacy.load("models/nlu/model-best")
df = pandas.read_csv('csv/nlu/bio/nlu_test.csv')
true_labels = []
predicted = []
for row_id, row in df.iterrows():
    doc = nlp(row.text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))
    predicted.append(biluo_tags_from_offsets(doc, entities))
    true_labels.append(row.labels.split())


results = dict(
    f1=metrics.f1_score(true_labels, predicted),
    precision=metrics.precision_score(true_labels, predicted),
    recall=metrics.recall_score(true_labels, predicted),
    # f1_span=f1_score_span(true_labels_final, predicted_labels_final),
    # precision_span=precision_score_span(true_labels_final, predicted_labels_final),
    # recall_span=recall_score_span(true_labels_final, predicted_labels_final),
)
print(results)
