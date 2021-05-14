import os
import pickle


# file_in = 'csv/wallet/bio/wallet_train.csv'
# file_in = 'csv/wallet/bio/wallet_dev.csv'
file_in = 'csv/wallet/bio/wallet_test.csv'

DATA = []
with open(file_in, mode="r", encoding="utf-8") as f_test:
    f_test.readline()
    labels_predict_all = []
    labels_true_all = []

    for line_id, line in enumerate(f_test):
        tokens_subword = []
        fields = line.strip().split("\t")
        if len(fields) == 2:
            labels, tokens = fields
        elif len(fields) == 3:
            labels, tokens, cls = fields
        else:
            print(f'The data is not in accepted format at line no:{line_id}.. Ignored')
            continue
        labels_true = labels.split()
        words = tokens.split()
        DATA.append((words, labels_true))

out_dir = 'iob/wallet/bio'
try:
    os.makedirs(out_dir)
except:
    pass

# with open(os.path.join(out_dir, 'train.iob'), 'w', encoding='utf-8') as fout:
# with open(os.path.join(out_dir, 'dev.iob'), 'w', encoding='utf-8') as fout:
with open(os.path.join(out_dir, 'test.iob'), 'w', encoding='utf-8') as fout:
    for example in DATA:
        tokens, tags = example
        ner_training = [tok + '|' + tag for tok, tag in zip(tokens, tags)]
        fout.write(' '.join(ner_training) + '\n')
