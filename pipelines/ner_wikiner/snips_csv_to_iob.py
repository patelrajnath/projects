import json
import logging
import pandas as pd


def split_at(s, delim, n):
    r = s.split(delim, n)[n]
    return s[:-len(r)-len(delim)], r


def tsv_to_json_format():
    try:
        entity_list = set()
        unknown_label = 'abc'
        train_df = pd.read_csv("SNIPS/200923_train_custom_snips_60.csv")  # input file
        test_df = pd.read_csv("SNIPS/200923_test_custom_snips_30.csv")  # input file
        valid_df = pd.read_csv("SNIPS/200923_val_custom_snips_10.csv")  # input file

        # df = pd.concat([train_df, valid_df, test_df])
        # df = pd.concat([train_df, valid_df])
        # df = train_df
        # df = test_df
        df = valid_df

        # fp=open('SNIPS/train.iob', 'w', encoding='utf8') # output file
        # fp = open('SNIPS/test.iob', 'w', encoding='utf8') # output file
        fp = open('SNIPS/dev.iob', 'w', encoding='utf8') # output file
        for index, row in df.iterrows():
            s = ''
            line = row.text
            for token in line.split(" "):
                token = token.strip()
                if token:
                    token_split = token.split(':')
                    if len(token_split) > 2:
                        token_split = split_at(token, ':', len(token_split) - 1)
                    word, entity = token_split
                    entity = entity.strip()
                    word = word.strip()
                    if word:
                        s += word + "|NOPE|" + entity + " "

            fp.write(s.strip() + '\n')
        fp.close()

    except Exception as e:
        logging.exception("Unable to process file" + "\n" + "error = " + str(e))
        return None


tsv_to_json_format()