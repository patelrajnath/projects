import pickle
import random
import time
import warnings
from pathlib import Path
import plac
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

ents_dict = dict()
with open('wallet_entities.txt') as f:
    for line in f:
        if line.strip():
            ent_type, ents = line.split(':')
            if ent_type == "CARD_TYPE" or ent_type == "HUAWEI_DEVICE":
                ents_list = {ent.strip().lower() for ent in ents.split(',')}
                # Append "huawei" to cover the examples like "Huawei Honor 7X" as an entity
                ents_list.update({"huawei " + ent.strip().lower() for ent in ents.split(',')})
            else:
                ents_list = {ent.strip().lower() for ent in ents.split(',')}
            print(ents_list)
            for entity in ents_list:
                ents_dict[ent_type] = ents_list
print(ents_dict)


with open ('train-wallet-v2', 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, new_model_name="wallet",
         output_dir="wallet-v2-model", n_iter=10):
    start_time = time.time()
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        # ner = nlp.create_pipe("ner")
        ner = nlp.add_pipe("ner")
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")
    for LABEL in ents_dict:
        ner.add_label(LABEL)  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    ner.add_label("VEGETABLE")
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["transformer", "ner"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        print(nlp.pipe_names)
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch

        def convert_examples(texts, annotations):
            examples = []
            for i, text in enumerate(texts):
                doc = nlp(text)
                example = Example.from_dict(doc, annotations[i])
                examples.append(example)
            return examples

        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                # print(batch)
                texts, annotations = zip(*batch)
                examples = convert_examples(texts, annotations)
                nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)
                print(losses)
            print("Losses", losses)
    print('Training time', time.time() - start_time)
    # test the trained model
    test_text = "The Beijing bus card in Huawei Pay is not Beijing-Tianjin-Hebei " \
                "intercommunication card."
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)
