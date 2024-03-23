# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *

from torch.utils.data import DataLoader
from torch import optim
import transformers


import os

if __name__ == "__main__":
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # set verbosity to not show BERT warnings
    transformers.utils.logging.set_verbosity_error()

    print("Loading datasets...")

    # Write the code to load the datasets and to run your functions
    tmp_train_raw = load_data(os.path.join('dataset', 'ATIS', 'train.json'))
    test_raw = load_data(os.path.join('dataset', 'ATIS', 'test.json'))

    # Split train_raw into train and dev
    train_raw, dev_raw, test_raw = get_train_dev_test_datasets(tmp_train_raw, test_raw)

    PAD_TOKEN = 0

    # Create the Lang object
    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw

    # We do not wat unk labels however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    # alphabetically sort the slots and intents
    # this will prevent subsequent run to change the order
    # and making the loaded bin not working
    slots = sorted(list(slots))
    intents = sorted(list(intents))

    lang = Lang(words, intents, slots, cutoff=0)

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    collate_fn = Collate(device, PAD_TOKEN)

    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)


    """ 
    print("\nEvaluate BERT model with freezed params...")
    # Evaluate base model
    full_evaluate(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        dropout=0.0,
        experiment_name="bert_freezed",
        lang=lang,
        lr=0.01,
        pad_token=PAD_TOKEN,
        device=device,
        freeze=True
    )
    """

    print("\nEvaluate BERT model with unfreezed params (finetuning)...")
    full_evaluate(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        dropout=0.0,
        experiment_name="bert_finetuning",
        lang=lang,
        lr=0.01,
        pad_token=PAD_TOKEN,
        device=device,
        freeze=False
    )


