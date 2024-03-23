# Add functions or classes used for data loading and preprocessing
import json
from collections import Counter
import torch.utils.data as data
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


def load_data(path):
    """
    Load dataset from json file
    :param path: path to json file
    :return: dataset
    """
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


def get_train_dev_test_datasets(train_raw, test_raw):
    """
    Split the dataset into train, dev and test
    :param train_raw: raw training dataset
    :param test_raw: raw test dataset
    :return: train, dev and test datasets
    """
    # First we get the 10% of dataset, then we compute the percentage of these examples
    # on the training set which is around 11%
    portion = round(((len(train_raw) + len(test_raw)) * 0.10) / (len(train_raw)), 2)

    intents = [x['intent'] for x in train_raw]  # We stratify on intents
    count_y = Counter(intents)

    Y, X, single_intent_X = [], [], []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # If some intents occur once only, we put them in training
            X.append(train_raw[id_y])
            Y.append(y)
        else:
            single_intent_X.append(train_raw[id_y])
    # Random Stratify
    train_raw, dev_raw, _, _ = train_test_split(X, Y, test_size=portion, random_state=42, shuffle=True, stratify=Y)

    # We add the single intent examples to the training set
    train_raw.extend(single_intent_X)

    return train_raw, dev_raw, test_raw


class Lang:
    def __init__(self, words, intents, slots, cutoff=0, pad_token=0):
        self.pad_token = pad_token
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': self.pad_token}
        if unk:
            # We use the unk token from the tokenizer
            vocab['unk'] = self.tokenizer.get_vocab()['[UNK]']
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                tokenized_word = self.tokenizer.tokenize(k)
                for i, subword in enumerate(tokenized_word):
                    if subword not in vocab:
                        token_id = self.tokenizer.convert_tokens_to_ids(subword)
                        vocab[subword] = token_id
        return {k: v for k, v in vocab.items()}

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = self.pad_token
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab


class IntentsAndSlots(data.Dataset):
    """
    Class to handle the dataset
    """

    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.lang = lang
        self.skip = -1

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids, self.slot_ids, self.skip_ids = self.mapping(self.utterances, self.slots)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        skip = self.skip_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent, 'skip': skip}
        return sample

    # Auxiliary methods

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping(self, utt, slt):  # Map sequences to number
        utt_result = []
        slots_result = []
        skip_result = []

        for utterance, slots in zip(utt, slt):
            # adding the CLS token
            utt_res = [self.lang.tokenizer.get_vocab()['[CLS]']]
            slots_res = [self.lang.slot2id['O']]
            skip_res = [0]
            for word, slot in zip(utterance.split(), slots.split()):
                tokens = self.lang.tokenizer.tokenize(word)
                for j, token in enumerate(tokens):
                    if token in self.lang.word2id:
                        utt_res.append(self.lang.word2id[token])
                    else:
                        utt_res.append(self.lang.word2id[self.unk])
                    if j > 0:
                        skip_res.append(len(utt_res) - 1)

                    if slot in self.lang.slot2id:
                        slots_res.append(self.lang.slot2id[slot])
                    else:
                        slots_res.append(self.lang.slot2id[self.unk])

            # adding the SEP token
            utt_res.append(self.lang.tokenizer.get_vocab()['[SEP]'])
            slots_res.append(self.lang.slot2id['O'])
            skip_res.append(len(utt_res) - 1)

            utt_result.append(utt_res)
            slots_result.append(slots_res)
            skip_result.append(skip_res)

        return utt_result, slots_result, skip_result


class Collate:
    """
    Class to handle the padding of the sequences, it is used in the DataLoader
    I need the class in order to pass the device and the pad_token as parameters
    """

    def __init__(self, device, pad_token):
        self.device = device
        self.pad_token = pad_token

    def __call__(self, data):
        def merge(sequences):
            """
            Merge from batch * sent_len to batch * max_len
            """
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            # Pad token is zero in our case
            # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
            # batch_size X maximum length of a sequence
            padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(self.pad_token)
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
            # print(padded_seqs)
            padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
            return padded_seqs, lengths

        # Sort data by seq lengths
        data.sort(key=lambda x: len(x['utterance']), reverse=True)
        new_item = {}
        for key in data[0].keys():
            new_item[key] = [d[key] for d in data]
        # We just need one length for packed pad seq, since len(utt) == len(slots)
        src_utt, _ = merge(new_item['utterance'])
        y_slots, y_lengths = merge(new_item["slots"])
        intent = torch.LongTensor(new_item["intent"])

        src_utt = src_utt
        y_slots = y_slots
        intent = intent
        y_lengths = torch.LongTensor(y_lengths)

        new_item["utterances"] = src_utt
        new_item["intents"] = intent
        new_item["y_slots"] = y_slots
        new_item["slots_len"] = y_lengths
        return new_item
