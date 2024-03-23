# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch
import torch.nn as nn
import json
from conll import evaluate
from sklearn.metrics import classification_report
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import os
import copy
from model import ModelIAS


def init_weights(mat):
    """
    Initialize the weights of the model, function from the lab
    """
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, device, clip=5):
    """
    Training loop for one epoch
    """
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        slots, intent = model(sample['utterances'].to(device), sample['slots_len'].to(device))
        loss_intent = criterion_intents(intent, sample['intents'].to(device))
        loss_slot = criterion_slots(slots, sample['y_slots'].to(device))
        loss = loss_intent + loss_slot  # In joint training we sum the losses.
        # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang, device):
    """
    Evaluate the model on the val/test set
    """
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []

    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'].to(device), sample['slots_len'].to(device))
            loss_intent = criterion_intents(intents, sample['intents'].to(device))
            loss_slot = criterion_slots(slots, sample['y_slots'].to(device))
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                # do not consider cls and sep tokens
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq][:length].tolist()

                # index to skip because subword or special tokens
                skip_idx = sample['skip'][id_seq]
                gt_slots = [lang.id2slot[elem] for i, elem in enumerate(gt_ids) if i not in skip_idx]
                utterance = [lang.id2word[elem] for i, elem in enumerate(utt_ids) if i not in skip_idx]
                to_decode = [s for i, s in enumerate(seq[:length].tolist()) if i not in skip_idx]

                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predics a class that is not in REF
        print(ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array


def save_model(model, experiment):
    """
    Save the model
    """
    # To save the model
    path = f'bin/{experiment}.pt'
    print(f"Saving model to {path}")
    torch.save(model.state_dict(), path)


def load_model(model, experiment, device):
    """
    Load the model
    """
    # To load the model you need to initialize it
    # Then you load it
    path = f'bin/{experiment}.pt'
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def evaluate_model(
        model,
        train_loader,
        dev_loader,
        test_loader,
        pad_token,
        lang,
        experiment,
        device,
        lr=0.001,
        clip=5,
        n_epochs=200,
        patience=5,
):
    """
    Evaluate the model, train it if needed
    """
    criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_token)
    criterion_intents = nn.CrossEntropyLoss()

    # check if there is a model to load
    if os.path.isfile(f"bin/{experiment}.pt"):
        best_model = load_model(model, experiment, device)

    # if not train it
    else:
        # model.apply(init_weights)
        optimizer = AdamW([
            {'params': model.slot_out.parameters(), 'lr': lr},
            {'params': model.intent_out.parameters(), 'lr': lr},
            {'params': filter(lambda p: p.requires_grad, model.bert.parameters()), 'lr': lr / 10},
        ])

        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = -1
        pbar = tqdm(range(1, n_epochs))
        for x in pbar:
            loss = train_loop(
                data=train_loader,
                optimizer=optimizer,
                criterion_slots=criterion_slots,
                criterion_intents=criterion_intents,
                model=model,
                clip=clip,
                device=device,
            )

            if x % 3 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(
                    data=dev_loader,
                    criterion_slots=criterion_slots,
                    criterion_intents=criterion_intents,
                    model=model,
                    lang=lang,
                    device=device
                )

                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']
                acc = intent_res['accuracy']

                pbar.set_description(f"Epoch {x} - F1 {f1} - Acc {acc}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = copy.deepcopy(model)
                else:
                    patience -= 1
                if patience <= 0:  # Early stoping with patience
                    break 

            model.reset_hebb()
            pbar.update(1)

        save_model(best_model, experiment)

    results_test, intent_test, _ = eval_loop(
        data=test_loader,
        criterion_slots=criterion_slots,
        criterion_intents=criterion_intents,
        model=best_model,
        lang=lang,
        device=device
    )

    return results_test, intent_test


def full_evaluate(
        train_loader,
        dev_loader,
        test_loader,
        dropout,
        experiment_name,
        lang,
        pad_token,
        freeze=True,
        lr=0.001,
        n_runs=5,
        device='cpu'
):
    """
    Evaluate the model n_runs times, it trains it if needed
    """
    slot_f1s, intent_acc = [], []
    for i in range(n_runs):
        model = ModelIAS(
            out_slot=len(lang.slot2id),
            out_int=len(lang.intent2id),
            freeze=freeze,
            dropout=dropout,
            device=device,
        ).to(device)

        results_test, intent_test = evaluate_model(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            pad_token=pad_token,
            lang=lang,
            lr=lr,
            n_epochs=200,
            patience=3,
            device=device,
            experiment=f'{experiment_name}_{i}',
        )

        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(), 3), '+-', round(slot_f1s.std(), 3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))
