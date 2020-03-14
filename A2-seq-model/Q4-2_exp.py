# ===
#
# ===

import collections
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn

from solution import RNN, GRU


# ============================================================================
# Helper methods from run_exp
# ============================================================================

def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word


# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.

    This prevents Pytorch from trying to backpropagate into previous input
    sequences when we use the final hidden states from one mini-batch as the
    initial hidden states for the next mini-batch.

    Using the final hidden states in this way makes sense when the elements of
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)


# ============================================================================
# Modified helper methods from run_exp
# ============================================================================

def generate(model, data, n_samples=128, gen_seq_len=35, device='cpu'):
    # Set to eval
    model.to(device)
    model.eval()

    # Initialize hidden
    hidden = model.init_hidden()
    hidden = hidden.to(device)
    hidden = repackage_hidden(hidden)

    rng = np.random.RandomState(seed=6135)
    rand_int = rng.randint(1, 100, size=(n_samples))
    inputs = torch.from_numpy(rand_int.astype(np.int64)).contiguous().to(device)

    # Generate (samples are of size (seq_len, batch_size))
    samples = model.generate(inputs,
                             hidden=hidden,
                             generated_seq_len=gen_seq_len)

    return samples.cpu().numpy()



# ==========
# My own helper methods
# ==========
def load_model(path, vocab_size, model_class='GRU'):
    """
    Load a model given saved states
    From: https://pytorch.org/tutorials/beginner/saving_loading_models.html

    Default model parameters:
    --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=128 --seq_len=35
                --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8
                --num_epochs=20 --save_best
    --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128
                --seq_len=35 --hidden_size=512 --num_layers=2
                --dp_keep_prob=0.5  --num_epochs=20 --save_best

    :return:
    """

    # Default

    if model_class is 'GRU':
        model = GRU(emb_size=200,
                    hidden_size=512,
                    seq_len=35,
                    batch_size=128,
                    vocab_size=vocab_size,
                    num_layers=2,
                    dp_keep_prob=0.8)
    else:
        model = RNN(emb_size=200,
                    hidden_size=512,
                    seq_len=35,
                    batch_size=128,
                    vocab_size=vocab_size,
                    num_layers=2,
                    dp_keep_prob=0.8)

    model.load_state_dict(torch.load(path))

    return model


# ==========
# My own helper methods
# ==========

def get_data(path):
    """
    Get the Penn Treebank data from pre-defined path
    :param path: string, path to the dataset
    :return: dataset in tuples
    """
    raw_data = ptb_raw_data(data_path=path)
    train_data, valid_data, test_data, word_to_id, id_2_word = raw_data

    return (train_data, valid_data, test_data), (word_to_id, id_2_word), len(word_to_id)


if __name__ == "__main__":
    DATA_PATH = '/network/home/chenant/class/IFT6135-DL/IFT6135-rep-learning/IFT6135H20_assignment/assignment2/data'
    MOD_PATH = {
        'RNN': '/network/home/chenant/class/IFT6135-DL/IFT6135-rep-learning/A2-seq-model/output/3_1/RNN_SGD_1.0_128_35_512_2_0.8_20_35_0/best_params.pt',
        'GRU': '/network/home/chenant/class/IFT6135-DL/IFT6135-rep-learning/A2-seq-model/output/3_2/GRU_ADAM_0.001_128_35_512_2_0.5_20_35_0/best_params.pt'
    }
    OUT_DIR = '/network/home/chenant/class/IFT6135-DL/IFT6135-rep-learning/A2-seq-model/output/4_2'

    MODEL_CLASS = 'GRU'  # RNN, GRU
    GEN_SEQ_LEN = 70  # 35, 70

    # Getting data
    Data, Ids, vocab_size = get_data(DATA_PATH)
    train_data, valid_data, test_data = Data
    word_to_id, id_2_word = Ids

    print('Vocab size:', vocab_size)

    # Get model
    model = load_model(MOD_PATH[MODEL_CLASS], vocab_size, model_class=MODEL_CLASS)

    print(model)

    # Get device
    if torch.cuda.is_available():
        print("Using the GPU")
        device = torch.device("cuda")
    else:
        print("WARNING: You are about to run on cpu, and this will likely run out \
          of memory. \n You can try setting batch_size=1 to reduce memory usage")
        device = torch.device("cpu")

    # Run generation
    print('\n==========\nRunning Model\n==========')
    samples_id = generate(model, train_data,
                          n_samples=128,
                          gen_seq_len=GEN_SEQ_LEN,
                          device=device)
    print(np.shape(samples_id))

    # Sentences
    print('\n==========\nSentence Samples\n==========')
    # Convert to text
    for n_sample in range(20):
        sent_list = []
        for t in range(GEN_SEQ_LEN):
            tok = id_2_word[samples_id[t, n_sample]]
            sent_list.append(tok)
        sent = ' '.join(sent_list)
        print(sent)

