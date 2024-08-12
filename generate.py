import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gzip
import pickle
import random
from collections import Counter

import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, vocab
from tqdm import tqdm

# tree = 'acacia'
# with gzip.open('acacia_A2_e_counter.pkl','rb') as f:
#     e_counter = pickle.load(f)
# with gzip.open('acacia_A2_s_counter.pkl','rb') as f:
#     s_counter = pickle.load(f)
# TABLE_PATH = "Acacia_30000_set_ratio10.pkl"
# model_path = "./252855-Acacia/acacia_e5_idx8000.pth"

# tree = 'oak'
# with gzip.open('Oak_E_counter.pkl','rb') as f:
#     e_counter = pickle.load(f)
# with gzip.open('Oak_S_counter.pkl','rb') as f:
#     s_counter = pickle.load(f)
# TABLE_PATH = "Oak_30000_set_ratio10.pkl"
# model_path = "./937683-oak/oak_e10_idx25000.pth"

# tree = 'maple'
# with gzip.open('Maple_E_counter.pkl','rb') as f:
#     e_counter = pickle.load(f)
# with gzip.open('Maple_S_counter.pkl','rb') as f:
#     s_counter = pickle.load(f)
# TABLE_PATH = "Maple_30000_set_ratio10.pkl"
# model_path = "./522261-maple/maple_e5_idx25000.pth"

tree = "birch"
with gzip.open("Birch_E_counter_2.pkl", "rb") as f:
    e_counter = pickle.load(f)
with gzip.open("Birch_S_counter_2.pkl", "rb") as f:
    s_counter = pickle.load(f)
TABLE_PATH = "Birch_30000_set_ratio10.pkl"
model_path = "./617403-birch-train_all/best.pth"
num_generate = 5


def get_table(path):
    import pickle

    with open(path, "rb") as f:
        table = pickle.load(f)
    # clusterVal to alphabet ID
    letterTable = dict()
    for idx, val in enumerate(sorted(table)):
        letterTable[number2letter(idx)] = val
    return letterTable


def get_value(values):
    temp = ""
    symbols = set(["^", "F", "R", "&", "+", "-", "\\", "/"])
    for idx, s in enumerate(values):
        temp += s
        if s == ")":
            temp += " "
    temp = temp.rstrip()
    return temp


def reverseParameter(strings, letterTable):
    temp = ""
    ob, cb = -1, -1
    isSafe = True
    letterSet = set()
    for idx, t in enumerate(strings):
        if t != "(" and t != ")":
            if isSafe:
                temp += t
        else:
            if t == "(":
                ob = idx
                temp += "("
                isSafe = False
            elif t == ")":
                cb = idx
                letter = strings[ob + 1 : idx]
                temp += str(abs(letterTable[letter]))
                letterSet.add(letter)
                temp += ")"
                isSafe = True
    print(f"unique set size: {len(letterSet)}")
    return temp


import string


def number2letter(n, b=string.ascii_uppercase):
    import string

    d, m = divmod(n, len(b))
    return number2letter(d - 1, b) + b[m] if d else b[m]


def letter2number(number):
    import string

    col_number = (
        sum([(ord(number.upper()[-i - 1]) - 64) * 26**i for i in range(len(number))])
        - 1
    )
    return col_number


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices
# def tensor_transform(token_ids):
#   return torch.tensor(token_ids)

# function to add BOS/EOS and create tensor for input sequence indices


def tensor_transform(token_ids):
    return torch.cat(
        (torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))
    )


import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 1000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


SRC_LANGUAGE = "start"
TGT_LANGUAGE = "data"

# Place-holders
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer(tokenizer=None)
token_transform[TGT_LANGUAGE] = get_tokenizer(tokenizer=None)

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ("<unk>", "<pad>", "<bos>", "<eos>")


vocab_transform[SRC_LANGUAGE] = vocab(s_counter, specials=special_symbols)
vocab_transform[TGT_LANGUAGE] = vocab(e_counter, specials=special_symbols)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
print(SRC_VOCAB_SIZE)

TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
print(TGT_VOCAB_SIZE)

# DEVICE = torch.device('cpu')

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(
        token_transform[ln],  # Tokenization
        vocab_transform[ln],  # Numericalization
        tensor_transform,
    )  # Add BOS/EOS and create tensor

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 128
NHEAD = 4
FFN_HID_DIM = 128
BATCH_SIZE = 1024
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    EMB_SIZE,
    NHEAD,
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
    FFN_HID_DIM,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = transformer.to(DEVICE)
transformer.load_state_dict(torch.load(model_path, map_location=torch.device("cuda")))
transformer.eval()


import numpy as np

a = None


# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    model = model.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    # for i in range(max_len-1):
    while True:
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            DEVICE
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])

        prob_cpu = prob.detach().cpu().numpy()
        prob_cpu = prob_cpu.flatten()
        prob_softmax = softmax(prob_cpu)
        next_word = sample(prob_softmax, temperature=0.8)
        _, is_next_word_eos = torch.max(prob, dim=1)
        is_next_word_eos = is_next_word_eos.item()
        if is_next_word_eos == EOS_IDX:
            ys = torch.cat(
                [ys, torch.ones(1, 1).type_as(src.data).fill_(is_next_word_eos)], dim=0
            )
            break
        else:
            ys = torch.cat(
                [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0
            )

        # _, next_word = torch.max(prob, dim=1)
        # next_word = next_word.item()

        # ys = torch.cat([ys,
        #                 torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        # 44 is [END]

    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX
    ).flatten()
    return (
        " ".join(
            vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))
        )
        .replace("<bos>", "")
        .replace("<eos>", "")
        .replace("[END]", "")
        .lstrip()
        .rstrip()
    )


def letter_cmp_key(a, b):
    a = a.replace(")", "")
    b = b.replace(")", "")
    a_num = int(a.split("@")[1])
    b_num = int(b.split("@")[1])
    if a_num > b_num:
        return 1
    elif a_num == b_num:
        return 0
    else:
        return -1


from functools import cmp_to_key

cmp_items_py3 = cmp_to_key(letter_cmp_key)


def get_rules(strings, treeToken):
    import re

    rules = re.findall(f"{treeToken}\(\w+@\w+\)", strings)
    rules = list(set(rules))
    rules.sort(key=cmp_items_py3)
    temp = []
    for r in rules:
        r = r.replace("(", "").replace(")", "")
        temp.append(r)
    return temp


def extract_rules(strings, treeToken):
    import re

    rules = re.findall(f"{treeToken}\(\w+@\w+\)", strings)
    rules = list(set(rules))
    rules.sort(key=cmp_items_py3)
    return rules


def export_lstring_file(rule_str, filename):
    import os

    text = ""
    for t in rule_str:
        if t in ["[", "]"]:
            text += t + "\n"
        elif t == ")":
            text += t + "\n"
        else:
            text += t
    global tree
    path = f"./{tree}_lstring_flow"
    isExist = os.path.exists(path)
    os.makedirs(path, exist_ok=True)
    save_path = f"./{path}/{filename}.lstring"
    with open(save_path, "w") as file:
        file.write(text)
    print(f"lstring file is saved to {save_path}")


def letter_cmp_key(a, b):
    a = a.replace(")", "")
    b = b.replace(")", "")
    a_num = int(a.split("@")[1])
    b_num = int(b.split("@")[1])
    if a_num > b_num:
        return 1
    elif a_num == b_num:
        return 0
    else:
        return -1


from functools import cmp_to_key

cmp_items_py3 = cmp_to_key(letter_cmp_key)


def get_rules(strings, treeToken):
    import re

    rules = re.findall(f"{treeToken}\(\w+@\w+\)", strings)
    rules = list(set(rules))
    rules.sort(key=cmp_items_py3)
    temp = []
    for r in rules:
        r = r.replace("(", "").replace(")", "")
        temp.append(r)
    return temp


def get_table(path):
    import pickle

    with open(path, "rb") as f:
        table = pickle.load(f)
    # clusterVal to alphabet ID
    letterTable = dict()
    for idx, val in enumerate(sorted(table)):
        letterTable[number2letter(idx)] = val
    return letterTable


def get_value(values):
    temp = ""
    symbols = set(["^", "F", "R", "&", "+", "-", "\\", "/"])
    for idx, s in enumerate(values):
        temp += s
        if s == ")":
            temp += " "
    temp = temp.rstrip()
    return temp


def reverseParameter(strings, letterTable):
    symbols = set(["^", "F", "R", "&", "+", "-", "\\", "/"])
    temp = []
    lstring_list = strings.split(" ")
    for l in lstring_list:
        if l[0] not in symbols:
            temp.append(l)
            continue
        open_idx = l.index("(") + 1
        close_idx = l.index(")")
        val = l[open_idx:close_idx]

        converted_string = f"{l[0]}({letterTable[val]})"
        temp.append(converted_string)
    return "".join(temp)


import string


def number2letter(n, b=string.ascii_uppercase):
    import string

    d, m = divmod(n, len(b))
    return number2letter(d - 1, b) + b[m] if d else b[m]


def letter2number(number):
    import string

    col_number = (
        sum([(ord(number.upper()[-i - 1]) - 64) * 26**i for i in range(len(number))])
        - 1
    )
    return col_number


def filter_rule_table(rule_table):
    for key in rule_table:
        val = rule_table[key]
        newKey = ""
        for idx, v in enumerate(key):
            if idx == 1:
                newKey += "("
            newKey += v
        newKey += ")"
        val = val.replace(newKey, "")
        rule_table[key] = val
    return rule_table


def get_new_rule_table(t, tree_token):
    new_rule_table = {}
    rule_table_keys = list(t.keys())
    rule_table_keys.reverse()
    for key in rule_table_keys:
        val = t[key]
        val = get_new_value(val, tree_token, t)
        new_rule_table[key] = val
    return new_rule_table


def get_new_value(val, tree_token, rule_table2):
    while True:
        gen_rules_lstring = extract_rules(val, tree_token)
        if len(gen_rules_lstring) == 0:
            break
        for g in gen_rules_lstring:
            key = g.replace("(", "").replace(")", "")
            val = val.replace(g, rule_table2[key])
    return val


letterTable = get_table(TABLE_PATH)
for i in tqdm(range(0, num_generate)):

    fixed_digits = 6

    model_number = random.randrange(111111, 999999, fixed_digits)
    tree_tokens = {"acacia": "A", "birch": "B", "pine": "P", "oak": "O", "maple": "M"}
    tree_token = tree_tokens[tree]
    rule_table = {}
    start_rule = f"{tree_token}0@0"
    res = translate(transformer, f"[{start_rule}]")

    parent = res.replace("[", "").replace("]", "").lstrip().rstrip()
    rule_table[start_rule] = res
    rules = get_rules(res, tree_token)
    processed_rules = set()
    processed_rules.add(start_rule)
    parent_table = {}
    history = set()
    for r in rules:
        parent_table[r] = parent
        history.add(r)

    while True:
        if len(rules) == 0:
            break
        rule = rules.pop(0)
        if rule in processed_rules:
            continue
        parent = parent_table[rule]
        sentence = f"[{rule}] {parent}"

        # Check if `rule` in `res`
        res = translate(transformer, sentence)

        rule_table[rule] = res
        processed_rules.add(rule)
        extracted_rules = get_rules(res, tree_token)

        parent = res.replace("[", "").replace("]", "").lstrip().rstrip()
        for e in extracted_rules:
            if e not in processed_rules:
                rules.append(e)
                parent_table[e] = parent
    rule_table = filter_rule_table(rule_table)

    rule_table = get_new_rule_table(rule_table, tree_token)

    new_rule_table = {}
    rule_table_keys = list(rule_table.keys())
    rule_table_keys.reverse()
    for key in rule_table_keys:
        val = rule_table[key]
        val = get_new_value(val, tree_token, rule_table)
        new_rule_table[key] = val
    strings = new_rule_table[start_rule]
    rev = reverseParameter(strings, letterTable)
    export_lstring_file(rev, f"{tree}-{model_number}")
