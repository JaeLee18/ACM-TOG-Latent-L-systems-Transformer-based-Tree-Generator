'''
Set Line #32 `e_counter_path`, Line#36 `s_counter_path`
Set Line #40, Line#41 for `train_path` and `test_path`



'''


import json
import gzip, pickle
import os
import numpy as np
import random
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# load and uncompress.
tree='birch'
e_counter_path = '../final_data/Birch_E_counter_2.pkl'
with gzip.open(e_counter_path,'rb') as f:
    e_counter = pickle.load(f)  
# load and uncompress.
s_counter_path = "../final_data/Birch_S_counter_2.pkl"
with gzip.open(s_counter_path,'rb') as f:
    s_counter = pickle.load(f)

train_path = "../final_data/Birch_train_total.csv" # Put {tree}_train_total.csv path
test_path = "../final_data/Birch_test_total.csv" # Put {tree}_test_total.csv path

EMB_SIZE = 128
NHEAD = 4
FFN_HID_DIM = 128
BATCH_SIZE = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

def set_seed(seed: int = 218) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


fixed_digits = 6 
model_number = random.randrange(111111, 999999, fixed_digits)
print(f"Model # : {model_number}")
model_path  = f"{model_number}-{tree}-train_all"
os.makedirs(f"{model_path}",exist_ok=True)
print(f"Model {model_number} is created.")


set_seed()


SRC_LANGUAGE = 'start'
TGT_LANGUAGE = 'data'

# Place-holders
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer(tokenizer=None)
token_transform[TGT_LANGUAGE] = get_tokenizer(tokenizer=None)

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ('<unk>', '<pad>', '<bos>', '<eos>')


vocab_transform[SRC_LANGUAGE] = vocab(s_counter,specials=special_symbols)
vocab_transform[TGT_LANGUAGE] = vocab(e_counter,specials=special_symbols)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
print(SRC_VOCAB_SIZE)

TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
print(TGT_VOCAB_SIZE)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 1000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

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
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask



SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])





transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
transformer = transformer.to(DEVICE)
transformer.train()


loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)



# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
      for transform in transforms:
          txt_input = transform(txt_input)
      return txt_input
    return func


def tensor_transform(token_ids):
  return torch.cat((torch.tensor([BOS_IDX]),
                    torch.tensor(token_ids),
                    torch.tensor([EOS_IDX])))


# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

import json
config={"EMB_SIZE": EMB_SIZE, "NHEAD": NHEAD, "FFN_HID_DIM":FFN_HID_DIM, "BATCH_SIZE":BATCH_SIZE, "NUM_ENCODER_LAYERS":NUM_ENCODER_LAYERS, "NUM_DECODER_LAYERS":NUM_DECODER_LAYERS, "maxlen": 1000, "vocab": s_counter_path, "model_number":model_number, "tree_num":"all"}



with open(f'{model_path}/config.json', 'w') as fp:
    json.dump(config, fp)
print(f"{model_path}/config.json created") 

def test_epoch(model, test_dataloader, val_epoch):
    model.eval()
    losses = 0
    cnt = 0

    for idx, (srcs, tgts) in enumerate(test_dataloader):
        srcs = srcs.to(DEVICE)
        tgts = tgts.to(DEVICE)

        tgt_input = tgts[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(srcs, tgt_input)

        logits = model(srcs, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgts[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        cnt += 1


    loss = losses/cnt
    print(f"[test {idx}] {loss}", end='\r')
    return losses/cnt


global_cnt = 0
def train_epoch(model, optimizer, train_dataloader, test_dataloader, epoch, val_epoch,scheduler):
    model.train()
    losses = 0
    cnt = 0
    for idx, (srcs, tgts) in enumerate(train_dataloader):
        model.train()
        srcs = srcs.to(DEVICE)
        tgts = tgts.to(DEVICE)
        tgt_input = tgts[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(srcs, tgt_input)
        logits = model(srcs, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()
        tgt_out = tgts[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        cnt += 1

        scheduler.step(loss.item())

        loss = losses / cnt
        print(f"[{epoch}-{idx}] {loss}", end='\r')




    return losses / len(train_dataloader)



class LStringDataset(Dataset):
  def __init__(self, path):
    self.df = pd.read_csv(path, usecols=['SRC', 'TRG'])

    self.src = self.df['SRC'].to_list()
    self.trg = self.df['TRG'].to_list()
  def __len__(self):
    return len(self.df)
  def __getitem__(self, index):
    return self.src[index], self.trg[index]



def main():
    
    
    dataset = LStringDataset(train_path)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, pin_memory=True)
    test_dataset = LStringDataset(test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False, pin_memory=True)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)



    from timeit import default_timer as timer
    NUM_EPOCHS = 100000
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=2000, threshold=0.0001,min_lr=1e-5, eps=1e-08, verbose=True)
    val_epoch = 0
    counter = 0
    MAX_TRAIN = 20
    current_test_loss = 999999999
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader, test_dataloader, epoch, val_epoch, scheduler)
        if(train_loss==-1):
            break
        test_loss = test_epoch(transformer, test_dataloader, val_epoch)
        if(test_loss < current_test_loss):
            current_test_loss = test_loss
            torch.save(transformer.state_dict(), f"{model_path}/best.pth")
            counter = 0
        else:
            counter += 1
            if(counter == MAX_TRAIN):
                print("Early stopping")
                return -1
            
        end_time = timer()

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Test loss: {test_loss:.3f} "f"Epoch time = {(end_time - start_time):.3f}s at {current_time}") 

if __name__ == "__main__":
    main()