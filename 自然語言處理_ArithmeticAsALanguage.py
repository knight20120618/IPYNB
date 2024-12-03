# %% [markdown]
# LSTM-arithmetic

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.utils.rnn
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
import opencc
import os
from sklearn.model_selection import train_test_split

data_path = './'

# %%
df_train = pd.read_csv(os.path.join(data_path, 'arithmetic_train.csv'))
df_eval = pd.read_csv(os.path.join(data_path, 'arithmetic_eval.csv'))

print(df_train.head())

# %%
# 輸入的資料型態轉換成 str
df_train['tgt'] = df_train['tgt'].apply(lambda x: str(x))
df_train['src'] = df_train['src'].add(df_train['tgt'])
df_train['len'] = df_train['src'].apply(lambda x: len(x))

df_eval['tgt'] = df_eval['tgt'].apply(lambda x: str(x))
df_eval['src'] = df_eval['src'].add(df_eval['tgt'])
df_eval['len'] = df_eval['src'].apply(lambda x: len(x))

# %% [markdown]
# Build Dictionary
# 
#     The model cannot perform calculations directly with plain text.
#     Convert all text (numbers/symbols) into numerical representations.
#     Special tokens
#         '<pad>'
#             Each sentence within a batch may have different lengths.
#             The length is padded with '<pad>' to match the longest sentence in the batch.
#         '<eos>'
#             Specifies the end of the generated sequence.
#             Without '<eos>', the model will not know when to stop generating.

# %%
# 建立字典
char_to_id = {
    "<pad>": 0,
    "<eos>": 1
}
id_to_char = {
    0: "<pad>",
    1: "<eos>"
}

chars = list("0123456789+-*()=")

char_to_id.update({char: idx +2 for idx, char in enumerate(chars)})
id_to_char.update({idx +2: char for idx, char in enumerate(chars)})

vocab_size = len(char_to_id)
print('Vocab size: {}'.format(vocab_size))

# %% [markdown]
# Data Preprocessing
# 
#     The data is processed into the format required for the model's input and output.
#     Example: 1+2-3=0
#         Model input: 1 + 2 - 3 = 0
#         Model output: / / / / / 0 <eos> (the '/' can be replaced with <pad>)
#         The key for the model's output is that the model does not need to predict the next character of the previous part. What matters is that once the model sees '=', it should start generating the answer, which is '0'. After generating the answer, it should also generate<eos>

# %%
# 資料預處理
def process_row(src, tgt):
    char_id_list = [char_to_id[char] for char in src if char in char_to_id] + [char_to_id['<eos>']]
    
    answer_id = char_to_id[tgt] if tgt in char_to_id else 17
    label_id_list = [0] * (len(src) - 1) + [answer_id, char_to_id['<eos>']]

    return char_id_list, label_id_list
# 資料轉換成 TOKEN
df_train['char_id_list'], df_train['label_id_list'] = zip(*df_train.apply(lambda row: process_row(row['src'], row['tgt']), axis=1))
df_eval['char_id_list'], df_eval['label_id_list'] = zip(*df_eval.apply(lambda row: process_row(row['src'], row['tgt']), axis=1))
# 顯示 char_id_list 和 label_id_list ， .to_string() 使 df 資料不斷行
print(df_eval.head().to_string())

# %% [markdown]
# Hyper Parameters

# %%
# 參數設定
batch_size = 64
epochs = 2
embed_dim = 256
hidden_dim = 256
lr = 0.001
grad_clip = 1

# %% [markdown]
# Data Batching
# 
#     Use torch.utils.data.Dataset to create a data generation tool called dataset.
#     The, use torch.utils.data.DataLoader to randomly sample from the dataset and group the samples into batches.

# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences): # 資料初始化
        self.sequences = sequences
    
    def __len__(self): # 資料長度回傳
        return len(self.sequences)
    
    def __getitem__(self, index): # 資料擷取
        x = self.sequences.iloc[index, 0][:-1] # 輸入元素移除最後一個
        y = self.sequences.iloc[index, 1][1:] # 輸出元素向右移動一個
        return x, y

def collate_fn(batch): # dataloader 函數
    batch_x = [torch.tensor(data[0]) for data in batch]
    batch_y = [torch.tensor(data[1]) for data in batch]
    batch_x_lens = torch.LongTensor([len(x) for x in batch_x])
    batch_y_lens = torch.LongTensor([len(y) for y in batch_y])
    
    # 將資料長度調整成一致
    pad_batch_x = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True,
                                                  padding_value=char_to_id['<pad>'])
    pad_batch_y = torch.nn.utils.rnn.pad_sequence(batch_y, batch_first=True,
                                                  padding_value=char_to_id['<pad>'])
    
    return pad_batch_x, pad_batch_y, batch_x_lens, batch_y_lens

# %%
# 定義資料集
ds_train = Dataset(df_train[['char_id_list', 'label_id_list']])
ds_eval = Dataset(df_eval[['char_id_list', 'label_id_list']])

# %%
# 載入資料集
from torch.utils.data import DataLoader
# shuffle 為是否打亂順序
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dl_eval = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# %% [markdown]
# Model Design
# 
#     Execution Flow
#         Convert all characters in the sentence into embeddings.
#         Pass the embeddings through an LSTM sequentially.
#         The output of the LSTM is passed into another LSTM, and additional layers can be added.
#         The output from all time steps of the final LSTM is passed through a Fully Connected layer.
#         The character corresponding to the maximum value across all output dimensions is selected as the next character.
#     Loss Function
#         Since this is a classification task, Cross Entropy is used as the loss function.
#     Gradient Update
#         Adam algorithm is used for gradient updates.

# %%
# CPU or GPU
dml = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharRNN, self).__init__()
        
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim,
                                            padding_idx=char_to_id['<pad>'])
        
        self.rnn_layer1 = torch.nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim,
                                        batch_first=True)
        
        self.rnn_layer2 = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                                        batch_first=True)
        
        self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(in_features=hidden_dim, out_features=vocab_size))

    def forward(self, batch_x, batch_x_lens):
        return self.encoder(batch_x, batch_x_lens)

    def encoder(self, batch_x, batch_x_lens): # 前向傳導模型
        batch_x = self.embedding(batch_x)
        
        batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x, batch_x_lens,
                                                            batch_first=True, enforce_sorted=False)
        
        batch_x, _ = self.rnn_layer1(batch_x)
        batch_x, _ = self.rnn_layer2(batch_x)
        
        batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x, batch_first=True)
        
        batch_x = self.linear(batch_x)
        
        return batch_x

    def generator(self, start_char, max_len=200):
        char_list = [char_to_id[c] for c in start_char]
        
        next_char = None
        
        while len(char_list) < max_len:
            input_tensor = torch.tensor([[char_list[-1]]]).to(dml)

            embedded = self.embedding(input_tensor)
            output, _ = self.rnn_layer1(embedded)
            output, _ = self.rnn_layer2(output)
            # 線性預測
            last_output = output[:, -1, :]
            y = self.linear(last_output)
            # 預測下個字元
            next_char = torch.argmax(y, dim=1).item()
            
            if next_char == char_to_id['<eos>']:
                break

            char_list.append(next_char)
            
        return [id_to_char[ch_id] for ch_id in char_list]

# %%
torch.manual_seed(2) # 設定隨機種子
model = CharRNN(vocab_size, embed_dim, hidden_dim).to(dml)

# %%
# 設定損失函數
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
# 設定優化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %% [markdown]
# Training
# 
#     The outer for loop controls the epoch
#         The inner for loop uses data_loader to retrieve batches.
#             Pass the batch to the model for training.
#             Compare the predicted results batch_pred_y with the true labels batch_y using Cross Entropy to calculate the loss loss
#             Use loss.backward to automatically compute the gradients.
#             Use torch.nn.utils.clip_grad_value_ to limit the gradient values between -grad_clip < and < grad_clip.
#             Use optimizer.step() to update the model (backpropagation).
#     After every 1000 batches, output the current loss to monitor whether it is converging.

# %%
from tqdm import tqdm
from copy import deepcopy

model.train()

i = 0 # 初始化
for epoch in range(1, epochs + 1):
    # 進度條
    bar = tqdm(dl_train, desc=f"Train epoch {epoch}")
    
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        batch_x = batch_x.to(dml)
        batch_y = batch_y.to(dml)

        # 清除梯度
        optimizer.zero_grad()
        
        # 前向傳導
        batch_pred_y = model(batch_x, batch_x_lens)
        
        # 計算損失
        loss = criterion(batch_pred_y.view(-1, batch_pred_y.size(-1)), batch_y.view(-1))
        
        # 後向傳導
        loss.backward()

        # 防止梯度爆炸
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        
        # 在模型使用優化器
        optimizer.step()
        
        i += 1
        if i % 50 == 0:
            bar.set_postfix(loss=loss.item())
    
    # 模型評估
    bar = tqdm(dl_eval, desc=f"Validation epoch {epoch}")
    matched = 0
    total = 0
    
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        batch_x = batch_x.to(dml)
        batch_y = batch_y.to(dml)

        predictions = model(batch_x, batch_x_lens)
        _, predicted_indices = torch.max(predictions, dim=-1)
        
        matched += (predicted_indices.view(-1) == batch_y.view(-1)).sum().item()
        total += batch_y.size(0)
    
    print(matched / total)

# %% [markdown]
# Generation
# 
#     Use model.generator and provide an initial character to automatically generate a sequence.

# %%
print("".join(model.generator('1+1='))) # 模型測試


