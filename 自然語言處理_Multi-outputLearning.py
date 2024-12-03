# %%
import transformers as T
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torchmetrics import SpearmanCorrCoef, Accuracy, F1Score
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# 有些中文的標點符號在tokenizer編碼以後會變成[UNK]，所以將其換成英文標點
token_replacement = [
    ["：" , ":"],
    ["，" , ","],
    ["“" , "\""],
    ["”" , "\""],
    ["？" , "?"],
    ["……" , "..."],
    ["！" , "!"]
]

# %%
class MultiLabelModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 使用BERT模型作為編碼器
        self.encoder = T.AutoModel.from_pretrained('bert-base-uncased')
        # 定義 relatedness_score ，輸出維度為1
        self.relatedness_head = torch.nn.Linear(self.encoder.config.hidden_size, 1)
        # 定義 entailment_judgment ，輸出維度為3
        self.entailment_head = torch.nn.Linear(self.encoder.config.hidden_size, 3)
        
    def forward(self, **kwargs):
        input_ids = kwargs.get('input_ids').to(device)
        attention_mask = kwargs.get('attention_mask').to(device)
        token_type_ids = kwargs.get('token_type_ids', None).to(device)
        # 將輸入傳入編碼器以獲取編碼器輸出
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 獲取CLS標記的輸出作為整個序列的表示
        cls_output = encoder_outputs.last_hidden_state[:, 0, :]
        # 通過 relatedness_score 計算相關性分數
        relatedness_score = self.relatedness_head(cls_output)
        # 通過 entailment_judgment 計算邊際分數
        entailment_logits = self.entailment_head(cls_output)
        
        return relatedness_score, entailment_logits

# %%
torch.manual_seed(2) # 設定隨機種子，避免每次訓練不一樣
model = MultiLabelModel().to(device)
tokenizer = T.BertTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/")

# %%
class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation", 'test'] # 加入 test 資料集
        self.data = load_dataset(                                     # 本機加入trust_remote_code=True以信任遠端
            "sem_eval_2014_task_1", split=split, cache_dir="./cache/", trust_remote_code=True
        ).to_list()

    def __getitem__(self, index):
        d = self.data[index]
        # 把中文標點替換掉
        for k in ["premise", "hypothesis"]:
            for tok in token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)

data_sample = SemevalDataset(split="train").data[:3]
print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")

# %%
# 定義參數
lr = 3e-5
epochs = 3
# 修改以下參數使 test set 達成 SpearmanCorrCoef over 0.77, Accuracy over 0.85 (需要在雲端環境)
train_batch_size = 64
validation_batch_size = 64

# %%
def collate_fn(batch):
    premises = [item['premise'] for item in batch]
    hypotheses = [item['hypothesis'] for item in batch]
    relatedness_scores = [item['relatedness_score'] for item in batch]
    entailment_labels = [item['entailment_judgment'] for item in batch]
    # 使用 tokenizer 進行分詞處理
    tokenized_inputs = tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors='pt')
    # 轉換為 PyTorch 張量
    relatedness_scores = torch.tensor(relatedness_scores)
    entailment_labels = torch.tensor(entailment_labels)

    return tokenized_inputs, relatedness_scores, entailment_labels
# 建立 DataLoader
dl_train = DataLoader(SemevalDataset(split='train'),
                    batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True)
dl_validation = DataLoader(SemevalDataset(split='validation'),
                    batch_size=validation_batch_size, collate_fn=collate_fn, shuffle=False)
dl_test = DataLoader(SemevalDataset(split='test'),
                    batch_size=validation_batch_size, collate_fn=collate_fn, shuffle=False)

# %%
optimizer = AdamW(model.parameters(), lr=lr) # 定義優化器

regression_loss_fn = torch.nn.MSELoss() # 回歸損失函數
classification_loss_fn = torch.nn.CrossEntropyLoss() # 分類損失函數

# 計算評分方法
spc = SpearmanCorrCoef().to(device)
acc = Accuracy(task="multiclass", num_classes=3).to(device)
f1 = F1Score(task="multiclass", num_classes=3, average='macro').to(device)

# %%
for ep in range(epochs):
    pbar = tqdm(dl_train) # 顯示進度條
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    model.train()
    
    total_loss = 0.0 # 初始化
    for tokenized_inputs, relatedness_scores, entailment_labels in pbar:
        tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
        relatedness_scores = relatedness_scores.to(device)
        entailment_labels = entailment_labels.to(device)
        # 清空之前的梯度
        optimizer.zero_grad()
        # 獲取模型輸出的回歸和分類結果
        output_reg, output_clf = model(**tokenized_inputs)
        # 計算損失
        loss_reg = regression_loss_fn(output_reg.squeeze(), relatedness_scores)
        loss_clf = classification_loss_fn(output_clf, entailment_labels)
        loss = loss_reg + loss_clf
        total_loss += loss.item()
        
        loss.backward() # 計算梯度
        optimizer.step() # 更新模型參數

    pbar = tqdm(dl_validation) # 顯示進度條
    pbar.set_description(f"Validation epoch [{ep+1}/{epochs}]")
    model.eval()
    # 初始化
    val_loss = 0.0
    spearman_corr = 0.0
    accuracy = 0.0
    f1_score = 0.0
    # 在評估模式下不計算梯度
    with torch.no_grad():
        for tokenized_inputs, relatedness_scores, entailment_labels in pbar:
            tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
            relatedness_scores = relatedness_scores.to(device)
            entailment_labels = entailment_labels.to(device)
            # 獲取模型輸出的回歸和分類結果
            output_reg, output_clf = model(**tokenized_inputs)
            # 計算損失
            loss_reg = regression_loss_fn(output_reg.squeeze(), relatedness_scores)
            loss_clf = classification_loss_fn(output_clf, entailment_labels)
            loss = loss_reg + loss_clf
            val_loss += loss.item()
            # 計算評分
            spearman_corr += spc(output_reg.squeeze(), relatedness_scores)
            accuracy += acc(output_clf, entailment_labels)
            f1_score += f1(output_clf, entailment_labels)
    
    avg_spearman = spearman_corr / len(dl_validation)
    avg_accuracy = accuracy / len(dl_validation)
    avg_f1 = f1_score / len(dl_validation)
    
    print('訓練損失: {}'.format(total_loss))
    print('評估損失: {}'.format(val_loss))
    print(f"Spearman Correlation: {avg_spearman:.4f}")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")

    torch.save(model, f'./saved_models/ep{ep+1}.ckpt') # 儲存模型

# %%
pbar = tqdm(dl_test) # 顯示進度條
pbar.set_description('TEST 資料集')
model.eval()
# 初始化
val_loss = 0.0
spearman_corr = 0.0
accuracy = 0.0
f1_score = 0.0
# 在評估模式下不計算梯度
with torch.no_grad():
    for tokenized_inputs, relatedness_scores, entailment_labels in pbar:
        tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
        relatedness_scores = relatedness_scores.to(device)
        entailment_labels = entailment_labels.to(device)
        # 獲取模型輸出的回歸和分類結果
        output_reg, output_clf = model(**tokenized_inputs)
        # 計算損失
        loss_reg = regression_loss_fn(output_reg.squeeze(), relatedness_scores)
        loss_clf = classification_loss_fn(output_clf, entailment_labels)
        loss = loss_reg + loss_clf
        val_loss += loss.item()
        # 計算評分
        spearman_corr += spc(output_reg.squeeze(), relatedness_scores)
        accuracy += acc(output_clf, entailment_labels)
        f1_score += f1(output_clf, entailment_labels)

avg_spearman = spearman_corr / len(dl_test)
avg_accuracy = accuracy / len(dl_test)
avg_f1 = f1_score / len(dl_test)

print(f"Spearman Correlation: {avg_spearman:.4f}")
print(f"Accuracy: {avg_accuracy:.4f}")
print(f"F1 Score: {avg_f1:.4f}")


