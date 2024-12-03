# %% [markdown]
# Part I: Data Pre-processing

# %%
!wget http://download.tensorflow.org/data/questions-words.txt
# 以上一個步驟在Google Colab執行

# %%
# 預處理資料集
with open("questions-words.txt", "r") as f:
    data = f.read().splitlines()

# 檢查前四筆
for entry in data[:5]:
    print(entry)

# %%
# 儲存以 : 開頭的句子
start_line = []
# 建立空LIST
questions = []
categories = []
sub_categories = []

# Question
for i in range(len(data)):
    if data[i].startswith(":"):
        # 紀錄行號及 : 後的句子
        start_line.append((i, data[i]))
    else:
        questions.append(data[i])
    
# Category
m = (start_line[5][0] - 5)
n = (len(data) - m  - len(start_line))
for o in range(0, m):
    # 前五筆的
    categories.append("semantic")
for o in range(0, n):
    # 後九筆的
    categories.append("syntatic")

# SubCategory
for j in range(len(start_line)):
    # 最後一句 : 的句子
    if j == (len(start_line) - 1):
        k = (len(data) - len(start_line) - start_line[j][0])
        for l in range(0, k):
            sub_categories.append(start_line[j][1])
    # 其餘的
    else:
        k = (start_line[j+1][0] - start_line[j][0])
        for l in range(0, k):
            sub_categories.append(start_line[j][1])

# %%
import pandas as pd
# 建立DataFrame
df = pd.DataFrame(
    {
        "Question": questions,
        "Category": categories,
        "SubCategory": sub_categories,
    }
)

# 顯示前五筆
print(df.head())

# 檔案存成csv格式
df.to_csv("questions-words.csv", index=False)

# %% [markdown]
# Part II: Use pre-trained word embeddings

# %%
import pandas as pd
import numpy as np
import gensim.downloader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 載入資料
data = pd.read_csv("questions-words.csv")

# 載入模型
model = gensim.downloader.load("glove-wiki-gigaword-100")
print("The Gensim model loaded successfully!")

# %%
preds = []
golds = []
for analogy in tqdm(data["Question"]):
    # 切割句子
    s = analogy.split()
    golds.append(s[3])
    # 詞向量函數
    def w2v(word_a, word_b, word_c, model):
        try:
            result_vector = model[word_b] + model[word_c] - model[word_a]
            closest_word = model.most_similar(positive=[result_vector], topn=1)[0][0]
            return closest_word
        except KeyError as e:
            return None
    preds.append(w2v(s[0], s[1], s[2], model))

# %%
# 定義calculate_accuracy函數
def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(gold == pred)

golds_np, preds_np = np.array(golds), np.array(preds)
data = pd.read_csv("questions-words.csv")

# Evaluation: categories
for category in data["Category"].unique():
    mask = data["Category"] == category
    golds_cat, preds_cat = golds_np[mask], preds_np[mask]
    acc_cat = calculate_accuracy(golds_cat, preds_cat)
    print(f"Category: {category}, Accuracy: {acc_cat * 100}%")

# Evaluation: sub-categories
for sub_category in data["SubCategory"].unique():
    mask = data["SubCategory"] == sub_category
    golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
    acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
    print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")

# %%
# 資料集
df = pd.DataFrame(data)
# 目標
SUB_CATEGORY = ": family"

filtered_df = df[df["SubCategory"] == SUB_CATEGORY]

words = []
for p in filtered_df["Question"]:
    words.append(p.split())

# 轉一維LIST及去除重複字詞
words = list(set([item for word in words for item in word]))

# 文字轉詞向量
word_vectors = [model[word] for word in words]

# 使用 t-SNE 進行降維
tsne = TSNE(n_components=2, random_state=0, perplexity=10)
word_vectors_2d = tsne.fit_transform(np.array(word_vectors))

# 繪製圖像
plt.figure(figsize=(10, 6))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])

# 標記每個點
for i, word in enumerate(words):
    plt.text(word_vectors_2d[i, 0], word_vectors_2d[i, 1], word, fontsize=10)

plt.title("Word Relationships from Google Analogy Task")
plt.savefig("word_relationships.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Part III: Train your own word embeddings

# %%
!gdown --id 1jiu9E1NalT2Y8EIuWNa1xf2Tw1f1XuGd -O wiki_texts_part_0.txt.gz
!gdown --id 1ABblLRd9HXdXvaNv8H9fFq984bhnowoG -O wiki_texts_part_1.txt.gz
!gdown --id 1z2VFNhpPvCejTP5zyejzKj5YjI_Bn42M -O wiki_texts_part_2.txt.gz
!gdown --id 1VKjded9BxADRhIoCzXy_W8uzVOTWIf0g -O wiki_texts_part_3.txt.gz
!gdown --id 16mBeG26m9LzHXdPe8UrijUIc6sHxhknz -O wiki_texts_part_4.txt.gz
!gdown --id 17JFvxOH-kc-VmvGkhG7p3iSZSpsWdgJI -O wiki_texts_part_5.txt.gz
!gdown --id 19IvB2vOJRGlrYulnTXlZECR8zT5v550P -O wiki_texts_part_6.txt.gz
!gdown --id 1sjwO8A2SDOKruv6-8NEq7pEIuQ50ygVV -O wiki_texts_part_7.txt.gz
!gdown --id 1s7xKWJmyk98Jbq6Fi1scrHy7fr_ellUX -O wiki_texts_part_8.txt.gz
!gdown --id 17eQXcrvY1cfpKelLbP2BhQKrljnFNykr -O wiki_texts_part_9.txt.gz
!gdown --id 1J5TAN6bNBiSgTIYiPwzmABvGhAF58h62 -O wiki_texts_part_10.txt.gz

!gunzip -k wiki_texts_part_*.gz
!cat wiki_texts_part_*.txt > wiki_texts_combined.txt

!head -n 10 wiki_texts_combined.txt

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import shutil
path0 = '/content/wiki_texts_combined.txt'
path1 = '/content/drive/MyDrive/wiki_texts_combined.txt'

shutil.copyfile(path0, path1)
print("OK")
# 以上三個步驟在Google Colab執行

# %%
import random
from gensim.models import Word2Vec
# 使用壓縮檔
with open("wiki_texts_combined.txt", "r", encoding="utf-8-sig") as f:
    print("#等待3~4分鐘")
    lines = f.readlines()
    print("#讀取OK")

samples = random.sample(lines, int(len(lines) * 0.2))

with open("wiki_texts_output.txt", "w", encoding="utf-8-sig") as f:
    print("#取得20%")
    for line in samples:
        f.write(line)

# %%
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import string
with open("wiki_texts_output.txt", "r", encoding="utf-8-sig") as f:
    samples = f.readlines()
#下載數據包
nltk.download("punkt_tab")

#預處理函數
def preprocess_texts(text):
    for i in range(len(samples)):
        #字詞分割
        words = word_tokenize(text[i])
        #去除標點符號
        words = [word for word in words if word not in string.punctuation]
        return words

print("#預處理")
corpus = [preprocess_texts(samples)]

print("#訓練開始")
model = Word2Vec(sentences=corpus, min_count=1, workers=4)

print("#儲存")
model.save("w2v.model")

# %%
from gensim.models import Word2Vec
import pandas as pd
from tqdm import  tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# 載入模型
model = Word2Vec.load("w2v.model")
# 載入資料
data = pd.read_csv("questions-words.csv")

preds = []
golds = []
for analogy in tqdm(data["Question"]):
    # 切割句子
    s = analogy.split()
    golds.append(s[3])
    # 詞向量函數
    def w2v(word_a, word_b, word_c, model):
        try:
            result_vector = model.wv[word_b] + model.wv[word_c] - model.wv[word_a]
            closest_word = model.wv.most_similar(positive=[result_vector], topn=1)[0][0]
            return closest_word
        except KeyError:
            return None
    preds.append(w2v(s[0], s[1], s[2], model))

# 資料集
df = pd.DataFrame(data)
# 目標
SUB_CATEGORY = ": family"

filtered_df = df[df["SubCategory"] == SUB_CATEGORY]

words = []
for p in filtered_df["Question"]:
    words.append(p.split())

# 轉一維LIST及去除重複字詞
words = list(set([item for word in words for item in word]))

# 文字轉詞向量
word_vectors = []
for word in words:
    try:
        word_vector = model.wv[word]
    except KeyError:
        # 找到與缺失詞語最相似的詞語
        all_words = model.wv.index_to_key  # 所有詞彙
        word_similarities = []
        
        random_vector = np.random.rand(model.vector_size)
        
        # 計算與模型中的每個詞語的餘弦相似度
        for candidate_word in all_words:
            candidate_vector = model.wv[candidate_word]
            # 使用餘弦相似度公式
            similarity = np.dot(candidate_vector, random_vector) / (np.linalg.norm(candidate_vector) * np.linalg.norm(random_vector))
            word_similarities.append((candidate_word, similarity))

        # 找到相似度最高的詞
        most_similar_word = sorted(word_similarities, key=lambda x: x[1], reverse=True)[0][0]
        word_vector = model.wv[most_similar_word]
    
    word_vectors.append(word_vector)

# 使用 t-SNE 進行降維
tsne = TSNE(n_components=2, random_state=0, perplexity=10)
word_vectors_2d = tsne.fit_transform(np.array(word_vectors))

# 繪製圖像
plt.figure(figsize=(10, 6))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])

# 標記每個點
for i, word in enumerate(words):
    plt.text(word_vectors_2d[i, 0], word_vectors_2d[i, 1], word, fontsize=10)

plt.title("Word Relationships from Google Analogy Task")
plt.savefig("w2v.png", bbox_inches="tight")
plt.show()


