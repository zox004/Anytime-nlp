import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import random
from tqdm import tqdm, tqdm_notebook
import pandas as pd
from sklearn.model_selection import train_test_split
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

# cpu or gpu device 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# KoBERT가 사용한 토크나이저, 사전 훈련된 모델, 모델의 어휘 세트(보카) 로드
# KoBERT가 감성 분석에서 사용한 모델 파라미터 그대로 적용
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
tok = tokenizer.tokenize

# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 100
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):

        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    
    return train_acc

def inference(model, sentence):
    category = {0:'관련 없는', 1:"학교 관련"}
    dataset = [[sentence, '0']]
    test = BERTDataset(dataset, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=2)

    model.eval()
    answer = 0

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)
        test_eval = []
        for logits in out:
            logits = logits.detach().cpu().numpy()
            test_eval.append(category[np.argmax(logits)])

        print(f"Question: {sentence}\nAnswer: {test_eval[0]} 질문입니다.")
        
def train():
    # 데이터 파일 경로
    data_file_path = '/content/drive/MyDrive/AnyTime/dataset/data.txt'

    # 데이터 읽어오기
    data = pd.read_csv(data_file_path, sep="\t")

    # 데이터 섞기
    data_shuffled = data.sample(frac=1, random_state=42)

    # 데이터 분할 (8:2 비율)
    train_data, test_data = train_test_split(data_shuffled, test_size=0.2, random_state=42)

    # 훈련 데이터와 테스트 데이터 저장
    train_data.to_csv("train_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)

    # 저장된 CSV 파일 다시 읽기
    train_set = pd.read_csv("train_data.csv")
    validation_set = pd.read_csv("test_data.csv")

    train_set_data = [[i, str(j)] for i, j in zip(train_set['query'], train_set['label'])]
    validation_set_data = [[i, str(j)] for i, j in zip(validation_set['query'], validation_set['label'])]

    train_set_data, test_set_data = train_test_split(train_set_data, test_size = 0.2, random_state=4)

    train_set_data = BERTDataset(train_set_data, 0, 1, tok, vocab, max_len, True, False)
    test_set_data = BERTDataset(test_set_data, 0, 1, tok, vocab, max_len, True, False)
    train_dataloader = torch.utils.data.DataLoader(train_set_data, batch_size=batch_size, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_set_data, batch_size=batch_size, num_workers=2)
    
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    best_val_acc = 0.0 # 가장 높은 테스트 정확도 초기화
    best_model = None

    for e in range(num_epochs):
        train_acc = 0.0
        val_acc = 0.0
        model.train()
        
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
                
        print("epoch {} [train acc {}]".format(e+1, train_acc / (batch_id+1)))
        model.eval()
        
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            
            valid_length= valid_length
            label = label.long().to(device)            
            out = model(token_ids, valid_length, segment_ids)
            val_acc += calc_accuracy(out, label)

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_model = model
            
        print("epoch {} [val acc {}, best acc {}] ".format(e+1, val_acc / (batch_id+1), best_val_acc / (batch_id+1)))
    
    torch.save(best_model, '/content/drive/MyDrive/AnyTime/models/kobert_data564.pt')
    # torch.save(model.state_dict(), f'/content/drive/MyDrive/AnyTime/models/AnyTimeofKoBert_StateDict.pt')

def main():
    model = torch.load('/content/drive/MyDrive/AnyTime/models/kobert_data564.pt', map_location=torch.device(device))
    
    while True:
        print("\n하고싶은 말을 입력해주세요(종료: 0)")
        sentence = input()
        if sentence == "0":
            break

        inference(model, sentence)

if __name__=="__main__":
    main()