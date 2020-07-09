import torch
from torch import nn
import os
import copy
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from transformers import BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel
from transformers import BertTokenizer,BertConfig,BertForTokenClassification,BertModel
from transformers import AlbertModel, AlbertTokenizer,AlbertConfig
import time,datetime
from sklearn.metrics import precision_score,classification_report,f1_score,recall_score
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaConfig, RobertaModel,RobertaTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from crf import CRF
import time
import sys,json,re
from sklearn.model_selection import KFold
import copy

base = '../data/raw'
base_path = '/root/pycharm/yy/berts/bert-base-uncased'
data_path = 'new_train.json'

def load_data(base,train_path):
    with open(os.path.join(base,data_path),'r',encoding='utf-8')as f:
        data = f.readlines()
    
    tokens,labels = [],[]
    for i,line in enumerate(data):
        line = json.loads(line)
        raw_text = line['text']
        text = list(raw_text)
        label = len(raw_text) * ['O']
        tokens.append(text)
        entities = line['entities']
        for i,en in enumerate(entities):
            entity,tag,start,end = en['entity'],en['type'],en['start'],en['end']
            if raw_text[start:end] == entity:
                label[start] = 'B-' + tag
                label[start +1:end] = ['I-' + tag] *(end - start -1)
        labels.append(label)
    return tokens,labels

def trans_data(tokens,labels):
    texts,tags = [],[]
    for (token,label) in zip(tokens,labels):
        res,tag = [],[]
        start,end = -1,-1
        for i,(word,_) in enumerate(zip(token,label)):
            if word != ' ' and start < 0 :
                start = i
                end = i
            elif word in ['\t',' ','.']:
                end = i
                res.append(''.join(token[start:end]))
                tag.append(label[start])
                start,end = -1,-1
#         res = [isolate(i) for i in res]
        texts.append(res)
        tags.append(tag)
    return texts,tags

def trans2id(labels):
    tag_set = set()
    for line in labels:
        for label  in line:
            if label not in tag_set:
                tag_set.add(label)
    tag_set.add('[CLS]')
    tag_set.add('[SEP]')
    tag_set = list(tag_set)
    idx = [i for i in range(len(tag_set))]
    tag2id = dict(zip(tag_set,idx))
    id2tag = dict(zip(idx,tag_set))
    return tag2id,id2tag

def isolate(x):
    x = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）)]+:",'',x)
    return x

def gen_features(tokens,labels,tokenizer,tag2id,max_len):
    input_ids,tags,masks,lengths = [],[],[],[]
    for i,(token,label) in enumerate(zip(tokens,labels)):
        lengths.append(len(token))
        if len(token) >= max_len - 2:
            token = token[0:max_len - 2]
            label = labels[i][0:max_len - 2]
        mask = [1] * len(token)
                
        token.insert(0,'[CLS]')
        token.append('[SEP]')
        
        input_id = tokenizer.convert_tokens_to_ids(token)
        label = [tag2id['[CLS]']] + [tag2id[i] for i in label] + [tag2id['[SEP]']]
        mask = [0] + mask + [0]
        # padding
        if len(input_id) < max_len:
            input_id = input_id + [0] * (max_len - len(input_id))
            label = label + [tag2id['O']] * (max_len - len(label))
            mask = mask + [0] * (max_len - len(mask))
        
        assert len(input_id) == max_len
        assert len(label) == max_len
        assert len(mask) == max_len
         
        input_ids.append(input_id)
        tags.append(label)
        masks.append(mask)
    return input_ids,tags,masks,lengths


def convert_examples_to_features(tokens,labels,tokenizer,tag2id,max_len,split_num):
    input_ids,tags,masks,lengths = [],[],[],[]
    for i,(token,label) in enumerate(zip(tokens,labels)):
        skip_len = len(token) / split_num
        for i in range(split_num):
            token_choice = token[int(i * skip_len) : int((i + 1) * skip_len)]
            label_choice = label[int(i * skip_len) : int((i + 1) * skip_len)]
            lengths.append(len(token_choice))
            
            if len(token_choice) >= max_len - 2:
                token_choice = token_choice[0:max_len - 2]
                label_choice = label_choice[0:max_len - 2]
            mask = [1] * len(token_choice)
            
            token_choice.insert(0,'[CLS]')
            token_choice.append('[SEP]')
            
            input_id = tokenizer.convert_tokens_to_ids(token_choice)
            label_choice = [tag2id['[CLS]']] + [tag2id[i] for i in label_choice] + [tag2id['[SEP]']]
            mask = [0] + mask + [0]
            
            if len(input_id) < max_len:
                input_id = input_id + [0] * (max_len - len(input_id))
                label_choice = label_choice + [tag2id['O']] * (max_len - len(label_choice))
                mask = mask + [0] * (max_len - len(mask))
        
            assert len(input_id) == max_len
            assert len(label_choice) == max_len
            assert len(mask) == max_len
            
            input_ids.append(input_id)
            tags.append(label_choice)
            masks.append(mask)
    return input_ids,tags,masks,lengths  
            
def trans_word_id(path):
    with open(path,'r')as f:
        vocab = f.readlines()
    vocab = [word.replace('\n','') for word in vocab]
    index = [i for i in range(len(vocab))]
    word2idx = dict(zip(vocab,index))
    idx2owrd = dict(zip(index,vocab))
    return word2idx,idx2owrd

def gen_word_set(base,data_path):
    with open(os.path.join(base,data_path),'r',encoding='utf-8')as f:
        data = f.readlines()
    all_data = []
    for line in data:
        all_data.append(json.loads(line))
    word_set = set()
    for line in all_data:
        entities = line['entities']
        for en in entities:
            for i in en['entity'].split(' '):
                if i != '':
                    word_set.add(i)
    return word_set

def AddOOVTokenizer(word_set,tokenizer):
    oov = []
    vocab = [word  for word in tokenizer.vocab]
    for i in word_set:
        if i not in vocab:
            oov.append(i)
    for i in oov:
        tokenizer.add_tokens(i)
    return tokenizer,oov

max_len = 128
bs = 32
split_num = 3
tokenizer = BertTokenizer.from_pretrained('/root/pycharm/yy/berts/bert-base-uncased')
# tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer.convert_tokens_to_ids("Quidditch")

tokens,labels = load_data(base,data_path)
tag2id,id2tag = trans2id(labels)
print(id2tag)
texts,tags =  trans_data(tokens,labels)
word_set = gen_word_set(base,data_path)

tokenizer,oov = AddOOVTokenizer(word_set,tokenizer)

input_ids,input_tags,input_masks,input_lengths = convert_examples_to_features(texts,tags,tokenizer,tag2id,max_len,split_num)


len(input_ids)

lengths = [len(i) for i in texts]
print('MAX:',max(lengths),"MIN:",min(lengths),"AVG:",np.mean(lengths),"MID:",np.median(lengths))
print(len(lengths))
print('num of len > 128:',len([i for i in lengths if i > 128]))


class Bert_CRF(nn.Module):
    def __init__(self,base_path,oov,num_labels,lstm_hidden_size = 128,dropout = 0.3,lm_flag = False):
        super(Bert_CRF,self).__init__()
        bert_config = BertConfig.from_json_file(os.path.join(base_path,'config.json'))
        bert_config.num_labels = num_labels
        #hidden_states (tuple(torch.FloatTensor), optional, returned when config.output_hidden_states=True):
        bert_config.output_hidden_states=True
        bert_config.output_attentions=True
        self.bert = BertModel.from_pretrained(os.path.join(base_path,'pytorch_model.bin'), config=bert_config)
        self.tokenizer = tokenizer
        self.oov = oov
        self._oov_embed()
        self.dropout = nn.Dropout(dropout)
        #lstm input_size = bert_config.hidden_size  hidden_size(第二个参数)= 跟Linear 的第一个参数对上
        # 尝试下双向LSTM
        self.lm_flag = lm_flag
        self.lstm = nn.LSTM(bert_config.hidden_size, lstm_hidden_size,
                            num_layers=1, bidirectional=True, dropout=0.3, batch_first=True)
        self.clf = nn.Linear(256,bert_config.num_labels + 2)
        self.layer_norm = nn.LayerNorm(lstm_hidden_size * 2)
        self.crf = CRF(target_size=bert_config.num_labels, average_batch=True, use_cuda=True)
        
    def _oov_embed(self):
        weight = self.bert.embeddings.word_embeddings.weight.data
        weight = weight.numpy()
        _,embed_size= weight.shape
        #生成随机正态分布的矩阵
        mean = np.mean(weight)
        var = np.var(weight)
        rand_oov = np.random.normal(loc=mean, scale=var, size=(len(self.oov),embed_size))
        embed = np.concatenate((weight,rand_oov),axis=0)
        embed = torch.Tensor(embed)
        self.bert.embeddings.word_embeddings.weight.data = embed
        
    def forward(self,input_ids,masks):
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        
        outputs = self.bert(input_ids, attention_mask=masks)
        # 方案一：
        #embeds = outputs[0]
        
        #方案二：倒数第二层hidden_states 的shape
        # bert_config的设置
        all_hidden_states, all_attentions = outputs[-2:]
        embeds = all_hidden_states[-2]

        lstm_out,hidden = self.lstm(embeds)
        lstm_out= lstm_out.contiguous().view(-1, 128*2)
        if self.lm_flag:
            lstm_out = self.layer_norm(lstm_out)
        logits = self.clf(lstm_out)
        logits = logits.contiguous().view(batch_size, seq_length, -1)
        return logits
    
    def loss(self,logits,mask,tag):
        loss_value = self.crf.neg_log_likelihood_loss(logits,mask,tag)
        bs = logits.size(0)
        loss_value  /= float(bs)
        return loss_value


model = Bert_CRF(base_path,oov,num_labels = len(tag2id),lm_flag = True)
model.to(device)
param = [{'params':model.bert.parameters(),'lr':5e-5},
         {'params':model.lstm.parameters(),'lr':1e-4},
         {'params':model.crf.parameters(),'lr':1e-4}
        ]
optimizer = AdamW(param)
# optimizer = AdamW(model.parameters(),
#                   lr = 5e-5, # default is 5e-5
#                   eps = 1e-8 # default is 1e-8
#                 )
epochs = 50

def trans2label(id2tag,data,lengths):
    new = []
    for i,line in enumerate(data):
        tmp = [id2tag[word] for word in line]
        tmp = tmp[1:1 + lengths[i]]    
        new.append(tmp)
    return new

def get_entities(tags):
    start, end = -1, -1
    prev = 'O'
    entities = []
    n = len(tags)
    tags = [tag.split('-')[1] if '-' in tag else tag for tag in tags]
    for i, tag in enumerate(tags):
        if tag != 'O':
            if prev == 'O':
                start = i
                prev = tag
            elif tag == prev:
                end = i
                if i == n -1 :
                    entities.append((start, i))
            else:
                entities.append((start, i - 1))
                prev = tag
                start = i
                end = i
        else:
            if start >= 0 and end >= 0:
                entities.append((start, end))
                start = -1
                end = -1
                prev = 'O'
    return entities

def measure(preds,trues,lengths,id2tag):
    correct_num = 0
    predict_num = 0
    truth_num = 0
    pred = trans2label(id2tag,preds,lengths)
#     print('pred',pred)
    true = trans2label(id2tag,trues,lengths)
    assert len(pred) == len(true)
    for p,t in zip(pred,true):
        pred_en = get_entities(p)
        true_en = get_entities(t)
#         print('pred_en',pred_en)
#         print('true_en',true_en)
        correct_num += len(set(pred_en) & set(true_en))
        predict_num += len(set(pred_en))
        truth_num += len(set(true_en))
    precision = correct_num / predict_num if predict_num else 0
    recall = correct_num / truth_num if truth_num else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return f1, precision, recall

def save_model(model,path,f1,epoch):
    if not os.path.exists(path):
        os.mkdir(path)


input_ids = np.array(input_ids)
input_tags = np.array(input_tags)
input_masks = np.array(input_masks)
input_lengths = np.array(input_lengths)


kf = KFold(n_splits=5)
for i,(train_index, test_index) in enumerate(kf.split(input_ids)):
    print("**************************** k Fold:{} Starting ****************************".format(i))
    train_ids, dev_ids = input_ids[train_index], input_ids[test_index]
    train_tags, dev_tags= input_tags[train_index], input_tags[test_index]
    train_masks, dev_masks = input_masks[train_index], input_masks[test_index]
    train_lengths, dev_lengths = input_lengths[train_index], input_lengths[test_index]
    
    print(train_ids.shape,train_tags.shape,train_masks.shape)
    
    train_ids = torch.tensor(train_ids)
    train_tags = torch.tensor(train_tags)
    train_masks = torch.tensor(train_masks)
    # train_lengths = torch.tensor(train_lengths)

    dev_ids = torch.tensor(dev_ids)
    dev_tags = torch.tensor(dev_tags)
    dev_masks = torch.tensor(dev_masks)
    # dev_lengths = torch.tensor(dev_lengths)
    
    train_data = TensorDataset(train_ids, train_masks, train_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(dev_ids, dev_masks, dev_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)
    
    max_grad_norm = 1.0
    
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                           num_warmup_steps = 0,
                                           num_training_steps = total_steps)

    tra_loss,train_steps = 0.0,0
    dev_loss,dev_steps = 0.0,0

    start = time.time()
    for i in range(epochs):
        model.train()
        for step ,batch in enumerate(train_dataloader):
            input_ids,masks,labels= (i.to(device) for i in batch)
#             print(input_ids.shape,masks.shape)
            outputs = model(input_ids,masks)
            loss = model.loss(outputs,masks,labels)

            loss.backward()

            tra_loss += loss.item()
            train_steps += 1

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            scheduler.step()
            optimizer.step()

            if step % 50 == 0:
                print("epoch :{},step :{} ,Train loss: {}".format(i,step,tra_loss/train_steps))
                if tra_loss/train_steps < 0:
                    end = time.time()
                    print('Training Time:',end - start)
                    print('Early Stop')
#                     break
                    sys.exit(0)

        print("Training Loss of epoch {}:{}".format(i,tra_loss / train_steps))

        model.eval()
        predictions , true_labels = [], []

        for step ,batch in enumerate(valid_dataloader):
            input_ids,masks,labels = (i.to(device) for i in batch)
            with torch.no_grad():
                logits = model(input_ids,masks)
                loss = model.loss(logits,masks,labels)
                path_score, best_path = model.crf(logits, input_ids.bool())

                dev_loss += loss.item()
                dev_steps += 1

                if step % 50 == 0:
                    print("epoch :{},step :{} ,Dev loss: {}".format(i,step,dev_loss/dev_steps))

            best_path = best_path.detach().cpu().numpy().tolist()
            predictions.extend(best_path)
            true_labels.extend(labels.to('cpu').numpy().tolist())
        f1, precision, recall = measure(predictions,true_labels,dev_lengths,id2tag)
        print('epoch {} : Acc : {},Recall : {},F1 :{}'.format(i,precision,recall,f1))
    end = time.time()
    print('Training Time:',end - start)
    sys.exit(0)
