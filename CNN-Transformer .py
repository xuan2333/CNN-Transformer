import torch
#import torchtext
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.metrics import roc_auc_score

data=pd.read_csv('train.csv')
data2=pd.read_csv('valid.csv')
data3=pd.read_csv('test.csv');
data=data[['labels','SequenceID']]
labels=data.labels
import re
import string
pat=re.compile('[A-Za-z]')
def pre_text(text):
    text=pat.findall(text)
    text=[w.lower() for w in text]
    return text
x=data.SequenceID.apply(pre_text)
#创建词表
word_set =set()
for t in x:
    for word in t:
        word_set.add(word)

max_word=len(word_set)+1
word_list=list(word_set)
word_index=dict((w,word_list.index(w)+1)for w in word_list)
text = x.apply(lambda x: [word_index.get(word, 0) for word in x])
text_len=800
pad_text = [l + (text_len-len(l))*[0] if len(l)<=text_len else l[ :text_len]
                                                               for l in text]
pad_text = np.array(pad_text)
labels = data.labels.values

labels=data.labels
import re
import string
pat=re.compile('[A-Za-z]')
def pre_text(text):
    text=pat.findall(text)
    text=[w.lower() for w in text]
    return text
x=data.SequenceID.apply(pre_text)
#创建词表
word_set =set()
for t in x:
    for word in t:
        word_set.add(word)

max_word=len(word_set)+1
word_list=list(word_set)
word_index=dict((w,word_list.index(w)+1)for w in word_list)
text = x.apply(lambda x: [word_index.get(word, 0) for word in x])
text_len=1200
pad_text = [l + (text_len-len(l))*[0] if len(l)<=text_len else l[ :text_len]
                                                               for l in text]
pad_text = np.array(pad_text)
labels = data.labels.values
x_train,  y_train = pad_text, labels



#-------------------------------
labels=data2.labels
import re
import string
pat=re.compile('[A-Za-z]')
def pre_text(text):
    text=pat.findall(text)
    text=[w.lower() for w in text]
    return text
x=data2.SequenceID.apply(pre_text)
#创建词表
word_set =set()
for t in x:
    for word in t:
        word_set.add(word)

max_word=len(word_set)+1
word_list=list(word_set)
word_index=dict((w,word_list.index(w)+1)for w in word_list)
text = x.apply(lambda x: [word_index.get(word, 0) for word in x])
text_len=1200
pad_text = [l + (text_len-len(l))*[0] if len(l)<=text_len else l[ :text_len]
                                                               for l in text]
pad_text = np.array(pad_text)
labels = data.labels.values

labels=data2.labels
import re
import string
pat=re.compile('[A-Za-z]')
def pre_text(text):
    text=pat.findall(text)
    text=[w.lower() for w in text]
    return text
x=data.SequenceID.apply(pre_text)
#创建词表
word_set =set()
for t in x:
    for word in t:
        word_set.add(word)

max_word=len(word_set)+1
word_list=list(word_set)
word_index=dict((w,word_list.index(w)+1)for w in word_list)
text = x.apply(lambda x: [word_index.get(word, 0) for word in x])
text_len=1200
pad_text = [l + (text_len-len(l))*[0] if len(l)<=text_len else l[ :text_len]
                                                               for l in text]
pad_text = np.array(pad_text)
labels = data2.labels.values
x_valid,  y_valid = pad_text, labels
#-------------------------------
labels=data3.labels
pat=re.compile('[A-Za-z]')
def pre_text(text):
    text=pat.findall(text)
    text=[w.lower() for w in text]
    return text
x=data3.SequenceID.apply(pre_text)
#创建词表
word_set =set()
for t in x:
    for word in t:
        word_set.add(word)

max_word=len(word_set)+1
word_list=list(word_set)
word_index=dict((w,word_list.index(w)+1)for w in word_list)
text = x.apply(lambda x: [word_index.get(word, 0) for word in x])
text_len=1200
pad_text = [l + (text_len-len(l))*[0] if len(l)<=text_len else l[ :text_len]
                                                               for l in text]
pad_text = np.array(pad_text)
labels = data3.labels.values

labels=data3.labels
pat=re.compile('[A-Za-z]')
def pre_text(text):
    text=pat.findall(text)
    text=[w.lower() for w in text]
    return text
x=data.SequenceID.apply(pre_text)
#创建词表
word_set =set()
for t in x:
    for word in t:
        word_set.add(word)

max_word=len(word_set)+1
word_list=list(word_set)
word_index=dict((w,word_list.index(w)+1)for w in word_list)
text = x.apply(lambda x: [word_index.get(word, 0) for word in x])
text_len=800
pad_text = [l + (text_len-len(l))*[0] if len(l)<=text_len else l[ :text_len]
                                                               for l in text]
pad_text = np.array(pad_text)
labels = data3.labels.values
x_test,  y_test = pad_text, labels
#-------------------------------

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, text_list, label_list):
        self.text_list = text_list
        self.label_list = label_list

    def __getitem__(self, index):
        text = torch.LongTensor(self.text_list[index])
        label = self.label_list[index]
        return text, label

    def __len__(self):
        return len(self.text_list)
train_ds = Mydataset(x_train, y_train)
valid_ds=Mydataset(x_valid, y_valid)
test_ds = Mydataset(x_test, y_test)
BTACH_SIZE = 128
train_dl = torch.utils.data.DataLoader(
                                       train_ds,
                                       batch_size=BTACH_SIZE,
                                       shuffle=True,
)
valid_dl = torch.utils.data.DataLoader(
                                       valid_ds,
                                       batch_size=BTACH_SIZE,
                                       shuffle=True,
)
test_dl = torch.utils.data.DataLoader(
                                       test_ds,
                                       batch_size=BTACH_SIZE
)
embeding_dim = 50
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.1, max_len=1200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.em = nn.Embedding(max_word, embeding_dim)
        self.pos = PositionalEncoding(embeding_dim)
        self.conv1 = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=2, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=3, stride=1, padding=2, bias=True)
        self.conv3 = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=4, stride=1, padding=3, bias=True)
        self.conv4 = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=5, stride=1, padding=4, bias=True)
        self.conv5 = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=6, stride=1, padding=5, bias=True)
        self.conv6 = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=7, stride=1, padding=6, bias=True)
        self.pool1 = nn.AvgPool1d(4, stride=1, padding=1)
        self.pool2 = nn.AvgPool1d(5, stride=1, padding=1)
        self.pool3 = nn.AvgPool1d(6, stride=1, padding=1)
        self.pool4 = nn.AvgPool1d(7, stride=1, padding=1)
        self.pool5 = nn.AvgPool1d(8, stride=1, padding=1)
        self.pool6 = nn.AvgPool1d(9, stride=1, padding=1)
        self.pool = nn.AvgPool1d(2, stride=2, )
        self.conv7 = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=3, stride=1, padding=4, bias=True)
        self.conv8 = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=6, stride=1, padding=5, bias=True)
        self.conv9 = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=9, stride=1, padding=6, bias=True)
        self.pool7 = nn.AvgPool1d(5, stride=1, padding=1)
        self.pool8 = nn.AvgPool1d(8, stride=1, padding=1)
        self.pool9 = nn.AvgPool1d(11, stride=1, padding=1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=192, nhead=10)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.fc1 = nn.Linear(300, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, inputs):
        #print(inputs.size())
        x = self.em(inputs)
        x = self.pos(x)
        #print(x.size())
        x = x.permute(0, 2, 1)
        #print(x.size())
        x1 = self.conv1(x)
        #print(x1.size())
        x1 = self.pool1(x1)
        #print(x1.size())

        x2 = self.conv2(x)
        #print(x2.size())
        x2 = self.pool2(x2)
        #print(x2.size())

        x3 = self.conv3(x)
        #print(x3.size())
        x3 = self.pool3(x3)
        #print(x3.size())
        x4 = self.conv4(x)
        #print(x4.size())
        x4 = self.pool4(x4)
        #print(x4.size())
        x5 = self.conv5(x)
        #print(x5.size())
        x5 = self.pool5(x5)
        #print(x5.size())
        x6 = self.conv6(x)
        #print(x6.size())
        x6 = self.pool6(x6)
        #print(x6.size())

        x = torch.cat([x1, x2, x3, x4, x5, x6], axis=1)
        x7 = self.conv7(x)
        # print(x7.size())
        x7 = self.pool7(x7)
        # print(x7.size())
        x8 = self.conv8(x)
        # print(x8.size())
        x8 = self.pool8(x8)
        # print(x8.size())
        x9 = self.conv8(x)
        # print(x8.size())
        x9 = self.pool8(x8)
        # print(x8.size())
        #print(x.size())
        x = torch.cat([x7,x8,x9], axis=1)
        x = x.permute(0, 2, 1)
        #print(x.size())
        x = self.transformer_encoder(x)
        # print(x.size())
        # x=x.permute(0,2,1)
        # print(x.size())
        x = torch.sum(x, dim=-1)
        # print(x.size())
        x = F.dropout(F.relu(self.fc1(x)))
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        return x
weight=torch.from_numpy(np.array([0.53, 10.5])).float()
model = Net()
loss_fn = nn.CrossEntropyLoss(weight=weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loss = []
train_acc = []
test_loss = []
test_acc = []
tra_ba=[]
test_ba=[]
tra_gm = []
tra_bm = []
test_gm = []
test_bm = []
tra_mcc=[]
test_mcc=[]
tra_auc=[]
te_auc=[]
train_loss = []
train_acc = []
test_loss = []
test_acc = []
#model=model.to('cuda')


def acu_curve(y, prob):
    # y真实prob预测
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真阳性率和假阳性率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")

    plt.show()

def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    model.train()
    for x, y in trainloader:
        #x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        # print('trian_y:',y)
        loss = loss_fn(y_pred, y)
        flood = (loss - 0.002).abs() + 0.002
        optimizer.zero_grad()
        flood.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            # print('train_y_pre',y_pred)
            correct += (y_pred == y).sum().item()
            # print(y_pred,y)
            tp += ((y_pred == y) & (y == 1)).sum().item()
            fn += ((y_pred != y) & (y == 1)).sum().item()
            fp += ((y_pred != y) & (y == 0)).sum().item()
            tn += ((y_pred == y) & (y == 0)).sum().item()
            # print(tp,fn,fp,tn)
            total += y.size(0)
            running_loss += loss.item()
    #    exp_lr_scheduler.step()
    print(tp, fn, fp, tn)
    ba=((tp/(tp+fn))+(tn/(fp+tn)))/2
    mcc = (tp * tn - tp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total
    gm = math.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
    bm = tp / (tp + fn) + tn / (tn + fp) - 1

    #-------------------------

    vl_tp = 0
    vl_fn = 0
    vl_fp = 0
    vl_tn = 0
    for x, y in valid_dl:
        #x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        # print('trian_y:',y)

        y_pred = torch.argmax(y_pred, dim=1)
        # print('train_y_pre',y_pred)
        correct += (y_pred == y).sum().item()
        # print(y_pred,y)
        vl_tp += ((y_pred == y) & (y == 1)).sum().item()
        vl_fn += ((y_pred != y) & (y == 1)).sum().item()
        vl_fp += ((y_pred != y) & (y == 0)).sum().item()
        vl_tn += ((y_pred == y) & (y == 0)).sum().item()
        # print(tp,fn,fp,tn)
        total += y.size(0)
        running_loss += loss.item()

    vl_ba=((vl_tp/(vl_tp+vl_fn))+(vl_tn/(vl_fp+vl_tn)))/2
    vl_mcc = (vl_tp * vl_tn - vl_tp * vl_fn) / (math.sqrt((vl_tp + vl_fp) * (vl_tp + vl_fn) * (vl_tn + vl_fp) * (vl_tn + vl_fn)))
    vl_gm = math.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
    vl_bm = vl_tp / (vl_tp + vl_fn) + vl_tn / (vl_tn + vl_fp) - 1
    print(vl_ba,vl_bm,vl_gm,vl_mcc)
    #-------------------------




    test_correct = 0
    test_total = 0
    test_running_loss = 0
    te_tp = 0
    te_fn = 0
    te_fp = 0
    te_tn = 0



    model.eval()
    with torch.no_grad():
        for x, y in testloader:

            #x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            # print('test_y:',y)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            # print('test_y_pre',y_pred)
            test_correct += (y_pred == y).sum().item()
            te_tp += ((y_pred == y) & (y == 1)).sum().item()
            te_fn += ((y_pred != y) & (y == 1)).sum().item()
            te_fp += ((y_pred != y) & (y == 0)).sum().item()
            te_tn += ((y_pred == y) & (y == 0)).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    print(te_tp, te_fn, te_fp, te_tn)
    te_ba = ((te_tp / (te_tp + te_fn)) + (te_tn / (te_fp + te_tn))) / 2
    te_mcc = (te_tp * te_tn - te_tp * te_fn) / math.sqrt(
        (te_tp + te_fp) * (te_tp + te_fn) * (te_tn + te_fp) * (te_tn + te_fn))
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
    te_gm = math.sqrt((te_tp / (te_tp + te_fn)) * (te_tn / (te_tn + te_fp)))
    te_bm = te_tp / (te_tp + te_fn) + te_tn / (te_tn + te_fp) - 1
    #acu_curve(y,y_pred)
    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 5),
          'ba:', round(ba, 5),
          tra_ba.append(ba),
          'test_loss： ', round(epoch_test_loss, 5),
          'test_ba: ', round(te_ba, 5),
          test_ba.append(te_ba),
          'GM: ', round(gm, 5),
          tra_gm.append(gm),
          'test_GM: ', round(te_gm, 5),
          test_gm.append(te_gm),
          'BM: ', round(bm, 5),
          tra_bm.append(bm),
          'te_BM: ', round(te_bm, 5),
          test_bm.append(te_bm),
          'mcc: ', round(mcc, 5),
          tra_mcc.append(mcc),
          'test_mcc: ', round(te_mcc, 5),
          test_mcc.append(te_mcc)
           #'auc: ',round(auc,5),
           #tra_auc.append(auc),
           #'test_auc: ',round(test_auc,5),
           #te_auc.append(test_auc)
          # 'tp: ',round(tp),
          # 'fn: ',round(fn),
          # 'fp: ',round(fp),
          # 'tn: ',round(tn),
          # 'te_tp: ',round(te_tp),
          # 'te_fn: ',round(te_fn),
          # 'te_fp: ',round(te_fp),
          # 'te_tn: ',round(te_tn)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc
epochs = 50
i=0
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    i=i+1
    print("save_model")
    torch.save(model.state_dict(), '\\mz_tragaijin_%d.pkl' % i)
    train_loss.append(epoch_loss)
    #train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    #test_acc.append(epoch_test_acc)
print(test_loss)
print(test_ba)
print(test_bm)
print(test_gm)
print(test_mcc)
print(te_auc)
print(train_loss)
print(tra_ba)
print(tra_bm)
print(tra_gm)
print(tra_mcc)
print(tra_auc)