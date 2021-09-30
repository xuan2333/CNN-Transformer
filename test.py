import sys
import torch
#import torchtext
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import re
import string
pat=re.compile('[A-Za-z]')
def pre_text(text):
    text= pat.findall(text)
    text=[w.lower() for w in text]
    return text
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
if __name__ == '__main__':
    data=pd.read_csv('test.csv');
    data = data[['labels', 'SequenceID']]
    x = data.SequenceID.apply(pre_text)
    word_set = set()
    for t in x:
        for word in t:
            word_set.add(word)

    max_word = len(word_set) + 1
    word_list = list(word_set)
    word_index = dict((w, word_list.index(w) + 1) for w in word_list)
    text = x.apply(lambda x: [word_index.get(word, 0) for word in x])
    text_len = 1200
    pad_text = [l + (text_len - len(l)) * [0] if len(l) <= text_len else l[:text_len]
                for l in text]
    pad_text = np.array(pad_text)
    model=Net().cuda()
    model = model.load_state_dict(torch.load('mz_tragaijin_24.pkl'))
    pad_text=torch.from_numpy(pad_text)
    pad_text=pad_text.to(torch.int64).cuda()
    pred_last=model(pad_text)
    pred_last=torch.argmax(pred_last,dim=1)
    pred_last= pred_last.detach().numpy()
    pred_last = pred_last.tolist()
    print(pred_last)