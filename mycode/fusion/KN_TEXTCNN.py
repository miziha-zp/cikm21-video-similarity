import torch.nn as nn
import torch
from torch.nn.functional import normalize, dropout
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, hiddenlist=[256,256,256], final_relu=False):
        '''
        hiddenlist must contain input size 
        '''
        super(MLP, self).__init__()
        modellist = [nn.BatchNorm1d(hiddenlist[0])]
        for idx, hdn in enumerate(hiddenlist[:-1]):

                modellist.append(nn.Linear(hiddenlist[idx], hiddenlist[idx+1])) 
                if idx == len(hiddenlist)-2:
                    if final_relu:
                        modellist.append(nn.ReLU())
#                         modellist.append(nn.Dropout(0.02, inplace=True))
                else:       
                    modellist.append(nn.ReLU())
                modellist.append(nn.BatchNorm1d(hiddenlist[idx+1]))
        self.model = nn.ModuleList(modellist)
    def forward(self, x):
        for l in  self.model:
            x = l(x)
        return x
        
class CNN(nn.Module):
    def __init__(self, outsz, seq_len=32, bert_embsz=768):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, 256, (size, bert_embsz)) for size in [3,4,5,6]])
        self.final_mlp = MLP([1024, 512, outsz])
        
    def forward(self, input_emb):
        cnn = input_emb.unsqueeze(1)#.cuda()
        cnn = [F.relu(conv(cnn)).squeeze(3) for conv in self.convs]
        # print(cnn[0].shape)
        cnn = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in cnn]
        # print(cnn[0].shape) # torch.Size([bs, 256])
        cnn = torch.cat(cnn, 1)
        out = self.final_mlp(cnn)
        # print(out.shape)
        return out

class SENet(nn.Module):
    def __init__(self, input_sz, ratio):
        super(SENet, self).__init__()
        self.channels = input_sz
        self.hidden_unit = input_sz // ratio
        self.BN = nn.BatchNorm1d(self.channels)
        self.channel_attention = nn.Sequential(
                nn.Linear(self.channels, self.hidden_unit, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_unit, self.channels, bias=False),
                nn.Sigmoid()
        )
    def forward(self, input_list):
        input_list = self.BN(input_list)
        atte_weight = self.channel_attention(input_list)
        final_emb = atte_weight * input_list
        return final_emb

class Network(nn.Module):
    def __init__(self, KNOW_NUM, inputszlist, hiddensz=256, outsz=256):
        super(Network, self).__init__()
        assert  KNOW_NUM == len(inputszlist)
        self.outsz = outsz

        mlplist = []
        for shape in inputszlist:
            if len(shape) == 1:
                 mlplist.append(MLP([shape[0]]+[hiddensz, hiddensz, hiddensz]))
            else:
                 mlplist.append(CNN(hiddensz, seq_len=shape[0], bert_embsz=shape[1]))

        self.MLPlist = nn.ModuleList(mlplist)
        self.enhanceNet = SENet(KNOW_NUM * hiddensz, 8)
        self.finalMLP = nn.Sequential(
                nn.BatchNorm1d((KNOW_NUM-0) * hiddensz),
                nn.Linear((KNOW_NUM-0)  * hiddensz, hiddensz),
                nn.ReLU(),
#                 nn.Dropout(0.2),
                nn.Linear(hiddensz, outsz, bias=True),
#                 nn.ReLU()
        )
        
        
    def forward(self, input_list):
        know_out_list = []
        for idx, input in enumerate(input_list):
            know_out_list.append(self.MLPlist[idx](input))

        knows = torch.cat(know_out_list, axis=1)
        # print
        knows = self.enhanceNet(knows)
        out = self.finalMLP(knows)
        return out



