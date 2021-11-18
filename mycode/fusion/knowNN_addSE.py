import torch.nn as nn
import torch
from torch.nn.functional import normalize, dropout
class ChannelSENet(nn.Module):
    def __init__(self, NUM_EMB, ratio):
        super(ChannelSENet, self).__init__()
        self.channels = NUM_EMB 
        self.hidden_unit = NUM_EMB // ratio
        self.global_pooling = torch.nn.AdaptiveMaxPool1d(1)
        
        self.channel_attention = nn.Sequential(
                nn.Linear(self.channels, self.hidden_unit, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_unit, self.channels, bias=False),
                nn.Sigmoid()
        )
    def forward(self, input_list):
        input_list = [tensor.unsqueeze(1) for tensor in input_list]
        input_list = torch.cat(input_list, axis=1)
        ave_pool_w = self.global_pooling(input_list).squeeze()
        atte_weight = self.channel_attention(ave_pool_w).unsqueeze(-1)
        final_emb = atte_weight * input_list
        return torch.mean(final_emb, 1)
    
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
#         input_list = torch.cat(input_list, axis=1)
#         input_list = self.BN(input_list)
        atte_weight = self.channel_attention(input_list)
        final_emb = atte_weight * input_list
        return final_emb

class MLP(nn.Module):
    def __init__(self, hiddenlist=[256,256,256], final_relu=True):
        '''
        hiddenlist must contain input size 
        '''
        super(MLP, self).__init__()
        modellist = [nn.BatchNorm1d(hiddenlist[0]), nn.Dropout(0.02)]
        for idx, hdn in enumerate(hiddenlist[:-1]):
                modellist.append(nn.Linear(hiddenlist[idx], hiddenlist[idx+1])) 
                if idx == len(hiddenlist)-2:
                    if final_relu:
                        modellist.append(nn.ReLU())
#                         modellist.append(nn.Dropout(0.02, inplace=True))
                else:       
                    modellist.append(nn.ReLU())
        self.model = nn.ModuleList(modellist)
    def forward(self, x):
        for l in  self.model:
            x = l(x)
        return x
        


class Network(nn.Module):
    def __init__(self, KNOW_NUM, inputszlist, hiddensz=256, outsz=256):
        super(Network, self).__init__()
        assert  KNOW_NUM == len(inputszlist)
        self.outsz = outsz

        mlplist = []
        for i in range(KNOW_NUM):
            mlplist.append(MLP([inputszlist[i]]+[hiddensz, hiddensz, hiddensz]))
        
        selist = []
        for i in range(KNOW_NUM):
            selist.append(SENet(inputszlist[i], 8))
        
        self.MLPlist = nn.ModuleList(mlplist)
        self.SElist = nn.ModuleList(selist)
        
        
        self.finalMLP = nn.Sequential(
                nn.BatchNorm1d(KNOW_NUM * hiddensz),
#                 SENet(KNOW_NUM * hiddensz, 4),
                nn.Linear(KNOW_NUM * hiddensz, hiddensz),
                nn.ReLU(),
                nn.Linear(hiddensz, outsz, bias=True),
#                 nn.BatchNorm1d(outsz)
        )
        self.senet = SENet(outsz, 4)
#         self.se_final_fc = nn.Linear(hiddensz * KNOW_NUM, outsz)
    def forward(self, input_list):
        know_out_list = []
        # print(len(input_list))
        for idx, input in enumerate(input_list):
            # print(input.shape)
            # print(self.MLPlist[idx](input).shape)
#             input_ = self.SElist[idx](input)
            know_out_list.append(self.MLPlist[idx](input))
        
        # MLP fusion
        knows = torch.cat(know_out_list, axis=1)
        out = self.finalMLP(knows)
        # SENet fusion
#         se_out = self.senet(out)
#         out = self.finalMLP(se_out)
        return out



