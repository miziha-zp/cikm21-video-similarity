import torch.nn as nn
import torch
from torch.nn.functional import normalize, dropout


class MLP(nn.Module):
    def __init__(self, hiddenlist=[256,256,256], final_relu=False):
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

        self.MLPlist = nn.ModuleList(mlplist)
        self.finalMLP = nn.Sequential(
                nn.BatchNorm1d(KNOW_NUM * hiddensz),
                nn.Linear(KNOW_NUM * hiddensz, hiddensz),
                nn.ReLU(),
#                 nn.Dropout(0.2),
                nn.Linear(hiddensz, outsz, bias=True),
#                 nn.ReLU()
        )

    def forward(self, input_list):
        know_out_list = []
        # print(len(input_list))
        for idx, input in enumerate(input_list):
            # print(input.shape)
            # print(self.MLPlist[idx](input).shape)
            know_out_list.append(self.MLPlist[idx](input))

        knows = torch.cat(know_out_list, axis=1)
        # print
        out = self.finalMLP(knows)
        return out



