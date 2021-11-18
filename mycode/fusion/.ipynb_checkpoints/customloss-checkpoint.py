import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

class dual_softmax_loss(nn.Module):
    def __init__(self,):
        super(dual_softmax_loss, self).__init__()
        
    def forward(self, sim_matrix, temp=1000):
        sim_matrix = sim_matrix * F.softmax(sim_matrix/temp, dim=0)*len(sim_matrix) #With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss

class NCELoss(nn.Module):
    def __init__(self, temp=0.001):
        super(NCELoss, self).__init__()
        self.temp = temp
        
    def forward(self, input1, input2):
        sim_matrix = cosine(input1, input2)
        logpt = F.log_softmax(sim_matrix / self.temp, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss
 
    
def cosine(input1, input2,):
    input1 = F.normalize(input1, dim=1)
    input1 = input1.unsqueeze(1)
    input2 = F.normalize(input2, dim=1)
    input2 = input2.unsqueeze(2)
    similarity = torch.bmm(input1, input2).squeeze().squeeze()
    return similarity

class SentenceBertLoss(nn.Module):
    def __init__(self, outsz):
        super(SentenceBertLoss, self).__init__()
        self.fc = nn.Linear(outsz * 3, 1)
        self.MSEloss = nn.MSELoss()
    def forward(self, input1, input2, label, return_sim=False):
#         cat_tensor = torch.cat([input1, input2, torch.abs(input1-input2)], dim=1)

        input1 = F.normalize(input1, dim=1)
        input1 = input1.unsqueeze(1)
        input2 = F.normalize(input2, dim=1)
        input2 = input2.unsqueeze(2)
        similarity = torch.bmm(input1, input2).squeeze().squeeze()
        input1 = input1.squeeze()
        input2 = input2.squeeze()
        
        cat_tensor = torch.cat([input1, input2, torch.abs(input1-input2)], dim=1)
#         print(cat_tensor.shape)
        out = self.fc(cat_tensor).squeeze()
        out = 1 - torch.sigmoid(out)
        label = (label - label.mean()) * 100
        label = torch.sigmoid(label)
        loss = -label * torch.log(out) - (1 - label)*torch.log(1-out)
        
#         out = 1 - torch.sigmoid(out)
#         loss = self.MSEloss(out, label)
#         loss = label * torch.log(out) + (1 - label)*torch.log(1-out)
        loss = torch.mean(loss)
        if return_sim:
            return loss, similarity
        return loss 
    
class wiseMSE(nn.Module):
    def __init__(self):
        super(wiseMSE, self).__init__()
    def forward(self, input1, input2, label, return_sim=False):
        input1 = F.normalize(input1, dim=1)
        input1 = input1.unsqueeze(1)
        input2 = F.normalize(input2, dim=1)
        input2 = input2.unsqueeze(2)
        similarity = torch.bmm(input1, input2).squeeze().squeeze()
#         similarity = (similarity + 1) / 2
        loss = label*(similarity-1)**2 + (1-label)*similarity**2
        loss = torch.mean(loss)
        if return_sim:
            return loss, similarity
        return loss

class WhiteMSE(nn.Module):
    def __init__(self):
        super(WhiteMSE, self).__init__()
        self.MSEloss = nn.MSELoss()
    def forward(self, input1, input2, label, return_sim=False):
        input1 = F.normalize(input1, dim=1)
        input1 = input1.unsqueeze(1)
        input2 = F.normalize(input2, dim=1)
        input2 = input2.unsqueeze(2)
        similarity = torch.bmm(input1, input2).squeeze().squeeze()
#         similarity = (similarity + 1) / 2
        if return_sim:
            return self.MSEloss(similarity, label), similarity
        
        return self.MSEloss(similarity, label)


class SpearmanCorrelationLoss(nn.Module):
    def __init__(self, temp=0.5):
        super(SpearmanCorrelationLoss, self).__init__() 
        self.temp = temp
    def forward(self, input1, input2, label, return_sim=False):
        similarity = cosine(input1, input2)
        similarity_sm = F.softmax(similarity / self.temp, dim=0)
#         similarity_sm = torch.sigmoid(similarity / self.temp)
        similarity_sm = similarity_sm -  torch.mean(similarity_sm)
        label = label - torch.mean(label)

        t_m1 = torch.sqrt(torch.sum(similarity_sm ** 2))#.detach() + 0.00001
        t_m2 = torch.sqrt(torch.sum(label ** 2))#.detach()+ 0.00001

        correlation = torch.sum(similarity_sm*label) / (t_m1 * t_m2 + 0.00001)
        if return_sim:
            return -correlation, similarity
        else:
            return -correlation   



if __name__ == '__main__':
    a = torch.rand(4,3)
    b = torch.rand(4,3)
    c = torch.rand(4)

    ans = loss(a,b, c)

    print(ans)
