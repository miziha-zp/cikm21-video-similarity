环境依赖见`requirement.py`
该目录负责产生zp的三个模型,分别是:

1. roberta+nextvlad -- tagloss

2. roberta + nextvlad --tagloss + video-text match loss + video contrastive loss

3. bert + nextvlad --tagloss + video-text match loss + video contrastive loss

如何训练：

通过一下命令来训练：
训练1.
```bash
python train1.py --bert-dir "hfl/chinese-roberta-wwm-ext-large"
python inference1.py --inference-name test
python inference1.py --inference-name valid
```
训练2.

```bash
python train2.py --bert-dir "hfl/chinese-roberta-wwm-ext-large"
python inference2.py --inference-name test
python inference2.py --inference-name valid
```
训练3.

```bash
python train3.py --bert-dir '/data03/yrqUni/Workspace/QQB/Data/chinese_L-12_H-768_A-12'
python inference3.py --inference-name test
python inference3.py --inference-name valid
```
