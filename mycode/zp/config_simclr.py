import argparse

EMBSZ = 512
parser = argparse.ArgumentParser(description="QQ Browser video embedding challenge")

parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--multi-label-file', type=str, default='/data03/yrqUni/Workspace/QQB/Data/tag_list.txt', help='supervised tag list')

# ========================= Dataset Configs ==========================
parser.add_argument('--train-record-pattern', type=str, default='/data03/yrqUni/Workspace/QQB/Data/pointwise/*.tfrecords')
parser.add_argument('--val-record-pattern', type=str, default='/data03/yrqUni/Workspace/QQB/Data/pairwise/pairwise.tfrecords')
parser.add_argument('--annotation-file', type=str, default='/data03/yrqUni/Workspace/QQB/Data/pairwise/label.tsv')
parser.add_argument('--validfile', type=str, default='/data03/yrqUni/Workspace/QQB/Data/pairwise/pairwise.tfrecords')
parser.add_argument('--test-a-file', type=str, default='/data03/yrqUni/Workspace/QQB/Data/test_a/test_a.tfrecords')
parser.add_argument('--test-b-file', type=str, default='/data03/yrqUni/Workspace/QQB/Data/test_b/test_b.tfrecords')
parser.add_argument('--output-json', type=str, default='result.json')
parser.add_argument('--output-zip', type=str, default='result.zip')
parser.add_argument('--inference-name', type=str, default='test')

parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--val-batch-size', default=192*2, type=int)
parser.add_argument('--test-batch-size', default=192*2, type=int)

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', default=200, type=int, help='print frequency')
parser.add_argument('--eval-freq', default=500, type=int, help='evaluation step frequency')

# ======================== SavedModel Configs =========================
parser.add_argument('--resume-training', default=0, type=int, help='resume training from checkpoints')
parser.add_argument('--savedmodel-path', type=str, default='save/set in train_simclr.py')
parser.add_argument('--ckpt-file', type=str, default='save/v1/ckpt-40000')
parser.add_argument('--max-to-keep', default=100, type=int, help='the number of checkpoints to keep')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--total-steps', default=400000, type=int)
parser.add_argument('--warmup-steps', default=100, type=int)
parser.add_argument('--minimum-lr', default=0., type=float, help='minimum learning rate')
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')

# ==================== Vision Modal Configs =======================
parser.add_argument('--frame-embedding-size', type=int, default=1536)
parser.add_argument('--max-frames', type=int, default=32)
parser.add_argument('--vlad-cluster-size', type=int, default=64)
parser.add_argument('--vlad-groups', type=int, default=8)
parser.add_argument('--vlad-hidden-size', type=int, default=EMBSZ, help='nextvlad output size using dense')
parser.add_argument('--se-ratio', type=int, default=16, help='reduction factor in se context gating')

# ========================== Title BERT =============================
# parser.add_argument('--bert-dir', type=str, default='/data03/yrqUni/Workspace/QQB/Data/chinese_L-12_H-768_A-12')
parser.add_argument('--bert-dir', type=str, default='hfl/chinese-roberta-wwm-ext-large')
parser.add_argument('--bert-seq-length', type=int, default=32)
parser.add_argument('--bert-lr', type=float, default=1e-5)
parser.add_argument('--bert-total-steps', type=int, default=40000)
parser.add_argument('--bert-warmup-steps', type=int, default=100)
parser.add_argument('--bert-fc-dim', type=int, default=EMBSZ)

# ====================== Fusion Configs ===========================
parser.add_argument('--hidden-size', type=int, default=EMBSZ, help='NO MORE THAN 256')