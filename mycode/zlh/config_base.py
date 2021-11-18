import argparse
data_path = '/data03/yrqUni/Workspace/QQB/Data'
parser = argparse.ArgumentParser(description="QQ Browser video embedding challenge")

parser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--multi-label-file', type=str, default='/data03/yrqUni/Workspace/QQB/Data/tag_list.txt', help='supervised tag list')
parser.add_argument('--hidden-size', default=256, type=int)
parser.add_argument('--ratio', default=8, type=int)

# ========================= Dataset Configs ==========================
parser.add_argument('--train-record-pattern', type=str, default='/data03/yrqUni/Workspace/QQB/Data/pointwise/*.tfrecords')
parser.add_argument('--val-record-pattern', type=str, default='/data03/yrqUni/Workspace/QQB/Data/pairwise/pairwise.tfrecords')
parser.add_argument('--pair-record-pattern', type=str, default='/data03/yrqUni/Workspace/QQB/Data/pairwise/pairwise.tfrecords')
parser.add_argument('--validfile', type=str, default=data_path+'/pairwise/pairwise.tfrecords')
parser.add_argument('--pair-all-record-pattern-train', type=str, default='/data03/yrqUni/Workspace/QQB/Data/zlh_pair/train_datalabel_all.tfrecord')
parser.add_argument('--pair-all-record-pattern-valid', type=str, default='/data03/yrqUni/Workspace/QQB/Data/zlh_pair/valid_datalabel_all.tfrecord')
parser.add_argument('--pair-record-pattern-root', type=str, default='/data03/yrqUni/Workspace/QQB/Data/zlh_pair/')
parser.add_argument('--annotation-file', type=str, default='/data03/yrqUni/Workspace/QQB/Data/pairwise/label.tsv')
parser.add_argument('--test-a-file', type=str, default='/data03/yrqUni/Workspace/QQB/Data/test_a/test_a.tfrecords')
parser.add_argument('--test-b-file', type=str, default='/data03/yrqUni/Workspace/QQB/Data/test_b/test_b.tfrecords')
parser.add_argument('--output-json', type=str, default='result.json')
parser.add_argument('--output-zip', type=str, default='result.zip')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--val-batch-size', default=512, type=int)
parser.add_argument('--test-batch-size', default=512, type=int)

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
parser.add_argument('--eval-freq', default=500, type=int, help='evaluation step frequency')

# ======================== SavedModel Configs =========================
parser.add_argument('--resume-training', default=0, type=int, help='resume training from checkpoints')
parser.add_argument('--savedmodel-path', type=str, default='save/v1_sim')
parser.add_argument('--ckpt-file', type=str, default='save/v1_sim/ckpt-13000')
parser.add_argument('--max-to-keep', default=100, type=int, help='the number of checkpoints to keep')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--total-steps', default=40000, type=int)
parser.add_argument('--warmup-steps', default=100, type=int)
parser.add_argument('--minimum-lr', default=1e-5, type=float, help='minimum learning rate')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')

# ==================== Vision Modal Configs (trans)=======================
parser.add_argument('--frame-embedding-size', type=int, default=1536, help='Embedding size for each frame')
parser.add_argument('--max-frames', type=int, default=32, help='maxlen of frames')
parser.add_argument('--vis-trans-num-heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--vis-num-trans-layers', type=int, default=6, help='Number of trans layer')
parser.add_argument('--vis-trans-dense-dim',  type=int, default=6144, help='Dense dim Transformer')
parser.add_argument('--vis-trans-dropout', type=float, default=0.1)
parser.add_argument('--vlad-cluster-size', type=int, default=64)
parser.add_argument('--vlad-groups', type=int, default=8)
parser.add_argument('--vlad-hidden-size', type=int, default=1024, help='nextvlad output size using dense')

# ==================== All Modal Configs (trans)=======================
parser.add_argument('--all-embedding-size', type=int, default=768, help='Embedding size for each fea')
parser.add_argument('--all-maxlen', type=int, default=96, help='maxlen of fea')
parser.add_argument('--all-trans-num-heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--all-num-trans-layers', type=int, default=4, help='Number of trans layer')
parser.add_argument('--all-trans-dense-dim',  type=int, default=3072, help='Dense dim Transformer')
parser.add_argument('--all-trans-dropout', type=float, default=0.1)

# ========================== Title&ASR BERT =============================
parser.add_argument('--bert-dir', type=str, default='/data03/yrqUni/Workspace/QQB/Data/chinese_L-12_H-768_A-12')
parser.add_argument('--bert-seq-length', type=int, default=32)
parser.add_argument('--bert-lr', type=float, default=3e-5)
parser.add_argument('--bert-total-steps', type=int, default=40000)
parser.add_argument('--bert-warmup-steps', type=int, default=200)
