import torch
from sklearn.metrics import roc_auc_score


class Config():
    def __init__(self):
        # ==== cuda environments ======= #
        self.gpu = '0,1,2,3'
        self.cuda = 'cuda'

        # ========= dataset ============ #
        self.dataPath = './dataset/MovieLens1M/'
        self.ratingBinThreshold = 3
        self.maxSequenceLen = 10
        self.splitRatio = 0.8
        self.batchSize = 256
        self.splitMethod = 'behavior'  # 'user' 分割数据集时按照用户分还是按照每个用户的行为序列分, 按 'behavior' 更加普遍

        # ========== model ============= #
        # embedding layer info:
        # embeddingGroups[embeddingGroupName] = (vocabulary size, embedding size)
        self.embeddingGroups = {'MovieId': (3953, 16), 'Genre': (19, 8)}
        # 不使用户信息
        # 如果使用: {'Gender': (2, 8), 'Age': (7, 8), 'Occupation': (21, 8), 'MovieId': (3953, 16), 'Genre': (19, 8)}
        # 1M {'MovieId': (3953, 16), 'Genre': (19, 8)}
        # 20M {'MovieId': (27279, 16), 'Genre': (21, 8)}
        self.MLPInfo = [48, 200, 80]  # 不使用用户、电影类别信息， 如果使用：[72, 200, 80]
        self.AttMLPInfo = [72, 36]  # 不使用用户、电影类别信息， 如果使用：[72, 36]
        self.isUseBN = True
        self.l2RegEmbedding = 0
        self.dropoutRate = 0.0

        # !!!这是个不合理的参数初始化方式，std太小会使得参数初始值都近乎为0，导致不管模型输入是什么，输出几乎一样(甚至直接一样)
        self.initStd = 0.0001

        # ============ train =========== #
        self.epoch = 30

        self.learningRate = 0.1

        # 0.2+ < 0.51 | 0.1 < 0.63 | 0.05 < 0.6 | 0.01 < 0.58 | 0.005 < 0.56 | 0.001 < 0.55
        # 0.1 + BM < 0.63 | 0.1 + BM + dropout < 0.62
        # + l2 1e-6 < 0.615 | 1e-5 : < 0.62 | 1e-4: < 0.62 | 1e-3: < 0.63 | 1e-2: < 0.61 | 1e-1: < 0.6 | 1: < 0.6
        # best lr 0.1, batchSize 100, BM/dropout/l2 arbitrary
        self.optimizer = torch.optim.SGD

        # 0.05+ < 0.51 | 0.02 0.61 | 0.01 0.65-0.707 | 0.007 < 0.7 | 0.005 < 0.7 | 0.001 0.65?
        # self.optimizer = torch.optim.Adagrad

        # 0.02+ < 0.51 | 0.01 0.72 | 0.005 0.72 | 0.001 0.72
        # self.optimizer = torch.optim.Adam

        self.decay = 0.1
        self.decayStep = 30
        self.lrSchedule = torch.optim.lr_scheduler.ExponentialLR

        self.lossFunc = torch.nn.functional.binary_cross_entropy

        # ============ metrics =========== #
        self.metricFunc = roc_auc_score
