import random
import pandas as pd

class ConstructData(object):

    """
    succeNum: 连续特征个数
    discretNum: 离散特征个数
    multiNum:  多值特征个数
    sampleNum: 样本个数
    """
    def __init__(self, succeNum=1, discretNum=1, multiNum=1, sampleNum=1):
        self.succeNum   = succeNum
        self.discretNum = discretNum
        self.multiNum   = multiNum
        self.sampleNum  = sampleNum

    def generateData(self):
        label = [random.randint(0, 1) for i in range(self.sampleNum)]
        succeMap, discretMap, multiMap = {}, {}, {}

        for i in range(self.succeNum):
            value = [round(random.uniform(0, 10), 2) for _ in range(self.sampleNum)]
            succeMap['s'+str(i)] = list(value)

        for i in range(self.discretNum):
            value = [random.randint(0, 10) for _ in range(self.sampleNum)]
            discretMap['d'+str(i)] = list(value)

        for i in range(self.multiNum):
            value = [self.generateMultiFeature() for _ in range(self.sampleNum)]
            multiMap['m'+str(i)] = value

        succeFeature   = pd.DataFrame(succeMap)
        discretFeature = pd.DataFrame(discretMap)
        multiFeature   = pd.DataFrame(multiMap)
        succeFeature.insert(loc=0, column='label', value=label)
        discretFeature.insert(loc=0, column='label', value=label)
        multiFeature.insert(loc=0, column='label', value=label)

        succeFeature.to_csv("../data/DenseFeature.csv", index=False)
        discretFeature.to_csv("../data/SparseFeature.csv", index=False)
        multiFeature.to_csv("../data/VarlenFeature.csv", index=False)

        sparse_cols = ['d'+str(i) for i in range(self.discretNum)]
        varlen_cols = ['m'+str(i) for i in range(self.multiNum)]
        df = pd.concat([succeFeature, discretFeature[sparse_cols], multiFeature[varlen_cols]], axis=1)
        train = df.sample(frac=0.9, replace=False, random_state=0, axis=0)
        valid = df[~df.index.isin(train.index)]

        train.to_csv("../data/train.csv", index=False)
        valid.to_csv("../data/valid.csv", index=False)


    def generateMultiFeature(self):
        ans = []
        for i in range(1, 6):
            ans.append(str(i)+':'+str(round(random.uniform(0.01, 0.99), 2)))

        return ';'.join(ans)