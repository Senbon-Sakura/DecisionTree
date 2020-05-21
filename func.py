import xlrd
import numpy as np
import matplotlib.pyplot as plt

# 从excel表格中获取西瓜数据集，为了在后续计算信息熵过程中方便，data和label放到一个数组中
def createDataSet(filename):
    workbook = xlrd.open_workbook(filename)
    name = workbook.sheet_names()
    print("workbook name: ", name)
    worksheet = workbook.sheet_by_index(0)
    print("worksheet is: ", worksheet)

    nrows = worksheet.nrows
    print("rows number is: ", nrows)
    ncols = worksheet.ncols
    print("cols number is: ", ncols)

    dataArr = []
    className = np.array(worksheet.row_values(0))[1:-1]
    for i in range(1, nrows):                           # 第一行为表头，去除
        rowVals = np.array(worksheet.row_values(i))
        dataArr.append(rowVals[1:])                     # 第一列为样本序号，去除
    #print(dataArr)
    dataArr = np.array(dataArr)
    featDict = {}
    for i in range(len(className)):
        featDict[className[i]] = np.unique(dataArr[:, i])

    return dataArr, featDict

# 计算信息熵
def InfoEntropy(dataArr):
    labelCounts = {}
    numEntries = len(dataArr)
    for featVec in dataArr:
        currentLabel =featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    infoEnt = 0
    for key in labelCounts.keys():
        prob = labelCounts[key]/numEntries
        infoEnt -= prob*np.log2(prob)
    return infoEnt

# 根据第i个特征进行划分，返回划分的字典类型
def splitDataSet(dataArr, featInd):
    numOfFeature = len(dataArr[0])-1                                # 最后一列为label，不是feature
    featSplit = {}                                                  # 字典用于存feture对应不同值所对应的子样本
    for featVec in dataArr:                                         # 遍历每一个样本
        if featVec[featInd] not in featSplit.keys():                # 如果样本的该特征值存在，则将其增加到字典的key对应的值下面，否则需要先建立一个新的数组
            featSplit[featVec[featInd]] = []
        tmp = np.hstack((featVec[:featInd], featVec[featInd+1:]))   # 将划分特征值从数据集中去除
        featSplit[featVec[featInd]].append(tmp)
    return featSplit

# 计算C.54决策树不同特征值增益率对应的分母IV(a)
def calcIVa(dataArr):
    numOfFeat = dataArr.shape[1]-1                      # 计算dataArr中Feature的个数，最后一个为label需要减去
    numOfSamp = dataArr.shape[0]                        # 计算dataArr中样本个数
    IVa = []
    for i in range(numOfFeat):
        featValCount = {}
        for featVal in dataArr[:,i]:
            if featVal not in featValCount.keys():
                featValCount[featVal] = 0
            featValCount[featVal] += 1
        featValCountArr = np.array(list(featValCount.values())) / numOfSamp
        IVa.append(np.sum(-1*featValCountArr*np.log2(featValCountArr)))
    return IVa

# 选择最优的feature进行划分，使结果的信息增益最高
# 返回最优的信息增益、最优的划分特征序号以及划分结果（字典）
def chooseBestFeatureToSplit(dataArr, classDict):
    numOfFeature = len(dataArr[0])-1                    # 计算dataArr中Feature的个数，最后一个为label需要减去
    numOfSample = len(dataArr)                          # 计算样本总数，用于计算信息熵
    originInfoEnt = InfoEntropy(dataArr)                # 计算划分之前的信息熵
    bestInfoEnt = originInfoEnt                         # 保存最佳的信息熵，越小越好
    bestFeatIndex = 0                                   # 保存最好划分的feature序号
    bestfeatSpitDict = {}                               # 保存最好划分的划分结果
    for featInd in range(numOfFeature):                 # 遍历dataArr中的每一个Feature
        featSplit = splitDataSet(dataArr, featInd)      # 根据当前featInd划分dataArr，返回划分的字典结果
        infoEnt = 0
        for key in featSplit.keys():                    # 计算信息熵
            prob = len(featSplit[key]) / numOfSample
            infoEnt += prob*InfoEntropy(featSplit[key])
        if infoEnt < bestInfoEnt:                       # 如果新划分的信息熵小于最有信息熵，则将最优结果更新为此次划分结果
            bestInfoEnt = infoEnt
            bestFeatIndex = featInd
            bestfeatSpitDict = featSplit
    infoEntGain = originInfoEnt - bestInfoEnt           # 计算最大的信息增益
    keyList = list(classDict.keys())
    bestClassName = keyList[bestFeatIndex]              # 返回当作当前数节点的值
    classDict.pop(bestClassName)
    return infoEntGain, bestFeatIndex, bestfeatSpitDict, bestClassName, classDict


# 选择最优的feature进行划分，使结果的信息增益率最高
# 返回最优的信息增益率、最优的划分特征序号以及划分结果（字典）
def chooseBestFeatureToSplitGainRatio(dataArr, IVa):
    numOfFeature = len(dataArr[0])-1                    # 计算dataArr中Feature的个数，最后一个为label需要减去
    numOfSample = len(dataArr)                          # 计算样本总数，用于计算信息熵
    originInfoEnt = InfoEntropy(dataArr)                # 计算划分之前的信息熵
    bestGainRatio = np.inf                              # 保存最佳的信息熵，越小越好
    bestFeatIndex = 0                                   # 保存最好划分的feature序号
    bestfeatSpitDict = {}                               # 保存最好划分的划分结果
    for featInd in range(numOfFeature):                 # 遍历dataArr中的每一个Feature
        featSplit = splitDataSet(dataArr, featInd)      # 根据当前featInd划分dataArr，返回划分的字典结果
        infoEnt = 0
        for key in featSplit.keys():                    # 计算信息熵
            prob = len(featSplit[key]) / numOfSample
            infoEnt += prob*InfoEntropy(featSplit[key])
        gainRatio = (originInfoEnt - infoEnt)/IVa[featInd]
        if gainRatio < bestGainRatio:                   # 如果新划分的信息熵小于最有信息熵，则将最优结果更新为此次划分结果
            bestGainRatio = gainRatio
            bestFeatIndex = featInd
            bestfeatSpitDict = featSplit
    IVa = np.hstack((IVa[:featInd], IVa[featInd+1:]))
    return bestGainRatio, bestFeatIndex, bestfeatSpitDict, IVa


# 由信息增益生成决策树
def DecisionTree(dataArr, classDict):
    myTree = {}
    # (1) D 中样本全属于同一类别 C,将 node 标记为 C 类叶结点
    if len(np.unique(dataArr[:,-1])) == 1:
        return dataArr[0,-1]
    # (2) A=空集 OR D 中样本在 A 上取值相同
    elif len(classDict) == 0 or len(np.unique(dataArr[:, :-1], axis=0)) == 1:
        # 将 node 标记为叶结点，其类别标记为 D 中样本数最多的类;
        labelCount = {}
        for label in dataArr[:,-1]:
            if label not in labelCount.keys():
                labelCount[label] = 0
            labelCount[label] += 1
        return max(labelCount, key=labelCount.get)
    # (3) 否则继续进行最优划分，并返回划分的子树
    else:
        # 计算父节点最多的类，当Dv为空时使用
        freqCount = {}
        for sampVec in dataArr:
            if sampVec[-1] not in freqCount:
                freqCount[sampVec[-1]] = 0
            freqCount[sampVec[-1]]  += 1
        maxLab = max(freqCount, key=freqCount.get)
        # 直接传classDict会相当于引用传参，从而导致程序错误，此处需要进一步斟酌
        bestEntGain, bestFeatIndex, bestfeatSplitDict, bestClassName, classDictReturn = chooseBestFeatureToSplit(dataArr, classDict.copy())
        myTree = {bestClassName:{}}
        for key in bestfeatSplitDict.keys():
            myTree[bestClassName][key] = DecisionTree(np.array(bestfeatSplitDict[key]), classDictReturn.copy())
        for featVal in classDict[bestClassName]:
            if featVal not in bestfeatSplitDict.keys():
                myTree[bestClassName][featVal] = maxLab
        return myTree

# 由增益率生成决策树
def DecisionTreeGainRatio(dataArr, IVa):
    myTree = {}
    # (1) D 中样本全属于同一类别 C,将 node 标记为 C 类叶结点
    if len(np.unique(dataArr[:,-1])) == 1:
        return dataArr[0,-1]
    # (2) A=空集 OR D 中样本在 A 上取值相同
    elif dataArr.shape[1] == 1 or len(np.unique(dataArr[:, :-1], axis=0)) == 1:
        # 将 node 标记为叶结点，其类别标记为 D 中样本数最多的类;
        labelCount = {}
        for label in dataArr[:,-1]:
            if label not in labelCount.keys():
                labelCount[label] = 0
            labelCount[label] += 1
        return max(labelCount, key=labelCount.get)
    # (3) 否则继续进行最优划分，并返回划分的子树
    else:
        bestGainRatio, bestFeatIndex, bestfeatSplitDict, IVa = chooseBestFeatureToSplitGainRatio(dataArr, IVa)
        for key in bestfeatSplitDict.keys():
            myTree[key] = DecisionTreeGainRatio(np.array(bestfeatSplitDict[key]), IVa)
        return myTree

'''
def classify(inputTree, testVec):
    #if len(inputTree) == 1:
    if isinstance(inputTree, dict):
        for key in inputTree.keys():
            for i in range(len(testVec)):
                if testVec[i] == key:
                    classify(inputTree[key], np.hstack((testVec[:i], testVec[i+1:])))
    else:
        print(inputTree)
'''

def classify(inputTree, featLabels, testVec):
    keyStr = []
    for key in inputTree.keys():
        keyStr.append(key)
    firstStr = keyStr[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = ""
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

