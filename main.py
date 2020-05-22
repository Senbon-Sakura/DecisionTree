import func
import numpy as np
import PlotTree as pt

# classDict为以后决策数节点的字典（划分依据）
dataArr, classDict = func.createDataSet('西瓜数据集2.0.xlsx')
print(classDict)
#myTree = func.DecisionTree(dataArr, classDict)
#myTree = func.DecisionTreeGainRatio(dataArr, classDict)
#myTree = func.DecisionTreeGainAndRatio(dataArr, classDict)
myTree = func.DecisionTreeGiniIndex(dataArr, classDict)
print(myTree)
pt.create_plot(myTree)

testDataArr, classDict = func.createDataSet('西瓜数据集测试集.xlsx')
featLabels = list(classDict.keys())
for testVec in testDataArr:
    #testVec = np.array(["青绿", "蜷缩","浊响","清晰","凹陷","硬滑"])
    #print("\n" + str(testVec) + "是否为好瓜：")
    print(testVec)
    result = func.classify(myTree, featLabels, testVec)
    print(result)





