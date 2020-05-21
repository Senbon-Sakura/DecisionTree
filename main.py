import func
import numpy as np
import PlotTree as pt

# classDict为以后决策数节点的字典（划分依据）
dataArr, classDict = func.createDataSet('西瓜数据集2.0.xlsx')
print(classDict)
IVa = func.calcIVa(dataArr)
myTree = func.DecisionTree(dataArr, classDict)
#myTree = func.DecisionTreeGainRatio(dataArr, IVa)
print(myTree)
pt.create_plot(myTree)

#testDataArr = func.createDataSet('西瓜数据集测试集.xlsx')
#for testVec in testDataArr:
#    #testVec = np.array(["青绿", "蜷缩","浊响","清晰","凹陷","硬滑"])
#    #print("\n" + str(testVec) + "是否为好瓜：")
#    print(testVec)
#    func.classify(myTree, testVec)





