import func
import numpy as np

dataArr = func.createDataSet('西瓜数据集2.0.xlsx')
IVa = func.calcIVa(dataArr)
#myTree = func.DecisionTree(dataArr)
myTree = func.DecisionTreeGainRatio(dataArr, IVa)
print(myTree)

testDataArr = func.createDataSet('西瓜数据集测试集.xlsx')
for testVec in testDataArr:
    #testVec = np.array(["青绿", "蜷缩","浊响","清晰","凹陷","硬滑"])
    #print("\n" + str(testVec) + "是否为好瓜：")
    print(testVec)
    func.classify(myTree, testVec)





