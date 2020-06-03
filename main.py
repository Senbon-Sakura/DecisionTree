import func
import numpy as np
import PlotTree as pt
import matplotlib.pyplot as plt

# classDict为以后决策数节点的字典（划分依据）
dataArr, classDict = func.createDataSet('西瓜数据集3.0a.xlsx')
numSamp = dataArr.shape[0]
for i in range(numSamp):
    if dataArr[i,-1] == '否':
        dataArr[i, -1] = float(0.0)
    else:
        dataArr[i, -1] = float(1.0)
#plt.figure()
#c = np.array(dataArr[:,2], dtype='float')
#plt.scatter(list(map(float,dataArr[:,0])), list(map(float, dataArr[:,1])), s=c*60+40, c=c*40+40)
#plt.savefig("西瓜数据集3.0a分布图.jpg")
#plt.show()
#exit(0)

myTree = func.DecisionTree(dataArr, classDict)
#myTree = func.DecisionTreeGainRatio(dataArr, classDict)
#myTree = func.DecisionTreeGainAndRatio(dataArr, classDict)
#myTree = func.DecisionTreeGiniIndex(dataArr, classDict)
print(myTree)
pt.create_plot(myTree)
# 计算Accuracy
featLabels = list(classDict.keys())
m = len(dataArr)
rightCount = 0
for testVec in dataArr:
    #testVec = np.array(["青绿", "蜷缩","浊响","清晰","凹陷","硬滑"])
    #print("\n" + str(testVec) + "是否为好瓜：")
    #print(testVec)
    result = func.classify(myTree, featLabels, testVec)
    if (testVec[-1] == result):
        rightCount += 1
print("Accuracy =", (rightCount/m))

#plt.figure()
#c = np.array(dataArr[:,2], dtype='float')
#x = list(map(float,dataArr[:,0]))
#y = list(map(float, dataArr[:,1]))
#plt.scatter(x, y, s=c*60+40, c=c*40+40)
##plt.savefig("西瓜数据集3.0a分布图.jpg")
#plt.show()



'''
testDataArr, classDict = func.createDataSet('西瓜数据集测试集.xlsx')
featLabels = list(classDict.keys())
m = len(testDataArr)
rightCount = 0
for testVec in testDataArr:
    #testVec = np.array(["青绿", "蜷缩","浊响","清晰","凹陷","硬滑"])
    #print("\n" + str(testVec) + "是否为好瓜：")
    #print(testVec)
    result = func.classify(myTree, featLabels, testVec)
    if (testVec[-1] == result):
        rightCount += 1
print("Accuracy =", (rightCount/m))
'''





