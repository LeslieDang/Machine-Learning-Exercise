#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author：Leslie Dang

from numpy import *
import numpy as np
from scipy import *
from math import *
import matplotlib.pyplot as plt

# 获取数据集
def loadSimpData():
    datMat = matrix([[1.0, 2.1], [2.0, 1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

'''
	单层决策树生成函数（弱分类器）思路：
	在特征0、情况1、下通过阈值生成1、-1矩阵，在用这个矩阵和classLabels矩阵一一对比计算加权错误率weightedError
	在特征0、情况2、下通过阈值生成1、-1矩阵，在用这个矩阵和classLabels矩阵一一对比计算加权错误率weightedError
	在特征1、情况1、下通过阈值生成1、-1矩阵，在用这个矩阵和classLabels矩阵一一对比计算加权错误率weightedError
	在特征1、情况2、下通过阈值生成1、-1矩阵，在用这个矩阵和classLabels矩阵一一对比计算加权错误率weightedError
	保留加权错误率weightedError最小的字典bestStump： bestStump, minError, bestClasEst。
	
# 构建单层分类器
# 单层分类器是基于最小加权分类错误率的树桩

    伪代码
    将最小错误率minError设为+∞
    对数据集中的每个特征(第一层特征)：
      对每个步长(第二层特征)：
          对每个不等号(第三层特征)：
              建立一颗单层决策树并利用加权数据集对它进行测试
              如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
    返回最佳单层决策树
'''


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    # ones函数设成shape(dataMatrix)[0]行，1列的矩阵。
    retArray = ones((shape(dataMatrix)[0], 1))
    # 'lt'为情况1、'gt'为情况2。
    # 举例：第dimen=0个特征矩阵为[[1],[2],[1.3],[1],[2]],阈值threshVal为1.3
    # 情况1下有：retArray为[[-1],[1],[-1],[-1],[1]]  在buildStump()函数下计算weightedError=0.2
    # 情况2下有：retArray为[[1],[-1],[1],[1],[-1]]   在buildStump()函数下计算weightedError=0.8
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0  # 情况1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0  # 情况2
    return retArray


def buildStump(dataArr, classLabels, D):
    # dataArr、classLabels变成矩阵，.T为转置
    dataMatrix = mat(dataArr);
    labelMat = mat(classLabels).T
    # m为行数 n为列数、numSteps用于特征所有可能的值的历遍
    # bestStump用于存储给定权值向量D时的所得到的最佳单层决策树信息
    # inf是无穷大的意思
    m, n = shape(dataMatrix)
    numSteps = 50.0;
    bestStump = {};
    bestClasEst = mat(zeros((m, 1)))
    minError = inf
    # 第一层循环 对数据集中的每一个特征 n为特征总数
    for i in range(n):
        # rangeMin、rangeMax第i个特征的最小值，最大值
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps # 移动步长，划分点的移动
        # 第二层循环 对每个步长
        for j in range(-1, int(numSteps) + 1):
            # 第三层循环 对两种选取情况
            for inequal in ['lt', 'gt']:
                # 计算阈值
                threshVal = rangeMin + float(j) * stepSize
                # stumpClassify(矩阵,特征,阈值,不等式)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 先假设所有的结果都是错的（标记为1）
                errArr = mat(ones((m, 1)))
                # 然后把预测结果正确的标记为0
                errArr[predictedVals == labelMat] = 0
                # 计算加权错误率\D是向量
                weightedError = D.T * errArr
                # print('split: dim %d, thresh %.2f, thresh inequal: %s, \
                # the weightederror is %.3f' % (i, threshVal, inequal, weightedError))
                # 将加权错误率最小的结果保存下来
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


# 输出结果试一试
# D=mat(ones((5,1))/5)
# datMat,classLabels=loadSimpData()
# print(buildStump(datMat,classLabels,D))


'''
这部分公式难以理解请看《统计学习方法》P138-P142(8.1.2节到-8.1.3节)
'''


# 参考https://www.cnblogs.com/zy230530/p/6909288.html
# 特征矩阵dataArr，类标签classLabels，循环次数。
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    # 弱分类器相关信息列表：存储特征、阈值、情况、分类器alpha
    weakClassArr = []
    m = shape(dataArr)[0]
    # 初始化权重向量的每一项值相等
    D = mat(ones((m, 1)) / m)
    # 累计估计值向量
    aggClassEst = mat(zeros((m, 1)))
    # 循环迭代次数
    for i in range(numIt):
        # 根据当前数据集，标签及权重建立最佳单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # 打印权重向量
        # print("D:", D.T)
        # 求单层决策树的系数alpha\max用法：error不能小于1e-16，小于就为1e-16
        alpha = float(0.5 * log((1.0 - error) / (max(error, 1e-16))))
        # 存储决策树的系数alpha到字典
        bestStump['alpha'] = alpha
        # 将该决策树存入列表
        weakClassArr.append(bestStump)
        # 打印决策树的预测结果
        # print("classEst:", classEst.T)
        # expon为列表
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        # print(expon)、、、、、、、、、、、、、、、、、、、、、、、
        # 更新权值向量，D为列表,python标准库里面也有exp，换成numpy的exp
        D = multiply(D, np.exp(expon))
        # print(D)
        D = D / D.sum()
        # 累加当前单层决策树的加权预测值,出现Cannot cast ufunc add output from dtype('float64') to dtype('int32')
        # 把 aggClassEst+=alpha*classEst改成 aggClassEst=aggClassEst+alpha*classEst
        aggClassEst = aggClassEst + alpha * classEst
        # print("aggClassEst", aggClassEst.T)
        # 求出分类错的样本个数
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        # 计算错误率
        errorRate = aggErrors.sum() / m
        print("total error:", errorRate, "\n")
        # 错误率为0.0退出循环
        if errorRate == 0.0: break
    # 返回弱分类器的组合列表
    return weakClassArr, aggClassEst


# 测试一下
# datMat,classLabels=loadSimpData()
# classifierArray=adaBoostTrainDS(datMat,classLabels,9)
# print(classifierArray)


# adaBoost分类函数
def adaClassify(datToClass, classifierArr):
    # datToClass转换为矩阵
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    # len(classifierArr)为字典数
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return sign(aggClassEst)


# 测试一下
# datArr,labelArr=loadSimpData()
# classifierArray, aggClassEst=adaBoostTrainDS(datArr,labelArr,30)
# print('classifierArray = ',classifierArray)
# A=adaClassify([[5,5],[0,0],[3,2],[2,4]],classifierArray)
# print(A)




# 自适应数据加载函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 绘制ROC
# 这部分代码参考https://blog.csdn.net/namelessml/article/details/52536514?locationNum=13
def plotROC(predStrengths, classLabels):

    # cur保留的是绘制光标的位置
    cur = (1.0, 1.0)
    # ySum则用于计算AUC的值
    ySum = 0.0
    # 通过数组过滤方式计算正例的数目，并赋给numPosClas，接着在x轴和y轴的0.0到1.0区间上绘点
    numPosClas = sum(array(classLabels) == 1.0)
    # 在y轴上的步长
    yStep = 1 / float(numPosClas)
    # 在x轴上的步长
    xStep = 1 / float(len(classLabels) - numPosClas)
    # 获取排好序的索引sortedIndicies，这些索引从小到大排序。需要从<1, 1>开始绘，一直到<0,0>
    sortedIndicies = predStrengths.argsort()
    print('sortedIndicies = ',sortedIndicies)
    print('sortedIndicies type = ', type(sortedIndicies))

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 在所有排序值上进行循环。这些值在一个NumPy数组或矩阵中进行排序，
    # python则需要一个表来进行迭代循环，因此需要调用tolist()方法
    for index in sortedIndicies.tolist()[0]:
        # 每得到一个标签为1.0的类，则要沿着y轴的方向下降一个步长，即降低真阳率
        if classLabels[index] == 1.0:
            delX = 0;
            delY = yStep
        # 对于每个其他的标签，则是x轴方向上倒退一个步长（假阴率方向），
        # 代码只关注1这个类别标签，采用1/0标签还是+1/-1标签就无所谓了
        else:
            delX = xStep;
            delY = 0
            # 所有高度的和ySum随着x轴的每次移动而渐次增加
            ySum += cur[1]
        # 一旦决定了在x轴还是y轴方向上进行移动，就可在当前点和新点之间画出一条线段
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='r')
        cur = (cur[0] - delX, cur[1] - delY)
    # 计算AUC需要对多个小矩形的面积进行累加，这些小矩形的宽度都是xStep，
    # 因此可对所有矩形的高度进行累加，然后再乘以xStep得到其总面积
    print("the Area Under the Curve is: ", ySum * xStep)
    ax.plot([0, 1], [0, 1], 'g--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()



# 测试一下
# 注意：horseColicTraining.txt中分类结果是0和1，不是-1和1
# dataMat, labelMat = loadDataSet("horseColicTraining.txt")
# labelMat = mat(labelMat)
# labelMat[labelMat == 0.0] = -1.0
# # print('labelMat = ',labelMat)
# weakClassArr, aggClassEst = adaBoostTrainDS(dataMat, labelMat, 50)

# 注意：horseColicTest.txt中分类结果是0和1，不是-1和1
# dataMat1, labelMat1 = loadDataSet("horseColicTest.txt")
# labelMat1 = mat(labelMat1).T
# labelMat1[labelMat1 == 0.0] = -1.0
#
# prediction2 = adaClassify(dataMat1, weakClassArr)
# print(prediction2)

# 计算测试错误率
# numTest = len(labelMat1)
# errArr = mat(ones((numTest,1)))
# errArr[prediction2 == labelMat1] = 0.0
# errorRate = errArr.sum()/numTest
# print('测试错误个数:%s,测试错误率：%f'%(errArr.sum(),errorRate))

# plotROC(aggClassEst.T, labelMat.T)
