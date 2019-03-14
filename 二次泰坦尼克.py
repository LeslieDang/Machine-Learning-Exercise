#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author：Leslie Dang

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

import time #
import pandas as pd # 数据分析
import numpy as np  # 科学计算
from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion() #开启interactive mode

def pause():
    plt.pause(0.3)

# 1、数据初步分析
# 每个/多个 属性和最后的Survived之间有着什么样的关系
data_train = pd.read_csv("train.csv")
print(data_train.head())
print(data_train.describe())

### 乘客各属性分布
fig = plt.figure(figsize = (15,10))
fig.set(alpha=0.2) # 设定图表颜色alpha参数

plt.subplot2grid((2,3),(0,0)) # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')# 柱状图
plt.title("获救情况 (1为获救)") # 标题
plt.ylabel("人数")

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("人数")
plt.title("乘客等级分布")

plt.subplot2grid((2,3),(0,2))

data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("年龄")# plots an axis lable
plt.ylabel("密度")
plt.title("各等级的乘客年龄分布")
plt.legend(('头等舱', '2等舱','3等舱'),loc='best') # sets our legend for our graph.

plt.subplot2grid((2,3),(1,0), colspan=2)

# data_train.Age[data_train.Survived == 1].counts().plot(kind='bar',color = 'r')
# data_train.Age[data_train.Survived == 0].counts().plot(kind='bar',color = 'g')
data_train.Age[data_train.Survived == 1].groupby(data_train.Age).count().plot(kind="line",color = 'g')
data_train.Age[data_train.Survived == 0].groupby(data_train.Age).count().plot(kind="line",color = 'r',alpha =0.2)
data_train.Age.groupby(data_train.Age).count().plot(kind="line",color = 'b',alpha =0.1)
plt.xlabel("年龄") # 设定x坐标名称
plt.ylabel("人数")
# plt.grid(b=True, which='major', axis='y')
plt.title("按年龄看获救分布 (1为获救)")
plt.legend(('获救', '未获救','总体分布'),loc='best')

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("各登船口岸上船人数")
plt.ylabel("人数")
# plt.savefig('01乘客各属性分布')
# plt.show()
pause()
plt.close()

# 看看各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.2) # 设定图表颜色alpha参数
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")
# plt.show()
pause()
plt.close()

# 看看各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.2) # 设定图表颜色alpha参数
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"按性别看获救情况")
plt.xlabel(u"性别")
plt.ylabel(u"人数")
# plt.show()
pause()
plt.close()

# 然后我们再来看看各种舱级别情况下各性别的获救情况
fig=plt.figure(figsize = (15,8))
fig.set(alpha=0.65) # 设置图像透明度，无所谓
plt.title(u"根据舱等级和性别的获救情况")
ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
ax1.legend([u"女性/高级舱"], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/高级舱"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/低级舱"], loc='best')
# plt.show()
pause()
plt.close()

fig = plt.figure()
fig.set(alpha=0.2) # 设定图表颜色alpha参数
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各登录港口乘客的获救情况")
plt.xlabel(u"登录港口")
plt.ylabel(u"人数")
# plt.show()
pause()
plt.close()


g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print (df)

#ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴把
#cabin只有204个乘客有值，我们先看看它的一个分布
data_train.Cabin.value_counts()

fig = plt.figure()
fig.set(alpha=0.2) # 设定图表颜色alpha参数
Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"按Cabin有无看获救情况")
plt.xlabel(u"Cabin有无")
plt.ylabel(u"人数")
# plt.show()
pause()
plt.close()

##2、 简单数据预处理
from sklearn.ensemble import RandomForestRegressor

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # y即目标年龄
    y = known_age[:, 0]
    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

# 对类目型的特征因子化-get_dummies
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
print(df.head())

# 用scikit-learn里面的preprocessing模块对数值型数据做一个scaling
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()

age_scale_param = scaler.fit(df[['Age']])
# preprocessing.StandardScaler()参数不接受一维数组怎么办:
# age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))

df['Age_scaled'] = scaler.fit_transform(df[['Age']], age_scale_param)
fare_scale_param = scaler.fit([df['Fare']])
df['Fare_scaled'] = scaler.fit_transform(df[['Fare']], fare_scale_param)
print(df.head())

##3、 逻辑回归建模
# from sklearn import linear_model
# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values
# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]

print('y = ',y)
print('X = ',X)
print('y type = ',type(y))
print('X type = ',type(X))

# 得到训练数据矩阵（dataMat、labelMat）
labelMat = np.mat(y)
labelMat[labelMat == 0.0] = -1.0
print('labelMat = ',labelMat)
print('labelMat type = ',type(labelMat))
dataMat = np.mat(X)
print('dataMat = ',dataMat)
print('dataMat type = ',type(dataMat))



# 测试数据做同样的处理
data_test = pd.read_csv("test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age']], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Fare']], fare_scale_param)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
testMat = np.mat(test.values)
print('testMat = ',testMat)
print('testMat type = ',type(testMat))

# 开始采用AdaBoost模型预测
import AdaBoost
weakClassArr, aggClassEst = AdaBoost.adaBoostTrainDS(dataMat, labelMat, 30)

prediction_AdaBoost = AdaBoost.adaClassify(testMat, weakClassArr)
# 此时prediction_AdaBoost为numpy下的matrix格式，shape =  (418, 1)

# print('prediction_AdaBoost type = ',type(prediction_AdaBoost))
# print('prediction_AdaBoost[0:10] = ',prediction_AdaBoost[0:10])
# print('prediction_AdaBoost.shape = ',prediction_AdaBoost.shape)
prediction_AdaBoost[prediction_AdaBoost == -1.0] = 0
# 匹配赋值需要为matrix格式

prediction_AdaBoost = prediction_AdaBoost.getA().flatten()
# getA()使prediction_AdaBoost为numpy下的array格式，shape =  (418, 1)
# flatten()使prediction_AdaBoost降低为一维，shape =  (418, )

# print('prediction_AdaBoost type = ',type(prediction_AdaBoost))
# print('prediction_AdaBoost[0:10] = ',prediction_AdaBoost[0:10])
# print('prediction_AdaBoost.shape = ',prediction_AdaBoost.shape)

# print("data_test['PassengerId'].values.shape = ",data_test['PassengerId'].values.shape)
# print("data_test['PassengerId'].values type = ",type(data_test['PassengerId'].values))
# print("data_test['PassengerId'].values[0:10] = ",data_test['PassengerId'].values[0:10])

# data_test['PassengerId'].values.shape =  (418,)
# data_test['PassengerId'].values type =  <class 'numpy.ndarray'>

result_AdaBoost = pd.DataFrame({'PassengerId':data_test['PassengerId'].values,
                                'Survived':prediction_AdaBoost.astype(np.int32)})

result_AdaBoost.to_csv("result_AdaBoost-30.csv", index=False)