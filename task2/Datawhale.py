#coding:utf-8
#导入warnings包，利用过滤器来实现忽略警告语句。
#载入各种数据科学以及可视化库
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# 载入数据
## 1) 载入训练集和测试集；
Train_data = pd.read_csv('./data/used_car_train_20200313.csv', sep=' ')
Test_data = pd.read_csv('./data/used_car_testA_20200313.csv', sep=' ')

print(Train_data)
print(Test_data)
# 2) 简略观察数据(head()+shape)
print(Train_data.head().append(Train_data.tail()))
#行，列
print(Train_data.shape)
Test_data.head().append(Test_data.tail())
Test_data.shape
# 总览数据概况
# describe种有每列的统计量，个数count、平均值mean、方差std、最小值min、中位数25% 50% 75% 、以及最大值 看这个信息主要是瞬间掌握数据的大概的范围以及每个值的异常值的判断，比如有的时候会发现999 9999 -1 等值这些其实都是nan的另外一种表达方式，有的时候需要注意下
# info 通过info来了解数据每列的type，有助于了解是否存在除了nan以外的特殊符号异常
## 1) 通过describe()来熟悉数据的相关统计量
print("总览数据概况")
print(Train_data.describe())
Test_data.describe()

# 2) 通过info()来熟悉数据类型
print(Train_data.info())
Test_data.info()

#判断数据缺失和异常
## 1) 查看每列的存在nan情况
print(Train_data.isnull().sum())
Test_data.isnull().sum()

# nan可视化
print("nan可视化")
missing = Train_data.isnull().sum()
missing = missing[missing > 0]
print(missing)
print("**************")
missing.sort_values(inplace=True)
missing.plot.bar()
plt.tight_layout()
plt.show()

#explain 通过以上两句可以很直观的了解哪些列存在 “nan”, 并可以把nan的个数打印，主要的目的在于 nan存在的个数是否真的很大，如果很小一般选择填充，如果使用lgb等树模型可以直接空缺，让树自己去优化，但如果nan存在的过多、可以考虑删掉
# 可视化看下缺省值
print("可视化看下缺省值")
msno.matrix(Train_data.sample(250))
plt.tight_layout()
plt.show()
msno.bar(Train_data.sample(1000))
plt.tight_layout()
plt.show()

# # 可视化看下缺省值
msno.matrix(Test_data.sample(250))
plt.tight_layout()
plt.show()

msno.bar(Test_data.sample(1000))
plt.tight_layout()
plt.show()

## 2) 查看异常值检测
print("查看异常值检测")
#通过info()来熟悉数据类型
print(Train_data.info())
print("可以发现除了notRepairedDamage 为object类型其他都为数字 这里我们把他的几个不同的值都进行显示就知道了")

print(Train_data['notRepairedDamage'].value_counts())
print("可以看出来‘ - ’也为空缺值，因为很多模型对nan有直接的处理，这里我们先不做处理，先替换成nan")
#对-缺省值进行替换nan
Train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
print(Train_data['notRepairedDamage'].value_counts())
Train_data.isnull().sum()

Test_data['notRepairedDamage'].value_counts()
Test_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
print("以下两个类别特征严重倾斜，一般不会对预测有什么帮助，故这边先删掉，当然你也可以继续挖掘，但是一般意义不大")
print(Train_data["seller"].value_counts())
print(Train_data["offerType"].value_counts())

del Train_data["seller"]
del Train_data["offerType"]
del Test_data["seller"]
del Test_data["offerType"]



#了解预测值的分布
print("了解预测值的分布")
print(Train_data['price'])

print(Train_data['price'].value_counts())

## 1) 总体分布概况（无界约翰逊分布等）
print("总体分布概况（无界约翰逊分布等）")
import scipy.stats as st
y = Train_data['price']
print("Johnson SU")
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.tight_layout()
plt.show()
print("Normal")
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.tight_layout()
plt.show()
print("Log Normal")
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
plt.tight_layout()
plt.show()
print("价格不服从正态分布，所以在进行回归之前，它必须进行转换。虽然对数变换做得很好，但最佳拟合是无界约翰逊分布")

## 2) 查看skewness and kurtosis
sns.distplot(Train_data['price'])
print("Skewness: %f" % Train_data['price'].skew())
print("Kurtosis: %f" % Train_data['price'].kurt())
plt.tight_layout()
plt.show()
print("******************")
print(Train_data.skew())

print(Train_data.kurt())


sns.distplot(Train_data.skew(),color='blue',axlabel ='Skewness')
plt.tight_layout()
plt.show()

sns.distplot(Train_data.kurt(),color='orange',axlabel ='Kurtness')
plt.tight_layout()
plt.show()

## 3) 查看预测值的具体频数
plt.hist(Train_data['price'], orientation = 'vertical',histtype = 'bar', color ='red')
plt.show()
print("查看频数, 大于20000得值极少，其实这里也可以把这些当作特殊得值（异常值）直接用填充或者删掉，再前面进行")
# log变换 z之后的分布较均匀，可以进行log变换进行预测，这也是预测问题常用的trick
plt.hist(np.log(Train_data['price']), orientation = 'vertical',histtype = 'bar', color ='red') 
plt.show()

#特征分为类别特征和数字特征，并对类别特征查看unique分布
print("2.3.6 特征分为类别特征和数字特征，并对类别特征查看unique分布")
# 分离label即预测值
Y_train = Train_data['price']

numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ]

categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode',]

# 特征nunique分布
for cat_fea in categorical_features:
    print(cat_fea + "的特征分布如下：")
    print("{}特征有个{}不同的值".format(cat_fea, Train_data[cat_fea].nunique()))
    print(Train_data[cat_fea].value_counts())


# 特征nunique分布
for cat_fea in categorical_features:
    print(cat_fea + "的特征分布如下：")
    print("{}特征有个{}不同的值".format(cat_fea, Test_data[cat_fea].nunique()))
    print(Test_data[cat_fea].value_counts())


#数字特征分析
print("数字特征分析")
numeric_features.append('price')
print(numeric_features)


print(Train_data.head())


## 1) 相关性分析
print("相关性分析")
price_numeric = Train_data[numeric_features]
correlation = price_numeric.corr()
print(correlation['price'].sort_values(ascending = False),'\n')

f , ax = plt.subplots(figsize = (7, 7))

plt.title('Correlation of Numeric Features with Price',y=1,size=16)

sns.heatmap(correlation,square = True,  vmax=0.8)
plt.tight_layout()
plt.show()

del price_numeric['price']


## 2) 查看几个特征得 偏度和峰值
print("查看几个特征得 偏度和峰值")
for col in numeric_features:
    print('{:15}'.format(col), 
          'Skewness: {:05.2f}'.format(Train_data[col].skew()) , 
          '   ' ,
          'Kurtosis: {:06.2f}'.format(Train_data[col].kurt())  
         )

## 3) 每个数字特征得分布可视化
print("每个数字特征得分布可视化")
f = pd.melt(Train_data, value_vars=numeric_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
plt.tight_layout()
plt.show()


## 4) 数字特征相互之间的关系可视化
print("数字特征相互之间的关系可视化")
sns.set()
columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
sns.pairplot(Train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()


Train_data.columns

print(Y_train)


## 5) 多变量互相回归关系可视化
print("此处是多变量之间的关系可视化，可视化更多学习可参考很不错的文章 https://www.jianshu.com/p/6e18d21a4cad")
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2, figsize=(24, 20))
# ['v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
v_12_scatter_plot = pd.concat([Y_train,Train_data['v_12']],axis = 1)
sns.regplot(x='v_12',y = 'price', data = v_12_scatter_plot,scatter= True, fit_reg=True, ax=ax1)

v_8_scatter_plot = pd.concat([Y_train,Train_data['v_8']],axis = 1)
sns.regplot(x='v_8',y = 'price',data = v_8_scatter_plot,scatter= True, fit_reg=True, ax=ax2)

v_0_scatter_plot = pd.concat([Y_train,Train_data['v_0']],axis = 1)
sns.regplot(x='v_0',y = 'price',data = v_0_scatter_plot,scatter= True, fit_reg=True, ax=ax3)

power_scatter_plot = pd.concat([Y_train,Train_data['power']],axis = 1)
sns.regplot(x='power',y = 'price',data = power_scatter_plot,scatter= True, fit_reg=True, ax=ax4)

v_5_scatter_plot = pd.concat([Y_train,Train_data['v_5']],axis = 1)
sns.regplot(x='v_5',y = 'price',data = v_5_scatter_plot,scatter= True, fit_reg=True, ax=ax5)

v_2_scatter_plot = pd.concat([Y_train,Train_data['v_2']],axis = 1)
sns.regplot(x='v_2',y = 'price',data = v_2_scatter_plot,scatter= True, fit_reg=True, ax=ax6)

v_6_scatter_plot = pd.concat([Y_train,Train_data['v_6']],axis = 1)
sns.regplot(x='v_6',y = 'price',data = v_6_scatter_plot,scatter= True, fit_reg=True, ax=ax7)

v_1_scatter_plot = pd.concat([Y_train,Train_data['v_1']],axis = 1)
sns.regplot(x='v_1',y = 'price',data = v_1_scatter_plot,scatter= True, fit_reg=True, ax=ax8)

v_14_scatter_plot = pd.concat([Y_train,Train_data['v_14']],axis = 1)
sns.regplot(x='v_14',y = 'price',data = v_14_scatter_plot,scatter= True, fit_reg=True, ax=ax9)

v_13_scatter_plot = pd.concat([Y_train,Train_data['v_13']],axis = 1)
sns.regplot(x='v_13',y = 'price',data = v_13_scatter_plot,scatter= True, fit_reg=True, ax=ax10)
plt.show()
#类别特征分析

## 1) unique分布
for fea in categorical_features:
    print(Train_data[fea].nunique())

print(categorical_features)

## 2) 类别特征箱形图可视化
print("类别特征箱形图可视化")
# 因为 name和 regionCode的类别太稀疏了，这里我们把不稀疏的几类画一下
categorical_features = ['model',
 'brand',
 'bodyType',
 'fuelType',
 'gearbox',
 'notRepairedDamage']
for c in categorical_features:
    Train_data[c] = Train_data[c].astype('category')
    if Train_data[c].isnull().any():
        Train_data[c] = Train_data[c].cat.add_categories(['MISSING'])
        Train_data[c] = Train_data[c].fillna('MISSING')

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "price")
plt.show()
print(Train_data.columns)


## 3) 类别特征的小提琴图可视化
print("类别特征的小提琴图可视化")
catg_list = categorical_features
target = 'price'
for catg in catg_list :
    sns.violinplot(x=catg, y=target, data=Train_data)
    plt.show()

categorical_features = ['model',
 'brand',
 'bodyType',
 'fuelType',
 'gearbox',
 'notRepairedDamage']
## 4) 类别特征的柱形图可视化
print("类别特征的柱形图可视化")
def bar_plot(x, y, **kwargs):
    sns.barplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(bar_plot, "value", "price")
plt.show()


##  5) 类别特征的每个类别频数可视化(count_plot)
print("类别特征的每个类别频数可视化(count_plot)")
def count_plot(x,  **kwargs):
    sns.countplot(x=x)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data,  value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(count_plot, "value")
plt.show()

# 用pandas_profiling生成数据报告 用pandas_profiling生成一个较为全面的可视化和数据报告(较为简单、方便) 最终打开html文件即可

import pandas_profiling
pfr = pandas_profiling.ProfileReport(Train_data)
pfr.to_file("./example.html")