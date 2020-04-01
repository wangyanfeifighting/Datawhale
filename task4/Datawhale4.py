import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
""" 
reduce_mem_usage 函数通过调整数据类型，帮助我们减少数据在内存中占用的空间
"""
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
#5.4.1 读取数据
sample_feature = reduce_mem_usage(pd.read_csv('./task3/data_for_tree.csv'))
print(sample_feature)

continuous_feature_names = [x for x in sample_feature.columns if x not in ['price','brand','model','brand']]

print(continuous_feature_names)

#5.4.2 线性回归 & 五折交叉验证 & 模拟真实业务情况
sample_feature = sample_feature.dropna().replace('-', 0).reset_index(drop=True)

print(sample_feature)
print("**************")
sample_feature['notRepairedDamage'] = sample_feature['notRepairedDamage'].astype(np.float32)
print(sample_feature['notRepairedDamage'] )
print("&&&&&&&&&&&&&&")
train = sample_feature[continuous_feature_names + ['price']]
print(train)
train_X = train[continuous_feature_names]
train_y = train['price']

#5.4.2 - 1 简单建模
from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
model = model.fit(train_X, train_y)
#查看训练的线性回归模型的截距（intercept）与权重(coef)
print('intercept:'+ str(model.intercept_))

print(sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True))


from matplotlib import pyplot as plt
subsample_index = np.random.randint(low=0, high=len(train_y), size=50)

print(subsample_index)
#绘制特征v_9的值与标签的散点图，图片发现模型的预测结果（蓝色点）与真实标签（黑色点）的分布差异较大，且部分预测值出现了小于0的情况，说明我们的模型存在一些问题
plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], model.predict(train_X.loc[subsample_index]), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price','Predicted Price'],loc='upper right')
print('The predicted price is obvious different from true price')
plt.show()

#通过作图我们发现数据的标签（price）呈现长尾分布，不利于我们的建模预测。原因是很多模型都假设数据误差项符合正态分布，而长尾分布的数据违背了这一假设。参考博客：https://blog.csdn.net/Noob_daniel/article/details/76087829

import seaborn as sns
print('It is clear to see the price shows a typical exponential distribution')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_y)
plt.show()
plt.subplot(1,2,2)
sns.distplot(train_y[train_y < np.quantile(train_y, 0.9)])
plt.show()

#在这里我们对标签进行了 $log(x+1)$ 变换，使标签贴近于正态分布
print("在这里我们对标签进行了 $log(x+1)$ 变换，使标签贴近于正态分布")
train_y_ln = np.log(train_y + 1)
import seaborn as sns
print('The transformed price seems like normal distribution')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_y_ln)
plt.show()
print("分界线")
plt.subplot(1,2,2)
sns.distplot(train_y_ln[train_y_ln < np.quantile(train_y_ln, 0.9)])
plt.show()


model = model.fit(train_X, train_y_ln)

print('intercept:'+ str(model.intercept_))
sortResult=sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)
print(sortResult)

#再次进行可视化，发现预测结果与真实值较为接近，且未出现异常状况
print("再次进行可视化，发现预测结果与真实值较为接近，且未出现异常状况")
plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], np.exp(model.predict(train_X.loc[subsample_index])), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price','Predicted Price'],loc='upper right')
print('The predicted price seems normal after np.log transforming')
plt.show()

#5.4.2 - 2 五折交叉验证
"""
在使用训练集对参数进行训练的时候，经常会发现人们通常会将一整个训练集分为三个部分
（比如mnist手写训练集）。一般分为：训练集（train_set），评估集（valid_set），
测试集（test_set）这三个部分。这其实是为了保证训练效果而特意设置的。其中测试集很好理解，
其实就是完全不参与训练的数据，仅仅用来观测测试效果的数据。而训练集和评估集则牵涉到下面的知识了。
因为在实际的训练中，训练的结果对于训练集的拟合程度通常还是挺好的（初始条件敏感），
但是对于训练集之外的数据的拟合程度通常就不那么令人满意了。
因此我们通常并不会把所有的数据集都拿来训练，而是分出一部分来（这一部分不参加训练）
对训练集生成的参数进行测试，相对客观的判断这些参数对训练集之外的数据的符合程度。
这种思想就称为交叉验证（Cross Validation）
"""
print("五折交叉验证")
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,  make_scorer
def log_transfer(func):
    def wrapper(y, yhat):
        result = func(np.log(y), np.nan_to_num(np.log(yhat)))
        return result
    return wrapper
scores = cross_val_score(model, X=train_X, y=train_y, verbose=1, cv = 5, scoring=make_scorer(log_transfer(mean_absolute_error)))

print(scores)

#使用线性回归模型，对未处理标签的特征数据进行五折交叉验证（Error 1.36）

print(print('AVG:', np.mean(scores)))

#使用线性回归模型，对处理过标签的特征数据进行五折交叉验证（Error 0.19）

scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=1, cv = 5, scoring=make_scorer(mean_absolute_error))
print(scores)
print('AVG:', np.mean(scores))

scores = pd.DataFrame(scores.reshape(1,-1))
scores.columns = ['cv' + str(x) for x in range(1, 6)]
scores.index = ['MAE']
print(scores)

#5.4.2 - 3 模拟真实业务情况
"""
但在事实上，由于我们并不具有预知未来的能力，五折交叉验证在某些与时间相关的数据集上
反而反映了不真实的情况。通过2018年的二手车价格预测2017年的二手车价格，这显然是不合理的，
因此我们还可以采用时间顺序对数据集进行分隔。在本例中，我们选用靠前时间的4/5样本当作训练集，
靠后时间的1/5当作验证集，最终结果与五折交叉验证差距不大
"""

import datetime
sample_feature = sample_feature.reset_index(drop=True)
split_point = len(sample_feature) // 5 * 4
train = sample_feature.loc[:split_point].dropna()
val = sample_feature.loc[split_point:].dropna()

train_X = train[continuous_feature_names]
train_y_ln = np.log(train['price'] + 1)
val_X = val[continuous_feature_names]
val_y_ln = np.log(val['price'] + 1)
model = model.fit(train_X, train_y_ln)
print(mean_absolute_error(val_y_ln, model.predict(val_X)))

#5.4.2 - 4 绘制学习率曲线与验证曲线
from sklearn.model_selection import learning_curve, validation_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_size=np.linspace(.1, 1.0, 5 )):  
    plt.figure()  
    plt.title(title)  
    if ylim is not None:  
        plt.ylim(*ylim)  
    plt.xlabel('Training example')  
    plt.ylabel('score')  
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size, scoring = make_scorer(mean_absolute_error))  
    train_scores_mean = np.mean(train_scores, axis=1)  
    train_scores_std = np.std(train_scores, axis=1)  
    test_scores_mean = np.mean(test_scores, axis=1)  
    test_scores_std = np.std(test_scores, axis=1)  
    plt.grid()#区域  
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,  
                     train_scores_mean + train_scores_std, alpha=0.1,  
                     color="r")  
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,  
                     test_scores_mean + test_scores_std, alpha=0.1,  
                     color="g")  
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',  
             label="Training score")  
    plt.plot(train_sizes, test_scores_mean,'o-',color="g",  
             label="Cross-validation score")  
    plt.legend(loc="best")  
    return plt 

plot_learning_curve(LinearRegression(), 'Liner_model', train_X[:1000], train_y_ln[:1000], ylim=(0.0, 0.5), cv=5, n_jobs=1)  
plt.show()

#5.4.3 多种模型对比
train = sample_feature[continuous_feature_names + ['price']].dropna()

train_X = train[continuous_feature_names]
train_y = train['price']
train_y_ln = np.log(train_y + 1)
print('多种模型对比')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_y_ln)
plt.show()

#5.4.3 - 1 线性模型 & 嵌入式特征选择
"""
本章节默认，学习者已经了解关于过拟合、模型复杂度、正则化等概念。
否则请寻找相关资料或参考如下连接：

用简单易懂的语言描述「过拟合 overfitting」？ 
https://www.zhihu.com/question/32246256/answer/55320482
模型复杂度与模型的泛化能力 http://yangyingming.com/article/434/
正则化的直观理解 https://blog.csdn.net/jinping_shi/article/details/52433975
在过滤式和包裹式特征选择方法中，特征选择过程与学习器训练过程有明显的分别。
而嵌入式特征选择在学习器训练过程中自动地进行特征选择。嵌入式选择最常用的是L1正则化与L2正则化。
在对线性回归模型加入两种正则化方法后，他们分别变成了岭回归与Lasso回归。
"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
models = [LinearRegression(),
          Ridge(),
          Lasso()]
result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')


#对三种方法的效果对比

result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
print(result)

model = LinearRegression().fit(train_X, train_y_ln)
print('intercept:'+ str(model.intercept_))
sns.barplot(abs(model.coef_), continuous_feature_names)
plt.show()

"""
L2正则化在拟合过程中通常都倾向于让权值尽可能小，最后构造一个所有参数都比较小的模型。
因为一般认为参数值小的模型比较简单，能适应不同的数据集，也在一定程度上避免了过拟合现象。
可以设想一下对于一个线性回归方程，若参数很大，那么只要数据偏移一点点，就会对结果造成很大的影响；
但如果参数足够小，数据偏移得多一点也不会对结果造成什么影响，专业一点的说法是『抗扰动能力强』
"""

model = Ridge().fit(train_X, train_y_ln)
print('intercept:'+ str(model.intercept_))
sns.barplot(abs(model.coef_), continuous_feature_names)
plt.show()
"""
L1正则化有助于生成一个稀疏权值矩阵，进而可以用于特征选择。如下图，
我们发现power与userd_time特征非常重要。
"""

model = Lasso().fit(train_X, train_y_ln)
print('intercept:'+ str(model.intercept_))
sns.barplot(abs(model.coef_), continuous_feature_names)
plt.show()
"""
除此之外，决策树通过信息熵或GINI指数选择分裂节点时，优先选择的分裂特征也更加重要，
这同样是一种特征选择的方法。XGBoost与LightGBM模型中的model_importance指标正是基于此计算的
"""

#5.4.3 - 2 非线性模型
"""
除了线性模型以外，还有许多我们常用的非线性模型如下，在此篇幅有限不再一一讲解原理。
我们选择了部分常用模型与线性模型进行效果比对。
"""

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
models = [LinearRegression(),
          DecisionTreeRegressor(),
          RandomForestRegressor(),
          GradientBoostingRegressor(),
          MLPRegressor(solver='lbfgs', max_iter=100), 
          XGBRegressor(n_estimators = 100, objective='reg:squarederror'), 
          LGBMRegressor(n_estimators = 100)]
result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')


result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
print(result)


#5.4.4 模型调参
"""
在此我们介绍了三种常用的调参方法如下：

贪心算法 https://www.jianshu.com/p/ab89df9759c8
网格调参 https://blog.csdn.net/weixin_43172660/article/details/83032029
贝叶斯调参 https://blog.csdn.net/linxid/article/details/81189154
"""
## LGB的参数集合：

objective = ['regression', 'regression_l1', 'mape', 'huber', 'fair']

num_leaves = [3,5,10,15,20,40, 55]
max_depth = [3,5,10,15,20,40, 55]
bagging_fraction = []
feature_fraction = []
drop_rate = []
#5.4.4 - 1 贪心调参
best_obj = dict()
for obj in objective:
    model = LGBMRegressor(objective=obj)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_obj[obj] = score
    
best_leaves = dict()
for leaves in num_leaves:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0], num_leaves=leaves)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_leaves[leaves] = score
    
best_depth = dict()
for depth in max_depth:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0],
                          num_leaves=min(best_leaves.items(), key=lambda x:x[1])[0],
                          max_depth=depth)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_depth[depth] = score

sns.lineplot(x=['0_initial','1_turning_obj','2_turning_leaves','3_turning_depth'], y=[0.143 ,min(best_obj.values()), min(best_leaves.values()), min(best_depth.values())])
plt.show()

#5.4.4 - 2 Grid Search 调参
from sklearn.model_selection import GridSearchCV
parameters = {'objective': objective , 'num_leaves': num_leaves, 'max_depth': max_depth}
model = LGBMRegressor()
clf = GridSearchCV(model, parameters, cv=5)
clf = clf.fit(train_X, train_y)
print(clf.best_params_)

model = LGBMRegressor(objective='regression',
                          num_leaves=55,
                          max_depth=15)

result=np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))

print(result)

#5.4.4 - 3 贝叶斯调参
from bayes_opt import BayesianOptimization
def rf_cv(num_leaves, max_depth, subsample, min_child_samples):
    val = cross_val_score(
        LGBMRegressor(objective = 'regression_l1',
            num_leaves=int(num_leaves),
            max_depth=int(max_depth),
            subsample = subsample,
            min_child_samples = int(min_child_samples)
        ),
        X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)
    ).mean()
    return 1 - val
rf_bo = BayesianOptimization(
    rf_cv,
    {
    'num_leaves': (2, 100),
    'max_depth': (2, 100),
    'subsample': (0.1, 1),
    'min_child_samples' : (2, 100)
    }
)
value=rf_bo.maximize()
print(value)

print("最后结果： "+1 - rf_bo.max['target'])
"""
在本章中，我们完成了建模与调参的工作，并对我们的模型进行了验证。
此外，我们还采用了一些基本方法来提高预测的精度，提升如下图所示。
"""

plt.figure(figsize=(13,5))
sns.lineplot(x=['0_origin','1_log_transfer','2_L1_&_L2','3_change_model','4_parameter_turning'], y=[1.36 ,0.19, 0.19, 0.14, 0.13])
plt.show()