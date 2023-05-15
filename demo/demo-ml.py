import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error,explained_variance_score, mean_absolute_error, mean_squared_error, r2_score ,mean_squared_log_error
# 批量导入指标算法
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error,explained_variance_score, mean_absolute_error, mean_squared_error, r2_score 
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,cross_val_score

from sklearn.ensemble import AdaBoostRegressor
#rmse=mse^(1/2)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LassoCV

models={}
models['ada']=AdaBoostRegressor()
models['etr']=ExtraTreesRegressor()
models['rfr']=RandomForestRegressor()
models['svr']=SVR()
models['lasso']=Lasso(alpha=0.1, normalize=True, max_iter=10000)
models['linear']=LinearRegression()

'''
num_splits分割份数
x y 自变量和因变量
fp1 文件存储地址1 fp2 文件存储地址2
'''
def val_tscv(num_splits,x,y,fp1,fp2):
    cv_score_list=[]
    pre_y_list=[]
    print('{0} folds'.format(num_splits))
    
    model_names=['ada','etr','rfr','svr','lasso','linear']
    for key in models:
        tscv=TimeSeriesSplit(n_splits=num_splits)
        cv_result=cross_val_score(models[key],x,y,cv=tscv,scoring='neg_mean_squared_error')
        cv_score_list.append(cv_result)
        pre_y_list.append(models[key].fit(x,y).predict(x))
        print('%s:%f(%f)'%(key,cv_result.mean(),cv_result.std()))
    n_samples,n_features=x.shape
    model_metrics_name=[explained_variance_score,mean_absolute_percentage_error,mean_absolute_error, mean_squared_error, r2_score,mean_squared_log_error]  # 回归评估指标对象集
    model_metrics_list=[]
    for i in range(6):
        tmp_list=[]
        for m in model_metrics_name:
            tmp_score=m(y,pre_y_list[i])
            tmp_list.append(tmp_score)
        model_metrics_list.append(tmp_list)
    df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
    df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['explained_variance','mape', 'mae', 'mse', 'r2','msle'])  # 建立回归指标的数据框
    out_score(n_samples,n_features,df1,df2)
    #df1.to_csv(fp1,encoding='utf_8_sig') #保存交叉验证的数据框
    #df2.to_csv(fp2,encoding='utf_8_sig') # 保存回归指标的数据框
    plot_pred(pre_y_list,x) # 展示图像

def out_score(n_samples,n_features,df1,df2):
    print ('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
    print (30 * '-')  # 打印分隔线
    print ('cross validation result:')  # 打印输出标题
    print (df1)  # 打印输出交叉检验的数据框
    print (70 * '-')  # 打印分隔线
    print ('regression metrics:')  # 打印输出标题
    print (df2)  # 打印输出回归指标的数据框    
    print (70 * '-')  # 打印分隔线

'''
x y 自变量和因变量
fp1 文件存储地址1 需要包含文件的格式 如 result.csv
'''
def val_ts(x,y,fp1):
    
    pre_y_list=[]#存放预测结果
    
    model_names=['ada','etr','rfr','svr','lasso','linear']
    for key in models:
        x_train,y_train,x_test,y_test=train_test_split(x,y,train_size=0.8,shuffle=False)#shuffle=False不打乱 因为是时间序列数据
        pre_y_list.append(models[key].fit(x,y).predict(x))
        
    n_samples,n_features=x.shape
    model_metrics_name=[explained_variance_score,mean_absolute_percentage_error,mean_absolute_error, mean_squared_error, r2_score,mean_squared_log_error]  # 回归评估指标对象集
    model_metrics_list=[]
    for i in range(6):
        tmp_list=[]
        for m in model_metrics_name:
            tmp_score=m(y,pre_y_list[i])
            tmp_list.append(tmp_score)
        model_metrics_list.append(tmp_list)
    df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['explained_variance','mape', 'mae', 'mse', 'r2','msle'])  # 建立回归指标的数据框
    out_score1(n_samples,n_features,df2)

    df2.to_csv(fp1,encoding='utf_8_sig')
    plot_pred(pre_y_list,x) # 展示图像

def plot_pred(pre_y_list,y,x,model_names):#展示图像
    plt.figure()  # 创建画布
    plt.plot(np.arange(x.shape[0]), y, color='k', label='true y')  # 画出原始值的曲线
    color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
    linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
    for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
            plt.plot(np.arange(x.shape[0]), pre_y_list[i], color_list[i], label=model_names[i]) # 画出每条预测结果线
    plt.title('regression result comparison')  # 标题
    plt.legend(loc='upper right')  # 图例位置
    plt.ylabel('real and predicted value')  # y轴标题
    plt.savefig('tmp.pdf', bbox_inches='tight')
    plt.show() 

def out_score1(n_samples,n_features,df1):
    print ('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
    print (30 * '-')  # 打印分隔线
    
    print (70 * '-')  # 打印分隔线
    print ('regression metrics:')  # 打印输出标题
    print (df1)  # 打印输出回归指标的数据框    
    print (70 * '-')  # 打印分隔线


from scipy.stats import jarque_bera
def info1(dat,fp1):
    #传入时去掉time列
    skew,kurt=[],[]
    outt=pd.DataFrame(columns=['Min','Max','Mean','Std','Skew','Kurt','JB-Test','JB-Test(p)'],index=dat.columns.tolist())
    print('    ','Min','Max','Mean','Std','Skew','Kurt','JB-Test','JB-Test(p)','\t')
    print('-'*50)
    for(coluName,columnData)in dat.iteritems():
        jb=jarque_bera(columnData)
        print(coluName,columnData.min(),columnData.max(),columnData.mean(),columnData.std(),columnData.skew(),columnData.kurt(),jb[0],jb[1])
        print('-'*50)
        
        outt[coluName]['Min']=columnData.min()
        outt[coluName]['Kurt']=columnData.kurt()
        outt[coluName]['Skew']=columnData.skew()
        outt[coluName]['Max']=columnData.max()
        outt[coluName]['Mean']=columnData.mean()
        outt[coluName]['Std']=columnData.std()
        jb=jarque_bera(columnData)
        outt[coluName]['JB-Test']=jb[0]
        outt[coluName]['JB-Test(p)']=jb[1]
    outt.to_csv(fp1,encoding='utf_8_sig')

def adjusted_r2(r2,train)->float:
    adj_r2=(1-(1-r2)*((train.shape[0]-1)/(train.shape[0]-train.shape[1])))
    return adj_r2
