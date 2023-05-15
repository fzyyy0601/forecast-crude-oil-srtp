from scipy.stats import jarque_bera
import pandas as pd
import re
import nltk.stem
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


dat=pd.read_csv(r'd:/0.csv',encoding='utf-8')
#去重+小写
def pre(dat,column):
    d=dat.drop_duplicates(column,keep='first')
    dat1=[i.lower() for i in d]
    #去停用词+标点符号+数字
    r0=[re.sub(r'\d+','',i)for i in dat1]
    r1=[i.translate(str.maketrans('','',string.punctuation))for i in r0]
    return r1

def token(dat):
    tokens=[]
    token_1=[]
    s=nltk.stem.SnowballStemmer('english')
    for i in dat:
        i=str(i)
        tokens.append(nltk.word_tokenize(i))
    #q去停用词
    sw=stopwords.words('english')
    for w in ['would','could','never','can','exclusive','column','corrected','wrapup','analysis','update',
    'timeline','insight','brief','rpt','factbox','breakingviews']:
        sw.append(w)
    for token in tokens:
        datt=[w for w in token if w not in sw]
        token_1.append(datt)
    cleaned=[]
    for t in token_1:
        datt=[s.stem(ws)for ws in t]
        cleaned.append(datt)
        return cleaned

#获得处理后的txt形式语料
def get_text(filepath,dat):
    f=open(filepath,'w',encoding='utf-8')
    for doc in dat:
        f.write(' '.join(doc)+'\n')
    f.close()


#获取情感指数sentiment index
def get_sentiment(dat):
    textb_p,textb_s=[],[]
    v_neu,v_pos,v_com,v_neg=[],[],[],[]
    sid=SentimentIntensityAnalyzer()
    for i in dat:
        t1=TextBlob(i)
        textb_p.append(t1.sentiment.polarity)
        textb_s.sppend(t1.sentiment.subjectivity)
        ss=sid.polarity_scores(i)
        v_neg.append(ss['neg'])
        v_neu.append(ss['neu'])
        v_pos.append(ss['pos'])
        v_com.append(ss['compound'])
    
    df_tb=pd.DataFrame(list(zip(textb_s,textb_p)))
    df_v=pd.DataFrame(list(zip(v_com,v_neg,v_neu,v_pos)))
    df_tb.columns=['tb_s','tb_p']
    df_v.columns=['v_com','v_neg','v_neu','v_pos']
    return df_tb,df_v

#画出相关关系的热图
def heatmap(dat):
    a=dat.corr('spearman')
    plt.subplots(figsize=(12,12))
    sns.heatmap(a,annot=True,vmax=True,square=True,cmap="Blues")
    plt.show()
    
#def pca(dat):
#   pca=PCA(n_components=1)
#   pca.fit(dat)
#   out=pca.transfrom(dat)
#   return out

def info(dat):
    #传入时去掉time列
    skew,kurt=[],[]
    outt=pd.DataFrame(columns=['Min','Max','Mean','Std','Skew','Kurt','JB-Test','JB-Test(p)'],index=dat.columns.tolist())
    for(coluName,columnData)in dat.iteritems():
        #if coluName=='time':
         #   continue
        outt[coluName]['Min']=columnData.min()
        outt[coluName]['Kurt']=columnData.kurt()
        outt[coluName]['Skew']=columnData.skew()
        outt[coluName]['Max']=columnData.max()
        outt[coluName]['Mean']=columnData.mean()
        outt[coluName]['Std']=columnData.std()
        jb=jarque_bera(columnData)
        outt[coluName]['JB-Test']=jb[0]
        outt[coluName]['JB-Test(p)']=jb[1]
        
def adftest(data,lag):
    adfResult = sm.tsa.stattools.adfuller(data,lag)
    output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used",
                                             "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"],
                                      columns=['value'])
    output['value']['Test Statistic Value'] = adfResult[0]
    output['value']['p-value'] = adfResult[1]
    output['value']['Lags Used'] = adfResult[2]
    output['value']['Number of Observations Used'] = adfResult[3]
    output['value']['Critical Value(1%)'] = adfResult[4]['1%']
    output['value']['Critical Value(5%)'] = adfResult[4]['5%']
    output['value']['Critical Value(10%)'] = adfResult[4]['10%']
    print(output)

