import requests
import re
import datetime
import time
import pretty_errors
import random

'''
using requests to request web content, then use regular expression to extract specific information
finally output to txt file
'''
def baidu(page):
    header={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36'}
    num=(page-1)*10
    url='https://www.baidu.com/s?tn=news&rtt=1&bsst=1&cl=2&wd=原油&pn='+str(num)
    res=requests.get(url,headers=header).text
    p_time='<span class="c-color-gray2 c-font-normal">(.*?)</span>'
    p_ab='<span class="c-font-normal c-color-text"><!--s-text-->(.*?)<!--/s-text--></span>'#简介
    p_site='<div><h3 class="news-title_1YtI1"><a href="(.*?)" target="_blank" class="c-font-big"'
    p_info1='<div><h3 class="news-title_1YtI1"><a href=.*?<!--s-text-->(.*?)<!--/s-text--></a>'
    sites=re.findall(p_site,res,re.S)
    ab=re.findall(p_ab,res,re.S)
    href=re.findall(p_info1,res,re.S)
    timess=re.findall(p_time,res,re.S)
    for i in range(len(href)):
        href[i]=re.sub('<.*?>','',href[i])
        ab[i]=re.sub('<.*?>','',ab[i])
     #   print(str(i+1)+'.'+href[i]+'\n'+sites[i]+'\n'+ab[i])
        with open('d:/srtp/py/bdnews_30P.txt','a',encoding='utf-8') as f:
            p=str(i+1)+'.'+href[i]+'\t'+timess[i]+'\n'+sites[i]+'\n'+ab[i]+'\n'
            f.write(p)

if __name__=="__main__":
    with open('d:/srtp/py/bdnews_30P.txt','a',encoding='utf-8') as f:
       f.write(str(datetime.datetime.now()))
    for i  in range(500):
        baidu(i+1)
        print('第'+str(i+1)+'页爬取成功')
        time.sleep(random.uniform(1.2,9.9))