from bs4 import BeautifulSoup     #web parser, get required information
import re       #regular expression
import urllib.request,urllib.error      #make URL，get webpage data
import xlwt     #manipulate excel

def main():
    baseurl="https://oilprice.com/Energy/Crude-Oil/Page-"
    
    datalist = getData(baseurl)
    
    #3.保存数据
    savepath=r'D:\study\毕设\基本数据-new\oilprice_1-new.xls'
    saveData(datalist,savepath)
    
findTime=re.compile(r'<p class="categoryArticle__meta">(.*)</p>')
findTitle=re.compile(r'<h2 class="categoryArticle__title">(.*)</h2>')


#爬取网页
def getData(baseurl):
    datalist = []
    for i in range(210): 

        print('第',i,'页')
        url = baseurl + str(i)+".html"
        html = askURL(url)      #save source code of webpage

         # 2.逐一解析数据
        soup = BeautifulSoup(html,"html.parser")
        
        for item in soup.find_all('div',class_="categoryArticle__content"):     #search for strings that meet our requirements and put them inside a list
            #print(item)   #测试：查看电影item全部信息
            data = []    #use list to save all information per an item
            item = str(item)

            #影片详情的链接
            title = re.findall(findTitle,item)[0]     #re library, search for strings that meet our requirements
            data.append(title)                       #add url
            #print(title)
            
            timee= re.findall(findTime,item)[0][:13]
            data.append(timee)                       
            #print(timee)
            
            datalist.append(data)       
    return datalist

#get webpage content of given URL
def askURL(url):
    head = {"User-Agent": "Mozilla / 5.0(Windows NT 10.0; Win64; x64) AppleWebKit / 537.36(KHTML, like Gecko) Chrome / 80.0.3987.122  Safari / 537.36"
    }
    request = urllib.request.Request(url,headers=head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
        #print(html)
    except urllib.error.URLError as e:
        if hasattr(e,"code"):
            print(e.code)
        if hasattr(e,"reason"):
            print(e.reason)
    return html



def saveData(datalist,savepath):
    print("save....")
    book = xlwt.Workbook(encoding="utf-8",style_compression=0)  #create workbook object
    sheet = book.add_sheet('oilnews',cell_overwrite_ok=True)    #create worksheet
    col = ("title","time")
    for i in range(0,2):
        sheet.write(0,i,col[i]) #column name
    for i in range(0,len(datalist)):
        print("第%d条" %(i+1))
        data = datalist[i]
        for j in range(0,2):
            sheet.write(i+1,j,data[j])      #write data into file

    book.save(savepath)       # save file


if __name__ == "__main__":          
    main()
    
    print("爬取完毕！")