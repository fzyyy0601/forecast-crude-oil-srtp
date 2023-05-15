import pandas as pd
import numpy as np
from pandas import DataFrame
import xlsxwriter
def delete_col(filepath,savefile_path,col_name):
    data=pd.read_excel(filepath)
    
    data=data.drop(col_name,axis=1)#删除指定列
    
    DataFrame(data).to_excel(savefile_path,index=False,header=False)
    print('done!')

def delete_dup(file_path,save_path,col_name):#去重

    #print(time.asctime(time.localtime(time.time())))
    data=pd.DataFrame(pd.read_excel(file_path)
    re_row=data.drop_duplicates([col_name])
    re_row.to_excel(save_path)
    #print(time.asctime(time.localtime(time.time())))

def delete_(filepath):
    wb=load_workbook(filepath)
    ws=wb.active
    ws.delete_cols(1)#删除序号列,即第一列
    wb.save(filepath)

if __name__=='__main__':
    file_path='d:\srtp\exp3.5.xlsx'
    
    col_name1='title'
    col_name2='times'
    delete_col(file_path,file_path,col_name2)
    delete_dup(file_path,file_path,col_name1)
    delete_(file_path)