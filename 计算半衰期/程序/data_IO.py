#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:14:03 2018

@author: jiyuyang
"""

#输入数据
#file=open('QAZ.txt','a')
#file.writelines(['A\t','Z\t','Q\n'])
#A=input('A = : ')
#while int(A)!=0:
#    file.write(A+'\t')
#    Z=input('Z = : ')
#    file.write(Z+'\t')
#    Q=input('Q = : ')
#    file.write(Q+'\n')
#    A=input('A = : ')

#输出数据
file=open('AZQ.txt','rU')
lines=file.readlines()
AZQList=[]
AZQ=[]
for i in range(len(lines)):
    if i==0:
        continue
    AZQStr=lines[i]
    AZQList=list(AZQStr)
    AZQ_list=[]
    AZQ_str=''
    for j in range(len(AZQList)):
        if AZQList[j]!='\t' and AZQList[j]!='\n':
            AZQ_str+=AZQList[j]
        if AZQList[j]=='\t' or AZQList[j]=='\n':
            AZQ_list.append(float(AZQ_str))
            AZQ_str=''
    AZQ.append(AZQ_list)
file.close()