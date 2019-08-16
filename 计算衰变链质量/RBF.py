# -*- coding: utf-8 -*-
"""
Created on Tue May 29 23:46:59 2018

@author: jiyuyang
"""
import numpy as np
#RBF重建函数(计算单个核)
def RBF(AList,ZList,M_expList,M_thList,X=[0,1,2,3]):
    #unknow nucleus X : X[0]:'A' ; X[1]:'Z' ; X[2]:'X_M_exp' ; X[3]:'X_M_th'
    X_Z=X[1]
    X_N=X[0]-X[1]
    X_M_exp=X[2]
    X_M_th=X[3]
    m=len(ZList)       #样点个数
    N=np.zeros(m)   #中子数数组
    for i in range(m):
        N[i]=AList[i]-ZList[i]
    #获取质量偏差一维数组
    D=np.zeros(m)
    for i in range(m):
        if N[i]==X_N and ZList[i]==X_Z:
            continue
        D[i]=M_expList[i]-M_thList[i]
        #获取基函数二维数组
    D=np.delete(D,i)
    phi_r=np.zeros((m,m))
    #行指数i，列指数j
    for i in range(m):
        if N[i]==X_N and ZList[i]==X_Z:
            continue
        for j in range(m):
            if N[j]==X_N and ZList[j]==X_Z:
                continue
            Z_ij2=pow((ZList[i]-ZList[j]),2)
            N_ij2=pow((N[i]-N[j]),2)
            phi_r[i][j]=pow((Z_ij2+N_ij2),0.5)
    phi_r=np.delete(phi_r,i,axis=0)   #axis = 0：表示删除数组的行
    D=np.mat(D)
    phi_r=np.mat(phi_r)
    omega=phi_r.I*D.T    #权重矩阵
    S=0   #重建函数
    for i in range(m):
        Z_xi2=pow((X_Z-ZList[i]),2)
        N_xi2=pow((X_N-N[i]),2)
        r=pow((Z_xi2+N_xi2),0.5)
        S+=float(omega[i])*r
    X_M_RBF=X_M_th+S
    dev=np.abs(X_M_RBF-X_M_exp)
    return X_M_RBF,dev

#计算RBF方法所有用到的核
def sum_RBF(AList,ZList,M_expList,M_thList):
    m=len(ZList)
    X_M_RBF_List=[]
    devList=[]
    for i in range(m):
        X=[AList[i],ZList[i],M_expList[i],M_thList[i]]
        X_M_RBF,dev=RBF(AList,ZList,M_expList,M_thList,X)
        X_M_RBF_List.append(X_M_RBF)
        devList.append(dev)
    return X_M_RBF_List,devList

#输出数据
def date_output():
    file=open('M_th.txt','r')
    lines=file.readlines()
    AZMList=[]
    AZM=[]
    for i in range(len(lines)):
        if i==0:
            continue
        AZMStr=lines[i]
        AZMList=list(AZMStr)
        AZM_list=[]
        AZM_str=''
        for j in range(len(AZMList)):
            if AZMList[j]!='\t' and AZMList[j]!='\n':
                AZM_str+=AZMList[j]
            if AZMList[j]=='\t' or AZMList[j]=='\n':
                AZM_list.append(float(AZM_str))
                AZM_str=''
        AZM.append(AZM_list)
    file.close()
    return AZM

def calculate():
    AZM=date_output()
    AList=[]
    ZList=[]
    M_expList=[]
    M_thList=[]
    for i in range(len(AZM)):
        AList.append(AZM[i][0])
        ZList.append(AZM[i][1])
        M_expList.append(AZM[i][2])
        M_thList.append(AZM[i][3])
    X_M_RBF_List,devList=sum_RBF(AList,ZList,M_expList,M_thList)
    RBF_list=[]
    nucleus_list=[]
    for j in range(len(devList)):
        nucleus_list.append((AList[j],ZList[j]))
        RBF_list.append((X_M_RBF_List[j],devList[j]))
    RBF_dict=dict(zip(nucleus_list,RBF_list))
    file=open('M_RBF.txt','w')
    file.writelines("A"+'\t'+"Z"+'\t'+"M_RBF"+'\t'+"dev"+'\n')
    for k in range(len(AList)):
        file.writelines(str(AList[k])+'\t'+str(ZList[k])+'\t'+\
                        str(X_M_RBF_List[k])+'\t'+str(devList[k])+'\n')
    file.close()
    return RBF_dict