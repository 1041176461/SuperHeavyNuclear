#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:39:13 2018

@author: jiyuyang
"""

#势能表达式
import math
import sympy as sp
from scipy import constants as C

#核势   单位：J
def nuclearPotential(r,R,A,Z,Q,L):
    """核势"""
    V0=162.3  #阱深，单位：MeV
    a=0.40    #常数，单位：fm    
    Vn=(-V0)*(1+sp.cosh(R/a))/(sp.cosh(r/a)+sp.cosh(R/a))
    return Vn

#库伦势  单位：J
    #r<=R
def coulombPotential(r,R,A,Z,Q,L):
    """库伦势(Z:母核电荷数)"""
    Vc1=(2.0*(Z-2)/(2.0*R))*(3-(r/float(R))**2)*1.43896
    return Vc1
    #r>=R
def cou(r,R,A,Z,Q,L):
    Vc2=1.43896*2.0*(Z-2)/r
    return Vc2

#离心势  单位：MeV
def centrifugationPotentials(r,R,A,Z,Q,L):
    """离心势(L:轨道角动量;A:母核质量数)"""
    u=4*(A-4)/A
    b=u/(C.hbar**2/(C.e*1e6*1e-30*1.66e-27*2))
    #b=u/20.936
    Vl=(1.0/b)*((L+1.0/2)**2)/(r**2)
    return Vl

#总势能  单位：MeV
def totalPotential(r,R,A,Z,Q,L):
    """总势能(Z:母核的电荷数;L:轨道角动量;A:母核质量数)"""
    Vn=nuclearPotential(r,R,A,Z,Q,L)
    Vl=centrifugationPotentials(r,R,A,Z,Q,L)
    Vc=coulombPotential(r,R,A,Z,Q,L)
    V=Vn+Vc+Vl
    return V
def total(r,R,A,Z,Q,L):
    """总势能(Z:母核的电荷数;L:轨道角动量;A:母核质量数)"""
    Vn=nuclearPotential(r,R,A,Z,Q,L)
    Vl=centrifugationPotentials(r,R,A,Z,Q,L)
    Vc=cou(r,R,A,Z,Q,L)
    V=Vn+Vc+Vl
    return V

#牛顿迭代法r2
def func(x0,R,A,Z,Q,L):
    f=totalPotential(x0,R,A,Z,Q,L)-Q
    return f
def func_diff(x0,R,A,Z,Q,L):
    x=sp.symbols("x")
    f=func(x,R,A,Z,Q,L)
    f_diff=sp.diff(f)
    diff=f_diff.evalf(subs = {x:x0})
    return diff
def newton(x0,R,A,Z,Q,L):
    f0=func(x0,R,A,Z,Q,L)
    diff0=func_diff(x0,R,A,Z,Q,L)
    x1=x0-float(f0)/diff0
    f1=func(x1,R,A,Z,Q,L)
    while sp.Abs(f1-f0)>R*1e-6:
        x0=x1
        f0=func(x0,R,A,Z,Q,L)
        diff0=func_diff(x0,R,A,Z,Q,L)
        x1=x0-float(f0)/diff0
        f1=func(x1,R,A,Z,Q,L)
    return x0

#二分法(解决出现奇点的问题)r1
def dichotomie(a,b,R,A,Z,Q,L):
    eps=R*1e-6
    if b>=R:
        b=R
    b_9=0.9*b
    fb_9=func(b_9,R,A,Z,Q,L)
    fb=func(b,R,A,Z,Q,L)
    while fb_9*fb>0:
        b=b_9
        b_9=0.9*b
        fb_9=func(b_9,R,A,Z,Q,L)
        fb=func(b,R,A,Z,Q,L)
    a=b_9
    fa=func(a,R,A,Z,Q,L)
    ab_2=(a+b)/2.0
    fb=func(ab_2,R,A,Z,Q,L)
    if fa*fb>0:
        a=ab_2
        b=(a+b)/2.0
    else:
        b=ab_2
    while sp.Abs(b-a)>eps:
        fb=func(b,R,A,Z,Q,L)
        ab_2=(a+b)/2
        fab_2=func(ab_2,R,A,Z,Q,L)
        if fab_2==0:
            return ab_2
            break
        if fb*fab_2<0:
            a=ab_2
        else:
            b=ab_2
    return ab_2

#牛顿迭代法r3
def gunc(x0,R,A,Z,Q,L):
    g=total(x0,R,A,Z,Q,L)-Q
    return g
def gunc_diff(x0,R,A,Z,Q,L):
    x=sp.symbols("x")
    g=gunc(x,R,A,Z,Q,L)
    g_diff=sp.diff(g)
    diff=g_diff.evalf(subs = {x:x0})
    return diff
def newton_r3(x0,R,A,Z,Q,L):
    g0=gunc(x0,R,A,Z,Q,L)
    diff0=gunc_diff(x0,R,A,Z,Q,L)
    x1=x0-float(g0)/diff0
    g1=gunc(x1,R,A,Z,Q,L)
    while sp.Abs(g1-g0)>R*1e-6:
        x0=x1
        g0=gunc(x0,R,A,Z,Q,L)
        diff0=gunc_diff(x0,R,A,Z,Q,L)
        x1=x0-float(g0)/diff0
        g1=gunc(x1,R,A,Z,Q,L)
    return x0

#波尔——索墨菲量子化条件被积函数
def quantization(r,R,A,Z,Q,L):
    u=4*(A-4)/A
    b=u/(C.hbar**2/(C.e*1e6*1e-30*1.66e-27*2))
    #b=u/20.936
    V_tot=totalPotential(r,R,A,Z,Q,L)
    f=b*(Q-V_tot)
    f=sp.sqrt(f)
    return f

def value(A,Z,Q,L):
    #计算G值
    N=A-Z
    if N>126:
        G=22
    if 82<N<=126:
        G=20
    if N<=82:
        G=18
    #写出积分方程    
    sub=(G-L+1)*C.pi/2
    return sub

# 1.复化求积公式
def intervals(N,a,b):
    '''Upper limit : b ; Lower limit : a'''
    h=(b-a)/float(N)   #子区间长度h,N为子区间个数
    xList=[]
    for i in range(N):
        xList.append(a+i*h)
    #得到的列表没有积分上限值
    return xList,h
#复化辛普森公式
def simpson(a,b,R,A,Z,Q,L):
    '''The Compound Simpson Formula'''
    N=5
    xList,h=intervals(N,a,b)
    I=0
    for i in range(len(xList)):
        if i<len(xList)-1:
            I1_2=quantization((xList[i]+xList[i+1])/2.0,R,A,Z,Q,L)
            I+=quantization(xList[i],R,A,Z,Q,L)+4*I1_2\
               +quantization(xList[i+1],R,A,Z,Q,L)
        else:
            I1_2=quantization((xList[i]+b)/2.0,R,A,Z,Q,L)
            I+=quantization(xList[i],R,A,Z,Q,L)+4*I1_2\
               +quantization(b,R,A,Z,Q,L)
    return I*h/6.0

#黄金分割法最优化选取R
def hunc(R,A,Z,Q,L):
    r2=float(newton(R,R,A,Z,Q,L))
    r1=dichotomie(0.0,r2,R,A,Z,Q,L)
    int_f=simpson(r1,r2,R,A,Z,Q,L)
    return complex(int_f).real
def golden_section(x_low,x_high,eps,A,Z,Q,L):
    sub=value(A,Z,Q,L)
    x_range=x_high-x_low
    x_low_try=x_low+(1-0.618)*x_range
    x_high_try=x_low+0.618*x_range
    while x_high_try-x_low_try>eps:
        y_low_try=hunc(x_low_try,A,Z,Q,L)-sub
        y_high_try=hunc(x_high_try,A,Z,Q,L)-sub
        if sp.Abs(y_low_try)<sp.Abs(y_high_try):
            x_high=x_high_try
            x_range=x_high-x_low
            x_low_try=x_low+(1-0.618)*x_range
            x_high_try=x_low+0.618*x_range
        else:
            x_low=x_low_try
            x_range=x_high-x_low
            x_low_try=x_low+(1-0.618)*x_range
            x_high_try=x_low+0.618*x_range      
    return (x_high+x_low)/2.0

def get_R(A,Z,Q,L=0):
    x_low=5
    x_high=10
    eps=1e-10
    R=golden_section(x_low,x_high,eps,A,Z,Q,L)
    r2=newton(R,R,A,Z,Q,L)
    r1=dichotomie(0.0,r2,R,A,Z,Q,L) 
    r3=newton_r3(30,R,A,Z,Q,L)
    return r1,r2,r3,R

#获取半衰期
def K(r,R,A,Z,Q,L):
    u=4*(A-4)/A
    b=u/(C.hbar**2/(C.e*1e6*1e-30*1.66e-27*2))
    #b=u/20.936
    V_tot=totalPotential(r,R,A,Z,Q,L)
    k=b*sp.Abs(Q-V_tot)
    k=sp.sqrt(k)
    return k
def K_rec(r,R,A,Z,Q,L):
    k=K(r,R,A,Z,Q,L)
    k=1.0/(2*k)
    return k

def K_r3(r,R,A,Z,Q,L):
    u=4*(A-4)/A
    b=u/(C.hbar**2/(C.e*1e6*1e-30*1.66e-27*2))
    #b=u/20.936
    V_tot=total(r,R,A,Z,Q,L)
    k=b*sp.Abs(Q-V_tot)
    k=sp.sqrt(k)
    return k

#复化辛普森公式 get_F
def simpson_rec(a,b,R,A,Z,Q,L):
    '''The Compound Simpson Formula'''
    N=5
    xList,h=intervals(N,a,b)
    I=0
    for i in range(len(xList)):
        if i !=0:
            if i<len(xList)-1:
                I1_2=K_rec((xList[i]+xList[i+1])/2.0,R,A,Z,Q,L)
                I+=K_rec(xList[i],R,A,Z,Q,L)+4*I1_2\
                +K_rec(xList[i+1],R,A,Z,Q,L)
            else:
                I1_2=K_rec((xList[i]+b)/2.0,R,A,Z,Q,L)
                I+=K_rec(xList[i],R,A,Z,Q,L)+4*I1_2
    return I*h/6.0
#修正因子
def get_F(r1,r2,R,A,Z,Q,L=0):
    int_12=simpson_rec(r1,r2,R,A,Z,Q,L)
    F=1.0/int_12
    return F
#形成几率
def get_P(A,Z,Q,L=0):
    #母核中子数
    N=A-Z
    if N%2==0 and Z%2==0:
        pee=1
        return pee
    if N%2!=0 and Z%2!=0:
        poo=0.35
        return poo
    else:
        poe=0.6
        return poe

#复化辛普森公式 get_L
def simpson_Rr3(a,b,R,A,Z,Q,L):
    '''The Compound Simpson Formula'''
    N=5
    xList,h=intervals(N,a,b)
    I=0
    for i in range(len(xList)):
        if i<len(xList)-1:
            I1_2=K_r3((xList[i]+xList[i+1])/2.0,R,A,Z,Q,L)
            I+=K_r3(xList[i],R,A,Z,Q,L)+4*I1_2+K_r3(xList[i+1],R,A,Z,Q,L)
        else:
            I1_2=K_r3((xList[i]+b)/2.0,R,A,Z,Q,L)
            I+=K_r3(xList[i],R,A,Z,Q,L)+4*I1_2+K_r3(b,R,A,Z,Q,L)
    return I*h/6.0
#束缚态衰变宽度
def get_L(r1,r2,r3,R,A,Z,Q,L=0):
    u=4*(A-4)/A
    b=u/(C.hbar**2/(C.e*1e6*1e-30*1.66e-27*2))
    #b=u/20.936
    int_k=simpson_Rr3(r2,r3,R,A,Z,Q,L)
    P=get_P(A,Z,Q,L)
    F=get_F(r1,r2,R,A,Z,Q,L)
    width=P*F*(1.0/(2*b))*math.exp(-2*int_k)
    return width
#获取半衰期
def get_T(r1,r2,r3,R,A,Z,Q,L=0):
    width=get_L(r1,r2,r3,R,A,Z,Q,L)
    T=(C.hbar/(C.e*1e6))*math.log(2)/width
    return T

#输出数据到文件‘T_acc_data.txt’
def data_AZQ():
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
    return AZQ

def output_T(L=0):
    AZQ=data_AZQ()
    file=open('T_acc_data.txt','a')
    file.writelines(['r1\t','r2\t','r3\t','R\t','T\n'])
    for i in range(len(AZQ)):
        A=AZQ[i][0]
        Z=AZQ[i][1]
        Q=AZQ[i][2]
        r1,r2,r3,R=get_R(A,Z,Q,L)
        T=get_T(r1,r2,r3,R,A,Z,Q,L)
        file.write(str(r1)+'\t'+str(r2)\
                   +'\t'+str(r3)+'\t'+str(R)+'\t'+str(T)+'\n')
    file.close()

#黄金分割法最优化选取Q
def iunc(Q,A,Z,L):
    r1,r2,r3,R=get_R(A,Z,Q,L)
    T=get_T(r1,r2,r3,R,A,Z,Q,L)
    return T
def golden_section_Q(x_low,x_high,eps,A,Z,L,T0):
    x_range=x_high-x_low
    x_low_try=x_low+(1-0.618)*x_range
    x_high_try=x_low+0.618*x_range
    while x_high_try-x_low_try>eps:
        y_low_try=iunc(x_low_try,A,Z,L)-T0
        y_high_try=iunc(x_high_try,A,Z,L)-T0
        if sp.Abs(y_low_try)<sp.Abs(y_high_try):
            x_high=x_high_try
            x_range=x_high-x_low
            x_low_try=x_low+(1-0.618)*x_range
            x_high_try=x_low+0.618*x_range
        else:
            x_low=x_low_try
            x_range=x_high-x_low
            x_low_try=x_low+(1-0.618)*x_range
            x_high_try=x_low+0.618*x_range      
    return (x_high+x_low)/2.0

def get_Q(T0,A,Z,L=0):
    x_low=5
    x_high=12
    eps=1e-3
    Q=golden_section_Q(x_low,x_high,eps,A,Z,L,T0)
    return Q

#输出数据到文件‘Q_acc_data.txt’
def data_AZT():
    file=open('AZT.txt','rU')
    lines=file.readlines()
    AZTList=[]
    AZT=[]
    for i in range(len(lines)):
        if i==0:
            continue
        AZTStr=lines[i]
        AZTList=list(AZTStr)
        AZT_list=[]
        AZT_str=''
        for j in range(len(AZTList)):
            if AZTList[j]!='\t' and AZTList[j]!='\n':
                AZT_str+=AZTList[j]
            if AZTList[j]=='\t' or AZTList[j]=='\n':
                AZT_list.append(float(AZT_str))
                AZT_str=''
        AZT.append(AZT_list)
    file.close()
    return AZT

def output_Q(L=0):
    AZT=data_AZT()
    file=open('Q_acc_data.txt','a')
    file.writelines(['r1\t','r2\t','r3\t','R\t','Q\n'])
    for i in range(len(AZT)):
        A=AZT[i][0]
        Z=AZT[i][1]
        T0=AZT[i][2]
        Q=get_Q(T0,A,Z,L)
        r1,r2,r3,R=get_R(A,Z,Q,L)
        file.write(str(r1)+'\t'+str(r2)\
                   +'\t'+str(r3)+'\t'+str(R)+'\t'+str(Q)+'\n')
    file.close()

def get_Mth(T0,A,Z,M_exp,M_dau,alpha_per,L=0):
    # alpha-dacay half-life = total half-life / alpha_percent
    T_alpha=T0/float(alpha_per*1e-2)   
    Q=get_Q(T_alpha,A,Z,L)
    M_alpha=2.424915
    M_mother=(Q+M_dau+M_alpha)
    file=open('M_th.txt','a')
    file.writelines(str(A)+'\t'+str(Z)+'\t'+\
                    str(M_exp)+'\t'+str(M_mother)+'\n')
    file.close()
    return Q,M_mother

#计算单个核的质量
def single_nucleus(L=0):
    """Calculate single nucleonic mass"""
    A=float(input("Mother nuclear mass number. : "))
    Z=float(input("Maternal proton number : "))
    T0=float(input("The total half-life of the mother nucleus : "))
    M_exp=float(input("The experimental date of mother quality : "))
    M_dau=float(input("The daughter nuclear quality : "))
    alpha_per=float(input("Alpha-dacay intensities : "))
    Q,M_th=get_Mth(T0,A,Z,M_exp,M_dau,alpha_per,L=0)
    return Q,M_th

#计算整条衰变链上的核的质量
#Mexp_dau对应质量数最小的核的质量
#衰变链数组元素必须排除质量数最小的核,然后根据质量数的大小，从小到大排列
def dacay_chain(T0List,AList,ZList,MexpList,Mexp_dau,alpha_perList,L=0):
    """Calculate each of nucleonic mass 
    in dacay_chain except the last daughter nucleus"""
    num=len(AList)
    QList=[]
    MthList=[0]*num
    if AList[0]<AList[num-1]:
        for i in range(num):
            if i==0:
                Q,M_th=get_Mth(T0List[i],AList[i],ZList[i],MexpList[i],\
                               Mexp_dau,alpha_perList[i],L=0)
                QList.append(Q)
                MthList[i]=M_th
            else:
                Q,M_th=get_Mth(T0List[i],AList[i],ZList[i],MexpList[i],\
                              MthList[i-1],alpha_perList[i],L=0)
                QList.append(Q)
                MthList[i]=M_th
        decay_list=[]
        nucleus_list=[]
        for j in range(len(QList)):
            nucleus_list.append((AList[j],ZList[j]))
            decay_list.append((QList[j],MthList[j]))
        decay_dict=dict(zip(nucleus_list,decay_list))
        return decay_dict
    else:
        print("Input As Wrong Order !")

#main program
T0List=[492290000000000.0,28401233400]
AList=[247,251]
ZList=[96,98]
MexpList=[65.533,74.135]
alpha_perList=[100,100]
Mexp_dau=57.7546
decay_dict=dacay_chain(T0List,AList,ZList,MexpList,Mexp_dau,alpha_perList,L=0)

