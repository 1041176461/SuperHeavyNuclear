#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:00:15 2018

@author: jiyuyang
"""

from matplotlib import pyplot as plt
from get_T import quantization
from get_T import newton
import numpy as np
import sympy as sp
#衰变能Q
Q=10.774
#轨道角动量
L=0
#母核质子数
Z=116
#母核质量数
A=292
u=(A-4)/A
V0=162.3
fig,ax=plt.subplots()
R=7.8
r2=float(newton(R,R))
r1=float(sp.sqrt(20.936/(4*u*(Q+V0-2.1584*2*(Z-2)/R))))
x=np.linspace(r1,r2,10)
fList=[]
for i in range(len(x)):
    fList.append(sp.Abs(quantization(x[i],R)))
plt.plot(x,fList)
plt.show()