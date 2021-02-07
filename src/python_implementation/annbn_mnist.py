import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.special import logit,expit
import time
from numpy import random,linalg,corrcoef,ones,float32,float64,c_,exp,log
from numpy import zeros,mean,where,array,unique,equal
import torch
import torchvision
import torchvision.transforms as transforms


from mnist import MNIST
mndata = MNIST('/mnist/')
images, labels = mndata.load_training()
y=np.array(labels).astype(float32)
x=np.array(images).astype(float32)
i0=y==0
in0=y!=0
y[i0]=1.0
y[in0]=0.0
obs=x.shape[0]
vars=x.shape[1]
mi=min(y);y-=mi;ma=max(y);y/=ma;y*=0.98;y+=0.01


t1=time.time();neurons=1000;xx=c_[ones((obs,1),float32), x];yy=logit(y);
ikeep=round(1.2*obs/neurons)
w=zeros((vars+1,neurons),float32)
t1=time.time()
for i in range(neurons):
    ira=random.randint(0, obs, ikeep)
    w[:,i],res1,rank1,s1=linalg.lstsq(xx[ira,:],yy[ira],rcond=-1)
    print("w-",i)
t2=time.time()
layer1=expit(xx @ w)
t2-t1

v,res1,rank1,s1=linalg.lstsq(layer1,y,rcond=-1)
pred=layer1@v
plt.scatter(y,pred)
co=np.corrcoef(pred,y)[0,1]
mo=mean(abs(pred-y))
mse=mean((pred-y)**2)
print("LinearRegression",co," ",mo," ",mse)
pred=np.round(pred)
y=np.round(y)
iok=y==1
iok2=y[iok]==pred[iok]
sum(iok2)/sum(iok)
ii=np.round(pred)==np.round(y)
100*sum(ii)/len(y)


# TEST
images, labels = mndata.load_testing()
yt=np.array(labels).astype(float32)
xt=np.array(images).astype(float32)
i0=yt==0
in0=yt!=0
yt[i0]=1.0
yt[in0]=0.0
obst=xt.shape[0]
yt-=mi;yt/=ma;yt*=0.98;yt+=0.01
xxt=c_[ones((obst,1),float32), xt]
layer1t=expit(xxt@w)
predt=layer1t@v
co=np.corrcoef(predt,yt)[0,1]
mo=mean(abs(predt-yt))
mse=mean((predt-yt)**2)
print("LinearRegression",co," ",mo," ",mse)
ii=np.round(predt)==np.round(yt)
100*sum(ii)/len(yt)


00000

