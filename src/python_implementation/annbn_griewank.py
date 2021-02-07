
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from sklearn import linear_model
import time
from numpy import random,linalg,corrcoef,ones,float32,float64,c_,exp,log,zeros,mean

t=np.atleast_2d 
r=np.repeat 
plot=plt.plot
sort=np.sort
fit=linear_model.LinearRegression().fit
cor=np.corrcoef
scatter=plt.scatter

obs=10_000
vars=100
x=20*(random.rand(obs,vars).astype(np.float32)-1/2)
y=np.zeros(obs).astype(np.float32)


def func(x):
    # return sum(x**2)
    return 1 + (1/4000)*sum((x+7)**2) - np.prod(np.cos((x+7) / range(1, len(x)+1, 1)))
v1=20*(np.random.rand(1000,2)-1/2)
ov1=np.zeros(1000)
for i in range(1000):
    ov1[i]=func(v1[i,:])
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(v1[:,0],v1[:,1],ov1, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
for i in range(obs):
    y[i]=func(x[i,:])

mi=min(y);y-=mi;ma=max(y);y/=ma;y*=0.98;y+=0.01
x=np.concatenate((x,x**2,x**3),axis=1)


#test
obst=2*obs
xt=20*(random.rand(obst,vars).astype(np.float32)-1/2)
yt=np.zeros(obst).astype(np.float32)

for i in range(obst):
    yt[i]=func(xt[i,:])
    
yt-=mi;yt/=ma;yt*=0.98;yt+=0.01
xt=np.concatenate((xt,xt**2,xt**3),axis=1)


vars=x.shape[1]

def sigm1(x): return log(exp(x)+1)
def isigm1(x): return log(exp(x)-1)
start1 = time.time()
neurons=1000
xx=c_[ones((obs,1),float32), x]
yy=isigm1(y)
ikeep=int(1.2*round(obs/neurons))
w=zeros((vars+1,neurons),float32)
for i in range(neurons):
    ira=random.randint(0, obs, ikeep)
    w[:,i],res1,rank1,s1=linalg.lstsq(xx[ira,:],yy[ira],rcond=-1)


end1 = time.time()
print("wwwww",end1 - start1)

start2 = time.time()
layer1=sigm1(xx@w)
end2 = time.time()
print("sigm1(xx@a_all)",end2 - start2)

start = time.time()
v,res1,rank1,s1=np.linalg.lstsq(layer1,y,rcond=-1)
end = time.time()


predi=layer1@v
scatter(y,predi)
co=cor(predi,y)[0,1]
mo=mean(abs(predi-y))
print("LinearRegression",co," ",mo," ",end - start)


# TEST
xxt=c_[ones((obst,1),float32), xt]
layer1t=sigm1(xxt@w)
predt=layer1t@v
scatter(yt,predt)
cot=cor(predt,yt)[0,1]
mot=mean(abs(predt-yt))
print("LinearRegression",cot," ",mot)



