
using Plots, Statistics, MLDatasets, LinearAlgebra, Clustering, Printf, Dates, Random, Debugger
if !(pwd() in LOAD_PATH) push!(LOAD_PATH, pwd()) end
# using ANNBN
# include("C:\\Dropbox\\julialangfiles\\ANNBN\\ANNBN.jl")

include("C:\\Dropbox\\julialangfiles\\ANNBN\\ANNBN.jl")
i_train=100
i_test=i_train-1
vars=1
xx_train=range(0.05,0.95,length=i_train)
yy_train=[zeros(50);ones(50)]
tol1=0.0000000001
ANNBN.isigm1(tol1)
ANNBN.isigm1(1.0-tol1)
mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1
scatter(xx_train,yy_train,label="input yy_train")
xx_test=(xx_train[2:end].+xx_train[1:end-1])./2
yy_test=(yy_train[2:end].+yy_train[1:end-1])./2





# neurons=Int64(floor(i_train/(vars+1)))
# neurons=4
# this automatically generates number of neurons which equals to the number of clusters (§3.1)
# inds_all,n_per_part,items_per_neuron=ANNBN.___clustering(neurons,xx_train,200)

inds_all=(1:i_train)[:,1]
# inds_all=randperm(i_train)
div1=20
neurons=Int64(i_train/div1)
items_per_neuron=div1*ones(Int64,neurons)   # (Int64(floor(i_train/(neurons))))
n_per_part=[0;cumsum(items_per_neuron)]

# using SpecialFunctions
# erfi(-3)
##### Train Sigmoid
# collect(hcat(a_all))
include("C:\\Dropbox\\julialangfiles\\ANNBN\\ANNBN.jl")

@time a_all,a_layer1,layer1=ANNBN.train_layer_1_sigmoid_fast(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train)
# i1=reduce(hcat, a_all).==[0;0]
# i2=i1[1,:].+i1[2,:]
# sum(i2.==2)
#
# @time ll1=layer1'*layer1
# @time det(ll1)
# @time ll1i=inv(ll1)
# @time a_layer1=ll1i*layer1'*y
# plot(a_all)
# det(layer1'*layer1)
predl1,layer1_train=ANNBN.predict_new(a_all,a_layer1,xx_train,i_train,neurons)
predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,xx_test,i_test,neurons)
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))
scatter!(xx_train,predl1,label=@sprintf("ANNBN-%1d-neurons",neurons))
scatter!(xx_test,predl1_test,label=@sprintf("ANNBN-%1d-neurons",neurons))
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).>0.01;maetr=sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;
maete=1.0-sum(i2)/length(i2)

# plot(ANNBN.isigm1.(yy_train))

@time bb=[ones(i_train) xx_train]\ANNBN.isigm1.(yy_train)
# # pinv([ones(i_train) xx_train])*[ones(i_train) xx_train]
# X=[ones(i_train) xx_train]
# Xt=[ones(i_train) xx_train]'
# XtX=Xt*X
# @time XtX_i=inv(XtX)
# det(XtX)
# @time bb=XtX_i*Xt*ANNBN.isigm1.(yy_train)
# # bb=[-500;1000]
predl1=ANNBN.sigm1.([ones(i_train) xx_train]*bb);
maetr=mean(abs.(yy_train-predl1))
# i1=predl1.<=0.5;predl1[i1].=0.0001;i1=predl1.>0.5;predl1[i1].=0.9998;i2=abs.(predl1.-yy_train).<0.01;maetr=sum(i2)/length(i2)
predl1_test=ANNBN.sigm1.([ones(i_test) xx_test]*bb);
maete=mean(abs.(yy_test-predl1_test))
# i1=predl1_test.<=0.5;predl1_test[i1].=0.0001;i1=predl1_test.>0.5;predl1_test[i1].=0.9998;i2=abs.(predl1_test.-yy_test).<0.01;maete=1.0-sum(i2)/length(i2)
scatter!(xx_train,predl1,label="Logistic",legend=false)
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).>0.01;maetr=sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;
maete=1.0-sum(i2)/length(i2)




xx_train=collect(xx_train)[:,:];xx_test=collect(xx_test)[:,:]
cc1=100; @time a_all,a_layer1,layer1=ANNBN.train_layer_1_rbf(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train,cc1);
predl1=[layer1 ones(i_train)]*a_layer1
maetr=mean(abs.(yy_train-predl1))
maximum(abs.(yy_train-predl1))
i1=predl1.<=0.5;predl1[i1].=0.0001;i1=predl1.>0.5;predl1[i1].=0.9998;i2=abs.(predl1.-yy_train).<0.01;maetr=sum(i2)/length(i2)
@time predl1_test,layer1_test=ANNBN.predict_new_rbf(a_all,a_layer1,xx_test,i_test,neurons,n_per_part,inds_all,xx_train,vars,cc1);
maete=mean(abs.(yy_test-predl1_test))
maximum(abs.(yy_test-predl1_test))
i1=predl1_test.<=0.5;predl1_test[i1].=0.0001;i1=predl1_test.>0.5;predl1_test[i1].=0.9998;i2=abs.(predl1_test.-yy_test).<0.01;maete=1.0-sum(i2)/length(i2)
