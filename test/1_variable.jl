
using Plots, Statistics, MLDatasets, LinearAlgebra, Clustering, Printf, Dates, Random, Debugger
path1=realpath(dirname(@__FILE__)*"/..")
include(string(path1,"/src/ANNBN.jl"))

# This is a simple example in 1 dimension. You may set the unknown function below,
# as well as the level of noise, and number of observations
ff(x)=0.3sin(exp(3x))+0.5; noise_multiplier=0.0; i_train=100

i_test=i_train-1
vars=1
xx_train=range(0.05,0.95,length=i_train)
yy_train_init=ff.(xx_train)
yy_train=yy_train_init .+ (rand(i_train).-1/2)*noise_multiplier
mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=0.8;yy_train.+=0.1
scatter(xx_train,yy_train,label="input yy_train",color=:black)
xx_test=(xx_train[2:end].+xx_train[1:end-1])./2
yy_test=ff.(xx_test)
yy_test.-=mi;yy_test./=ma;yy_test.*=0.8;yy_test.+=0.1
scatter!(xx_test,yy_test,label="initial yy_test, without noise",color=:black,markershape=:diamond,legend=:bottomleft)
xx_train=collect(xx_train)[:,:];xx_test=collect(xx_test)[:,:]

##### Clustering
neurons=Int64(floor(i_train/(vars+1))) # this automatically generates number of neurons which equals to the number of clusters (§3.1)
inds_all,n_per_part=ANNBN.___clustering(neurons,xx_train,200)

##### Train Sigmoid
a_all,a_layer1,layer1=ANNBN.train_layer_1_sigmoid(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train)
predl1,layer1_train=ANNBN.predict_new(a_all,a_layer1,xx_train,i_train,neurons)
predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,xx_test,i_test,neurons)
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))
scatter!(xx_test,predl1_test,label=@sprintf("ANNBN-%1d-neurons",neurons),color=:black,markershape=:rect)
# savefig("1D-noise.pdf")

##### Train RBF
# Select 1 Gaussian from calc_phi.jl
cc1=0.01
a_all,a_layer1,layer1=ANNBN.train_layer_1_rbf(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train,cc1)
predl1,layer1_train=ANNBN.predict_new_rbf(a_all,a_layer1,xx_train,i_train,neurons,n_per_part,inds_all,xx_train,vars,cc1)
predl1_test,layer1_test=ANNBN.predict_new_rbf(a_all,a_layer1,xx_test,i_test,neurons,n_per_part,inds_all,xx_train,vars,cc1)
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))
