using Flux
using Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated

include("C:\\Dropbox\\julialangfiles\\ANNBN\\ANNBN.jl")
using Plots, Statistics, MLDatasets, LinearAlgebra, Clustering, Printf, Dates, Random, MLDatasets, Plots, Printf, Debugger
train_x, yy_train_all = MLDatasets.MNIST.traindata()
test_x,  yy_test_all  = MLDatasets.MNIST.testdata()
xx_train=MLDatasets.MNIST.convert2features(MLDatasets.MNIST.traintensor())'
xx_test=MLDatasets.MNIST.convert2features(test_x)'
xx_train=convert(Array{Float64,2},xx_train)
xx_test=convert(Array{Float64,2},xx_test)
yy_train_all=convert(Array{Float64,1},yy_train_all)
yy_test_all=convert(Array{Float64,1},yy_test_all)
i_train=size(xx_train,1)
i_test=size(xx_test,1)
vars=size(xx_train,2)



tol1=0.0
num_to_test=0
yy_train=copy(yy_train_all)
i1=abs.(yy_train.-num_to_test).<0.01;i2=abs.(yy_train.-num_to_test).>0.01;yy_train[i1].=1;;yy_train[i2].=0;
mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1

yy_test=copy(yy_test_all)
i1=abs.(yy_test.-num_to_test).<0.01;i2=abs.(yy_test.-num_to_test).>0.01;yy_test[i1].=1;yy_test[i2].=0;
yy_test.-=mi;yy_test./=ma;yy_test.*=1.0-2tol1;yy_test.+=tol1


# neurons=Int64(floor(i_train/(vars+1)))
# neurons=100
# inds_all,n_per_part=ann_by_parts.___clustering(neurons,xx_train,200)
# x1=copy(xx_train[inds_all[n_per_part[1]+1:n_per_part[1+1]],:])
# obs1=size(x1,1)
# phi1=ANNBN.calc_phi_deriv(x1,x1,obs1,obs1,cc1,vars,rand(1:vars))
# phi_deriv=ANNBN.calc_phi(x1,x1,obs1,obs1,cc1,vars)
# aa=[phi1;phi_deriv]\[yy_train[inds_all[n_per_part[1]+1:n_per_part[1+1]]];zeros(obs1)]
# mean(abs.([phi1;phi_deriv]*aa-[yy_train[inds_all[n_per_part[1]+1:n_per_part[1+1]]];zeros(obs1)]))


# i_train=10000
include("C:\\Dropbox\\julialangfiles\\ANNBN\\ANNBN.jl")
inds_all=1:i_train;neurons=10000;items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
cc1=1.0
a_all,itmps=ANNBN.train_layer_1_rbf_deriv(neurons,vars,i_train,n_per_part,inds_all,xx_train[1:i_train,:],yy_train[1:i_train],cc1)
# phi_tr=ANNBN.calc_phi(xx_train[1:i_train,:],xx_train[1:i_train,:],i_train,i_train,cc1,vars)
layer1=Array{Float64}(undef,i_train,0)
@time for i=1:length(a_all)
    global layer1,inds_all,n_per_part,a_all,phi_tr
    # layer1=[layer1 phi_tr[:,inds_all[n_per_part[i]+1:n_per_part[i+1]]]*a_all[i]]
    layer1=[layer1 phi_tr[:,itmps[i]]*a_all[i]]
    println(i)
end
# @time a_layer1=inv([layer1 ones(i_train)]'*[layer1 ones(i_train)])*[layer1 ones(i_train)]'*yy_train[1:i_train]
@time a_layer1=[layer1 ones(i_train)]\yy_train[1:i_train]
predl1=[layer1 ones(i_train)]*a_layer1
maetr=mean(abs.(yy_train[1:i_train]-predl1))
maximum(abs.(yy_train[1:i_train]-predl1))


# phi_te=ANNBN.calc_phi(xx_train,xx_test,i_train,i_test,cc1,vars)
layer1_test=Array{Float64}(undef,i_test,0)
for i=1:length(a_all)
    global layer1_test,inds_all,n_per_part,a_all,phi_te
    # layer1_test=[layer1_test phi_te[:,inds_all[n_per_part[i]+1:n_per_part[i+1]]]*a_all[i]]
    layer1_test=[layer1_test phi_te[:,itmps[i]]*a_all[i]]
end
predl1_test=[layer1_test ones(i_test)]*a_layer1
maete=mean(abs.(yy_test-predl1_test))
maximum(abs.(yy_test-predl1_test))


i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train[1:i_train]).<0.01;maetr=100sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;
maete=100sum(i2)/length(i2)


using JLD2, JSON
@save "phis.jld2" phi_tr phi_te
@load "phis.jld2" phi_tr phi_te


##### APPROXIMATION OF DERIVATIVES
# first variable
yy_train_deriv=-3xx_train[:,1].^2+10xx_train[:,1]+5xx_train[:,2]
yy_train_deriv_deriv=-6xx_train[:,1]+10ones(i_train)
yy_test_deriv=-3xx_test[:,1].^2+10xx_test[:,1]+5xx_test[:,2]
yy_test_deriv_deriv=-6xx_test[:,1]+10ones(i_test)

predl1_deriv=ann_by_parts.predict_new_rbf_deriv(a_all,a_layer1,xx_train,i_train,neurons,
            n_per_part,inds_all,xx_train,vars,cc1,1)
mean(abs.(predl1_deriv-yy_train_deriv))
predl1_deriv_deriv=ann_by_parts.predict_new_rbf_deriv_deriv(a_all,a_layer1,xx_train,i_train,neurons,
            n_per_part,inds_all,xx_train,vars,cc1,1)
mean(abs.(predl1_deriv_deriv-yy_train_deriv_deriv))
predl1_test_deriv=ann_by_parts.predict_new_rbf_deriv(a_all,a_layer1,xx_test,i_test,neurons,
            n_per_part,inds_all,xx_train,vars,cc1,1)
mean(abs.(predl1_test_deriv-yy_test_deriv))
predl1_test_deriv_deriv=ann_by_parts.predict_new_rbf_deriv_deriv(a_all,a_layer1,xx_test,i_test,neurons,
            n_per_part,inds_all,xx_train,vars,cc1,1)
mean(abs.(predl1_test_deriv_deriv-yy_test_deriv_deriv))



# second variable
yy_train_deriv=5xx_train[:,1]
yy_train_deriv_deriv=zeros(i_train)
yy_test_deriv=5xx_test[:,1]
yy_test_deriv_deriv=zeros(i_test)

predl1_deriv=ann_by_parts.predict_new_rbf_deriv(a_all,a_layer1,xx_train,i_train,neurons,
            n_per_part,inds_all,xx_train,vars,cc1,2)
mean(abs.(predl1_deriv-yy_train_deriv))
predl1_deriv_deriv=ann_by_parts.predict_new_rbf_deriv_deriv(a_all,a_layer1,xx_train,i_train,neurons,
            n_per_part,inds_all,xx_train,vars,cc1,2)
mean(abs.(predl1_deriv_deriv-yy_train_deriv_deriv))

predl1_test_deriv=ann_by_parts.predict_new_rbf_deriv(a_all,a_layer1,xx_test,i_test,neurons,
            n_per_part,inds_all,xx_train,vars,cc1,2)
mean(abs.(predl1_test_deriv-yy_test_deriv))
predl1_test_deriv_deriv=ann_by_parts.predict_new_rbf_deriv_deriv(a_all,a_layer1,xx_test,i_test,neurons,
            n_per_part,inds_all,xx_train,vars,cc1,2)
mean(abs.(predl1_test_deriv_deriv-yy_test_deriv_deriv))
