


using Plots, Statistics, MLDatasets, LinearAlgebra, Clustering, Printf, Dates, Random, MLDatasets, Plots
using Printf, Debugger, IterativeSolvers, CuArrays
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

include("C:\\Dropbox\\julialangfiles\\ANNBN\\ANNBN.jl")

maetrs=Array{Float64}(undef,0)
maetes=Array{Float64}(undef,0)
dts=Array{Float64}(undef,0)
predl1_train_all=Array{Float64}(undef,i_train,0)
predl1_test_all=Array{Float64}(undef,i_test,0)
@time for num_to_test=0:9
    global yy_train_all,yy_test_all,predl1_train_all,predl1_test_all
    yy_train=copy(yy_train_all)
    yy_test=copy(yy_test_all)
    tol1=0.02
    yy_train=copy(yy_train_all)
    i1=abs.(yy_train.-num_to_test).<0.01;i2=abs.(yy_train.-num_to_test).>0.01;yy_train[i1].=1;;yy_train[i2].=0;
    mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1
    yy_test=copy(yy_test_all)
    i1=abs.(yy_test.-num_to_test).<0.01;i2=abs.(yy_test.-num_to_test).>0.01;yy_test[i1].=1;yy_test[i2].=0;
    yy_test.-=mi;yy_test./=ma;yy_test.*=1.0-2tol1;yy_test.+=tol1
    inds_all=1:i_train;neurons=1000;items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
    t1=now()
    a_all,a_layer1,layer1,mat1=ANNBN.train_layer_1_sigmoid_fast(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train)
    yy_train=cat(yy_train,dims=[1,2]);yy_test=cat(yy_test,dims=[1,2])
    nodes=neurons.*ones(Int64,1,10)
    nodes=cat(nodes,dims=[1,2])
    n_weigths = vars*nodes[1]
    for i=2:length(nodes) n_weigths+=nodes[i]*nodes[i-1] end; n_weigths+=sum(nodes)
    for i=1:size(yy_train,2) n_weigths+=1+nodes[end] end; opti_ww=rand(n_weigths)
    opti_ww,layer_prev=ANNBN.initialize_deep_weights_per_leyer(neurons,vars,a_all,nodes,a_layer1,n_weigths,xx_train,yy_train,i_train,layer1,mat1)
    t2=now();dt=convert(Float64,Dates.value(t2-t1))/1000;push!(dts,dt)
    predl1=ANNBN.deep_nnm(opti_ww,xx_train,nodes,i_train,vars,size(yy_train,2))
    predl1_test=ANNBN.deep_nnm(opti_ww,xx_test,nodes,i_test,vars,size(yy_test,2))
    maetr=100mean(abs.(round.(predl1).-round.(yy_train)).<0.1)
    maete=100mean(abs.(round.(predl1_test).-round.(yy_test)).<0.1)


    predl1_train_all=[predl1_train_all predl1]
    predl1_test_all=[predl1_test_all predl1_test]

    push!(maetrs,maetr)
    push!(maetes,maete)
    println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",num_to_test," ",maete)
end
println(dts)
println(maetes)



yy_train_pred=zeros(i_train)
for i=1:i_train
    yy_train_pred[i]=sortperm(predl1_train_all[i,:],rev=true)[1]-1
end
CC=100-100mean(yy_train_all-yy_train_pred.>0.01)


yy_test_pred=zeros(i_test)
for i=1:i_test
    yy_test_pred[i]=sortperm(predl1_test_all[i,:],rev=true)[1]-1
end
CC=100-100mean(yy_test_all-yy_test_pred.>0.01)
