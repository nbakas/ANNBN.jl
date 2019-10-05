
# With this file you may reproduce the results for cases 1, 2, 3, 4 in Table 2,
# by changing the number of neurons (variable neurons below), and ϵ (tol1).
# The activation function and it's inverse may be changed in the main file ANNBN.jl
# The Random Forests and Gradient Boosting calls utilized in the manuscript are aplso appended bellow

# 1st Step: Read the MNIST Database
using Plots, Statistics, MLDatasets, LinearAlgebra, Clustering, Printf, Dates, Random, MLDatasets, Plots
using Printf, Debugger, IterativeSolvers
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

# Import ANNBN module
path1=realpath(dirname(@__FILE__)*"/../..")
include(string(path1,"/src/ANNBN.jl"))


# 2nd Step: Train for a single digit num_to_test. tol1=ϵ normalization in manuscript
num_to_test=0;tol1=0.02;
yy_train=copy(yy_train_all)
i1=abs.(yy_train.-num_to_test).<0.01;i2=abs.(yy_train.-num_to_test).>0.01;yy_train[i1].=1;;yy_train[i2].=0;
mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1
yy_test=copy(yy_test_all)
i1=abs.(yy_test.-num_to_test).<0.01;i2=abs.(yy_test.-num_to_test).>0.01;yy_test[i1].=1;yy_test[i2].=0;
yy_test.-=mi;yy_test./=ma;yy_test.*=1.0-2tol1;yy_test.+=tol1

neurons=1000; # For higher accuracy, increase the number of neurons. RAM demands will increase as well.
inds_all=1:i_train;items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
@time a_all,a_layer1,layer1,mat1=ANNBN.train_layer_1_sigmoid_fast(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train)
predl1=[layer1 ones(i_train)]*a_layer1
maetr=mean(abs.(yy_train-predl1))
predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,xx_test,i_test,neurons);
maete=mean(abs.(yy_test-predl1_test))
cctr=100mean(abs.(round.(predl1).-round.(yy_train)).<0.1)
ccte=100mean(abs.(round.(predl1_test).-round.(yy_test)).<0.1)



# 3rd Step: Train all digits
cctrs=Array{Float64}(undef,0)
cctes=Array{Float64}(undef,0)
dts=Array{Float64}(undef,0)
predl1_train_all=Array{Float64}(undef,i_train,0)
predl1_test_all=Array{Float64}(undef,i_test,0)
neurons=1000;tol1=0.02;
@time for num_to_test=0:9
    global yy_train_all,yy_test_all,predl1_train_all,predl1_test_all,neurons,tol1
    yy_train=copy(yy_train_all)
    i1=abs.(yy_train.-num_to_test).<0.01;i2=abs.(yy_train.-num_to_test).>0.01;yy_train[i1].=1;;yy_train[i2].=0;
    mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1
    yy_test=copy(yy_test_all)
    i1=abs.(yy_test.-num_to_test).<0.01;i2=abs.(yy_test.-num_to_test).>0.01;yy_test[i1].=1;yy_test[i2].=0;
    yy_test.-=mi;yy_test./=ma;yy_test.*=1.0-2tol1;yy_test.+=tol1
    inds_all=1:i_train;items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
    t1=now();a_all,a_layer1,layer1=ANNBN.train_layer_1_sigmoid_fast(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train)
    t2=now();dt=convert(Float64,Dates.value(t2-t1))/1000;push!(dts,dt)
    predl1=[layer1 ones(i_train)]*a_layer1; maetr=mean(abs.(yy_train-predl1))
    predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,xx_test,i_test,neurons); maete=mean(abs.(yy_test-predl1_test))

    predl1_train_all=[predl1_train_all predl1]; predl1_test_all=[predl1_test_all predl1_test]
    cctr=100mean(abs.(round.(predl1).-round.(yy_train)).<0.1)
    ccte=100mean(abs.(round.(predl1_test).-round.(yy_test)).<0.1)
    push!(cctrs,cctr);push!(cctes,ccte)
    println(num_to_test," ",ccte)
end
println(dts)
println(cctes)


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




# # # # # # # # # # # Random Forests
using DecisionTree
num_trees=261
regr_3 = RandomForestClassifier(n_trees=num_trees);
t1=now()
DecisionTree.fit!(regr_3, xx_train, yy_train);
t2=now()
convert(Float64,Dates.value(t2-t1))/1000
predl1 = DecisionTree.predict(regr_3, xx_train)
predl1_test = DecisionTree.predict(regr_3, xx_test)
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<0.01;maetr=100sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;
maete=100sum(i2)/length(i2)


maetrs=Array{Float64}(undef,0)
maetes=Array{Float64}(undef,0)
dts=Array{Float64}(undef,0)
@time for num_to_test=0:9
    global yy_train_all,yy_test_all

    yy_train=copy(yy_train_all)
    yy_test=copy(yy_test_all)
    tol1=0.0
    yy_train=copy(yy_train_all)
    i1=abs.(yy_train.-num_to_test).<0.01;i2=abs.(yy_train.-num_to_test).>0.01;yy_train[i1].=1;;yy_train[i2].=0;
    mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1
    yy_test=copy(yy_test_all)
    i1=abs.(yy_test.-num_to_test).<0.01;i2=abs.(yy_test.-num_to_test).>0.01;yy_test[i1].=1;yy_test[i2].=0;
    yy_test.-=mi;yy_test./=ma;yy_test.*=1.0-2tol1;yy_test.+=tol1

    num_trees=261
    regr_3 = RandomForestClassifier(n_trees=num_trees);
    t1=now()
    DecisionTree.fit!(regr_3, xx_train, yy_train);
    t2=now();dt=convert(Float64,Dates.value(t2-t1))/1000;push!(dts,dt)
    predl1 = DecisionTree.predict(regr_3, xx_train)
    predl1_test = DecisionTree.predict(regr_3, xx_test)

    i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<0.01;maetr=100sum(i2)/length(i2)
    i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;maete=100sum(i2)/length(i2)
    push!(maetrs,maetr)
    push!(maetes,maete)
    println(" > > > > > > > > > > > > >",num_to_test," ",maete," ",dt)
end
println(dts)
println(maetes)




# # # # # # # # # # # # # # #  XGBoost
using XGBoost
num_round=200
@time bst = xgboost(xx_train, num_round, label = yy_train, eta = 1, max_depth = 7, verbose=false)
predl1 = XGBoost.predict(bst, copy(xx_train))
predl1_test = XGBoost.predict(bst, copy(xx_test))
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<0.01;maetr=100sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;
maete=100sum(i2)/length(i2)


num_round = 200
maetrs=Array{Float64}(undef,0)
maetes=Array{Float64}(undef,0)
dts=Array{Float64}(undef,0)
@time for num_to_test=0:9
    global yy_train_all,yy_test_all,num_round

    yy_train=copy(yy_train_all)
    yy_test=copy(yy_test_all)
    tol1=0.0
    yy_train=copy(yy_train_all)
    i1=abs.(yy_train.-num_to_test).<0.01;i2=abs.(yy_train.-num_to_test).>0.01;yy_train[i1].=1;;yy_train[i2].=0;
    mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1
    yy_test=copy(yy_test_all)
    i1=abs.(yy_test.-num_to_test).<0.01;i2=abs.(yy_test.-num_to_test).>0.01;yy_test[i1].=1;yy_test[i2].=0;
    yy_test.-=mi;yy_test./=ma;yy_test.*=1.0-2tol1;yy_test.+=tol1
    t1=now()
    bst = xgboost(xx_train, num_round, label = yy_train, eta = 1, max_depth = 7)
    t2=now();dt=convert(Float64,Dates.value(t2-t1))/1000;push!(dts,dt)
    predl1 = XGBoost.predict(bst, xx_train)
    predl1_test = XGBoost.predict(bst, xx_test)

    i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<0.01;maetr=100sum(i2)/length(i2)
    i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;maete=100sum(i2)/length(i2)
    push!(maetrs,maetr)
    push!(maetes,maete)
    println(" > > > > > > > > > > > > >",num_to_test," ",maete," ",dt)
end
println(dts)
println(maetes)
