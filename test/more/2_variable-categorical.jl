
using Plots, Statistics, MLDatasets, LinearAlgebra, Clustering, Printf, Dates, Random, Debugger
# if !(pwd() in LOAD_PATH) push!(LOAD_PATH, pwd()) end
# using ANNBN
include("C:\\Dropbox\\julialangfiles\\ANNBN\\ANNBN.jl")

tol1=0.01;ANNBN.isigm1(tol1);ANNBN.isigm1(1.0-tol1)
i_train=1000;i_test=i_train;vars=2
xx_train=rand(i_train,2);yy_train=zeros(i_train)
i1=xx_train[:,1].<0.5;i2=xx_train[:,2].<0.5;i12=(i1+i2).==2;yy_train[i12].=1.0
mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1
scatter3d(xx_train[:,1],xx_train[:,2],yy_train[:,1])
xx_test=rand(i_test,2);yy_test=zeros(i_test)
i1=xx_test[:,1].<0.5;i2=xx_test[:,2].<0.5;i12=(i1+i2).==2;yy_test[i12].=1.0
yy_test.-=mi;yy_test./=ma;yy_test.*=1.0-2tol1;yy_test.+=tol1
scatter3d!(xx_test[:,1],xx_test[:,2],yy_test)


neurons=Int64(floor(i_train/(vars+1)))
# neurons=100
inds_all,n_per_part,items_per_neuron=ANNBN.___clustering(neurons,xx_train,200)
# inds_all,n_per_part,items_per_neuron=ANNBN.___clustering(neurons,yy_train,200)
# inds_all,n_per_part,items_per_neuron=ANNBN.___clustering(neurons,[xx_train yy_train],200)
# plot(yy_train[inds_all])
# minimum(items_per_neuron)
inds_all=(1:i_train)[:,1]
# inds_all=randperm(i_train)
neurons=Int64(i_train/10)
items_per_neuron=10*ones(Int64,neurons)   # (Int64(floor(i_train/(neurons))))
n_per_part=[0;cumsum(items_per_neuron)]

##### Train Sigmoid
a_all,a_layer1,layer1=ANNBN.train_layer_1_sigmoid_fast(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train)
predl1=[layer1 ones(i_train)]*a_layer1
predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,xx_test,i_test,neurons)
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))
scatter3d!(xx_train[:,1],xx_train[:,2],predl1)
scatter3d!(xx_test[:,1],xx_test[:,2],predl1_test)
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).>0.01;maetr=sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;
maete=1.0-sum(i2)/length(i2)

@time bb=[ones(i_train) xx_train]\ANNBN.isigm1.(yy_train)
predl1=ANNBN.sigm1.([ones(i_train) xx_train]*bb);
maetr=mean(abs.(yy_train-predl1))
# i1=predl1.<=0.5;predl1[i1].=0.0001;i1=predl1.>0.5;predl1[i1].=0.9998;i2=abs.(predl1.-yy_train).<0.01;maetr=sum(i2)/length(i2)
predl1_test=ANNBN.sigm1.([ones(i_test) xx_test]*bb);
maete=mean(abs.(yy_test-predl1_test))
# i1=predl1_test.<=0.5;predl1_test[i1].=0.0001;i1=predl1_test.>0.5;predl1_test[i1].=0.9998;i2=abs.(predl1_test.-yy_test).<0.01;maete=1.0-sum(i2)/length(i2)
scatter3d!(xx_train[:,1],xx_train[:,2],predl1)



# 2nd layer
vars_f=copy(neurons)
xx_train_f=copy(layer1)
neurons_f=copy(neurons)
inds_all_f,n_per_part_f,items_per_neuron_f=ANNBN.___clustering(neurons_f,xx_train_f,200)
# inds_all_f=copy(inds_all)
# n_per_part_f=copy(n_per_part)

@time a_all_f,a_layer1_f,layer1_f=ANNBN.train_layer_1_sigmoid(neurons_f,vars_f,i_train,n_per_part_f,inds_all_f,xx_train_f,yy_train);
predl1_f=[layer1_f ones(i_train)]*a_layer1_f
maetr=mean(abs.(yy_train-predl1_f))
maximum(abs.(yy_train-predl1_f))
# i1=predl1_f.<=0.5;predl1_f[i1].=0.0001;i1=predl1_f.>0.5;predl1_f[i1].=0.9998;i2=abs.(predl1_f.-yy_train).>0.01;maetr=sum(i2)/length(i2)

xx_test_f=copy(layer1_test)
@time predl1_test_f,layer1_test_f=ANNBN.predict_new(a_all_f,a_layer1_f,xx_test_f,i_test,neurons_f);
predl1_test_f=[layer1_test_f ones(i_test)]*a_layer1_f
maete=mean(abs.(yy_test-predl1_test_f))
maximum(abs.(yy_test-predl1_test_f))
# i1=predl1_test_f.<=0.5;predl1_test_f[i1].=0.0001;i1=predl1_test_f.>0.5;predl1_test_f[i1].=0.9998;i2=abs.(predl1_test_f.-yy_test).<0.01;maete=1.0-sum(i2)/length(i2)
# ir=randperm(size(xx_test,1))[1:50]; plot(yy_test[ir]); plot!(predl1_test[ir])
i1=predl1_f.<=0.5;predl1_f[i1].=0.1;i1=predl1_f.>0.5;predl1_f[i1].=0.9;i2=abs.(predl1_f.-yy_train).>0.01;maetr=sum(i2)/length(i2)
i1=predl1_test_f.<=0.5;predl1_test_f[i1].=0.1;i1=predl1_test_f.>0.5;predl1_test_f[i1].=0.9;i2=abs.(predl1_test_f.-yy_test).<0.01;maete=1.0-sum(i2)/length(i2)




# 3nd layer


for i=1:20
    global neurons_f,layer1_f,layer1_test_f
    vars_f=copy(neurons_f)
    xx_train_f=copy(layer1_f)
    neurons_f=copy(neurons_f)
    inds_all_f,n_per_part_f,items_per_neuron_f=ANNBN.___clustering(neurons_f,xx_train_f,200)

    @time a_all_f,a_layer1_f,layer1_f=ANNBN.train_layer_1_sigmoid(neurons_f,vars_f,i_train,n_per_part_f,inds_all_f,xx_train_f,yy_train);
    predl1_f=[layer1_f ones(i_train)]*a_layer1_f
    maetr=mean(abs.(yy_train-predl1_f))
    maximum(abs.(yy_train-predl1_f))

    xx_test_f=copy(layer1_test_f)
    @time predl1_test_f,layer1_test_f=ANNBN.predict_new(a_all_f,a_layer1_f,xx_test_f,i_test,neurons_f);
    # predl1_test_f=[layer1_test_f ones(i_test)]*a_layer1_f
    maete=mean(abs.(yy_test-predl1_test_f))
    maximum(abs.(yy_test-predl1_test_f))

    i1=predl1_f.<=0.5;predl1_f[i1].=0.1;i1=predl1_f.>0.5;predl1_f[i1].=0.9;i2=abs.(predl1_f.-yy_train).>0.01;maetr=sum(i2)/length(i2)
    i1=predl1_test_f.<=0.5;predl1_test_f[i1].=0.1;i1=predl1_test_f.>0.5;predl1_test_f[i1].=0.9;i2=abs.(predl1_test_f.-yy_test).<0.01;maete=1.0-sum(i2)/length(i2)
    println(i,"   >>>>>>>>>>>>>>>>>>>>>>maetr=",maetr,"    >>>>>>>>>>>>>>>>>>>>>>>>>    maete=",maete)
end
# ir=randperm(size(xx_test,1))[1:50]; plot(yy_test[ir]); plot!(predl1_test[ir])
# scatter3d!(xx_train[:,1],xx_train[:,2],predl1_f)
# scatter3d!(xx_test[:,1],xx_test[:,2],predl1_test_f)










































using Plots, Statistics, MLDatasets, LinearAlgebra, Clustering, Printf, Dates, Random, MLDatasets, Plots, Printf

train_x, yy_train_all = MLDatasets.MNIST.traindata()
test_x,  yy_test_all  = MLDatasets.MNIST.testdata()
xx_train=MNIST.convert2features(MNIST.traintensor())'
xx_test=MNIST.convert2features(test_x)'
xx_train=convert(Array{Float64,2},xx_train)
xx_test=convert(Array{Float64,2},xx_test)
i_train=size(xx_train,1)
i_test=size(xx_test,1)
vars=size(xx_train,2)
num_to_test=3
yy_train=copy(yy_train_all)
i1=abs.(yy_train.-num_to_test).<0.01;i2=abs.(yy_train.-num_to_test).>0.01;yy_train[i1].=1;;yy_train[i2].=0;
yy_train=0.8*(convert(Array{Float64},yy_train));yy_train.+=0.1

yy_test=copy(yy_test_all)
i1=abs.(yy_test.-num_to_test).<0.01;i2=abs.(yy_test.-num_to_test).>0.01;yy_test[i1].=1;yy_test[i2].=0;
yy_test=0.8*(convert(Array{Float64},yy_test));yy_test.+=0.1














bb=[ones(i_train) xx_train]\ANNBN.isigm1.(yy_train)
bb=[-500;500;500]
predl1=ANNBN.sigm1.([ones(i_train) xx_train]*bb);
maetr=mean(abs.(yy_train-predl1))
# i1=predl1.<=0.5;predl1[i1].=0.0001;i1=predl1.>0.5;predl1[i1].=0.9998;i2=abs.(predl1.-yy_train).<0.01;maetr=sum(i2)/length(i2)
predl1_test=ANNBN.sigm1.([ones(i_test) xx_test]*bb);
maete=mean(abs.(yy_test-predl1_test))
# i1=predl1_test.<=0.5;predl1_test[i1].=0.0001;i1=predl1_test.>0.5;predl1_test[i1].=0.9998;i2=abs.(predl1_test.-yy_test).<0.01;maete=1.0-sum(i2)/length(i2)
scatter3d!(xx_train[:,1],xx_train[:,2],predl1)
scatter3d!(xx_test[:,1],xx_test[:,2],predl1_test)




# xx_train=collect(xx_train)[:,:];xx_test=collect(xx_test)[:,:]
cc1=100.0; @time a_all,a_layer1,layer1=ANNBN.train_layer_1_rbf(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train,cc1);
predl1=[layer1 ones(i_train)]*a_layer1
maetr=mean(abs.(yy_train-predl1))
maximum(abs.(yy_train-predl1))
# i1=predl1.<=0.5;predl1[i1].=0.0001;i1=predl1.>0.5;predl1[i1].=0.9998;i2=abs.(predl1.-yy_train).<0.01;maetr=sum(i2)/length(i2)
@time predl1_test,layer1_test=ANNBN.predict_new_rbf(a_all,a_layer1,xx_test,i_test,neurons,n_per_part,inds_all,xx_train,vars,cc1);
maete=mean(abs.(yy_test-predl1_test))
maximum(abs.(yy_test-predl1_test))
# i1=predl1_test.<=0.5;predl1_test[i1].=0.0001;i1=predl1_test.>0.5;predl1_test[i1].=0.9998;i2=abs.(predl1_test.-yy_test).<0.01;maete=1.0-sum(i2)/length(i2)
scatter3d!(xx_train[:,1],xx_train[:,2],predl1)
scatter3d!(xx_test[:,1],xx_test[:,2],predl1_test)







using DecisionTree, Statistics
num_trees=100;
regr_3 = RandomForestClassifier(n_trees=num_trees);
tmp_xx_train=copy(xx_train);tmp_yy_train=copy(yy_train);
tmp_xx_test=copy(xx_test);tmp_yy_test=copy(yy_test);
DecisionTree.fit!(regr_3, tmp_xx_train, tmp_yy_train[:,1]);
predl1 = DecisionTree.predict(regr_3, tmp_xx_train);
predl1_test = DecisionTree.predict(regr_3, tmp_xx_test);
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))
i1=predl1.<=0.5;predl1[i1].=0.0001;i1=predl1.>0.5;predl1[i1].=0.9998;i2=abs.(predl1.-yy_train).<0.01;maetr=sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=0.0001;i1=predl1_test.>0.5;predl1_test[i1].=0.9998;i2=abs.(predl1_test.-yy_test).<0.01;maete=1.0-sum(i2)/length(i2)
scatter3d!(xx_train[:,1],xx_train[:,2],predl1)
scatter3d!(xx_test[:,1],xx_test[:,2],predl1_test)


using PyCall,ScikitLearn
@sk_import ensemble: AdaBoostClassifier
sci_model = fit!(AdaBoostClassifier(n_estimators=100),xx_train,string.(yy_train))
predl1=parse.(Float64,predict(sci_model,xx_train))
predl1_test=parse.(Float64,predict(sci_model,xx_test))
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))
scatter3d!(xx_train[:,1],xx_train[:,2],predl1)
scatter3d!(xx_test[:,1],xx_test[:,2],predl1_test)

using XGBoost
num_round = 100
bst = xgboost(copy(xx_train), num_round, label = copy(yy_train), eta = 1, max_depth = 7)
predl1 = XGBoost.predict(bst, copy(xx_train))
predl1_test = XGBoost.predict(bst, copy(xx_test))
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))

i1=predl1.<=0.5;predl1[i1].=0.0001;i1=predl1.>0.5;predl1[i1].=0.9998;i2=abs.(predl1.-yy_train).<0.01;maetr=sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=0.0001;i1=predl1_test.>0.5;predl1_test[i1].=0.9998;i2=abs.(predl1_test.-yy_test).<0.01;maete=1.0-sum(i2)/length(i2)

scatter3d!(xx_train[:,1],xx_train[:,2],predl1)
scatter3d!(xx_test[:,1],xx_test[:,2],predl1_test)
