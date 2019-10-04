using Plots, Statistics, MLDatasets, LinearAlgebra, Clustering, Printf, Dates, Random
path1=realpath(dirname(@__FILE__)*"/..")
include(string(path1,"/src/ANNBN.jl"))

function ff(x)
    xx=200x .- 100
    n = length(xx)
    1 + (1/4000)*sum(abs2, (xx)) - prod(cos.((xx) ./ sqrt.(1:n)))
end
noise_perc=1;vars=100;
i_train=1000; xx_train=0.8*rand(i_train,vars).+0.1
yy_train=zeros(i_train); for i=1:i_train yy_train[i]=ff(xx_train[i,:]) end
# plot(yy_train[1:100]);yy_train.+=noise_perc.*(1/2 .-rand(i_train));plot!(yy_train[1:100])
mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=0.8;yy_train.+=0.1
i_test=1000;xx_test=0.8*rand(i_test,vars).+0.1
yy_test=zeros(i_test); for i=1:i_test yy_test[i]=ff(xx_test[i,:]) end
# yy_test.+=noise_perc.*rand(i_test)
yy_test.-=mi;yy_test./=ma;yy_test.*=0.8;yy_test.+=0.1

##### ANNBN RBF
neurons=Int64(floor(i_train/(vars+1)))
neurons=100 # this automatically generates number of neurons== number of clusters (§3.1)
inds_all,n_per_part=ANNBN.___clustering(neurons,xx_train,200)
cc1=100.0
a_all,a_layer1,layer1=ANNBN.train_layer_1_rbf(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train,cc1)
predl1,layer1_train=ANNBN.predict_new_rbf(a_all,a_layer1,xx_train,i_train,neurons,n_per_part,inds_all,xx_train,vars,cc1)
predl1_test,layer1_test=ANNBN.predict_new_rbf(a_all,a_layer1,xx_test,i_test,neurons,n_per_part,inds_all,xx_train,vars,cc1)
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))
err=yy_test-predl1_test;ier=sortperm(err)
plot(err[ier],label="ANNBN",color=:black,linestyle=:dashdotdot,legend=:topleft,linewidth=3)
# savefig("Grienwak.pdf")


using DecisionTree, Statistics
num_trees=100;
regr_3 = RandomForestRegressor(n_trees=num_trees);
tmp_xx_train=copy(xx_train);tmp_yy_train=copy(yy_train);
tmp_xx_test=copy(xx_test);tmp_yy_test=copy(yy_test);
DecisionTree.fit!(regr_3, tmp_xx_train, tmp_yy_train[:,1]);
predl1 = DecisionTree.predict(regr_3, tmp_xx_train);
predl1_test = DecisionTree.predict(regr_3, tmp_xx_test);
mean(abs.(predl1-yy_train))
mae=mean(abs.(predl1_test-yy_test))
err=yy_test-predl1_test;ier=sortperm(err)
plot!(err[ier],label="Random Forests",color=:black,linestyle=:dash,legend=:topleft,linewidth=3)


using PyCall,ScikitLearn
@sk_import ensemble: AdaBoostRegressor
sci_model = fit!(AdaBoostRegressor(n_estimators=100),xx_train,yy_train)
predl1=predict(sci_model,xx_train)
predl1_test=predict(sci_model,xx_test)
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))
err=yy_test-predl1_test;ier=sortperm(err)
plot!(err[ier],label="AdaBoost",color=:black,linestyle=:dashdot,legend=:topleft,linewidth=3)

using XGBoost
num_round = 10
bst = xgboost(copy(xx_train), num_round, label = copy(yy_train), eta = 1, max_depth = 7)
predl1 = XGBoost.predict(bst, copy(xx_train))
predl1_test = XGBoost.predict(bst, copy(xx_test))
mean(abs.(predl1-yy_train))
mae=mean(abs.(predl1_test-yy_test))
err=yy_test-predl1_test;ier=sortperm(err)
plot!(err[ier],label="XGBoost",color=:black,linestyle=:dot,legend=:topleft,linewidth=3)
