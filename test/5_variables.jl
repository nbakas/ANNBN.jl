
using Plots, Statistics, MLDatasets, LinearAlgebra, Clustering, Printf, Dates, Random
if !(pwd() in LOAD_PATH) push!(LOAD_PATH, pwd()) end
using ANNBN

noise_perc=0.1;vars=5;
ff(x)=-x[1]+(1/2)x[2].^2-(1/3)x[3].^3+(1/4)x[4].^4-(1/5)x[5].^5
i_train=1000; xx_train=0.8*rand(i_train,vars).+0.1
yy_train=zeros(i_train); for i=1:i_train yy_train[i]=ff(xx_train[i,:]) end
yy_train.+=noise_perc.*(1/2 .-rand(i_train))
mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=0.8;yy_train.+=0.1
i_test=1000;xx_test=0.8*rand(i_test,vars).+0.1
yy_test=zeros(i_test); for i=1:i_test yy_test[i]=ff(xx_test[i,:]) end
# yy_test.+=noise_perc.*rand(i_test)
yy_test.-=mi;yy_test./=ma;yy_test.*=0.8;yy_test.+=0.1



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
scatter(yy_test,predl1_test,label="Random Forests",markersize=4,legend=:topleft)

using PyCall,ScikitLearn
@sk_import ensemble: AdaBoostRegressor
sci_model = fit!(AdaBoostRegressor(n_estimators=100),xx_train,yy_train)
predl1=predict(sci_model,xx_train)
predl1_test=predict(sci_model,xx_test)
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))
scatter!(yy_test,predl1_test,label="AdaBoost",markersize=4,legend=:topleft)

using XGBoost
num_round = 10
bst = xgboost(copy(xx_train), num_round, label = copy(yy_train), eta = 1, max_depth = 7)
predl1 = XGBoost.predict(bst, copy(xx_train))
predl1_test = XGBoost.predict(bst, copy(xx_test))
mean(abs.(predl1-yy_train))
mae=mean(abs.(predl1_test-yy_test))
scatter!(yy_test,predl1_test,label="XGBoost",markersize=4,legend=:topleft)


##### ANNBN RBF
neurons=Int64(floor(i_train/(vars+1))) # this automatically generates number of neurons== number of clusters (§3.1)
inds_all,n_per_part=___clustering(neurons,xx_train,200)
cc1=1.0
a_all,a_layer1,layer1=ANNBN.train_layer_1_rbf(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train,cc1)
predl1,layer1_train=ANNBN.predict_new_rbf(a_all,a_layer1,xx_train,i_train,neurons,n_per_part,inds_all,xx_train,vars,cc1)
predl1_test,layer1_test=ANNBN.predict_new_rbf(a_all,a_layer1,xx_test,i_test,neurons,n_per_part,inds_all,xx_train,vars,cc1)
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))
scatter!(yy_test,predl1_test,label="ANNBN-RBF",markersize=4,legend=:topleft)
# savefig("sort-err.pdf")






using ANNBN
# RBF 
nof_folds=10 # you may change these
i_fold=Int64(floor(0.99i_train))
maetrs,a_all_all,a_layer1_all,n_per_part_all,inds_all_all,xx_fold_all,neurons_all=
fit_nfolds!(xx_train,yy_train,nof_folds,vars,i_train,i_fold,cc1,neurons);
predl1_all=predict_nfolds(a_all_all,a_layer1_all,n_per_part_all,inds_all_all,xx_train,i_train,vars,cc1,xx_fold_all,nof_folds,maetrs,neurons_all)
predl1_train=sum(predl1_all)./nof_folds# ./sum(1 ./maetrs)
maetr=mean(abs.(yy_train-predl1_train))
predl1_all=predict_nfolds(a_all_all,a_layer1_all,n_per_part_all,inds_all_all,xx_test,i_test,vars,cc1,xx_fold_all,nof_folds,maetrs,neurons_all)
predl1_test=sum(predl1_all)./nof_folds#./sum(1 ./maetrs)
maete=mean(abs.(yy_test-predl1_test))
maes_all=Vector{Float64}() # check folds history
for i=1:nof_folds 
    predl1_i=sum(predl1_all[1:i])./i#./sum(1 ./maetrs[1:i])
    maete_i=mean(abs.(yy_test-predl1_i))
    push!(maes_all,maete_i) 
end
scatter(maes_all)

using ANNBN
# Sigmoid 
nof_folds=100 # you may change these
i_fold=Int64(floor(0.95i_train))
maetrs,a_all_all,a_layer1_all,n_per_part_all,inds_all_all,xx_fold_all,neurons_all=
fit_nfolds_sigmoid(xx_train,yy_train,nof_folds,vars,i_train,i_fold,neurons);
predl1_all=predict_nfolds_sigmoid(a_all_all,a_layer1_all,n_per_part_all,inds_all_all,xx_train,i_train,vars,xx_fold_all,nof_folds,maetrs,neurons_all)
predl1_train=sum(predl1_all)./nof_folds# ./sum(1 ./maetrs)
maetr=mean(abs.(yy_train-predl1_train))
predl1_all=predict_nfolds_sigmoid(a_all_all,a_layer1_all,n_per_part_all,inds_all_all,xx_test,i_test,vars,xx_fold_all,nof_folds,maetrs,neurons_all)
predl1_test=sum(predl1_all)./nof_folds#./sum(1 ./maetrs)
maete=mean(abs.(yy_test-predl1_test))
maes_all=Vector{Float64}() # check folds history
for i=1:nof_folds 
    predl1_i=sum(predl1_all[1:i])./i#./sum(1 ./maetrs[1:i])
    maete_i=mean(abs.(yy_test-predl1_i))
    push!(maes_all,maete_i) 
end
scatter(maes_all)



