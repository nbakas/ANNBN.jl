


vars=5
i_train=10_000
i_test=10_000
cc1=10.0
neurons=Int64(floor(i_train/50)) # this automatically
# generates number of neurons== number of clusters (§3.1)
_case_noise_=5


ff(x)=-x[1]+(1/2)x[2].^2-(1/3)x[3].^3+(1/4)x[4].^4-(1/5)x[5].^5
xx_train=0.8*rand(i_train,vars).+0.1
yy_train=zeros(i_train); for i=1:i_train yy_train[i]=ff(xx_train[i,:]) end
mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=0.8;yy_train.+=0.1



if _case_noise_==1
    __noise__=rand(i_train)
elseif _case_noise_==2
    __noise__=rand(Normal(0.0, 1.0),i_train)
elseif _case_noise_==3
    __noise__=rand(GeneralizedPareto(0.0,1.0,0.25),i_train)
elseif _case_noise_==4
    __noise__=rand(LogNormal(0.25, 0.8),i_train)
elseif _case_noise_==5
    __noise__=[rand(LogNormal(0.25, 0.8),i_train);10.0.-rand(Exponential(3.0),i_train)]
    __noise__=[__noise__;15.0.+rand(Frechet(10.0,10.0),i_train)]
    __noise__=__noise__[1:3:end]
end
__noise__.-=mean(__noise__)
__noise__./=mean(abs.(__noise__))/(0.05*0.45)
histogram(__noise__,label=sum(__noise__)/sum(abs.(__noise__)),
        title=mean(abs.(__noise__)))
iso=sortperm(yy_train)
plot(yy_train[iso])
yy_train.+=__noise__
display(plot!(yy_train[iso],legend=false))


xx_test=0.8*rand(i_test,vars).+0.1
yy_test=zeros(i_test); for i=1:i_test yy_test[i]=ff(xx_test[i,:]) end
# yy_test.+=noise_perc.*rand(i_test)
yy_test.-=mi;yy_test./=ma;yy_test.*=0.8;yy_test.+=0.1

scatter3d(xx_train[:,1],xx_train[:,2],yy_train)
display(scatter3d!(xx_test[:,1],xx_test[:,2],yy_test))



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
display(plot(err[ier],label="Random Forests",color=:black,
    linestyle=:dash,legend=:topleft,linewidth=3))
__err__=string(mean(abs.(err)))




sci_model = fit!(AdaBoostRegressor(n_estimators=100),xx_train,yy_train)
predl1=ScikitLearn.predict(sci_model,xx_train)
predl1_test=ScikitLearn.predict(sci_model,xx_test)
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))
err=yy_test-predl1_test;ier=sortperm(err)
display(plot!(err[ier],label="AdaBoost",color=:black,
    linestyle=:dashdot,legend=:topleft,linewidth=3))
__err__=string(__err__," ",mean(abs.(err)))


num_round = 10
bst = xgboost(copy(xx_train), num_round, label = copy(yy_train), eta = 1, max_depth = 7)
predl1 = XGBoost.predict(bst, copy(xx_train))
predl1_test = XGBoost.predict(bst, copy(xx_test))
mean(abs.(predl1-yy_train))
mae=mean(abs.(predl1_test-yy_test))
err=yy_test-predl1_test;ier=sortperm(err)
display(plot!(err[ier],label="XGBoost",color=:black,
    linestyle=:dot,legend=:topleft,linewidth=3))
__err__=string(__err__," ",mean(abs.(err)))

##### ANNBN RBF

include(string(path1,"/src/ANNBN.jl"))
inds_all,n_per_part=ANNBN.___clustering(neurons,xx_train,200)

a_all,a_layer1,layer1=ANNBN.train_layer_1_rbf(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train,cc1)
predl1,layer1_train=ANNBN.predict_new_rbf(a_all,a_layer1,xx_train,i_train,neurons,n_per_part,inds_all,xx_train,vars,cc1)
predl1_test,layer1_test=ANNBN.predict_new_rbf(a_all,a_layer1,xx_test,i_test,neurons,n_per_part,inds_all,xx_train,vars,cc1)
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))
err=yy_test-predl1_test;ier=sortperm(err)
display(plot!(err[ier],label="ANNBN",color=:black,linestyle=:dashdotdot,
    legend=:topleft,linewidth=3))
__err__=string(__err__," ",mean(abs.(err)))


println(__err__)

#
