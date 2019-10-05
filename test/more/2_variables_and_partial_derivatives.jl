
using Plots, Statistics, MLDatasets, LinearAlgebra, Clustering, Printf, Dates, Random
path1=realpath(dirname(@__FILE__)*"/../..")
include(string(path1,"/src/ANNBN.jl"))

# Polynomial
function ff(x)
    -x[1]^3+5x[1]^2+5x[1]*x[2]
end


i_train=1000
vars=2
xx_train=0.8*rand(i_train,vars).+0.1
yy_train=zeros(i_train); for i=1:i_train yy_train[i]=ff(xx_train[i,:]) end
# mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=0.8;yy_train.+=0.1
i_test=1000
xx_test=0.8*rand(i_test,vars).+0.1
yy_test=zeros(i_test); for i=1:i_test yy_test[i]=ff(xx_test[i,:]) end
# yy_test.-=mi;yy_test./=ma;yy_test.*=0.8;yy_test.+=0.1


neurons=Int64(floor(i_train/(vars+1)))
neurons=100 
inds_all,n_per_part=ann_by_parts.___clustering(neurons,xx_train,200)
cc1=1.0
# Use # 2 r^4
a_all,a_layer1,layer1=ann_by_parts.train_layer_1_rbf(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train,cc1)
predl1=ann_by_parts.predict_new_rbf(a_all,a_layer1,xx_train,i_train,neurons,n_per_part,inds_all,xx_train,vars,cc1)
predl1_test=ann_by_parts.predict_new_rbf(a_all,a_layer1,xx_test,i_test,neurons,n_per_part,inds_all,xx_train,vars,cc1)
maetr=mean(abs.(yy_train-predl1))
maete=mean(abs.(yy_test-predl1_test))





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