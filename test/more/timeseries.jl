
using Plots, Statistics, MLDatasets, LinearAlgebra, Clustering, Printf, Dates, Random, XLSX
path1=realpath(dirname(@__FILE__)*"/../..")
include(string(path1,"/src/ANNBN.jl"))

path_project="~\\finance\\"
file1="GE.xlsx"
sheet1="GE"
range1="A1:G1260"
all_vars = XLSX.readdata(string(path_project,file1),sheet1,range1)


yy_train=float.(all_vars[2:end,5])
plot(yy_train)
xx_train=[yy_train[5:end-1] yy_train[4:end-2] yy_train[3:end-3] yy_train[2:end-4] yy_train[1:end-5]]

yy_train=yy_train[6:end]
aa=xx_train\yy_train
pred=xx_train*aa

plot(yy_train[1:100])
plot!(pred[1:100])

mean(abs.(yy_train-pred))
mean(abs.(yy_train[1:end-1]-yy_train[2:end]))


i_train=size(xx_train,1)
vars=size(xx_train,2)
neurons=Int64(floor(i_train/(vars+1)))
# neurons=50
inds_all,n_per_part=ann_by_parts.___clustering(neurons,xx_train,200)
cc1=100.0
a_all,a_layer1,layer1=ann_by_parts.train_layer_1_rbf(neurons,vars,i_train,n_per_part,
            inds_all,xx_train,yy_train,cc1)
predl1=ann_by_parts.predict_new_rbf(a_all,a_layer1,xx_train,i_train,neurons,
            n_per_part,inds_all,xx_train,vars,cc1)
# predl1_test=ann_by_parts.predict_new_rbf(a_all,a_layer1,xx_test,i_test,neurons,
            # n_per_part,inds_all,xx_train,vars,cc1)
maetr=mean(abs.(yy_train-predl1))
# scatter(yy_train,predl1,label=@sprintf("fann %1.4f",maetr))
# maete=mean(abs.(yy_test-predl1_test))


plot!(predl1[1:100])


nof_folds=20 # you may change these
i_fold=Int64(floor(0.8i_train))
include("ann_by_parts.jl")
maetrs,a_all_all,a_layer1_all,n_per_part_all,inds_all_all,xx_fold_all,neurons_all=ann_by_parts.fit_nfolds!(xx_train,yy_train,nof_folds,vars,i_train,i_fold,cc1);
predl1_all=ann_by_parts.predict_nfolds(a_all_all,a_layer1_all,n_per_part_all,inds_all_all,xx_train,i_train,vars,cc1,xx_fold_all,nof_folds,maetrs,neurons_all)
predl1_train=sum(predl1_all)./sum(1 ./maetrs)
maetr=mean(abs.(yy_train-predl1_train))
# predl1_all=ann_by_parts.predict_nfolds(a_all_all,a_layer1_all,n_per_part_all,inds_all_all,xx_test,i_test,vars,cc1,xx_fold_all,nof_folds,maetrs,neurons_all)
# predl1_test=sum(predl1_all)./sum(1 ./maetrs)
# maete=mean(abs.(yy_test-predl1_test))

plot!(predl1_train[1:100])
