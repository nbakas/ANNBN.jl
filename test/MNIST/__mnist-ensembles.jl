

predl1_train_ens=Array{Float64}(undef,i_train,0)
predl1_test_ens=Array{Float64}(undef,i_test,0)
maetrs=Vector{Float64}(undef,0)
maetes=Vector{Float64}(undef,0)
for i=1:5
    global predl1_train_ens,predl1_test_ens,maetrs,maetes
    inds_all=1:i_train
    neurons=7000
    items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
    @time include("C:\\Dropbox\\julialangfiles\\ANNBN\\__EXAMPLES\\more\\mnist_distort.jl")
    @time a_all,a_layer1,layer1=ANNBN.train_layer_1_sigmoid_fast(neurons,vars,i_train,n_per_part,inds_all,xx_train2,yy_train)
    predl1=[layer1 ones(i_train)]*a_layer1
    maetr=mean(abs.(yy_train-predl1))
    maximum(abs.(yy_train-predl1))
    predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,xx_test2,i_test,neurons);
    maete=mean(abs.(yy_test-predl1_test))
    maximum(abs.(yy_test-predl1_test))


    predl1_train_ens=[predl1_train_ens predl1]
    predl1_test_ens=[predl1_test_ens predl1_test]

    i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<1e-10;maetr=100sum(i2)/length(i2)
    i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<1e-10;
    maete=100sum(i2)/length(i2)

    println(i," ",maetr," ",maete)



    predl1=mean(predl1_train_ens,dims=2)
    maetr=mean(abs.(yy_train-predl1))
    maximum(abs.(yy_train-predl1))
    predl1_test=mean(predl1_test_ens,dims=2)
    maete=mean(abs.(yy_test-predl1_test))
    maximum(abs.(yy_test-predl1_test))
    i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<1e-10;maetr=100sum(i2)/length(i2)
    i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<1e-10;
    maete=100sum(i2)/length(i2)
    println(i," ",maetr," ",maete)
    push!(maetrs,maetr)
    push!(maetes,maete)
    # display(plot(maetrs))
    # display(plot!(maetes))


end

predl1=mean(predl1_train_ens,dims=2)
maetr=mean(abs.(yy_train-predl1))
maximum(abs.(yy_train-predl1))
predl1_test=mean(predl1_test_ens,dims=2)
maete=mean(abs.(yy_test-predl1_test))
maximum(abs.(yy_test-predl1_test))
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<1e-10;maetr=100sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<1e-10;
maete=100sum(i2)/length(i2)

findfirst(i2.==false)
yy_test[260]
plot(predl1_test_ens[260,:])


plot(xx_train[1,180:250])



#
# nr=10
# a_layer1_nr=Vector{Vector{Float64}}()
# varsl1=size(layer1,2)
# predl1_ir=Array{Float64}(undef,i_train,0)
# for i=1:nr
#     global varsl1,predl1_ir
#     ir=randperm(varsl1)[1:1000]
#     aa=layer1[:,ir]\yy_train
#     predl1_ir=[predl1_ir layer1[:,ir]*aa]
#     println(i)
# end
#
# # predl1=mean(predl1_ir,dims=2)
# layer1.+=0.7
# layer1./=2
# layer1.+=0.1
plot(layer1[1,:])
# plot(ANNBN.sigm1.(1000layer1[1,:]))
varsl1=2size(layer1,2)
inds_all=1:i_train;neurons=10000;items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);
n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
@time a_all,a_layer1,layer1=ANNBN.train_layer_1_sigmoid_fast(neurons,4,i_train,n_per_part,inds_all,[predl1_1 predl1_2 predl1_3 predl1_4],yy_train)
predl1=[layer1 ones(i_train)]*a_layer1
maetr=mean(abs.(yy_train-predl1))
maximum(abs.(yy_train-predl1))
predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,abs.(layer1_test),i_test,neurons);
maete=mean(abs.(yy_test-predl1_test))
maximum(abs.(yy_test-predl1_test))
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<1e-15;maetr=100sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<1e-15;
maete=100sum(i2)/length(i2)



maetrs=Array{Float64}(undef,0)
maetes=Array{Float64}(undef,0)
dts=Array{Float64}(undef,0)
predl1_train_all=Array{Float64}(undef,i_train,0)
predl1_test_all=Array{Float64}(undef,i_test,0)
@time for num_to_test=0:9
    global yy_train_all,yy_test_all,predl1_train_all,predl1_test_all
    yy_train=copy(yy_train_all)
    yy_test=copy(yy_test_all)
    tol1=0.0
    yy_train=copy(yy_train_all)
    i1=abs.(yy_train.-num_to_test).<0.01;i2=abs.(yy_train.-num_to_test).>0.01;yy_train[i1].=1;;yy_train[i2].=0;
    mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1
    yy_test=copy(yy_test_all)
    i1=abs.(yy_test.-num_to_test).<0.01;i2=abs.(yy_test.-num_to_test).>0.01;yy_test[i1].=1;yy_test[i2].=0;
    yy_test.-=mi;yy_test./=ma;yy_test.*=1.0-2tol1;yy_test.+=tol1
    inds_all=1:i_train;neurons=5000;items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
    t1=now()
    @time a_all,a_layer1,layer1=ANNBN.train_layer_1_sigmoid_fast(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train)
    t2=now();dt=convert(Float64,Dates.value(t2-t1))/1000;push!(dts,dt)
    predl1=[layer1 ones(i_train)]*a_layer1
    maetr=mean(abs.(yy_train-predl1))
    maximum(abs.(yy_train-predl1))
    predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,xx_test,i_test,neurons);
    maete=mean(abs.(yy_test-predl1_test))
    maximum(abs.(yy_test-predl1_test))


    predl1_train_all=[predl1_train_all predl1]
    predl1_test_all=[predl1_test_all predl1_test]

    i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<0.01;maetr=100sum(i2)/length(i2)
    i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;maete=100sum(i2)/length(i2)
    push!(maetrs,maetr)
    push!(maetes,maete)
    println(num_to_test," ",maete)
end
println(dts)
println(maetes)

yy_test_pred=zeros(i_test)
for i=1:i_test
    yy_test_pred[i]=sortperm(predl1_test_all[i,:],rev=true)[1]-1

    # pp=round.(predl1_test_all[i,:])
    # if length(unique(pp))==1
    #     yy_test_pred[i]=rand(0:9)
    #     println(i)
    # else
    #     yy_test_pred[i]=sortperm(pp,rev=true)[1]-1
    # end
end
ER=100-100mean(round.(yy_test_all-yy_test_pred).>0.1)
# ER=100-100mean(abs.(yy_test_all-yy_test_pred).>0)

yy_train_pred=zeros(i_train)
for i=1:i_train
    yy_train_pred[i]=sortperm(predl1_train_all[i,:],rev=true)[1]-1
end
ER=100-100mean(round.(yy_train_all-yy_train_pred).>0)
# ER=100-100mean(abs.(yy_train_all-yy_train_pred).>0)







# inds_all=1:i_train;neurons=100;items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
# @time a_all,a_layer1,layer1=ANNBN.train_layer_1_sigmoid_fast(neurons,10,i_train,n_per_part,inds_all,predl1_train_all,yy_train)
# predl1=[layer1 ones(i_train)]*a_layer1
# maetr=mean(abs.(yy_train-predl1))
# maximum(abs.(yy_train-predl1))
# predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,predl1_test_all,i_test,neurons);
# maete=mean(abs.(yy_test-predl1_test))
# maximum(abs.(yy_test-predl1_test))
# i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<0.01;maetr=100sum(i2)/length(i2)
# i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;
# maete=100sum(i2)/length(i2)


predl1_train_all
100-100*329/60000
iii=(1:i_train)[abs.(yy_train_all-yy_train_pred).>0.1]
predl1_train_all[iii[2],:]
sortperm(predl1_train_all[iii[2],:],rev=true).-1
yy_train_all[iii[2]]

plot(yy_train_all[iii])
mean(yy_train_all[iii])
histogram(yy_train_all[iii])
predl1_train_all[49,10]


predl1
i1=(1:i_train)[abs.(round.(predl1).-yy_train).>0.01]
ii1=i1[1]
predl1[ii1]
predl1_train_all[ii1,:]
yy_train_all[ii1]

predl1_c=(predl1.-sum(predl1_train_all[:,2:end],dims=2))[:,1]
predl1_c=round.(predl1_c)
plot(predl1_c)
ir=predl1_c.!=1.0
predl1_c[ir].=0.0
i1=(1:i_train)[abs.(predl1_c.-yy_train).>0.01]
ii1=i1[1]
predl1_c[ii1]
predl1_train_all[ii1,:]
yy_train_all[ii1]


# aa=predl1_train_all\yy_train
# ptr=predl1_train_all*aa
# 100-100mean(abs.(yy_train-round.(ptr)).>0)
# pte=predl1_test_all*aa
# 100-100mean(abs.(yy_test-round.(pte)).>0)

# 100-100mean(abs.(yy_train-round.(predl1_train_all[:,1])).>0)
# 100-100mean(abs.(yy_test-round.(predl1_test_all[:,1])).>0)












# rr=rand(7,100)
# ir=inv(rr)
# irp=pinv(rr)
# ir-irp

AA=[layer1 ones(i_train)]
# RR=rand(700,60000)
RR=[layer1 ones(i_train)]'
@time a_layer1=inv(RR*AA)*RR*yy_train


i1=randperm(5000)[1:1000]
i2=randperm(i_train)[1:10000]
AA=[layer1[i2,i1] ones(10000)]
RR=[layer1[i2,i1] ones(10000)]'
@time a_layer1=inv(RR*AA)*RR*yy_train[i2]
predl1=[layer1[:,i1] ones(i_train)]*a_layer1
maetr=mean(abs.(yy_train-predl1))
maximum(abs.(yy_train-predl1))
predl1_test,layer1_test=ANNBN.predict_new(a_all[i1],a_layer1,xx_test,i_test,neurons);
maete=mean(abs.(yy_test-predl1_test))
maximum(abs.(yy_test-predl1_test))
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<1e-10;maetr=100sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<1e-10;
maete=100sum(i2)/length(i2)


# AA=[layer1 ones(i_train)]
predl1_train_ens=Array{Float64}(undef,i_train,0)
predl1_test_ens=Array{Float64}(undef,i_test,0)
maetrs=Vector{Float64}(undef,0)
maetes=Vector{Float64}(undef,0)
for i=1:1000
    global predl1_train_ens,predl1_test_ens,maetrs,maetes

    # RR=rand(700,60000)
    # @time a_layer1=pinv(RR*AA)*RR*yy_train
    # predl1=[layer1 ones(i_train)]*a_layer1
    # maetr=mean(abs.(yy_train-predl1))
    # maximum(abs.(yy_train-predl1))
    # predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,xx_test,i_test,neurons);
    # maete=mean(abs.(yy_test-predl1_test))
    # maximum(abs.(yy_test-predl1_test))

    i1=randperm(5000)[1:100]
    i2=randperm(i_train)[1:1000]
    AA=[layer1[i2,i1] ones(1000)]
    RR=[layer1[i2,i1] ones(1000)]'
    @time a_layer1=inv(RR*AA)*RR*yy_train[i2]
    predl1=[layer1[:,i1] ones(i_train)]*a_layer1
    maetr=mean(abs.(yy_train-predl1))
    maximum(abs.(yy_train-predl1))
    predl1_test,layer1_test=ANNBN.predict_new(a_all[i1],a_layer1,xx_test,i_test,neurons);
    maete=mean(abs.(yy_test-predl1_test))
    maximum(abs.(yy_test-predl1_test))

    predl1_train_ens=[predl1_train_ens predl1]
    predl1_test_ens=[predl1_test_ens predl1_test]

    i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<1e-10;maetr=100sum(i2)/length(i2)
    i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<1e-10;
    maete=100sum(i2)/length(i2)



    println(i," ",maetr," ",maete)

    predl1=mean(predl1_train_ens,dims=2)
    maetr=mean(abs.(yy_train-predl1))
    maximum(abs.(yy_train-predl1))
    predl1_test=mean(predl1_test_ens,dims=2)
    maete=mean(abs.(yy_test-predl1_test))
    maximum(abs.(yy_test-predl1_test))
    i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<1e-10;maetr=100sum(i2)/length(i2)
    i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<1e-10;
    maete=100sum(i2)/length(i2)

    push!(maetrs,maetr)
    push!(maetes,maete)
    display(plot(maetrs,legend=false))
    display(plot!(maetes,legend=false))

    println(i," ",maetr," ",maete)
end

aa=predl1_train_ens\yy_train
predl1=predl1_train_ens*aa
predl1_test=predl1_test_ens*aa

# predl1=mean(predl1_train_ens,dims=2)
maetr=mean(abs.(yy_train-predl1))
maximum(abs.(yy_train-predl1))
# predl1_test=mean(predl1_test_ens,dims=2)
maete=mean(abs.(yy_test-predl1_test))
maximum(abs.(yy_test-predl1_test))
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<1e-10;maetr=100sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<1e-10;
maete=100sum(i2)/length(i2)








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







# # # # # # # # # # # # # # Flux
using Flux
using Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
imgs = Flux.Data.MNIST.images()
labels = Flux.Data.MNIST.labels()
test_X = hcat(float.(reshape.(Flux.Data.MNIST.images(:test), :))...)

tol1=0.0
num_to_test=0
yy_train=copy(labels)
i1=abs.(yy_train.-num_to_test).<0.01;i2=abs.(yy_train.-num_to_test).>0.01;yy_train[i1].=1;;yy_train[i2].=0;
mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1

yy_test=Flux.Data.MNIST.labels(:test)
i1=abs.(yy_test.-num_to_test).<0.01;i2=abs.(yy_test.-num_to_test).>0.01;yy_test[i1].=1;yy_test[i2].=0;
yy_test.-=mi;yy_test./=ma;yy_test.*=1.0-2tol1;yy_test.+=tol1



X = hcat(float.(reshape.(imgs, :))...)
Y = onehotbatch(yy_train,0:1)
m = Chain(
  Dense(28^2, 1000, relu),
  Dense(1000, 1000, relu),
  Dense(1000, 1000, relu),
  Dense(1000, 1000, relu),
  Dense(1000, 1000, relu),
  Dense(1000, 1000, relu),
  Dense(1000, 1000, relu),
  Dense(1000, 1000, relu),
  Dense(1000, 1000, relu),
  Dense(1000, 1000, relu),
  Dense(1000, 2),
  softmax)
loss(x, y) = crossentropy(m(x), y)
opt = ADAM();
dataset = repeated((X,Y),30)
evalcb = () -> @show(loss(X, Y))
@time for ep=1:3 Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10)); end

predl1=zeros(60000); mmm=m(X); predl1=Tracker.data(mmm[2,:])
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<0.01;maetr=100sum(i2)/length(i2)
predl1_test=zeros(10000); mmm=m(test_X); predl1_test=Tracker.data(mmm[2,:]);
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;
maete=100sum(i2)/length(i2)

weights = Tracker.data.(params(m))

maetrs=Array{Float64}(undef,0)
maetes=Array{Float64}(undef,0)
dts=Array{Float64}(undef,0)
@time for num_to_test=0:9
    global imgs,labels,test_X, tol1

    yy_train=copy(labels)
    i1=abs.(yy_train.-num_to_test).<0.01;i2=abs.(yy_train.-num_to_test).>0.01;yy_train[i1].=1;;yy_train[i2].=0;
    mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1

    yy_test=Flux.Data.MNIST.labels(:test)
    i1=abs.(yy_test.-num_to_test).<0.01;i2=abs.(yy_test.-num_to_test).>0.01;yy_test[i1].=1;yy_test[i2].=0;
    yy_test.-=mi;yy_test./=ma;yy_test.*=1.0-2tol1;yy_test.+=tol1

    X = hcat(float.(reshape.(imgs, :))...)
    Y = onehotbatch(yy_train,0:1)
    m = Chain(
      Dense(28^2, 5000, relu),
      # Dense(5000, 5000, relu),
      Dense(5000, 2),
      softmax)
    loss(x, y) = crossentropy(m(x), y)
    opt = ADAM();
    dataset = repeated((X,Y),30)
    evalcb = () -> @show(loss(X, Y))

    t1=now()
    for ep=1:3 Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10)); end
    t2=now();dt=convert(Float64,Dates.value(t2-t1))/1000;push!(dts,dt)

    predl1=zeros(60000); mmm=m(X); predl1=Tracker.data(mmm[2,:])
    i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<0.01;maetr=100sum(i2)/length(i2)
    predl1_test=zeros(10000); mmm=m(test_X); predl1_test=Tracker.data(mmm[2,:]);
    i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;
    maete=100sum(i2)/length(i2)

    push!(maetrs,maetr)
    push!(maetes,maete)
    println(" > > > > > > > > > > > > >",num_to_test," ",maete," ",dt)
end
println(dts)
println(maetes)


maetrs=Array{Float64}(undef,0)
maetes=Array{Float64}(undef,0)
dts=Array{Float64}(undef,0)
@time for num_to_test=0:9
    global imgs,labels,test_X, tol1

    yy_train=copy(labels)
    i1=abs.(yy_train.-num_to_test).<0.01;i2=abs.(yy_train.-num_to_test).>0.01;yy_train[i1].=1;;yy_train[i2].=0;
    mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1

    yy_test=Flux.Data.MNIST.labels(:test)
    i1=abs.(yy_test.-num_to_test).<0.01;i2=abs.(yy_test.-num_to_test).>0.01;yy_test[i1].=1;yy_test[i2].=0;
    yy_test.-=mi;yy_test./=ma;yy_test.*=1.0-2tol1;yy_test.+=tol1

    X = hcat(float.(reshape.(imgs, :))...)
    Y = onehotbatch(yy_train,0:1)
    m = Chain(
    Dense(28^2, 1000, relu),
    Dense(1000, 1000, relu),
    Dense(1000, 1000, relu),
    Dense(1000, 1000, relu),
    Dense(1000, 1000, relu),
    Dense(1000, 1000, relu),
    Dense(1000, 1000, relu),
    Dense(1000, 1000, relu),
    Dense(1000, 1000, relu),
    Dense(1000, 1000, relu),
    Dense(1000, 2),
      softmax)
    loss(x, y) = crossentropy(m(x), y)
    opt = ADAM();
    dataset = repeated((X,Y),50)
    evalcb = () -> @show(loss(X, Y))

    t1=now()
    for ep=1:3 Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10)); end
    t2=now();dt=convert(Float64,Dates.value(t2-t1))/1000;push!(dts,dt)

    predl1=zeros(60000); mmm=m(X); predl1=Tracker.data(mmm[2,:])
    i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<0.01;maetr=100sum(i2)/length(i2)
    predl1_test=zeros(10000); mmm=m(test_X); predl1_test=Tracker.data(mmm[2,:]);
    i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;
    maete=100sum(i2)/length(i2)

    push!(maetrs,maetr)
    push!(maetes,maete)
    println(" > > > > > > > > > > > > >",num_to_test," ",maete," ",dt)
end
println(dts)
println(maetes)
