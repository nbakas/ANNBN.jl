
# With this file you may reproduce the results for cases 1, 2, 3, 4 in Table 2,
# as well as for Random Forests and Gradient Boosting
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

tol1=0.02
num_to_test=0
yy_train=copy(yy_train_all)
i1=abs.(yy_train.-num_to_test).<0.01;i2=abs.(yy_train.-num_to_test).>0.01;yy_train[i1].=1;;yy_train[i2].=0;
mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1

yy_test=copy(yy_test_all)
i1=abs.(yy_test.-num_to_test).<0.01;i2=abs.(yy_test.-num_to_test).>0.01;yy_test[i1].=1;yy_test[i2].=0;
yy_test.-=mi;yy_test./=ma;yy_test.*=1.0-2tol1;yy_test.+=tol1


path1=realpath(dirname(@__FILE__)*"/../..")
include(string(path1,"/src/ANNBN.jl"))

inds_all=1:i_train
# For higher accuracy, increase the number of neurons. However memory demands increases as well.
neurons=1000;items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
@time a_all,a_layer1,layer1,mat1=ANNBN.train_layer_1_sigmoid_fast(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train)
predl1=[layer1 ones(i_train)]*a_layer1
maetr=mean(abs.(yy_train-predl1))
maximum(abs.(yy_train-predl1))
predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,xx_test,i_test,neurons);
maete=mean(abs.(yy_test-predl1_test))
maximum(abs.(yy_test-predl1_test))
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<1e-10;maetr=100sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<1e-10;
maete=100sum(i2)/length(i2)
100mean(abs.(round.(predl1).-round.(yy_train)).<0.1)
100mean(abs.(round.(predl1_test).-round.(yy_test)).<0.1)




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
    inds_all=1:i_train;neurons=7000;items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
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



using Base.Threads

nthreads()
x=zeros(10)
@time begin
    Threads.@threads for i=1:10
        for j=1:1e7
            global x
            x[i]+=1
            # println("i = $i on thread $(Threads.threadid())")
        end
    end
end
@time begin
    for i=1:10
        for j=1:1e7
            global x
            x[i]+=1
            # println("i = $i on thread $(Threads.threadid())")
        end
    end
end



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
