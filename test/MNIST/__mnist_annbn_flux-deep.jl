using Flux
using Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated

path1=realpath(dirname(@__FILE__)*"/../..")
include(string(path1,"/src/ANNBN.jl"))
using Plots, Statistics, MLDatasets, LinearAlgebra, Clustering, Printf, Dates, Random, MLDatasets, Plots, Printf, Debugger
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

# layer1_ok=Array{Float64}(undef,1000)
# layer1_ok=[layer1_ok  zeros(1000)]

# i_train=20000
# xx_train=xx_train[1:i_train,:]
# yy_train=yy_train[1:i_train]



inds_all=1:i_train;neurons=1000;items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
@time a_all,a_layer1,layer1=ANNBN.train_layer_1_sigmoid_fast(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train)
predl1=[layer1 ones(i_train)]*a_layer1
maetr=mean(abs.(yy_train-predl1))
maximum(abs.(yy_train-predl1))
predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,xx_test,i_test,neurons);
maete=mean(abs.(yy_test-predl1_test))
maximum(abs.(yy_test-predl1_test))
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<0.01;maetr=100sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;
maete=100sum(i2)/length(i2)



neurons_all=Vector{Int64}(undef,0)
push!(neurons_all,length(a_layer1)-1)
weights=Vector{Array{Float32}}(undef,0)
aa=reduce(hcat, a_all)'[:,2:end]
push!(weights,convert(Array{Float32,2},aa))
bb=reduce(hcat, a_all)'[:,1]
push!(weights,convert(Array{Float32,1},bb))
# aa=reshape(a_layer1[1:end-1],1,neurons)
# push!(weights,convert(Array{Float32,2},aa))
# bb=convert(Vector{Float32},[a_layer1[end]])
# push!(weights,bb)

# _todo_Eq_15
nof_layers=1
for i=2:nof_layers
    global neurons_all,layer1,layer1_test,weights,a_layer1
    neurons=neurons_all[end]
    # if i==2
    #     neurons=500
    # else
    #     neurons=100
    # end

    inds_all=1:i_train;items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train

    # inds_all=1:i_train;neurons=10000;items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train

    @time a_all,a_layer1,layer1=ANNBN.train_layer_1_sigmoid_fast(neurons,neurons,i_train,n_per_part,inds_all,layer1,yy_train)
    predl1=[layer1 ones(i_train)]*a_layer1
    maetr=mean(abs.(yy_train-predl1))
    maximum(abs.(yy_train-predl1))
    predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,layer1_test,i_test,neurons);
    maete=mean(abs.(yy_test-predl1_test))
    maximum(abs.(yy_test-predl1_test))
    i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<0.01;maetr=100sum(i2)/length(i2)
    i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;
    maete=100sum(i2)/length(i2)

    println(" >>>>>>>>>>>>>>>>>>>>    ",i," ",maetr," ",maete," ",length(a_layer1)-1)


    push!(neurons_all,length(a_layer1)-1)
    aa=reduce(hcat, a_all)'[:,2:end]
    push!(weights,convert(Array{Float32,2},aa))
    bb=reduce(hcat, a_all)'[:,1]
    push!(weights,convert(Array{Float32,1},bb))

end

aa=reshape(a_layer1[1:end-1],1,neurons_all[end])
push!(weights,convert(Array{Float32,2},aa))
bb=convert(Vector{Float32},[a_layer1[end]])
push!(weights,bb)

weights00=copy(weights)





X=convert(Array{Float32,2},xx_train')
Y=convert(Array{Float32,2},yy_train')
# X=rand(Float32,5,1000)
# Y=sum(X,dims=1)
m = Chain(
  Dense(vars, neurons_all[1], ANNBN.sigm1),
  # Dense(neurons_all[1], neurons_all[2], ANNBN.sigm1),
  # Dense(neurons_all[2], neurons_all[3], ANNBN.sigm1),
  # Dense(neurons_all[3], neurons_all[4], ANNBN.sigm1),
  # Dense(neurons_all[4], neurons_all[5], ANNBN.sigm1),
  # Dense(neurons_all[5], neurons_all[6], ANNBN.sigm1),
  # Dense(neurons_all[6], neurons_all[7], ANNBN.sigm1),
  # Dense(neurons_all[7], neurons_all[8], ANNBN.sigm1),
  # Dense(neurons_all[8], neurons_all[9], ANNBN.sigm1),
  # Dense(neurons_all[9], neurons_all[10], ANNBN.sigm1),
  Dense(neurons_all[1], 1))
loss(x, y) = mean((m(x).-y).^2)
# opt = ADAM()
opt = Descent(1e-2)
dataset = repeated((X,Y),10)
# evalcb = () -> @show(loss(X,Y),loss(Xte,Yte))
evalcb = () -> @show(100mean(abs.(round.(Tracker.data(m(X)[1,:])).-round.(yy_train)).<0.1), 100mean(abs.(round.(Tracker.data(m(Xte)[1,:])).-round.(yy_test)).<0.1))
# weights = Tracker.data.(params(m))


Flux.loadparams!(m, weights00)




loss(X, Y)
predl1=Tracker.data(m(X)[1,:])
weights = Tracker.data.(params(m))
maetr=mean(abs.(yy_train-predl1))
maximum(abs.(yy_train-predl1))
Xte=convert(Array{Float32,2},xx_test')
Yte=convert(Array{Float32,2},yy_test')
loss(Xte, Yte)
predl1_test=Tracker.data(m(Xte)[1,:])
maete=mean(abs.(yy_test-predl1_test))
maximum(abs.(yy_test-predl1_test))
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<0.01;maetr=100sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<0.01;
maete=100sum(i2)/length(i2)

100mean(abs.(round.(predl1).-round.(yy_train)).<0.1)
100mean(abs.(round.(predl1_test).-round.(yy_test)).<0.1)

println(">>>>>>>>>>>>>>>>>>>>>>>>>>START>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
@time for ep=1:1 Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 1)); end

# weights_out = Tracker.data.(params(m))
# weights[1][1,2]=1e5
# Flux.loadparams!(m, weights)
#
#
# typeof(weights)
#
# weights2=Vector{Array{Float32}}(undef,0)
# push!(weights2,rand(Float32,10,5))
# push!(weights2,rand(Float32,10))
