


neurons1=200 # sub-sampling for each digit
inds_all=Array{Int64}(undef,0)
items_per_neuron=Array{Int64}(undef,0)
@time for i=0:9
    global inds_all,items_per_neuron
    println(Dates.format(now(), "HH:MM:SS")," ",i)
    i1=yy_train_all.==i
    inds_all1,n_per_part1,items_per_neuron1=ANNBN.___clustering(neurons1,xx_train[i1,:],300)
    ia1=(1:i_train)[i1]
    inds_all=[inds_all;ia1[inds_all1]]
    items_per_neuron=[items_per_neuron;items_per_neuron1]
end
# plot(unique(sort(inds_all)))
n_per_part=[0;cumsum(items_per_neuron)]
neurons=10neurons1 # all neurons
plot(items_per_neuron)
minimum(items_per_neuron)
# xx_train=xx_train[inds_all,:]
# yy_train=yy_train[inds_all]
# yy_train_all=yy_train_all[inds_all]
# inds_all=1:i_train
