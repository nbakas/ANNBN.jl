

# # plot(predl1[1:50])
# # plot!(yy_train[1:50])
# # 10,3,4,5,20
# i1=findall(abs.(predl1-yy_train).>0.1)[1]
# # plot(0:9,predl1_train_all[i1,:])
# yy_train_all[i1]
# yy_train[i1]
# predl1[i1]
# using Images
# train_tensor = MNIST.traintensor()
# train_images = MNIST.convert2image(1.0 .-train_tensor)
# train_images[:,:,i1]



using Images
test_tensor = MNIST.testtensor()
test_images = MNIST.convert2image(1.0 .-test_tensor)
i1=findall(abs.(yy_test_pred-yy_test_all).>0.1)
nn=9
test_images[:,:,i1[nn]]
yy_test_all[i1[nn]]
yy_test_pred[i1[nn]]




# x1=Vector{Float64}(undef,0)
# for i=1:28
#     global x1
#     x1=[x1;float.(train_images[i,1:28,1])]
# end
#
# maximum(abs.(xx_train[1,:].-x1))
#
# plot()
# for i=1:28
#     display(plot!(xx_train[1,(i-1)*28+1:i*28]))
# end
#
#
#
#
#
# train_images[:,:,1]
# x1=Array{Float64}(undef,0,14)
# for i=1:2:28
#     global x1
#     x1=[x1;float.(train_images[i,1:2:end,1])']
# end
# Gray.(x1)
#
#
# im1=train_images[:,:,1]
# im1f=float.(im1.+.1)
# img=Gray.(im1f)
