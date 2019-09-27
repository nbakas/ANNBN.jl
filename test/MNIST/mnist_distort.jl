


using Images, MLDatasets,CoordinateTransformations
train_tensor = MNIST.traintensor()
test_tensor = MNIST.testtensor()
# @show summary(train_tensor);

# train_images = MNIST.convert2image(train_tensor)
# img_1 = train_images[:,:,1] # show first image
# typeof(train_images)

train_images = MNIST.convert2image(1.0 .-train_tensor)
test_images = MNIST.convert2image(1.0 .-test_tensor)
# train_tensor2= MNIST.convert2features(1.0 .-Float64.(train_images))
# maximum(abs.(1.0 .- MNIST.convert2features(train_tensor).-train_tensor2))


img_1 = train_images[:,:,1]
img_1r=imrotate(img_1, (rand()-1/2))[1:28,1:28]
img_1r=imrotate(img_1, 0.0)[1:28,1:28]
iim1=isnan.(img_1r)
img_1r[iim1].=0.0
maximum(abs.(Float64.(img_1)-Float64.(img_1r)))

img_1 = train_images[:,:,48]
img_1z=imresize(img_1,29,29)[1:28,1:28]
img_1z=imresize(img_1,16,16)

# imgs = imstretch(img_1, 2)
rot = LinearMap(RotMatrix(-pi/4))
# rot = LinearMap([0.5 0.5; 1. 1.])
imgw = warp(img_1, rot)
img_1

# f = fill(0,(9,9));
# f[5,5] = 1;
# w = centered([1 2 3; 4 5 6 ; 7 8 9]);
# convolution = imfilter(f,reflect(w),Fill(0,w))
i1=-2+4rand()
i2=-2+4rand()
i3=-2+4rand()
kernel = [i1 0 i1;i2 0 i2;i3 0 i3]
kernel./=sum(kernel)
sum(kernel)
sobel_x = imfilter(img_1, kernel)
# grad = imfilter(sobel_x, kernel')
img_1
float.(sobel_x)



rot1=(rand()-1/2)*10
# rot1=0.0
img_new=Array{Gray{Float64},3}(undef,28,28,60000)
for i=1:size(train_images,3)
     # rot1=0.0
    im1=train_images[:,:,i]
    # if rand(1:100)==1
    #     im1=imresize(im1,29,29)[1:28,1:28]
    # end
    # if rand(1:100)==2
        # rot1=(rand()-1/2)/4
        im1=imrotate(im1, rot1)[1:28,1:28]
    # end

    # i1=rand()
    # i2=rand()
    # i3=rand()
    # kernel = [i1 0 i1;i2 0 i2;i3 0 i3]
    # kernel./=sum(kernel)
    # im1 = imfilter(im1, kernel)

    img_new[:,:,i]=im1
    # img_new[:,:,i]=train_images[:,:,i]
    # println(i)
end
img_new_test=Array{Gray{Float64},3}(undef,28,28,10000)
for i=1:size(test_images,3)
     # rot1=0.0
    im1=test_images[:,:,i]
    # if rand(1:100)==1
    #     im1=imresize(im1,29,29)[1:28,1:28]
    # end
    # if rand(1:100)==2
        # rot1=(rand()-1/2)/4
        im1=imrotate(im1, rot1)[1:28,1:28]
    # end

    # i1=rand()
    # i2=rand()
    # i3=rand()
    # kernel = [i1 0 i1;i2 0 i2;i3 0 i3]
    # kernel./=sum(kernel)
    # im1 = imfilter(im1, kernel)

    img_new_test[:,:,i]=im1
    # img_new[:,:,i]=train_images[:,:,i]
    # println(i)
end


# typeof(img_new[:,:,500])
# img_new[:,:,500]
# minimum(im1)

# im1=channelview(img_new[:,:,500])
# iim1=isnan.(im1)
# im1[iim1].=0.0
# im1
# maximum(im1)
# mean(im1)

# train_tensor2=MNIST.convert2features(im1)
# maximum(train_tensor2)
# mean(train_tensor2)
# convert(Array{Float64,1},train_tensor2)'
# xx_train2[1,:].=convert(Array{Float64,1},train_tensor2)

xx_train2=0.0 .* xx_train
for i=1:size(xx_train2,1)
    # im1=channelview(img_new[:,:,i])
    im1=Float64.(img_new[:,:,i])'
    iim1=isnan.(im1)
    im1[iim1].=0.0
    train_tensor_i=1.0 .- MNIST.convert2features(im1)
    xx_train2[i,:].=1.0 .-convert(Array{Float64,1},train_tensor_i)# .+rand(28*28)/20
end
mean(xx_train2)
mean(xx_train)



xx_test2=0.0 .* xx_test
for i=1:size(xx_test2,1)
    # im1=channelview(img_new[:,:,i])
    im1=Float64.(img_new_test[:,:,i])'
    iim1=isnan.(im1)
    im1[iim1].=0.0
    test_tensor_i=1.0 .- MNIST.convert2features(im1)
    xx_test2[i,:].=1.0 .-convert(Array{Float64,1},test_tensor_i)# .+rand(28*28)/20
end
mean(xx_test2)
mean(xx_test)































# mm=names(MNIST)
# for i=1:length(mm) println(mm[i]) end
# typeof(train_tensor)









# using ImageTransformations
# img_1_rot=ImageTransformations.rotate


# # These two package will provide us with the capabilities
# # to perform interactive visualisations in a jupyter notebook
# using Augmentor, Interact, Reactive

# # The manipulate macro will turn the parameters of the
# # loop into interactive widgets.
# @manipulate for
#         unpaused = true,
#         ticks = fpswhen(signal(unpaused), 5.),
#         image_index = 1:100,
#         grid_size = 3:20,
#         scale = .1:.1:.5,
#         sigma = 1:5,
#         iterations = 1:6,
#         free_border = true
#     op = ElasticDistortion(grid_size, grid_size, # equal width & height
#                            sigma = sigma,
#                            scale = scale,
#                            iter = iterations,
#                            border = free_border)
#     augment(train_images[:, :, image_index], op)
# end
# nothing # hide
