


using Plots, Statistics

function sigm1(x) 1/(1+exp(-x)) end
function isigm1(x) -log((x-1)/(x)) end



x=range(0,1,length=100)
y=sigm1.(10x.-7) # vari apo 0 ws 100, b apo 0 ws -100
plot(x,y)
y=sigm1.(100x.-70) # vari apo 0 ws 100, b apo 0 ws -100
plot!(x,y)

y=sigm1.(100x.-30)
plot(x,y)
y=sigm1.(100x.-50)
plot!(x,y)
y=sigm1.(100x.-30)+sigm1.(100x.-50)
plot!(x,y)
y=sigm1.(100x.-30)-sigm1.(100x.-50)
plot!(x,y)


y=sigm1.(100x.-30)-2sigm1.(100x.-50)
plot(x,y)
y=sigm1.(100x.-30)-2sigm1.(100x.-50)+3sigm1.(100x.-70)
plot(x,y)
y=sigm1.(10x.-3)-2sigm1.(10x.-5)+3sigm1.(10x.-7)
plot!(x,y)


x1=rand(1000)
x2=rand(1000)
y=0sigm1.(100x1.-0)+sigm1.(100x2.-80)
scatter3d(x1,x2,y,zcolor=y,camera=(50,40))
y=sigm1.(100x1.-40)+sigm1.(100x2.-80)
scatter3d(x1,x2,y,zcolor=y,camera=(5,40))
y=3sigm1.(100x1.-40)-20sigm1.(100x2.-80)
scatter3d(x1,x2,y,zcolor=y,camera=(7,40))
y=3sigm1.(10x1.-4)-20sigm1.(10x2.-8)
scatter3d(x1,x2,y,zcolor=y,camera=(7,40))

# countr example, why regression cannot make smooth curves (+... higher order polynomial)
x=range(0,1,length=100)
y=(2x.-0.8).^2 
plot(x,y)
y=(2x.-1.2).^2 
plot!(x,y)
y=(2x.-0.8).^2-0.2(2x.-1.2).^2 
plot!(x,y)



obs=1000
vars=2
x=rand(obs,vars)
y=sin.(exp.(x[:,1])) .+cos.(5x[:,2])
y=(y.-minimum(y))./(maximum(y)-minimum(y))
y.*=0.8;y.+=0.1
scatter3d(x[:,1],x[:,2],y)
pred=1.3sigm1.(10x.-8)
scatter!(x,pred)
pred=1.2sigm1.(10x.-9).+0.07
scatter!(x,pred)
pred=0.7sigm1.(10x.-8)+0.3sigm1.(10x.-9).+0.01
scatter!(x,pred)
mean(abs.(y-pred))

opti_ww=rand(7)
opti_ww=[-9.91842, -1.59284, 4.58076, 1.2246, 8.79918, 8.4979, 0.185864]
pred=opti_ww[1]sigm1.(opti_ww[2]x[:,1].-opti_ww[3])+opti_ww[4]sigm1.(opti_ww[5]x[:,2].-opti_ww[6]).+opti_ww[7]
mean(abs.(y-pred))
scatter(x,y);scatter!(x,pred)


# func_evals=Int64(10000)
# include("run_its_4.jl")
# lb=-10ones(7)
# ub=-lb
# opti_ww,lb,ub=run_its_4(7,x,y,obs,func_evals,lb,ub,vars,opti_ww)
# pred=opti_ww[1]sigm1.(opti_ww[2]x.-opti_ww[3])+opti_ww[4]sigm1.(opti_ww[5]x.-opti_ww[6]).+opti_ww[7]
# mean(abs.(y-pred))
# scatter!(x,pred)