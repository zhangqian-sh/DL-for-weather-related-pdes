include("usingDG.jl")
include("EulerEqs.jl")
# Example 1 , sine initial condition
function bdr(xᵧ,yᵧ,time_now,nx,ny,ρ⁻,m⁻,n⁻,E⁻)
    ρ⁺=ρ⁻
    m⁺=m⁻
    n⁺=n⁻
    E⁺=E⁻
    return ρ⁺,m⁺,n⁺,E⁺
end
γ=1.4
ρ₀(x::Real,y::Real)=1+0.5sin(x+y); ρ₀(x::AbstractArray,y::AbstractArray)=ρ₀.(x,y);
m₀(x::Real,y::Real)=1+0.5sin(x+y); m₀(x::AbstractArray,y::AbstractArray)=m₀.(x,y);
n₀(x::Real,y::Real)=1+0.5sin(x+y); n₀(x::AbstractArray,y::AbstractArray)=n₀.(x,y);
E₀(x::Real,y::Real)=(1+0.5sin(x+y))+1/(γ-1); E₀(x::AbstractArray,y::AbstractArray)=E₀.(x,y);
ρ(x,y,t)=ρ₀(x-t,y-t)
CFL=0.1
P_order=2
T=1.0
α=1.1
xmin=0.0
ymin=0.0
xmax=2π
ymax=2π

Cell_N=60
Cell_M=60
mesh=Grid2DRect(xmin,xmax,ymin,ymax,Cell_N,Cell_M;Periodic=true)
mesh,Coef_ρ,Coef_m,Coef_n,Coef_E,time_now,num_iter,basisP = main(mesh,ρ₀,m₀,n₀,E₀,γ;
            CFL=CFL,P_order=P_order,T=T,α=α,bdr=bdr)

L2err=errLp(ρ,mesh,Coef_ρ,time_now;p=2)

# Example 2 , isentropic vortex advection problem
γ=1.4
ρ₀(x::Real,y::Real)=TT(x,y)^(1/(γ-1)); ρ₀(x::AbstractArray,y::AbstractArray)=ρ₀.(x,y);
m₀(x::Real,y::Real)=ρ₀(x,y)*(1-(y-5)*phi(x,y)); m₀(x::AbstractArray,y::AbstractArray)=m₀.(x,y);
n₀(x::Real,y::Real)=ρ₀(x,y)*(1+(x-5)*phi(x,y)); n₀(x::AbstractArray,y::AbstractArray)=n₀.(x,y);
E₀(x::Real,y::Real)=1/2*m₀(x,y)^2/ρ₀(x,y)+1/2*n₀(x,y)^2/ρ₀(x,y)+TT(x,y)^(γ/(γ-1))/(γ-1); E₀(x::AbstractArray,y::AbstractArray)=E₀.(x,y);
TT(x,y)=1-(γ-1)/(2γ)*phi(x,y)^2
phi(x,y)=5/(2π)*exp(0.5*(1-(x-5)^2-(y-5)^2))
ρ(x,y,t)=ρ₀(x-t,y-t)
function bdr(xᵧ,yᵧ,time_now,nx,ny,ρ⁻,m⁻,n⁻,E⁻)
    ρ⁺=ρ₀(xᵧ-time_now,yᵧ-time_now)
    m⁺=m₀(xᵧ-time_now,yᵧ-time_now)
    n⁺=n₀(xᵧ-time_now,yᵧ-time_now)
    E⁺=E₀(xᵧ-time_now,yᵧ-time_now)
    return ρ⁺,m⁺,n⁺,E⁺
end
CFL=0.1
P_order=2
T=1.0
α=1.1
xmin=0.0
ymin=0.0
xmax=10.0
ymax=10.0

Cell_N=60
Cell_M=60
mesh=Grid2DRect(xmin,xmax,ymin,ymax,Cell_N,Cell_M;Periodic=false)
mesh,Coef_ρ,Coef_m,Coef_n,Coef_E,time_now,num_iter,basisP = main(mesh,ρ₀,m₀,n₀,E₀,γ;
            CFL=CFL,P_order=P_order,T=T,α=α,bdr=bdr)

L2err=errLp(ρ,mesh,Coef_ρ,time_now;p=2)


# Example 3 , Burgers' like initial condition
function bdr(xᵧ,yᵧ,time_now,nx,ny,ρ⁻,m⁻,n⁻,E⁻)
    ρ⁺=ρ⁻
    m⁺=m⁻
    n⁺=n⁻
    E⁺=E⁻
    return ρ⁺,m⁺,n⁺,E⁺
end
γ=3
ρ₀(x::Real,y::Real)=(1+0.2sin((x+y)/2))/sqrt(6); ρ₀(x::AbstractArray,y::AbstractArray)=ρ₀.(x,y);
m₀(x::Real,y::Real)=ρ₀(x,y)*sqrt(γ/2)*ρ₀(x,y); m₀(x::AbstractArray,y::AbstractArray)=m₀.(x,y);
n₀(x::Real,y::Real)=ρ₀(x,y)*sqrt(γ/2)*ρ₀(x,y); n₀(x::AbstractArray,y::AbstractArray)=n₀.(x,y);
E₀(x::Real,y::Real)=1/2*m₀(x,y)^2/ρ₀(x,y)+1/2*n₀(x,y)^2/ρ₀(x,y)+ρ₀(x,y)^γ/(γ-1); E₀(x::AbstractArray,y::AbstractArray)=E₀.(x,y);
ρ(x,y,t)=Burgers(x/2,y/2,t/2;α=0.2,β=1.0)/sqrt(6)
CFL=0.1
P_order=2
T=0.5
α=1.1
xmin=0.0
ymin=0.0
xmax=4π
ymax=4π

Cell_N=60
Cell_M=60
mesh=Grid2DRect(xmin,xmax,ymin,ymax,Cell_N,Cell_M;Periodic=true)
mesh,Coef_ρ,Coef_m,Coef_n,Coef_E,time_now,num_iter,basisP = main(mesh,ρ₀,m₀,n₀,E₀,γ;
            CFL=CFL,P_order=P_order,T=T,α=α,bdr=bdr)

L2err=errLp(ρ,mesh,Coef_ρ,time_now;p=2)


# Exact Riemann solver
x=collect(range(-1,1,length=500))
t=0.15
left_rho=1.
left_u=-2.
left_p=0.4
left_gamma=1.4
right_rho=1.
right_u=2.
right_p=0.4
right_gamma=1.4
profiles=RiemannSolver(x, t,left_rho,left_u,left_p,left_gamma,right_rho,right_u,right_p,right_gamma)

density = [status.rho for status in profiles]
velocity = [status.u for status in profiles]
pressure = [status.p for status in profiles]
plot(x,density)
#=
file = matopen("Riemann.mat", "w")
write(file, "density", density)
write(file, "velocity", velocity)
write(file, "pressure", pressure)
write(file, "x", x)
close(file)
=#

#=
#  2D Double Rarefaction Riemann Problem
function bdr(x,y,t,nx,ny,ρ⁻,m⁻,n⁻,E⁻)
    if nx>0.5 || nx<-0.5
        if x<0 # left
            m⁺=-2.0
        else # right
            m⁺=2.0
        end
        ρ⁺=1.0
        n⁺=0.0
        E⁺=3.0
    else
        ρ⁺=ρ⁻
        m⁺=m⁻
        n⁺=-n⁻
        E⁺=E⁻
    end
    return ρ⁺,m⁺,n⁺,E⁺
end       
γ=1.4
ρ₀(x,y)=1.0
m₀(x,y)=(x<0 ? -2.0 : 2.0)
n₀(x,y)=0.0
E₀(x,y)=3.0
ρ₀(x::AbstractArray,y::AbstractArray)=ρ₀.(x,y)
m₀(x::AbstractArray,y::AbstractArray)=m₀.(x,y)
n₀(x::AbstractArray,y::AbstractArray)=n₀.(x,y)
E₀(x::AbstractArray,y::AbstractArray)=E₀.(x,y)
xmin = -1.
xmax = 1.
ymin = -1.
ymax = 1.
Cell_N=80
Cell_M=80
mesh=Grid2DRect(xmin,xmax,ymin,ymax,Cell_N,Cell_M;Periodic=false)

mesh,Coef_ρ,Coef_m,Coef_n,Coef_E,time_now,num_iter,basisP = main(mesh,ρ₀,m₀,n₀,E₀,γ;
            CFL=0.01,P_order=2,T=0.15,α=1.1,bdr=bdr)

=#


