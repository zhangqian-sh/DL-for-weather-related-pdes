function Burgers(x,t;α::Real=1.0,β::Real=0.0)::Float64
    y=mod(x-β*t,2π)
    if y≤π
        flag=2
    else
        flag=1
        y=2π-y
    end
    N=100
    y0=range(0,π,length=N)
    y1=similar(y0)
    @.y1=y0+α*sin(y0)*t
    ℓ= α≥0 ? findlast(y1.≤y) : findfirst(y1.≥y)
    v=α*sin(y0[ℓ])
    err=abs(v-α*sin(y-t*v))
    while err > 1e-13
        v=v-(v-α*sin(y-t*v))/(1+α*t*cos(y-t*v))
        err=abs(v-α*sin(y-t*v))
    end
    u=(-1)^flag*v+β
end
function Burgers(x::AbstractArray,t;α::Real=1.0,β::Real=0.0)
    N=100
    y0=range(0,π,length=N)
    y1=y0+α*sin.(y0)*t
    y=mod.(x.-β*t,2π)
    u=similar(y)
    @inbounds Threads.@threads for k=1:length(u)
        if y[k]≤π
            flag=2
        else
            flag=1
            y[k]=2π-y[k]
        end
        ℓ= α≥0 ? findlast(y1.≤y[k]) : findfirst(y1.≥y[k])
        v=α*sin(y0[ℓ])
        err=abs(v-α*sin(y[k]-t*v))
        while err > 1e-13
            v=v-(v-α*sin(y[k]-t*v))/(1+α*t*cos(y[k]-t*v))
            err=abs(v-α*sin(y[k]-t*v))
        end
        u[k]=(-1)^flag*v+β
    end
    u
end

function Burgers(x,y,t;α::Real=1.0,β::Real=0.0)
    Burgers(x+y,2t;α=α,β=β)
end



function errLp(u::Function,mesh::Grid2DRect,U::AbstractMatrix,
    t,quad1D_num::Integer=10;p=2)::Float64
    P_dof=size(U,1)
    P_order=isqrt(P_dof)-1
    quad1D_p=range(-1,1,length=quad1D_num)
    quad_num=quad1D_num^2
    quad_p=Array{Float64,2}(undef,quad_num,2)
    quad_w=1/quad_num
    for ℓ₁=1:quad1D_num, ℓ₂=1:quad1D_num
        ℓ=(ℓ₂-1)*quad1D_num+ℓ₁ 
        quad_p[ℓ,1]=quad1D_p[ℓ₁]
        quad_p[ℓ,2]=quad1D_p[ℓ₂]
    end
    basisP = Array{Float64,2}(undef,P_dof,quad_num)
    for i=0:P_order, j=0:P_order, ℓ=1:quad_num
        basisP[i*(P_order+1)+j+1,ℓ]=JacobiP(i,0,0,quad_p[ℓ,1])*JacobiP(j,0,0,quad_p[ℓ,2])
    end
    quad_p = ( quad_p .+1 ) / 2 # uniform_p now is in [0,1].
    err = 0.0
    uₕ=basisP'*U
    if p<Inf
        @inbounds for j=1:mesh.NumofE
            x₁=mesh.Coordinates[mesh.Elements[1,j],1]
            x₂=mesh.Coordinates[mesh.Elements[2,j],1]
            y₁=mesh.Coordinates[mesh.Elements[1,j],2]
            y₂=mesh.Coordinates[mesh.Elements[4,j],2]
            for ℓ=1:quad_num
                x_ℓj=(1-quad_p[ℓ,1])*x₁+quad_p[ℓ,1]*x₂
                y_ℓj=(1-quad_p[ℓ,2])*y₁+quad_p[ℓ,2]*y₂
                u_ℓj=u(x_ℓj,y_ℓj,t)
                err+=mesh.Tarea[j]*quad_w*abs(u_ℓj-uₕ[ℓ,j])^p
            end
        end
        err=err^(1/p)
    else
        @inbounds for j=1:mesh.NumofE
            x₁=mesh.Coordinates[mesh.Elements[1,j],1]
            x₂=mesh.Coordinates[mesh.Elements[2,j],1]
            y₁=mesh.Coordinates[mesh.Elements[1,j],2]
            y₂=mesh.Coordinates[mesh.Elements[4,j],2]
            for ℓ=1:quad_num
                x_ℓj=(1-quad_p[ℓ,1])*x₁+quad_p[ℓ,1]*x₂
                y_ℓj=(1-quad_p[ℓ,2])*y₁+quad_p[ℓ,2]*y₂
                u_ℓj=u(x_ℓj,y_ℓj,t)
                Δu_ℓj=abs(u_ℓj-uₕ[ℓ,j])
                err=max(err,Δu_ℓj)
            end
        end
    end
    err
end
