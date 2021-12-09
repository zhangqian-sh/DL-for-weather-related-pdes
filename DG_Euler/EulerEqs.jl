# Euler equations in 2D space
function main(mesh::AbstractGrid,ρ₀::Function,m₀::Function,n₀::Function,E₀::Function,γ;
            CFL=0.04,P_order=2,T=0.0,α=1.1,bdr::Function=+)
quad1D_num=P_order*3+3 
quad1D_p,quad1D_w=gausslegendre(quad1D_num) 
P_dof=(P_order+1)^2
quad_num=quad1D_num^2
quad_p=Array{Float64,2}(undef,quad_num,2)
quad_w=Array{Float64,1}(undef,quad_num)
for ℓ₂=1:quad1D_num, ℓ₁=1:quad1D_num
    ℓ=(ℓ₂-1)*quad1D_num+ℓ₁ 
    quad_p[ℓ,1]=quad1D_p[ℓ₁]
    quad_p[ℓ,2]=quad1D_p[ℓ₂]
    quad_w[ℓ]=quad1D_w[ℓ₁]*quad1D_w[ℓ₂]
end
basis1DP=Array{Float64,2}(undef,P_order+1,quad1D_num) 
Gradbasis1DP=Array{Float64,2}(undef,P_order+1,quad1D_num) 
Grad2basis1DP=Array{Float64,2}(undef,P_order+1,quad1D_num) 
for i = 0 : P_order
    for ℓ = 1 : quad1D_num
        basis1DP[i+1,ℓ]     = JacobiP(i,0,0,quad1D_p[ℓ])
        Gradbasis1DP[i+1,ℓ] = GradnJacobiP(i,0,0,quad1D_p[ℓ],1)
        Grad2basis1DP[i+1,ℓ] = GradnJacobiP(i,0,0,quad1D_p[ℓ],2)
    end
end
basisP=Array{Float64,2}(undef,P_dof,quad_num)
GradrbasisP=Array{Float64,2}(undef,P_dof,quad_num)
GradsbasisP=Array{Float64,2}(undef,P_dof,quad_num)
Grad2rrbasisP=Array{Float64,2}(undef,P_dof,quad_num)
Grad2rsbasisP=Array{Float64,2}(undef,P_dof,quad_num)
Grad2ssbasisP=Array{Float64,2}(undef,P_dof,quad_num)
boundaryP=Array{Float64,2}(undef,P_dof,quad1D_num*4)
GradrboundaryP=Array{Float64,2}(undef,P_dof,quad1D_num*4)
GradsboundaryP=Array{Float64,2}(undef,P_dof,quad1D_num*4)
Grad2rrboundaryP=Array{Float64,2}(undef,P_dof,quad1D_num*4)
Grad2rsboundaryP=Array{Float64,2}(undef,P_dof,quad1D_num*4)
Grad2ssboundaryP=Array{Float64,2}(undef,P_dof,quad1D_num*4)
for i=0:P_order, j=0:P_order, ℓ=1:quad_num
    basisP[i*(P_order+1)+j+1,ℓ]=JacobiP(i,0,0,quad_p[ℓ,1])*JacobiP(j,0,0,quad_p[ℓ,2])
    GradrbasisP[i*(P_order+1)+j+1,ℓ]=GradnJacobiP(i,0,0,quad_p[ℓ,1],1)*JacobiP(j,0,0,quad_p[ℓ,2])
    GradsbasisP[i*(P_order+1)+j+1,ℓ]=JacobiP(i,0,0,quad_p[ℓ,1])*GradnJacobiP(j,0,0,quad_p[ℓ,2],1)
    Grad2rrbasisP[i*(P_order+1)+j+1,ℓ]=GradnJacobiP(i,0,0,quad_p[ℓ,1],2)*JacobiP(j,0,0,quad_p[ℓ,2])
    Grad2rsbasisP[i*(P_order+1)+j+1,ℓ]=GradnJacobiP(i,0,0,quad_p[ℓ,1],1)*GradnJacobiP(j,0,0,quad_p[ℓ,2],1)
    Grad2ssbasisP[i*(P_order+1)+j+1,ℓ]=JacobiP(i,0,0,quad_p[ℓ,1])*GradnJacobiP(j,0,0,quad_p[ℓ,2],2)
end
for i=0:P_order, j=0:P_order, ℓ=1:quad1D_num
    boundaryP[i*(P_order+1)+j+1,ℓ]=JacobiP(i,0,0,quad1D_p[ℓ])*JacobiP(j,0,0,-1.)
    boundaryP[i*(P_order+1)+j+1,ℓ+quad1D_num]=JacobiP(i,0,0,1.)*JacobiP(j,0,0,quad1D_p[ℓ])
    boundaryP[i*(P_order+1)+j+1,ℓ+2*quad1D_num]=JacobiP(i,0,0,quad1D_p[ℓ])*JacobiP(j,0,0,1.)
    boundaryP[i*(P_order+1)+j+1,ℓ+3*quad1D_num]=JacobiP(i,0,0,-1.)*JacobiP(j,0,0,quad1D_p[ℓ])

    GradrboundaryP[i*(P_order+1)+j+1,ℓ]=GradnJacobiP(i,0,0,quad1D_p[ℓ],1)*JacobiP(j,0,0,-1.)
    GradrboundaryP[i*(P_order+1)+j+1,ℓ+quad1D_num]=GradnJacobiP(i,0,0,1.,1)*JacobiP(j,0,0,quad1D_p[ℓ])
    GradrboundaryP[i*(P_order+1)+j+1,ℓ+2*quad1D_num]=GradnJacobiP(i,0,0,quad1D_p[ℓ],1)*JacobiP(j,0,0,1.)
    GradrboundaryP[i*(P_order+1)+j+1,ℓ+3*quad1D_num]=GradnJacobiP(i,0,0,-1.,1)*JacobiP(j,0,0,quad1D_p[ℓ])

    GradsboundaryP[i*(P_order+1)+j+1,ℓ]=JacobiP(i,0,0,quad1D_p[ℓ])*GradnJacobiP(j,0,0,-1.,1)
    GradsboundaryP[i*(P_order+1)+j+1,ℓ+quad1D_num]=JacobiP(i,0,0,1.)*GradnJacobiP(j,0,0,quad1D_p[ℓ],1)
    GradsboundaryP[i*(P_order+1)+j+1,ℓ+2*quad1D_num]=JacobiP(i,0,0,quad1D_p[ℓ])*GradnJacobiP(j,0,0,1.,1)
    GradsboundaryP[i*(P_order+1)+j+1,ℓ+3*quad1D_num]=JacobiP(i,0,0,-1.)*GradnJacobiP(j,0,0,quad1D_p[ℓ],1)

    Grad2rrboundaryP[i*(P_order+1)+j+1,ℓ]=GradnJacobiP(i,0,0,quad1D_p[ℓ],2)*JacobiP(j,0,0,-1.)
    Grad2rrboundaryP[i*(P_order+1)+j+1,ℓ+quad1D_num]=GradnJacobiP(i,0,0,1.,2)*JacobiP(j,0,0,quad1D_p[ℓ])
    Grad2rrboundaryP[i*(P_order+1)+j+1,ℓ+2*quad1D_num]=GradnJacobiP(i,0,0,quad1D_p[ℓ],2)*JacobiP(j,0,0,1.)
    Grad2rrboundaryP[i*(P_order+1)+j+1,ℓ+3*quad1D_num]=GradnJacobiP(i,0,0,-1.,2)*JacobiP(j,0,0,quad1D_p[ℓ])

    Grad2rsboundaryP[i*(P_order+1)+j+1,ℓ]=GradnJacobiP(i,0,0,quad1D_p[ℓ],1)*GradnJacobiP(j,0,0,-1.,1)
    Grad2rsboundaryP[i*(P_order+1)+j+1,ℓ+quad1D_num]=GradnJacobiP(i,0,0,1.,1)*GradnJacobiP(j,0,0,quad1D_p[ℓ],1)
    Grad2rsboundaryP[i*(P_order+1)+j+1,ℓ+2*quad1D_num]=GradnJacobiP(i,0,0,quad1D_p[ℓ],1)*GradnJacobiP(j,0,0,1.,1)
    Grad2rsboundaryP[i*(P_order+1)+j+1,ℓ+3*quad1D_num]=GradnJacobiP(i,0,0,-1.,1)*GradnJacobiP(j,0,0,quad1D_p[ℓ],1)

    Grad2ssboundaryP[i*(P_order+1)+j+1,ℓ]=JacobiP(i,0,0,quad1D_p[ℓ])*GradnJacobiP(j,0,0,-1.,2)
    Grad2ssboundaryP[i*(P_order+1)+j+1,ℓ+quad1D_num]=JacobiP(i,0,0,1.)*GradnJacobiP(j,0,0,quad1D_p[ℓ],2)
    Grad2ssboundaryP[i*(P_order+1)+j+1,ℓ+2*quad1D_num]=JacobiP(i,0,0,quad1D_p[ℓ])*GradnJacobiP(j,0,0,1.,2)
    Grad2ssboundaryP[i*(P_order+1)+j+1,ℓ+3*quad1D_num]=JacobiP(i,0,0,-1.)*GradnJacobiP(j,0,0,quad1D_p[ℓ],2)
end
@.quad1D_p = ( quad1D_p + 1 ) / 2 
@.quad1D_w = quad1D_w / 2 
@.quad_p = ( quad_p + 1 ) / 2 
@.quad_w = quad_w / 4 
Coef_ρ = zeros(Float64,P_dof,mesh.NumofE) 
Coef_m = zeros(Float64,P_dof,mesh.NumofE) 
Coef_n = zeros(Float64,P_dof,mesh.NumofE) 
Coef_E = zeros(Float64,P_dof,mesh.NumofE) 
x = Array{Float64,2}(undef,quad_num,mesh.NumofE)
y = Array{Float64,2}(undef,quad_num,mesh.NumofE)
@inbounds Threads.@threads for j=1:mesh.NumofE
    x₁=mesh.Coordinates[mesh.Elements[1,j],1]
    x₂=mesh.Coordinates[mesh.Elements[2,j],1]
    y₁=mesh.Coordinates[mesh.Elements[1,j],2]
    y₂=mesh.Coordinates[mesh.Elements[4,j],2]
    for ℓ=1:quad_num
        x[ℓ,j]=(1-quad_p[ℓ,1])*x₁+quad_p[ℓ,1]*x₂
        y[ℓ,j]=(1-quad_p[ℓ,2])*y₁+quad_p[ℓ,2]*y₂
    end
end
ρ0=ρ₀(x,y)
m0=m₀(x,y)
n0=n₀(x,y)
E0=E₀(x,y)
@inbounds for j=1:mesh.NumofE
    for ℓ=1:quad_num
        for mm=1:P_dof
            Coef_ρ[mm,j]+=quad_w[ℓ]*basisP[mm,ℓ]*ρ0[ℓ,j]
            Coef_m[mm,j]+=quad_w[ℓ]*basisP[mm,ℓ]*m0[ℓ,j]
            Coef_n[mm,j]+=quad_w[ℓ]*basisP[mm,ℓ]*n0[ℓ,j]
            Coef_E[mm,j]+=quad_w[ℓ]*basisP[mm,ℓ]*E0[ℓ,j]
        end
    end
    for mm=1:P_dof
        Coef_ρ[mm,j]=Coef_ρ[mm,j]/(1/4)
        Coef_m[mm,j]=Coef_m[mm,j]/(1/4)
        Coef_n[mm,j]=Coef_n[mm,j]/(1/4)
        Coef_E[mm,j]=Coef_E[mm,j]/(1/4)
    end
end
time_init  = 0.0
time_final = T
time_now = time_init

num_iter=0
while time_now < time_final
    Coef_ρₜ,Coef_mₜ,Coef_nₜ,Coef_Eₜ,λₘₐₓ,μₘₐₓ =
        RHS(mesh,time_now,Coef_ρ,Coef_m,Coef_n,Coef_E,basisP,GradrbasisP,GradsbasisP,Grad2rrbasisP,Grad2rsbasisP,Grad2ssbasisP,
            boundaryP,GradrboundaryP,GradsboundaryP,Grad2rrboundaryP,Grad2rsboundaryP,Grad2ssboundaryP,
            quad_num,quad_p,quad_w,quad1D_num,quad1D_p,quad1D_w,γ,α,bdr)
    Δt=(CFL/(λₘₐₓ+μₘₐₓ)*sqrt(minimum(mesh.Tarea)))^((P_order+1)/3)
    Coef_ρ1=Coef_ρ+Δt*Coef_ρₜ
    Coef_m1=Coef_m+Δt*Coef_mₜ
    Coef_n1=Coef_n+Δt*Coef_nₜ
    Coef_E1=Coef_E+Δt*Coef_Eₜ

    Coef_ρ1ₜ,Coef_m1ₜ,Coef_n1ₜ,Coef_E1ₜ,λₘₐₓ,μₘₐₓ =
        RHS(mesh,time_now+Δt,Coef_ρ1,Coef_m1,Coef_n1,Coef_E1,basisP,GradrbasisP,GradsbasisP,Grad2rrbasisP,Grad2rsbasisP,Grad2ssbasisP,
            boundaryP,GradrboundaryP,GradsboundaryP,Grad2rrboundaryP,Grad2rsboundaryP,Grad2ssboundaryP,
            quad_num,quad_p,quad_w,quad1D_num,quad1D_p,quad1D_w,γ,α,bdr)
    Coef_ρ2=3/4*Coef_ρ+1/4*(Coef_ρ1+Δt*Coef_ρ1ₜ)
    Coef_m2=3/4*Coef_m+1/4*(Coef_m1+Δt*Coef_m1ₜ)
    Coef_n2=3/4*Coef_n+1/4*(Coef_n1+Δt*Coef_n1ₜ)
    Coef_E2=3/4*Coef_E+1/4*(Coef_E1+Δt*Coef_E1ₜ)

    Coef_ρ2ₜ,Coef_m2ₜ,Coef_n2ₜ,Coef_E2ₜ,λₘₐₓ,μₘₐₓ =
        RHS(mesh,time_now+Δt/2,Coef_ρ2,Coef_m2,Coef_n2,Coef_E2,basisP,GradrbasisP,GradsbasisP,Grad2rrbasisP,Grad2rsbasisP,Grad2ssbasisP,
            boundaryP,GradrboundaryP,GradsboundaryP,Grad2rrboundaryP,Grad2rsboundaryP,Grad2ssboundaryP,
            quad_num,quad_p,quad_w,quad1D_num,quad1D_p,quad1D_w,γ,α,bdr)
    Coef_ρ=1/3*Coef_ρ+2/3*(Coef_ρ2+Δt*Coef_ρ2ₜ)
    Coef_m=1/3*Coef_m+2/3*(Coef_m2+Δt*Coef_m2ₜ)
    Coef_n=1/3*Coef_n+2/3*(Coef_n2+Δt*Coef_n2ₜ)
    Coef_E=1/3*Coef_E+2/3*(Coef_E2+Δt*Coef_E2ₜ)

    time_now+=Δt
    num_iter+=1
end
return mesh,Coef_ρ,Coef_m,Coef_n,Coef_E,time_now,num_iter,basisP
end

function RHS(mesh,time_now,Coef_ρ,Coef_m,Coef_n,Coef_E,ϕ,∂ᵣϕ,∂ₛϕ,∂ᵣᵣϕ,∂ᵣₛϕ,∂ₛₛϕ,
    ϕᵧ,∂ᵣϕᵧ,∂ₛϕᵧ,∂ᵣᵣϕᵧ,∂ᵣₛϕᵧ,∂ₛₛϕᵧ,Nq,x̂,ω,Nqᵧ,x̂ᵧ,ωᵧ,γ,α,bdr::Function)
    Coef_ρₜ=zeros(eltype(Coef_ρ),size(Coef_ρ))
    Coef_mₜ=zeros(eltype(Coef_ρ),size(Coef_ρ))
    Coef_nₜ=zeros(eltype(Coef_ρ),size(Coef_ρ))
    Coef_Eₜ=zeros(eltype(Coef_ρ),size(Coef_ρ))
    λₘₐₓ=0.0
    μₘₐₓ=0.0
    P_dof=size(ϕ,1)
    nx=[0.,1.,0.,-1.]
    ny=[-1.,0.,1.,0.]
    EdtoVmap=[1 2;2 3;4 3;1 4]
    for j=1:mesh.NumofE
        for ℓ=1:Nq
            ρ=0.0;m=0.0;n=0.0;E=0.0;
            for mm=1:P_dof
                ρ+=Coef_ρ[mm,j]*ϕ[mm,ℓ]
                m+=Coef_m[mm,j]*ϕ[mm,ℓ]
                n+=Coef_n[mm,j]*ϕ[mm,ℓ]
                E+=Coef_E[mm,j]*ϕ[mm,ℓ]
            end
            c=sqrt(γ*(γ-1)*(E-1/2*m^2/ρ-1/2*n^2/ρ)/ρ)
            u=m/ρ
            v=n/ρ
            λₘₐₓ=max(λₘₐₓ,abs(u)+c)
            μₘₐₓ=max(μₘₐₓ,abs(v)+c)
        end
        for i=1:4 
            EE=mesh.EtoEmap[i,j]
            if EE == -1 
                V₁ = EdtoVmap[i,1]
                V₂ = EdtoVmap[i,2]
                xᵥ₁ = mesh.Coordinates[mesh.Elements[V₁,j],1]
                xᵥ₂ = mesh.Coordinates[mesh.Elements[V₂,j],1]
                yᵥ₁ = mesh.Coordinates[mesh.Elements[V₁,j],2]
                yᵥ₂ = mesh.Coordinates[mesh.Elements[V₂,j],2]
                for ℓ=1:Nqᵧ
                    ℓ⁻=ℓ+(i-1)*Nqᵧ
                    xᵧ=(1-x̂ᵧ[ℓ])*xᵥ₁+x̂ᵧ[ℓ]*xᵥ₂
                    yᵧ=(1-x̂ᵧ[ℓ])*yᵥ₁+x̂ᵧ[ℓ]*yᵥ₂
                    ρ⁻=0.0;m⁻=0.0;n⁻=0.0;E⁻=0.0;
                    for mm=1:P_dof
                        ρ⁻+=Coef_ρ[mm,j]*ϕᵧ[mm,ℓ⁻]
                        m⁻+=Coef_m[mm,j]*ϕᵧ[mm,ℓ⁻]
                        n⁻+=Coef_n[mm,j]*ϕᵧ[mm,ℓ⁻]
                        E⁻+=Coef_E[mm,j]*ϕᵧ[mm,ℓ⁻]
                    end
                    ρ⁺,m⁺,n⁺,E⁺ =
                        bdr(xᵧ,yᵧ,time_now,nx[i],ny[i],ρ⁻,m⁻,n⁻,E⁻)
                    c⁺=sqrt(γ*(γ-1)*(E⁺-1/2*m⁺^2/ρ⁺-1/2*n⁺^2/ρ⁺)/ρ⁺)
                    u⁺=m⁺/ρ⁺
                    v⁺=n⁺/ρ⁺
                    λₘₐₓ=max(λₘₐₓ,abs(u⁺)+c⁺)
                    μₘₐₓ=max(μₘₐₓ,abs(v⁺)+c⁺)
                end
            end
        end
    end

    for j=1:mesh.NumofE
        Aⱼ=mesh.Tarea[j]
        ∂r∂x=mesh.Jacobians[1,1,j]
        ∂s∂y=mesh.Jacobians[2,2,j]
        for ℓ=1:Nq
            ρ=0.0;m=0.0;n=0.0;E=0.0;
            for mm=1:P_dof
                ρ+=Coef_ρ[mm,j]*ϕ[mm,ℓ]
                m+=Coef_m[mm,j]*ϕ[mm,ℓ]
                n+=Coef_n[mm,j]*ϕ[mm,ℓ]
                E+=Coef_E[mm,j]*ϕ[mm,ℓ]
            end
            u=m/ρ
            v=n/ρ        
            □₁ₓ=m
            □₁ₙ=n
            □₂ₓ=(γ-1)*E+(3-γ)/2*m*u-(γ-1)/2*n*v
            □₂ₙ=m*v
            □₃ₓ=n*u
            □₃ₙ=(γ-1)*E-(γ-1)/2*m*u+(3-γ)/2*n*v
            □₄ₓ=γ*E*u-(γ-1)/2*m*u^2-(γ-1)/2*m*v^2
            □₄ₙ=γ*E*v-(γ-1)/2*n*u^2-(γ-1)/2*n*v^2
            Ψ₁ₓ=Aⱼ*ω[ℓ]*∂r∂x*□₁ₓ
            Ψ₁ₙ=Aⱼ*ω[ℓ]*∂s∂y*□₁ₙ
            Ψ₂ₓ=Aⱼ*ω[ℓ]*∂r∂x*□₂ₓ
            Ψ₂ₙ=Aⱼ*ω[ℓ]*∂s∂y*□₂ₙ
            Ψ₃ₓ=Aⱼ*ω[ℓ]*∂r∂x*□₃ₓ
            Ψ₃ₙ=Aⱼ*ω[ℓ]*∂s∂y*□₃ₙ
            Ψ₄ₓ=Aⱼ*ω[ℓ]*∂r∂x*□₄ₓ
            Ψ₄ₙ=Aⱼ*ω[ℓ]*∂s∂y*□₄ₙ
            for mm=1:P_dof
                Coef_ρₜ[mm,j]+=∂ᵣϕ[mm,ℓ]*Ψ₁ₓ+∂ₛϕ[mm,ℓ]*Ψ₁ₙ
                Coef_mₜ[mm,j]+=∂ᵣϕ[mm,ℓ]*Ψ₂ₓ+∂ₛϕ[mm,ℓ]*Ψ₂ₙ
                Coef_nₜ[mm,j]+=∂ᵣϕ[mm,ℓ]*Ψ₃ₓ+∂ₛϕ[mm,ℓ]*Ψ₃ₙ
                Coef_Eₜ[mm,j]+=∂ᵣϕ[mm,ℓ]*Ψ₄ₓ+∂ₛϕ[mm,ℓ]*Ψ₄ₙ
            end
        end
        for i=1:4
            hᵢ=mesh.Elength[i,j]
            EE=mesh.EtoEmap[i,j]
            pp=mod(i+2-1,4)+1
            if EE != -1 
                for ℓ=1:Nqᵧ
                    ∂r′∂x′=mesh.Jacobians[1,1,EE]
                    ∂s′∂y′=mesh.Jacobians[2,2,EE]
                    ℓ⁻=ℓ+(i-1)*Nqᵧ
                    ℓ⁺=ℓ+(pp-1)*Nqᵧ
                    ρ⁻=0.0;m⁻=0.0;n⁻=0.0;E⁻=0.0;
                    ρ⁺=0.0;m⁺=0.0;n⁺=0.0;E⁺=0.0;
                    for mm=1:P_dof
                        ρ⁻+=Coef_ρ[mm,j]*ϕᵧ[mm,ℓ⁻]
                        m⁻+=Coef_m[mm,j]*ϕᵧ[mm,ℓ⁻]
                        n⁻+=Coef_n[mm,j]*ϕᵧ[mm,ℓ⁻]
                        E⁻+=Coef_E[mm,j]*ϕᵧ[mm,ℓ⁻]

                        ρ⁺+=Coef_ρ[mm,EE]*ϕᵧ[mm,ℓ⁺]
                        m⁺+=Coef_m[mm,EE]*ϕᵧ[mm,ℓ⁺]
                        n⁺+=Coef_n[mm,EE]*ϕᵧ[mm,ℓ⁺]
                        E⁺+=Coef_E[mm,EE]*ϕᵧ[mm,ℓ⁺]
                    end
                    u⁻=m⁻/ρ⁻
                    v⁻=n⁻/ρ⁻
                    u⁺=m⁺/ρ⁺
                    v⁺=n⁺/ρ⁺

                    □₁ₓ⁻=m⁻
                    □₁ₙ⁻=n⁻
                    □₂ₓ⁻=(γ-1)*E⁻+(3-γ)/2*m⁻*u⁻-(γ-1)/2*n⁻*v⁻
                    □₂ₙ⁻=m⁻*v⁻
                    □₃ₓ⁻=n⁻*u⁻
                    □₃ₙ⁻=(γ-1)*E⁻-(γ-1)/2*m⁻*u⁻+(3-γ)/2*n⁻*v⁻
                    □₄ₓ⁻=γ*E⁻*u⁻-(γ-1)/2*m⁻*u⁻^2-(γ-1)/2*m⁻*v⁻^2
                    □₄ₙ⁻=γ*E⁻*v⁻-(γ-1)/2*n⁻*u⁻^2-(γ-1)/2*n⁻*v⁻^2
        
                    □₁ₓ⁺=m⁺
                    □₁ₙ⁺=n⁺
                    □₂ₓ⁺=(γ-1)*E⁺+(3-γ)/2*m⁺*u⁺-(γ-1)/2*n⁺*v⁺
                    □₂ₙ⁺=m⁺*v⁺
                    □₃ₓ⁺=n⁺*u⁺
                    □₃ₙ⁺=(γ-1)*E⁺-(γ-1)/2*m⁺*u⁺+(3-γ)/2*n⁺*v⁺
                    □₄ₓ⁺=γ*E⁺*u⁺-(γ-1)/2*m⁺*u⁺^2-(γ-1)/2*m⁺*v⁺^2
                    □₄ₙ⁺=γ*E⁺*v⁺-(γ-1)/2*n⁺*u⁺^2-(γ-1)/2*n⁺*v⁺^2

                    □₁ₓ=1/2*(□₁ₓ⁻+□₁ₓ⁺+α*λₘₐₓ*(ρ⁻-ρ⁺)*nx[i])
                    □₁ₙ=1/2*(□₁ₙ⁻+□₁ₙ⁺+α*μₘₐₓ*(ρ⁻-ρ⁺)*ny[i])
                    □₂ₓ=1/2*(□₂ₓ⁻+□₂ₓ⁺+α*λₘₐₓ*(m⁻-m⁺)*nx[i])
                    □₂ₙ=1/2*(□₂ₙ⁻+□₂ₙ⁺+α*μₘₐₓ*(m⁻-m⁺)*ny[i])
                    □₃ₓ=1/2*(□₃ₓ⁻+□₃ₓ⁺+α*λₘₐₓ*(n⁻-n⁺)*nx[i])
                    □₃ₙ=1/2*(□₃ₙ⁻+□₃ₙ⁺+α*μₘₐₓ*(n⁻-n⁺)*ny[i])
                    □₄ₓ=1/2*(□₄ₓ⁻+□₄ₓ⁺+α*λₘₐₓ*(E⁻-E⁺)*nx[i])
                    □₄ₙ=1/2*(□₄ₙ⁻+□₄ₙ⁺+α*μₘₐₓ*(E⁻-E⁺)*ny[i])
                    Ψ₁=-hᵢ*ωᵧ[ℓ]*(□₁ₓ*nx[i]+□₁ₙ*ny[i])
                    Ψ₂=-hᵢ*ωᵧ[ℓ]*(□₂ₓ*nx[i]+□₂ₙ*ny[i])
                    Ψ₃=-hᵢ*ωᵧ[ℓ]*(□₃ₓ*nx[i]+□₃ₙ*ny[i])
                    Ψ₄=-hᵢ*ωᵧ[ℓ]*(□₄ₓ*nx[i]+□₄ₙ*ny[i])
                    for mm=1:P_dof
                        Coef_ρₜ[mm,j]+=ϕᵧ[mm,ℓ⁻]*Ψ₁
                        Coef_mₜ[mm,j]+=ϕᵧ[mm,ℓ⁻]*Ψ₂
                        Coef_nₜ[mm,j]+=ϕᵧ[mm,ℓ⁻]*Ψ₃
                        Coef_Eₜ[mm,j]+=ϕᵧ[mm,ℓ⁻]*Ψ₄
                    end
                end
            else 
                V₁ = EdtoVmap[i,1]
                V₂ = EdtoVmap[i,2]
                xᵥ₁ = mesh.Coordinates[mesh.Elements[V₁,j],1]
                xᵥ₂ = mesh.Coordinates[mesh.Elements[V₂,j],1]
                yᵥ₁ = mesh.Coordinates[mesh.Elements[V₁,j],2]
                yᵥ₂ = mesh.Coordinates[mesh.Elements[V₂,j],2]
                for ℓ=1:Nqᵧ
                    ℓ⁻=ℓ+(i-1)*Nqᵧ
                    ℓ⁺=ℓ+(pp-1)*Nqᵧ
                    xᵧ=(1-x̂ᵧ[ℓ])*xᵥ₁+x̂ᵧ[ℓ]*xᵥ₂
                    yᵧ=(1-x̂ᵧ[ℓ])*yᵥ₁+x̂ᵧ[ℓ]*yᵥ₂
                    ρ⁻=0.0;m⁻=0.0;n⁻=0.0;E⁻=0.0;
                    for mm=1:P_dof
                        ρ⁻+=Coef_ρ[mm,j]*ϕᵧ[mm,ℓ⁻]
                        m⁻+=Coef_m[mm,j]*ϕᵧ[mm,ℓ⁻]
                        n⁻+=Coef_n[mm,j]*ϕᵧ[mm,ℓ⁻]
                        E⁻+=Coef_E[mm,j]*ϕᵧ[mm,ℓ⁻]
                    end
                    ρ⁺,m⁺,n⁺,E⁺=bdr(xᵧ,yᵧ,time_now,nx[i],ny[i],ρ⁻,m⁻,n⁻,E⁻)

                    u⁻=m⁻/ρ⁻
                    v⁻=n⁻/ρ⁻
                    u⁺=m⁺/ρ⁺
                    v⁺=n⁺/ρ⁺

                    □₁ₓ⁻=m⁻
                    □₁ₙ⁻=n⁻
                    □₂ₓ⁻=(γ-1)*E⁻+(3-γ)/2*m⁻*u⁻-(γ-1)/2*n⁻*v⁻
                    □₂ₙ⁻=m⁻*v⁻
                    □₃ₓ⁻=n⁻*u⁻
                    □₃ₙ⁻=(γ-1)*E⁻-(γ-1)/2*m⁻*u⁻+(3-γ)/2*n⁻*v⁻
                    □₄ₓ⁻=γ*E⁻*u⁻-(γ-1)/2*m⁻*u⁻^2-(γ-1)/2*m⁻*v⁻^2
                    □₄ₙ⁻=γ*E⁻*v⁻-(γ-1)/2*n⁻*u⁻^2-(γ-1)/2*n⁻*v⁻^2
        
                    □₁ₓ⁺=m⁺
                    □₁ₙ⁺=n⁺
                    □₂ₓ⁺=(γ-1)*E⁺+(3-γ)/2*m⁺*u⁺-(γ-1)/2*n⁺*v⁺
                    □₂ₙ⁺=m⁺*v⁺
                    □₃ₓ⁺=n⁺*u⁺
                    □₃ₙ⁺=(γ-1)*E⁺-(γ-1)/2*m⁺*u⁺+(3-γ)/2*n⁺*v⁺
                    □₄ₓ⁺=γ*E⁺*u⁺-(γ-1)/2*m⁺*u⁺^2-(γ-1)/2*m⁺*v⁺^2
                    □₄ₙ⁺=γ*E⁺*v⁺-(γ-1)/2*n⁺*u⁺^2-(γ-1)/2*n⁺*v⁺^2

                    □₁ₓ=1/2*(□₁ₓ⁻+□₁ₓ⁺+α*λₘₐₓ*(ρ⁻-ρ⁺)*nx[i])
                    □₁ₙ=1/2*(□₁ₙ⁻+□₁ₙ⁺+α*μₘₐₓ*(ρ⁻-ρ⁺)*ny[i])
                    □₂ₓ=1/2*(□₂ₓ⁻+□₂ₓ⁺+α*λₘₐₓ*(m⁻-m⁺)*nx[i])
                    □₂ₙ=1/2*(□₂ₙ⁻+□₂ₙ⁺+α*μₘₐₓ*(m⁻-m⁺)*ny[i])
                    □₃ₓ=1/2*(□₃ₓ⁻+□₃ₓ⁺+α*λₘₐₓ*(n⁻-n⁺)*nx[i])
                    □₃ₙ=1/2*(□₃ₙ⁻+□₃ₙ⁺+α*μₘₐₓ*(n⁻-n⁺)*ny[i])
                    □₄ₓ=1/2*(□₄ₓ⁻+□₄ₓ⁺+α*λₘₐₓ*(E⁻-E⁺)*nx[i])
                    □₄ₙ=1/2*(□₄ₙ⁻+□₄ₙ⁺+α*μₘₐₓ*(E⁻-E⁺)*ny[i])
                    Ψ₁=-hᵢ*ωᵧ[ℓ]*(□₁ₓ*nx[i]+□₁ₙ*ny[i])
                    Ψ₂=-hᵢ*ωᵧ[ℓ]*(□₂ₓ*nx[i]+□₂ₙ*ny[i])
                    Ψ₃=-hᵢ*ωᵧ[ℓ]*(□₃ₓ*nx[i]+□₃ₙ*ny[i])
                    Ψ₄=-hᵢ*ωᵧ[ℓ]*(□₄ₓ*nx[i]+□₄ₙ*ny[i])
                    for mm=1:P_dof
                        Coef_ρₜ[mm,j]+=ϕᵧ[mm,ℓ⁻]*Ψ₁
                        Coef_mₜ[mm,j]+=ϕᵧ[mm,ℓ⁻]*Ψ₂
                        Coef_nₜ[mm,j]+=ϕᵧ[mm,ℓ⁻]*Ψ₃
                        Coef_Eₜ[mm,j]+=ϕᵧ[mm,ℓ⁻]*Ψ₄
                    end
                end
            end
        end
        for mm=1:P_dof
            Coef_ρₜ[mm,j]=Coef_ρₜ[mm,j]/(1/4*Aⱼ)
            Coef_mₜ[mm,j]=Coef_mₜ[mm,j]/(1/4*Aⱼ)
            Coef_nₜ[mm,j]=Coef_nₜ[mm,j]/(1/4*Aⱼ)
            Coef_Eₜ[mm,j]=Coef_Eₜ[mm,j]/(1/4*Aⱼ)
        end
    end
    return Coef_ρₜ,Coef_mₜ,Coef_nₜ,Coef_Eₜ,λₘₐₓ,μₘₐₓ
end


