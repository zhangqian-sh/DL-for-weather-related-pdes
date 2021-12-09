abstract type AbstractGrid end

struct Grid2DRect <: AbstractGrid
    NumofE::Int64
    Coordinates::Matrix{Float64}
    Elements::Matrix{Int64}
    EtoEmap::Matrix{Int64}
    Jacobians::Array{Float64,3}
    Tarea::Vector{Float64}
    Elength::Matrix{Float64}
end

function Grid2DRect(xmin::Real,xmax::Real,ymin::Real,ymax::Real,
    Cell_N::Integer,Cell_M::Integer;Periodic::Bool=false)
    N=Int64(Cell_N)
    M=Int64(Cell_M)
    coor_x = range(xmin,xmax,length=N+1)
    coor_y = range(ymin,ymax,length=M+1)
    Δx = (xmax-xmin)/N
    Δy = (ymax-ymin)/M
    NumofE = N*M
    Coordinates = Array{Float64,2}(undef,(N+1)*(M+1),2)
    Elements = Array{Int64,2}(undef,4,NumofE)
    EtoEmap = Array{Int64,2}(undef,4,NumofE)
    Jacobians = Array{Float64,3}(undef,2,2,NumofE)
    Tarea = Array{Float64,1}(undef,NumofE)
    Elength = Array{Float64,2}(undef,4,NumofE)
    @inbounds for i=1:M+1, j=1:N+1
        node = (i-1)*(N+1)+j
        Coordinates[node,:]=[coor_x[j] coor_y[i]]
    end
    @inbounds for i=1:M, j=1:N
    elemt = (i-1)*N+j
    node = (i-1)*(N+1)+j
    Elements[:,elemt] = [node;node+1;node+N+2;node+N+1]
    EtoEmap[:,elemt] = [elemt-N;elemt+1;elemt+N;elemt-1]
    Jacobians[:,:,elemt] = [2.0/Δx 0.0; 0.0 2.0/Δy]
    Tarea[elemt] = Δx*Δy
    Elength[:,elemt] = [Δx; Δy; Δx; Δy]
    end
    if Periodic
        EtoEmap[1,1:N] = collect(Int64,(M-1)*N+1:M*N)
        EtoEmap[3,(M-1)*N+1:M*N] = collect(Int64,1:N)
        EtoEmap[4,1:N:end] = collect(Int64,N:N:M*N)
        EtoEmap[2,N:N:end] = collect(Int64,1:N:M*N)
    else
        @.EtoEmap[1,1:N] = -1
        @.EtoEmap[3,(M-1)*N+1:M*N] = -1
        @.EtoEmap[4,1:N:end] = -1
        @.EtoEmap[2,N:N:end] = -1
    end
    return Grid2DRect(NumofE,Coordinates,Elements,EtoEmap,
    Jacobians,Tarea,Elength)
end