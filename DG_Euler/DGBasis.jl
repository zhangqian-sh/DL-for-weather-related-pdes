function JacobiP(m::Integer,α::Real,β::Real,z::Real) :: Float64
    sqrt((2*m+α+β+1)/(2^(α+β+1)))*
    sqrt(gamma(m+1)/(gamma(m+α+1)*gamma(m+β+1)*gamma(m+α+β+1)))*
    gamma(m+α+β+1)*jacobip(m, α, β, Float64(z))
end

function GradnJacobiP(m::Integer,α::Real,β::Real,z::Real,n::Integer=1) :: Float64
    if m<n
        return zero(Float64)
    else
        return sqrt((2*m+α+β+1)/(2^(α+β+2*n+1)))*
        sqrt(gamma(m+1)/(gamma(m+α+1)*gamma(m+β+1)*gamma(m+α+β+1)))*
        gamma(m+α+β+n+1)*jacobip(m-n, α+n, β+n, Float64(z))
    end
end
