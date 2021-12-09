using SpecialFunctions
using FastGaussQuadrature
using ClassicalOrthogonalPolynomials
using Plots
using JLD2
using FileIO
using LinearAlgebra
using SIMD
using LoopVectorization 
using BenchmarkTools 
using StaticArrays
using SparseArrays
using GeometryBasics
using MAT
include("DGBasis.jl")

include("FEMGeometry.jl")

include("DGToolbox.jl")
