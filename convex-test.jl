# Make the Convex.jl module available
using Convex, SCS

using Distributions
using Plots
using SlicedNormals
using MinimumVolumeEllipsoids
using LinearAlgebra

n = 500

# Step 1
θ = rand(Normal(π / 2, 1.3), n)
r = 3 .+ rand(Uniform(0, 0.2), n) .* (θ .- π / 2)

δ1 = r .* cos.(θ)
δ2 = r .* sin.(θ)

idx = δ1 .< 0
δ2[idx] = δ2[idx] * -1

δ = [δ1 δ2]

# Step 2
θ = rand(Normal(π / 2, 1.3), n)
r = 3 .+ rand(Uniform(0, 0.2), n) .* (θ .- π / 2)

δ1 = r .* cos.(θ)
δ2 = r .* sin.(θ)

idx = δ1 .< 0
δ2[idx] = δ2[idx] * -1

δ = vcat(δ, [δ1 δ2] .* -1)

# Fit Sliced Normal Distribution
d = 3 # degree
b = 10000 # number of points to use  for estimation of the normalisation constant

ϵ = minimum_volume_ellipsoid(δ')
s = rand(ϵ, b)

zδ = mapreduce(r -> transpose(SlicedNormals.Z(r, d)), vcat, eachrow(δ))
zΔ = mapreduce(r -> transpose(SlicedNormals.Z(r, d)), vcat, eachcol(s))

μ, P = SlicedNormals.mean_and_covariance(zδ)

M = cholesky(P).U

zsosδ = transpose(mapreduce(z -> SlicedNormals.Zsos(z, μ, M), hcat, eachrow(zδ)))
zsosΔ = transpose(mapreduce(z -> SlicedNormals.Zsos(z, μ, M), hcat, eachrow(zΔ)))

m = size(δ, 1)
n = size(zδ, 1)

@show m, n

vol_m = log(volume(ϵ) / b)

nz = size(zδ, 2)

@show nz

l = Variable(nz)

con = l >= 0
problem = minimize(
    n * (vol_m + logsumexp((Matrix(zsosΔ) * l) / -2)) + sum(Matrix(zsosδ) * l) / 2, con
)

solve!(problem, SCS.Optimizer)

@show problem.status

@show problem.optval

@show Convex.evaluate(l)
