# Make the Convex.jl module available
using JuMP, Ipopt

using Distributions
using Plots
using SlicedNormals
using LinearAlgebra
using IntervalArithmetic

using QuasiMonteCarlo

using Random

Random.seed!(8128)

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

lb = vec(minimum(δ; dims=1))
ub = vec(maximum(δ; dims=1))

s = QuasiMonteCarlo.sample(b, lb, ub, SobolSample())

zδ = mapreduce(r -> transpose(SlicedNormals.Z(r, 2d)), vcat, eachrow(δ))
zΔ = mapreduce(r -> transpose(SlicedNormals.Z(r, 2d)), vcat, eachcol(s))

μ, P = SlicedNormals.mean_and_covariance(zδ)

M = cholesky(P).U

zsosδ = Matrix(transpose(mapreduce(z -> SlicedNormals.Zsos(z, μ, M), hcat, eachrow(zδ))))
zsosΔ = Matrix(transpose(mapreduce(z -> SlicedNormals.Zsos(z, μ, M), hcat, eachrow(zΔ))))

n = size(δ, 1)

nz = size(zδ, 2)

model = Model(Ipopt.Optimizer)

set_silent(model)

@variable(model, λ[i=1:nz] .>= 0)

@objective(
    model, Min, n * log(prod(ub - lb) / b * sum(exp.(zsosΔ * λ / -2))) + sum(zsosδ * λ) / 2
)

optimize!(model)

@show objective_value(model)

@show value.(λ)

# cΔ = b * (log(prod(ub - lb) / b) + log(sum(exp.(zsosΔ * value.(λ) / -2))))

# Δ = IntervalBox(interval.(lb, ub)...)

# sn = SlicedNormal(d, value.(λ), μ, M, Δ, cΔ)

# samples = rand(sn, 1000)

# p = scatter(
#     δ[:, 1], δ[:, 2]; aspect_ratio=:equal, lims=[-4, 4], xlab="δ1", ylab="δ2", label="data"
# )
# scatter!(p, samples[:, 1], samples[:, 2]; label="samples")

# display(p)

# # Plot density
# xs = range(-4, 4; length=200)
# ys = range(-4, 4; length=200)

# contour!(xs, ys, (x, y) -> SlicedNormals.pdf(sn, [x, y]))
