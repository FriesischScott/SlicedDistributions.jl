using Distributions
using Plots
using SlicedNormals
using JuMP
using NLopt
using Ipopt
# using Optim
using MinimumVolumeEllipsoids
using LinearAlgebra
using Random
using IntervalArithmetic

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
d = 5
b = 10000

Random.seed!(8128)
sn, lh = SlicedNormal(δ, d, b)

Random.seed!(8128)
ϵ = minimum_volume_ellipsoid(δ')
s = rand(ϵ, b)

zδ = mapreduce(r -> transpose(SlicedNormals.Z(r, d)), vcat, eachrow(δ))
zΔ = mapreduce(r -> transpose(SlicedNormals.Z(r, d)), vcat, eachcol(s))

μ, P = SlicedNormals.mean_and_covariance(zδ)

M = cholesky(P).U

zsosδ = transpose(mapreduce(z -> SlicedNormals.Zsos(z, μ, M), hcat, eachrow(zδ)))
zsosΔ = transpose(mapreduce(z -> SlicedNormals.Zsos(z, μ, M), hcat, eachrow(zΔ)))

m = size(δ, 1)

D = λ -> sum([SlicedNormals.ϕE(x, λ) for x in eachrow(zsosδ)] / 2)
cΔ = λ -> volume(ϵ) / b * sum([exp.(-SlicedNormals.ϕE(δ, λ) / 2) for δ in eachrow(zsosΔ)])

function f(λ...)
    return -1 * (-m * log(cΔ(λ)) - D(λ))
end

nz = size(zδ, 2)

model = Model(NLopt.Optimizer)
set_optimizer_attribute(model, "algorithm", NLopt.LD_SLSQP)
@variable(model, λ[1:nz] >= 0.0)
register(model, :f, nz, f; autodiff=true)
@NLobjective(model, Min, f(λ...))
JuMP.optimize!(model)

solution_summary(model)

lb = vec(minimum(δ; dims=1)) .* 1.1
ub = vec(maximum(δ; dims=1)) .* 1.1

Δ = IntervalBox(interval.(lb, ub)...)

cΔ = volume(ϵ) / b * sum([exp(-SlicedNormals.ϕE(δ, value.(λ)) / 2) for δ in eachrow(zsosΔ)])

sn2 = SlicedNormal(d, value.(λ), μ, M, Δ, cΔ)

samples1 = rand(sn, 1000)
samples2 = rand(sn2, 1000)

p = scatter(
    δ[:, 1], δ[:, 2]; aspect_ratio=:equal, lims=[-4, 4], xlab="δ1", ylab="δ2", label="data"
)
scatter!(p, samples1[:, 1], samples1[:, 2]; label="optim")

scatter!(p, samples2[:, 1], samples2[:, 2]; label="nlopt")

display(p)

@show abs.(sn.λ .- value.(λ))
