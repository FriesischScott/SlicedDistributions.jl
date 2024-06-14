# Make the Convex.jl module available
using Optim

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

function f(λ)
    return n * log(prod(ub - lb) / b * sum(exp.(zsosΔ * λ / -2))) + sum(zsosδ * λ) / 2
end

function ∇f!(g, λ)
    exp_Δ = exp.(zsosΔ * λ / -2)
    sum_exp_Δ = sum(exp_Δ)
    for i in eachindex(g)
        g[i] = @views n * sum(exp_Δ .* -0.5zsosΔ[:, i]) / sum_exp_Δ + sum(zsosδ[:, i]) / 2
    end
    return nothing
end

function ∇²f!(H, λ)
    exp_Δ = exp.(zsosΔ * λ / -2)
    sum_exp_Δ = sum(exp_Δ)
    sum_exp_Δ² = sum_exp_Δ^2

    for (i, Δ_i) in enumerate(eachcol(zsosΔ))
        exp_Δ_i = exp_Δ .* -0.5Δ_i
        sum_exp_Δ_i = sum(exp_Δ_i)

        for (j, Δ_j) in enumerate(eachcol(zsosΔ))
            H[i, j] =
                n * (
                    sum(exp_Δ_i .* -0.5Δ_j) * sum_exp_Δ -
                    sum_exp_Δ_i * sum(exp_Δ .* -0.5Δ_j)
                ) / sum_exp_Δ²
        end
    end
    return nothing
end

result = Optim.optimize(f, ∇f!, ∇²f!, zeros(nz), fill(Inf, nz), ones(nz), IPNewton())

@show result.minimum

@show result.minimizer

# Δ = IntervalBox(interval.(lb, ub)...)

# cΔ =
#     n * log(
#         prod(ub - lb) / b *
#         sum(exp.([dot(x, result.minimizer) for x in eachrow(zsosΔ)] ./ -2)),
#     )

# sn = SlicedNormal(d, result.minimizer, μ, M, Δ, cΔ)

# samples = rand(sn, 1000)

# p = scatter(
#     δ[:, 1], δ[:, 2]; aspect_ratio=:equal, lims=[-4, 4], xlab="δ1", ylab="δ2", label="data"
# )
# scatter!(p, samples[:, 1], samples[:, 2]; label="samples")

# display(p)

# # Plot density
# xs = range(-4, 4; length=1000)
# ys = range(-4, 4; length=1000)

# contour!(xs, ys, (x, y) -> SlicedNormals.pdf(sn, [x, y]))
