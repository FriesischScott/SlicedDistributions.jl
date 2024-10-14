module SlicedDistributions

using CovarianceEstimation
using Distributions
using LinearAlgebra
using Monomials
using TransitionalMCMC
using QuasiMonteCarlo
using Optim

import Base: rand

export SlicedNormal, SlicedExponential, rand, pdf

abstract type SlicedDistribution end

function Distributions.pdf(sn::SlicedDistribution, δ::AbstractMatrix)
    n, m = size(δ)
    if n == 1 || m == 1
        return pdf(sn, vec(δ))
    end
    if n < m
        return [pdf(sn, c) for c in eachcol(δ)]
    end
    return return [pdf(sn, c) for c in eachrow(δ)]
end

function rand(sd::SlicedDistribution, n::Integer)
    prior = Uniform.(sd.lb, sd.ub)

    logprior(x) = sum(logpdf.(prior, x))
    sampler(n) = mapreduce(u -> rand(u, n), hcat, prior)
    loglikelihood(x) = log(SlicedDistributions.pdf(sd, x))

    samples, _ = tmcmc(loglikelihood, logprior, sampler, n)

    return samples
end

function get_likelihood(
    zδ::Matrix{<:Real}, zΔ::Matrix{<:Real}, n::Integer, vol::Real, b::Integer
)
    f = λ -> n * log(vol / b * sum(exp.(zΔ * λ / -2))) + sum(zδ * λ) / 2
    return f
end

function get_gradient(zδ::Matrix{<:Real}, zΔ::Matrix{<:Real}, n::Integer)
    f =
        (g, λ) -> begin
            exp_Δ = exp.(zΔ * λ / -2)
            sum_exp_Δ = sum(exp_Δ)
            for i in eachindex(g)
                g[i] = @views n * sum(exp_Δ .* -0.5zΔ[:, i]) / sum_exp_Δ + sum(zδ[:, i]) / 2
            end
            return nothing
        end
    return f
end

function get_hessian(zΔ::Matrix{<:Real}, n::Integer)
    f =
        (H, λ) -> begin
            exp_Δ = exp.(zΔ * λ / -2)
            sum_exp_Δ = sum(exp_Δ)
            sum_exp_Δ² = sum_exp_Δ^2

            for (i, Δ_i) in enumerate(eachcol(zΔ))
                exp_Δ_i = exp_Δ .* -0.5Δ_i
                sum_exp_Δ_i = sum(exp_Δ_i)

                for (j, Δ_j) in enumerate(eachcol(zΔ))
                    H[i, j] =
                        n * (
                            sum(exp_Δ_i .* -0.5Δ_j) * sum_exp_Δ -
                            sum_exp_Δ_i * sum(exp_Δ .* -0.5Δ_j)
                        ) / sum_exp_Δ²
                end
            end
            return nothing
        end
    return f
end

include("exponentials/poly.jl")
include("normals/sum-of-squares.jl")

end
