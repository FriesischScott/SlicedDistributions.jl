module SlicedDistributions

using CovarianceEstimation
using Distributions
using LinearAlgebra
using Monomials
using TransitionalMCMC
using QuasiMonteCarlo
using Optim
using Random

import Base: eltype, length
import Distributions: _logpdf, insupport

export SlicedNormal, SlicedExponential

export pdf, insupport

abstract type SlicedDistribution <: ContinuousMultivariateDistribution end

function Distributions.rand!(rng::AbstractRNG, sd::SlicedDistribution, x::AbstractMatrix)
    prior = Uniform.(sd.lb, sd.ub)

    logprior(x) = sum(logpdf.(prior, x))
    sampler(n) = mapreduce(u -> rand(rng, u, n), hcat, prior)
    loglikelihood(x) = Distributions.logpdf(sd, x)

    samples, _ = tmcmc(loglikelihood, logprior, sampler, size(x, 2))

    x[:] = permutedims(samples)

    return x
end

function Distributions.insupport(sd::SlicedDistribution, x::AbstractVector)
    return all(sd.lb .<= x .<= sd.ub)
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

            for i in axes(zΔ, 2)
                exp_Δ_i = exp_Δ .* -0.5zΔ[:, i]
                sum_exp_Δ_i = sum(exp_Δ_i)

                for j in i:size(zΔ, 2)
                    Δ_j = @view zΔ[:, j]
                    H[i, j] =
                        n * (
                            sum(exp_Δ_i .* -0.5Δ_j) * sum_exp_Δ -
                            sum_exp_Δ_i * sum(exp_Δ .* -0.5Δ_j)
                        ) / sum_exp_Δ²
                end
            end

            H[:] = Symmetric(H)
            return nothing
        end
    return f
end

include("exponentials/poly.jl")
include("normals/sum-of-squares.jl")

Base.broadcastable(sd::SlicedDistribution) = Ref(sd)

end
