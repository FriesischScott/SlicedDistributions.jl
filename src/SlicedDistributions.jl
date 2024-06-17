module SlicedDistributions

using CovarianceEstimation
using Distributions
using DynamicPolynomials
using IntervalArithmetic
using LinearAlgebra
using TransitionalMCMC
using QuasiMonteCarlo
using Optim

import Base: rand

export SlicedNormal, rand, pdf

struct SlicedNormal
    d::Integer
    λ::AbstractVector
    μ::AbstractVector
    M::AbstractMatrix
    Δ::IntervalBox
    c::Float64
end

function SlicedNormal(δ::AbstractMatrix, d::Integer, b::Integer=10000)
    lb = vec(minimum(δ; dims=1))
    ub = vec(maximum(δ; dims=1))

    Δ = IntervalBox(interval.(lb, ub)...)

    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    zδ = mapreduce(r -> transpose(Z(r, 2d)), vcat, eachrow(δ))
    zΔ = mapreduce(r -> transpose(Z(r, 2d)), vcat, eachcol(s))

    μ, P = mean_and_covariance(zδ)

    M = cholesky(P).U

    zsosδ = transpose(mapreduce(z -> Zsos(z, μ, M), hcat, eachrow(zδ)))
    zsosΔ = transpose(mapreduce(z -> Zsos(z, μ, M), hcat, eachrow(zΔ)))

    n = size(δ, 1)
    nz = size(zδ, 2)

    function f(λ)
        return n * log(prod(ub - lb) / b * sum(exp.(zsosΔ * λ / -2))) + sum(zsosδ * λ) / 2
    end

    function ∇f!(g, λ)
        exp_Δ = exp.(zsosΔ * λ / -2)
        sum_exp_Δ = sum(exp_Δ)
        for i in eachindex(g)
            g[i] = @views n * sum(exp_Δ .* -0.5zsosΔ[:, i]) / sum_exp_Δ +
                sum(zsosδ[:, i]) / 2
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

    result = optimize(f, ∇f!, ∇²f!, zeros(nz), fill(Inf, nz), ones(nz), IPNewton())

    cΔ = prod(ub - lb) / b * sum(exp.(zsosΔ * result.minimizer / -2))

    sn = SlicedNormal(d, result.minimizer, μ, M, Δ, cΔ)
    return sn, -result.minimum
end

function pdf(sn::SlicedNormal, δ::AbstractVector)
    if δ ∈ sn.Δ
        z = Zsos(Z(δ, 2sn.d), sn)
        return exp(-dot(z, sn.λ) / 2) / sn.c
    else
        return 0
    end
end

function Distributions.pdf(sn::SlicedNormal, δ::AbstractMatrix)
    n, m = size(δ)
    if n == 1 || m == 1
        return pdf(sn, vec(δ))
    end
    if n < m
        return [pdf(sn, c) for c in eachcol(δ)]
    end
    return return [pdf(sn, c) for c in eachrow(δ)]
end

function Z(δ::AbstractVector, d::Integer)
    x = @polyvar x[1:length(δ)]
    z = mapreduce(p -> monomials(x..., p), vcat, 1:d)

    return map(p -> p(δ), z)
end

function Zsos(z::AbstractVector, μ::AbstractVector, M::AbstractMatrix)
    return (M * (z - μ)) .^ 2
end

Zsos(z::AbstractVector, sn::SlicedNormal) = Zsos(z, sn.μ, sn.M)

function bounds(Δ::IntervalBox)
    lb = getfield.(Δ, :lo)
    ub = getfield.(Δ, :hi)

    return lb, ub
end

function mean_and_covariance(z::AbstractMatrix)
    μ = vec(mean(z; dims=1))
    P = inv(cov(LinearShrinkage(ConstantCorrelation()), z))

    return μ, P
end

function rand(sn::SlicedNormal, n::Integer)
    lb, ub = bounds(sn.Δ)

    prior = Uniform.(lb, ub)

    logprior(x) = sum(logpdf.(prior, x))
    sampler(n) = mapreduce(u -> rand(u, n), hcat, prior)
    loglikelihood(x) = log(SlicedDistributions.pdf(sn, x))

    samples, _ = tmcmc(loglikelihood, logprior, sampler, n)

    return samples
end

end
