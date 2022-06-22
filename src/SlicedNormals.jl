module SlicedNormals

using CovarianceEstimation
using Distributions
using DynamicPolynomials
using IntervalArithmetic
using LinearAlgebra
using MinimumVolumeEllipsoids
using Optim
using TransitionalMCMC

import Base: rand

export SlicedNormal, Z, rand, pdf, fit_baseline, fit_scaling, fit_augmentation

struct SlicedNormal
    d::Integer
    μ::AbstractVector
    P::AbstractMatrix
    Δ::IntervalBox
    c::Float64
end

function pdf(sn::SlicedNormal, δ::AbstractVector)
    if δ ∈ sn.Δ
        return exp(-_ϕ(δ, sn.μ, sn.P, sn.d)) / sn.c
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

    # double reverse to achieve graded lexographic order
    return map(p -> p(δ), z)
end

function c(
    μ::AbstractVector, P::AbstractMatrix, d::Integer, x::AbstractMatrix, b::Integer=10000
)
    ϵ = minimum_volume_ellipsoid(x')

    V = volume(ϵ)
    u = rand(ϵ, b)

    return V / b * sum([exp(-_ϕ(δ, μ, P, d)) for δ in eachcol(u)])
end

function bounds(Δ::IntervalBox)
    lb = getfield.(Δ, :lo)
    ub = getfield.(Δ, :hi)

    return lb, ub
end

function mean_and_covariance(x::AbstractMatrix, d::Integer)
    z = mapreduce(r -> transpose(Z(r, d)), vcat, eachrow(x))

    μ = vec(mean(z; dims=1))
    P = inv(cov(LinearShrinkage(ConstantCorrelation()), z))

    return μ, P
end

function fit_baseline(x::AbstractMatrix, d::Integer)
    lb = vec(minimum(x; dims=1))
    ub = vec(maximum(x; dims=1))

    Δ = IntervalBox(interval.(lb, ub)...)

    return fit_baseline(x, d, Δ)
end

function fit_baseline(x::AbstractMatrix, d::Integer, Δ::IntervalBox)
    μ, P = mean_and_covariance(x, d)
    D = sum([_ϕ(δ, μ, P, d) for δ in eachrow(x)])

    m = size(x, 1)

    cΔ = c(μ, P, d, x)
    lh = m * log(1 / cΔ) - D

    return SlicedNormal(d, μ, P, Δ, cΔ), lh
end

function fit_scaling(x::AbstractMatrix, d::Integer)
    lb = vec(minimum(x; dims=1))
    ub = vec(maximum(x; dims=1))

    Δ = IntervalBox(interval.(lb, ub)...)

    return fit_scaling(x, d, Δ)
end

function fit_scaling(x::AbstractMatrix, d::Integer, Δ::IntervalBox, b::Integer=10000)
    μ, P = mean_and_covariance(x, d)

    D = sum([_ϕ(δ, μ, P, d) for δ in eachrow(x)])

    m = size(x, 1)

    ϵ = minimum_volume_ellipsoid(x')

    U = rand(ϵ, b)

    sᵢ = [_ϕ(δ, μ, P, d) for δ in eachcol(U)]

    f = γ -> -m * log(sum(exp.(-γ[1] .* sᵢ))) - γ[1] * D

    opt = maximize(f, [1.0], LBFGS())

    γ = Optim.maximizer(opt)[1]

    D = sum([_ϕ(δ, μ, γ .* P, d) for δ in eachrow(x)])
    cΔ = volume(ϵ) / b * sum([exp(-_ϕ(δ, μ, γ .* P, d)) for δ in eachcol(U)])
    lh = m * log(1 / cΔ) - D

    return SlicedNormal(d, μ, γ * P, Δ, cΔ), lh
end

function fit_augmentation(x::AbstractMatrix, d::Integer, b::Integer=10000)
    μ, P = mean_and_covariance(x, d)

    # Augmentation
    ϵ = minimum_volume_ellipsoid(x')
    s = rand(ϵ, b)

    Γ = ones(size(P))
    m = size(x, 1)

    for ν in Iterators.filter(ν -> ν[1] <= ν[2], CartesianIndices(P))
        M = zeros(size(P))
        M[ν] = P[ν] .* Γ[ν]

        # mirror off-diagonal entries (symmetry)
        if ν[1] < ν[2]
            M[ν[2], ν[1]] = P[ν[2], ν[1]] .* Γ[ν[2], ν[1]]
        end

        N = P .* Γ - M

        β = [_ϕ(sᵢ, μ, N, d) for sᵢ in eachcol(s)]
        α = [_ϕ(sᵢ, μ, M, d) for sᵢ in eachcol(s)]

        κ = [_ϕ(δᵢ, μ, M, d) for δᵢ in eachrow(x)]
        φ = [_ϕ(δᵢ, μ, N, d) for δᵢ in eachrow(x)]

        f = γ -> -m * log(sum(exp.(-γ[1] .* α + β))) - sum(γ[1] .* κ + φ)

        function g!(G, γ)
            return G[1] =
                -m * sum(-α .* exp.(-γ[1] .* α .+ β)) / sum(exp.(-γ[1] .* α .+ β)) - sum(κ)
        end

        opt = maximize(f, g!, [0.0], LBFGS())

        γ = Optim.maximizer(opt)[1]

        Γ[ν] += γ

        # mirror off-diagonal entries (symmetry)
        if ν[1] < ν[2]
            Γ[ν[2], ν[1]] += γ
        end
    end

    D = sum([_ϕ(δ, μ, Γ .* P, d) for δ in eachrow(x)])
    cΔ = volume(ϵ) / b * sum([exp(-_ϕ(δ, μ, Γ .* P, d)) for δ in eachcol(s)])
    lh = m * log(1 / cΔ) - D

    lb = vec(minimum(x; dims=1))
    ub = vec(maximum(x; dims=1))

    Δ = IntervalBox(interval.(lb, ub)...)

    return SlicedNormal(d, μ, Γ .* P, Δ, cΔ), lh
end

function rand(sn::SlicedNormal, n::Integer)
    lb, ub = bounds(sn.Δ)

    prior = Uniform.(lb, ub)

    logprior(x) = sum(logpdf.(prior, x))
    sampler(n) = mapreduce(u -> rand(u, n), hcat, prior)
    loglikelihood(x) = log(SlicedNormals.pdf(sn, x))

    samples, _ = tmcmc(loglikelihood, logprior, sampler, n)

    return samples
end

function _ϕ(δ, μ, P, d)
    z = Z(δ, d)
    return ((z - μ)' * P * (z - μ)) / 2
end

end # module
