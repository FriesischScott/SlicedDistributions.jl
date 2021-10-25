module SlicedNormals

using CovarianceEstimation
using Distributions
using DynamicPolynomials
using HCubature
using IntervalArithmetic
using LinearAlgebra
using Memoize
using NLopt
using PDMats
using TransitionalMCMC

import Base: rand

export SlicedNormal, Z, rand, pdf, fit_baseline, fit_scaling

struct SlicedNormal
    d::Integer
    μ::AbstractVector
    P::AbstractPDMat
    Δ::IntervalBox
end

function pdf(sn::SlicedNormal, δ::AbstractVector, normalize::Bool=true)
    if δ ∈ sn.Δ
        f = exp(-_ϕ(δ, sn.μ, sn.P, sn.d))

        if normalize
            return f / c(sn.μ, sn.P, sn.Δ, sn.d)
        else
            return f
        end
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
    z = monomials(x..., 1:d)

    # double reverse to achieve graded lexographic order
    return reverse(map(p -> p(reverse(δ)), z))
end

@memoize function c(μ, P, Δ, d)
    lb, ub = bounds(Δ)

    normalization, _ = hcubature(δ -> exp(-_ϕ(Z(δ, d), μ, P, d)), lb, ub)

    return normalization
end

function bounds(Δ::IntervalBox)
    lb = [map(x -> x.lo, Δ.v.data)...]
    ub = [map(x -> x.hi, Δ.v.data)...]

    return lb, ub
end

function mean_and_covariance(x::AbstractMatrix, d::Integer)
    z = mapreduce(r -> transpose(Z(r, d)), vcat, eachrow(x))

    μ = vec(mean(z; dims=1))
    P = PDMat(inv(cov(LinearShrinkage(ConstantCorrelation()), z)))

    return μ, P
end

function fit_baseline(x::AbstractMatrix, d::Integer)
    μ, P = mean_and_covariance(x, d)

    lb = vec(minimum(x; dims=1))
    ub = vec(maximum(x; dims=1))

    Δ = IntervalBox(interval.(lb, ub)...)

    D = sum([_ϕ(δ, μ, P, d) for δ in eachrow(x)])

    m = size(x, 1)

    cΔ = hcubature(δ -> exp(-_ϕ(δ, μ, P.mat, d)), lb, ub)[1]
    lh = m * log(1 / cΔ) - D

    return SlicedNormal(d, μ, P, Δ), lh
end

function fit_baseline(x::AbstractMatrix, d::Integer, Δ :: IntervalBox)

    μ, P = mean_and_covariance(x, d)
    D = sum([_ϕ(δ, μ, P, d) for δ in eachrow(x)])

    lb = getfield.(Δ, :lo)
    ub = getfield.(Δ, :hi)

    m = size(x, 1)

    cΔ = hcubature(δ -> exp(-_ϕ(δ, μ, P.mat, d)), lb, ub)[1]
    lh = m * log(1 / cΔ) - D

    return SlicedNormal(d, μ, P, Δ), lh
end

function fit_scaling(x::AbstractMatrix, d::Integer)
    μ, P = mean_and_covariance(x, d)

    lb = vec(minimum(x; dims=1))
    ub = vec(maximum(x; dims=1))

    Δ = IntervalBox(interval.(lb, ub)...)

    D = sum([_ϕ(δ, μ, P, d) for δ in eachrow(x)])

    m = size(x, 1)

    opt = Opt(:LN_NELDERMEAD, 1)
    opt.lower_bounds = [0.0]
    opt.xtol_rel = 1e-5
    opt.min_objective =
        (s, grad) -> begin
            cΔ = hcubature(δ -> exp(-_ϕ(δ, μ, s[1] * P.mat, d)), lb, ub)[1]
            lh = m * log(1 / cΔ) - s[1] * D
            return -1 * lh
        end

    (lh, factor, _) = optimize(opt, [1.0])
    return SlicedNormal(d, μ, factor[1] * P, Δ), -1 * lh
end

function fit_scaling(x::AbstractMatrix, d::Integer, Δ :: IntervalBox)
    μ, P = mean_and_covariance(x, d)

    lb = getfield.(Δ, :lo)
    ub = getfield.(Δ, :hi)

    D = sum([_ϕ(δ, μ, P, d) for δ in eachrow(x)])

    m = size(x, 1)

    opt = Opt(:LN_NELDERMEAD, 1)
    opt.lower_bounds = [0.0]
    opt.xtol_rel = 1e-5
    opt.min_objective =
        (s, grad) -> begin
            cΔ = hcubature(δ -> exp(-_ϕ(δ, μ, s[1] * P.mat, d)), lb, ub)[1]
            lh = m * log(1 / cΔ) - s[1] * D
            return -1 * lh
        end

    (lh, factor, _) = optimize(opt, [1.0])
    return SlicedNormal(d, μ, factor[1] * P, Δ), -1 * lh
end

function rand(sn::SlicedNormal, n::Integer)
    lb, ub = bounds(sn.Δ)

    prior = Uniform.(lb, ub)

    logprior(x) = sum(logpdf.(prior, x))
    sampler(n) = mapreduce(u -> rand(u, n), hcat, prior)
    loglikelihood(x) = log(SlicedNormals.pdf(sn, x, false))

    samples, _ = tmcmc(loglikelihood, logprior, sampler, n)

    return samples
end

function _ϕ(δ, μ, P, d)
    z = Z(δ, d)
    return ((z - μ)' * P * (z - μ)) / 2
end

end # module
