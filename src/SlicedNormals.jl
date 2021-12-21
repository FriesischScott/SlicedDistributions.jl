module SlicedNormals

using CovarianceEstimation
using Distributions
using DynamicPolynomials
using IntervalArithmetic
using LinearAlgebra
using Memoize
using MinimumVolumeEllipsoids
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
    δ::AbstractMatrix
end

function pdf(sn::SlicedNormal, δ::AbstractVector, normalize::Bool=true)
    if δ ∈ sn.Δ
        f = exp(-_ϕ(δ, sn.μ, sn.P, sn.d))

        if normalize
            return f / c(sn.μ, sn.P, sn.d, sn.δ)
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

@memoize function c(
    μ::AbstractVector, P::AbstractPDMat, d::Integer, x::AbstractMatrix, b::Integer=10000
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
    P = PDMat(inv(cov(LinearShrinkage(ConstantCorrelation()), z)))

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

    return SlicedNormal(d, μ, P, Δ, x), lh
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
    V = volume(ϵ)

    opt = Opt(:LN_NELDERMEAD, 1)
    opt.lower_bounds = [0.0]
    opt.xtol_rel = 1e-5
    opt.min_objective =
        (s, _) -> begin
            cΔ = V / b * sum([exp(-_ϕ(δ, μ, s[1] * P, d)) for δ in eachcol(U)])
            lh = m * log(1 / cΔ) - s[1] * D
            return -1 * lh
        end

    (lh, factor, _) = optimize(opt, [1.0])
    return SlicedNormal(d, μ, factor[1] * P, Δ, x), -1 * lh
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
