module SlicedNormals

using CovarianceEstimation
using Distributions
using DynamicPolynomials
using HCubature
using IntervalArithmetic
using JuMP
using LinearAlgebra
using Memoize
using NLopt
using PDMats
using TransitionalMCMC

import Base:rand

export SlicedNormal, Z, rand, pdf, fit_baseline, fit_scaling

struct SlicedNormal
    d::Integer
    μ::AbstractVector
    P::AbstractPDMat
    Δ::IntervalBox
end

function pdf(sn::SlicedNormal, δ::AbstractVector, normalize::Bool=true)
    if δ ∈ sn.Δ
        mvn = MvNormal(sn.μ, inv(sn.P))
        z = Z(δ, sn.d)

        if normalize
            return (γ(sn.μ, sn.P) / c(sn.μ, sn.P, sn.Δ, sn.d)) * Distributions.pdf(mvn, z)
        else
            return Distributions.pdf(mvn, z)
        end
    else
        return 0
    end
end

function Z(δ::AbstractVector, d::Integer)
    x = @polyvar x[1:length(δ)]
    z = monomials(x..., 1:d)

    # double reverse to achieve graded lexographic order
    map(p -> p(reverse(δ)), z) |> reverse
end

@memoize function c(μ, P, Δ, d)
    mvn = MvNormal(μ, inv(P))

    lb, ub = bounds(Δ)

    normalization, _ = hcubature(x -> γ(μ, P) * Distributions.pdf(mvn, Z(x, d)), lb, ub)

    return normalization
end

function γ(μ, P)
    (2 * π)^(length(μ) / 2) * sqrt(det(inv(P)))
end

function bounds(Δ::IntervalBox)
    lb = [map(x -> x.lo, Δ.v.data)...]
    ub = [map(x -> x.hi, Δ.v.data)...]

    return lb, ub
end

function fit_baseline(x::AbstractMatrix, d::Integer)
    z  = mapreduce(r -> Z(r, d) |> transpose, vcat, eachrow(x))

    μ = mean(z, dims=1) |> vec
    P = cov(LinearShrinkage(ConstantCorrelation()), z) |> inv |> PDMat

    return μ, P
end

function fit_scaling(x::AbstractMatrix, d::Integer)
    μ, P = fit_baseline(x, d)

    lb = minimum(x, dims=1) |> vec
    ub = maximum(x, dims=1) |> vec

    Δ = IntervalBox(interval.(lb, ub)...)

    D = [_ϕ(δ, μ, P, d) for δ in eachrow(x)] |> sum

    m = size(x, 1)

    opt = Opt(:LN_NELDERMEAD, 1)
    opt.lower_bounds = [0.0]
    opt.xtol_rel = 1e-5
    opt.min_objective = (s, grad) -> begin
        try
            cΔ = hcubature(δ -> exp(-_ϕ(δ, μ, s[1] * P.mat, d)), lb, ub)[1]
            lh = m * log(1 / cΔ) - s[1] * D
            return -1 * lh
        catch e
            @show e
            rethrow(e)
        end
    end

    (lh, factor, _) = optimize(opt, [1.0])
    @show factor
    return SlicedNormal(d, μ, factor[1] * P, Δ), -1 * lh
end

function rand(sn::SlicedNormal, n::Integer)
    lb, ub = bounds(sn.Δ)

    prior = Uniform.(lb, ub)

    logprior(x) = logpdf.(prior, x) |> sum
    sampler(n) = mapreduce(u -> rand(u, n), hcat, prior)
    loglikelihood(x) = SlicedNormals.pdf(sn, x, false) |> log

    samples, _ = tmcmc(loglikelihood, logprior, sampler, n)

    return samples
end

function _ϕ(δ, μ, P, d)
    z = Z(δ, d)
    return ((z - μ)' * P * (z - μ)) / 2
end

end # module
