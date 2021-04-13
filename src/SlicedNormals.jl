module SlicedNormals

using CovarianceEstimation
using Distributions
using DynamicPolynomials
using IntervalArithmetic
using LinearAlgebra
using PDMats
using TransitionalMCMC

import Base:rand

export SlicedNormal, Z, rand, pdf, fit

struct SlicedNormal
    d::Integer
    μ::AbstractVector
    P::AbstractPDMat
    Δ::IntervalBox
end

function pdf(sn::SlicedNormal, δ::AbstractVector)
    if δ ∈ sn.Δ
        mvn = MvNormal(sn.μ, inv(sn.P))
        z = Z(δ, sn.d)
        return Distributions.pdf(mvn, z) # TODO: Normalization
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

function fit(x::AbstractMatrix, d::Integer)
    z  = mapreduce(r -> Z(r, d) |> transpose, vcat, eachrow(x))

    μ = mean(z, dims=1) |> vec
    P = cov(LinearShrinkage(ConstantCorrelation()), z) |> inv |> PDMat

    return μ, P
end

function rand(sn::SlicedNormal, n::Integer)
    lb = [map(x -> x.lo, sn.Δ.v.data)...]
    ub = [map(x -> x.hi, sn.Δ.v.data)...]

    prior = Uniform.(lb, ub)

    logprior(x) = logpdf.(prior, x) |> sum
    sampler(n) = mapreduce(u -> rand(u, n), hcat, prior)
    loglikelihood(x) = SlicedNormals.pdf(sn, x) |> log

    tmcmc(loglikelihood, logprior, sampler, n)
end

end # module
