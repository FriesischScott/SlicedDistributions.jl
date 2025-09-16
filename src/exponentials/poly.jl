struct SlicedExponential <: SlicedDistribution
    d::Integer
    t::Vector{Monomial}
    λ::AbstractVector
    lb::AbstractVector{<:Real}
    ub::AbstractVector{<:Real}
    c::Float64
end

using Convex
using SCS

function SlicedExponential(
    δ::AbstractMatrix,
    d::Integer,
    b::Integer=10000,
    lb::AbstractVector{<:Real}=vec(minimum(δ; dims=2)),
    ub::AbstractVector{<:Real}=vec(maximum(δ; dims=2)),
)
    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    t = monomials(["δ$i" for i in 1:size(δ, 1)], d, GradedLexicographicOrder())

    zδ = permutedims(t(δ))
    zΔ = permutedims(t(s))

    n = size(δ, 2)
    nz = size(zδ, 2)

    function f(λ)
        return n * logsumexp(zΔ * -λ) + sum(zδ * λ)
    end

    x = Variable(nz)

    problem = minimize(n * logsumexp(zΔ * -x) + sum(zδ * x))

    result = optimize(f, zeros(nz), Newton(); autodiff=AutoEnzyme())

    λ = result.minimizer

    cΔ = exp(log(prod(ub - lb)) - log(b) + logsumexp(zΔ * -λ))

    se = SlicedExponential(d, t, λ, lb, ub, cΔ)
    return se, -result.minimum
end

function _logpdf(se::SlicedExponential, δ::AbstractArray)
    if all(se.lb .<= δ .<= se.ub)
        return log(exp(-dot(se.t(δ), se.λ) / 2) / se.c)
    else
        return log(0)
    end
end

function Base.length(se::SlicedExponential)
    return length(se.t[1].x)
end

function Base.eltype(se::SlicedExponential)
    return eltype(se.λ)
end
