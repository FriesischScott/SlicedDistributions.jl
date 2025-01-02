struct SlicedExponential <: SlicedDistribution
    d::Integer
    t::Vector{Monomial}
    λ::AbstractVector
    lb::AbstractVector{<:Real}
    ub::AbstractVector{<:Real}
    c::Float64
end

function SlicedExponential(δ::AbstractMatrix, d::Integer, b::Integer=10000)
    lb = vec(minimum(δ; dims=1))
    ub = vec(maximum(δ; dims=1))

    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    t = monomials(["δ$i" for i in 1:size(δ, 2)], 2d, GradedLexicographicOrder())

    zδ = permutedims(t(transpose(δ)))
    zΔ = permutedims(t(s))

    n = size(δ, 1)
    nz = size(zδ, 2)

    f = get_likelihood(zδ, zΔ, n, prod(ub - lb), b)

    ∇f! = get_gradient(zδ, zΔ, n)

    ∇²f! = get_hessian(zΔ, n)

    result = optimize(f, ∇f!, ∇²f!, ones(nz), Newton())

    cΔ = prod(ub - lb) / b * sum(exp.(zΔ * result.minimizer / -2))

    se = SlicedExponential(d, t, result.minimizer, lb, ub, cΔ)
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
