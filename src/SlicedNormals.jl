module SlicedNormals

using LinearAlgebra, Distributions, PDMats, DynamicPolynomials

import Base: in, ∈

export Z, pdf

include("support.jl")

struct SlicedNormal
    d::Integer
    μ::AbstractVector
    P::AbstractPDMat
    Δ::SupportSet # TODO: Use IntervalArithmetics.jl?
end

function pdf(sn::SlicedNormal, δ::AbstractVector)
    if δ ∈ sn.Δ
        mvn = MvNormal(sn.μ, inv(sn.P))
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

end # module
