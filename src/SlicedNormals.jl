module SlicedNormals

using LinearAlgebra, Distributions, PDMats, DynamicPolynomials, IntervalArithmetic

import Base: in, ∈

export SlicedNormal, Z, pdf

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

end # module
