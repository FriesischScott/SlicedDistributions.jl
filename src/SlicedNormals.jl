using LinearAlgebra, Distributions, PDMats

import Base: in, ∈

module SlicedNormals

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

end # module
