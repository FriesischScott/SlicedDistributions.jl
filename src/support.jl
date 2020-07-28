struct SupportSet
    lb::AbstractVector
    ub::AbstractVector
end

function in(x::AbstractVector, y::SupportSet)
    y.lb .<= x .<= y.ub
end

function ∈(x::AbstractVector, y::SupportSet) = in(x, y)