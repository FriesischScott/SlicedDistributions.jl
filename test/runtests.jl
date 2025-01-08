using DelimitedFiles
using HCubature
using Logging
using SlicedDistributions
using Test

Logging.disable_logging(Logging.Info)

circle = readdlm("../demo/data/circle.csv", ',')

d = 3
b = 10000

@testset "SlicedNormal" begin
    sn, _ = SlicedNormal(circle, d, b)

    @test all(insupport.(sn, eachcol(circle)))

    @test hcubature(x -> pdf(sn, x), sn.lb, sn.ub)[1] ≈ 1.0 atol = 1e-3

    samples = rand(sn, 1000)

    @test all(insupport.(sn, eachcol(samples)))

    @test repr(sn) ==
        "SlicedNormal(nδ=2, d=3, nz=27,\n  lb=[-3.2929135595124106, -3.453912509293032],\n  ub=[3.3151850678141344, 3.3768332192657207])"
end

@testset "SlicedExponential" begin
    se, _ = SlicedExponential(circle, d, b)

    @test all(insupport.(se, eachcol(circle)))

    @test hcubature(x -> pdf(se, x), se.lb, se.ub)[1] ≈ 1.0 atol = 1e-3

    samples = rand(se, 1000)

    @test all(insupport.(se, eachcol(samples)))

    @test repr(se) ==
        "SlicedExponential(nδ=2, d=3, nz=27,\n  lb=[-3.2929135595124106, -3.453912509293032],\n  ub=[3.3151850678141344, 3.3768332192657207])"
end
