using SlicedNormals
using PDMats

S = PDMat([1000 500; 500 1000])
z = [100, 100]
m = 10000

U = SlicedNormals._sample_ellipsoid(S, z, m)

scatter(U[1, :], U[2, :], lims=[0, 200], aspect_ratio=:equal, legend=:none)
