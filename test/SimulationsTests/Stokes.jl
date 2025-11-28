
filename = projdir("test/data/Stokes.jl")
include(filename)

x = Stokes(writevtk=false, verbose=false)

@test norm(x) ≈ 4199.925167995936
