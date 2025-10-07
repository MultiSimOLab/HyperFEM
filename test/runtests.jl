using HyperFEM
using Test
using Gridap


@testset "HyperFEMTests" verbose = true begin

    @time begin
        include("../test/TestConstitutiveModels/runtests.jl")
    end
    @time begin
        include("./TestTensorAlgebra/runtests.jl")
    end
    @time begin
        include("../test/TestWeakForms/runtests.jl")
    end

end;
