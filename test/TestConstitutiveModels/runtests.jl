using HyperFEM
using Test

@testset "ConstitutiveModels" begin

    @time begin
        include("PhysicalModels.jl")
    end

    @time begin
        include("ViscousModelsTests.jl")
    end

    @time begin
        include("ElectroMechanicalTests.jl")
    end

end
