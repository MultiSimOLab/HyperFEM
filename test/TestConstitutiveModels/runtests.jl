using HyperFEM

@testset "ConstitutiveModels" begin

    @time begin
        include("PhysicalModels.jl")
    end

    @time begin
        include("ViscousModelsTests.jl")
    end

end