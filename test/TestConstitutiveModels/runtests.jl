using HyperFEM

@testset "ConstitutiveModels" verbose = true begin

    @time begin
        include("PhysicalModels.jl")
    end

    @time begin
        include("ViscousModelsTests.jl")
    end

end