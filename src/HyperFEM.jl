module HyperFEM

using TimerOutputs


include("TensorAlgebra/TensorAlgebra.jl")
include("PhysicalModels/PhysicalModels.jl")
include("WeakForms/WeakForms.jl")
include("Solvers/Solvers.jl")
include("ComputationalModels/ComputationalModels.jl")

include("Io.jl")
export setupfolder

include("Exports.jl")
end
