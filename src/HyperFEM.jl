module HyperFEM

using TimerOutputs


include("TensorAlgebra/TensorAlgebra.jl")
include("PhysicalModels/PhysicalModels.jl")
include("WeakForms/WeakForms.jl")
include("ComputationalModels/ComputationalModels.jl")
include("Solvers/Solvers.jl")

include("Io.jl")
export setupfolder

include("Exports.jl")
end
