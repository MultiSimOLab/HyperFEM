include("../src/Mimosa.jl")

using Gridap
using LazilyInitializedFields


abstract type PhysicalDomain end
abstract type ConstitutiveModel end

struct Solid{Kind} <: PhysicalDomain 
    ConstitutiveModel :: ConstitutiveModel
    tag :: String
    Ω :: Gridap.Geometry.BodyFittedTriangulation
end

