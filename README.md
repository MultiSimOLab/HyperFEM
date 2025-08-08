 <p align="center"> 
&nbsp; &nbsp; &nbsp; &nbsp;
<img alt="Dark"
src="https://github.com/jmartfrut/HyperFEM/blob/main/docs/imgs/logo.png?raw=true" width="30%">
</p>
 

<!-- # HyperFEM :construction: :construction: :construction: **Work in progress** :construction: :construction: :construction: -->

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jmartfrut.github.io/HyperFEM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jmartfrut.github.io/HyperFEM.jl/dev/)
[![Build Status](https://github.com/jmartfrut/HyperFEM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jmartfrut/HyperFEM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jmartfrut/HyperFEM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jmartfrut/HyperFEM.jl)

# **M**ultiphysics-informed **D**esign of **T**unable **S**mart **M**aterials

This is an application repository with a collection of drivers for the simulation of Thermo-Electro-Magneto-Mechanical problems. It is based on [Gridap](https://github.com/gridap/Gridap.jl), a package for grid-based approximation of PDEs with Finite Element.

## Installation
Open the Julia REPL, type `]` to enter package mode, and install as follows
```julia
pkg> add https://github.com/jmartfrut/HyperFEM
```

## Usage
First, include the main HyperFEM module:
```julia
using HyperFEM
```

```julia
using HyperFEM
using HyperFEM: jacobian, solve!
using Gridap, GridapGmsh, GridapSolvers, DrWatson
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using WriteVTK


simdir = datadir("sims", "Static_ElectroMechanical")
setupfolder(simdir)

geomodel = GmshDiscreteModel(datadir("models", "ex2_mesh.msh"))

physmodel_mec = NeoHookean3D(λ=10.0, μ=1.0)
physmodel_elec = IdealDielectric(ε=1.0)
physmodel = ElectroMechModel(Mechano=physmodel_mec, Electro=physmodel_elec)

# Setup integration
order = 1
degree = 2 * order
Ω = Triangulation(geomodel)
dΩ = Measure(Ω, degree)

# Dirichlet conditions 
evolu(Λ) = 1.0
dir_u_tags = ["fixedup"]
dir_u_values = [[0.0, 0.0, 0.0]]
dir_u_timesteps = [evolu]
Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

evolφ(Λ) = Λ
dir_φ_tags = ["midsuf", "topsuf"]
dir_φ_values = [0.0, 0.1]
dir_φ_timesteps = [evolφ, evolφ]
Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

D_bc = MultiFieldBC([Du, Dφ])

# FE spaces
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffeφ = ReferenceFE(lagrangian, Float64, order)

# Test FE Spaces
Vu = TestFESpace(geomodel, reffeu, D_bc.BoundaryCondition[1], conformity=:H1)
Vφ = TestFESpace(geomodel, reffeφ, D_bc.BoundaryCondition[2], conformity=:H1)

# Trial FE Spaces
Uu = TrialFESpace(Vu, D_bc.BoundaryCondition[1], 1.0)
Uφ = TrialFESpace(Vφ, D_bc.BoundaryCondition[2], 1.0)

# Multifield FE Spaces
V = MultiFieldFESpace([Vu, Vφ])
U = MultiFieldFESpace([Uu, Uφ])

# residual and jacobian function of load factor
res(Λ) = ((u, φ), (v, vφ)) -> residual(physmodel, (u, φ), (v, vφ), dΩ)
jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) -> jacobian(physmodel, (u, φ), (du, dφ), (v, vφ), dΩ)

# nonlinear solver
ls = LUSolver()
nls_ = NewtonSolver(ls; maxiter=20, atol=1.e-10, rtol=1.e-8, verbose=true)

# Computational model
comp_model = StaticNonlinearModel(res, jac, U, V, D_bc; nls=nls_)


# Postprocessor to save results
function driverpost(post; Ω=Ω, U=U)
    # get from postprocessor 
    state = post.comp_model.caches[3]
    Λ_ = post.iter
    Λ = post.Λ[Λ_]

    xh = FEFunction(U, state)
    uh = xh[1]
    φh = xh[2]
    pvd = post.cachevtk[3]
    filePath = post.cachevtk[2]

    if post.cachevtk[1]
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        pvd[Λ_] = createvtk(Ω,
            filePath * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
            cellfields=["u" => uh, "φ" => φh]
        )
    end
end

post_model = PostProcessor(comp_model, driverpost; is_vtk=true, filepath=simdir)

# Solve
x = solve!(comp_model; stepping=(nsteps=5, maxbisec=5), post=post_model)
```

## How to cite HyperFEM

In order to give credit to the HyperFEM contributors, we ask that you please reference the paper:

> ------

along with the required citations for [Gridap](https://github.com/gridap/Gridap.jl).


# Project funded by:
 
- Grants PID2022-141957OA-C22/PID2022-141957OB-C22  funded by MCIN/AEI/ 10.13039/501100011033  and by ''ERDF A way of making Europe''


 <p align="center"> 
&nbsp; &nbsp; &nbsp; &nbsp;
<img alt="Dark"
src="https://github.com/jmartfrut/HyperFEM/blob/main/docs/imgs/aei.png?raw=true" width="70%">
</p>
 
#  Contact

Contact the project administrator [Jesús Martínez-Frutos](jesus.martinez@upct.es) for further questions about licenses and terms of use.