using GridapMultiSimO
using Gridap
using DrWatson
using GridapGmsh, Gridap.FESpaces

pname = "Static_Mechanical"
meshfile = "cantilever.msh"
is_vtk = true
simdir = datadir("sims", pname)
setupfolder(simdir)

geomodel     = GmshDiscreteModel(datadir("models", meshfile))

evolu(Λ) = Λ
dir_u_tags = ["fixedup"]
dir_u_values = [[0.0, 0.0, 1.0]]
dir_u_timesteps = [evolu]
Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

evolφ(Λ) = Λ
dir_φ_tags = ["midsuf", "topsuf"]
dir_φ_values = [0.0, 0.25]
dir_φ_timesteps = [evolφ, evolφ]
Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

dirichletbc = MultiFieldBC([Du, Dφ])

# elements
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, 1)
reffeφ = ReferenceFE(lagrangian, Float64, 1)
# Test Spaces
Vu = TestFESpace(geomodel, reffeu, dirichletbc.BoundaryCondition[1], :H1)
Vφ = TestFESpace(geomodel, reffeφ, dirichletbc.BoundaryCondition[2], :H1)
# Trial Spaces
Uu = TrialFESpace(Vu, dirichletbc.BoundaryCondition[1],1.0)
Uφ = TrialFESpace(Vφ, dirichletbc.BoundaryCondition[2],1.0)

# Multifield Spaces
U=MultiFieldFESpace([Uu,Uφ])
V=MultiFieldFESpace([Vu,Vφ])


 

function update_dirichlet!(space::TrialFESpace, dir_bc::DirichletBC, Λ)
    TrialFESpace!(space,dir_bc,Λ)
   end
   
   function update_dirichlet!(U::MultiFieldFESpace, dir_bc::MultiFieldBC, Λ)
     @inbounds for (i,space) in enumerate(U.spaces)
       TrialFESpace!(space,dir_bc.BoundaryCondition[i],Λ)
     end
   end


   update_dirichlet!(U, dirichletbc, 0.5)

   update_dirichlet!( Uu, Du, 0.5)

   Uu.dirichlet_values
   U.spaces[1].dirichlet_values


   

# Update spaces
λ=0.6
@inbounds for (i,space) in enumerate(U.spaces)
    TrialFESpace!(space,dirichletbc.BoundaryCondition[i],λ)
end

 