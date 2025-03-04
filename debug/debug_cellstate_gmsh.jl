
using Gridap
using GridapGmsh
using Gridap.TensorValues
using Gridap.CellData
 
using Mimosa

 


function Gridap.CellData.evaluate!(cache,f::CellState,x::CellPoint)
 
  # f.values


  if f.points === x
    f.values
  else
    println("f_")
    @show sum(get_data(f.points))
    println("x_")
    @show sum(get_data(x))
    error("dd")

 
  end
end


mesh_file = joinpath("./data/models/Column_mesh.msh")
model = GmshDiscreteModel(mesh_file)
reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, 1)
Ω = Triangulation(model)
dΩ = Measure(Ω, 2)
TRIANG = (; Ω, dΩ)
V = TestFESpace(Ω, reffe, conformity=:H1)
# UCell = CellState(1.0, dΩ)
UCell = CellState(SymTensorValue(1.0,1.0,1.0,1.0,1.0,1.0), dΩ)
res(v) = ∫(∇(v)' ⊙ (UCell))dΩ
T = assemble_vector(res, V)

UCell2 = CellState(VectorValue(1.0,2.0,3.0), dΩ)
res2(v) = ∫(v⋅ UCell2)dΩ
vec=zeros(get_num_dofs(V))
T2 = assemble_vector(res2, V)
 
A=reshape(T2,3,400)
 Tx=sum(A[1,:])
 Ty=sum(A[2,:])
 Tz=sum(A[3,:])


function up(A)
A+=0.2
return true, A
end

update_state!(up, UCell2)

 









function L2_CellState_Projection(CellField_, model, dΩ, order)
  reffCell = ReferenceFE(lagrangian, Float64, order)
  VCell = FESpace(model, reffCell, conformity=:L2)
  a(u, v) = ∫(u * v)dΩ
  l(v) = ∫(v * CellField_) * dΩ
  op = AffineFEOperator(a, l, VCell, VCell)
  return solve(op)
end

function L2_VectorCellState_Projection(CellField_, model, dΩ, order)
  reffCell = ReferenceFE(lagrangian, VectorValue{6,Float64}, order)
  VCell = FESpace(model, reffCell, conformity=:L2)
  a(u, v) = ∫(u ⊙ v)dΩ
  l(v) = ∫(v ⊙ CellField_) * dΩ
  op = AffineFEOperator(a, l, VCell, VCell)
  return solve(op)
end


reffeUv = ReferenceFE(lagrangian, VectorValue{6,Float64}, FEM.order)
VUv = TestFESpace(TRIANG.Ω, reffeUv, conformity=:L2)

Uvh = FEFunction(VUv, zeros(Float64, num_free_dofs(VUv)))
UvCell = CellState(VectorValue(1.0, 0.0, 0.0, 1.0, 0.0, 1.0), dΩ)

v1 = get_free_dof_values(Uvh)
u1 = get_free_dof_values(L2_VectorCellState_Projection(UvCell, MODEL.model, TRIANG.dΩ, FEM.order))
v1[:] = u1

#---------------------------------------------
# Weak forms
#---------------------------------------------
u = zeros(Float64, num_free_dofs(FEM.Vu))
uh = FEFunction(FEM.Vu, u)
unh = FEFunction(FEM.Vu, u)

function residualViscoh(u, Uvn, dΩ)
  res(v) = ∫(∇(v)' ⊙ (Uvn))dΩ
end
Tv = assemble_vector(residualViscoh(uh, Uvh, TRIANG.dΩ), FEM.Vu)

error("ddddd")

VectorState(θh, ch, α, Λh) = (u, v) -> ∫(∇(v)' ⊙ (∂Ψ_∂F ∘ (∇(u)', α, N ∘ (θh), ch))) * dΩ - ∫(Λh * τ ⋅ v) * dΓ
MatrixState(θh, ch, α) = (u, du, v) -> ∫(∇(v)' ⊙ (inner42_2D ∘ ((∂²Ψ_∂F∂F ∘ (∇(u)', α, N ∘ (θh), ch)), ∇(du)'))) * dΩ


