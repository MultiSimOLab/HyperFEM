abstract type AbstractPostProcessor end
get_pvd(::AbstractPostProcessor) = @abstractmethod
vtk_save(::AbstractPostProcessor) = @abstractmethod




mutable struct PostProcessor{A,B,C} <:AbstractPostProcessor
    comp_model::A
    cache::B
    cachevtk::C
    iter::Int64
    Λ ::Vector{Float64}

    function PostProcessor() 
        cachevtk = (false, nothing, nothing)
        cache = ((x...)->())
        Λ = Vector{Float64}()
        A, B, C = typeof(nothing), typeof(cache), typeof(cachevtk)
        new{A,B,C}(nothing, cache, cachevtk, 0, Λ)
    end

    function PostProcessor(comp_model,driver;
        is_vtk=true,
        filepath=datadir("sims", "Temp"),
        kwargs...)

        pvd = paraview_collection(filepath * "/Results", append=false)
        cache = (driver, kwargs)
        cachevtk = (is_vtk, filepath, pvd)
        Λ = Vector{Float64}()

        A, B, C = typeof(comp_model), typeof(cache), typeof(cachevtk)
        new{A,B,C}(comp_model, cache, cachevtk, 0,Λ)
    end


end
get_pvd(p::PostProcessor{<:Any,<:Any,<:Any}) = p.cachevtk[3]
function vtk_save(p::PostProcessor{<:Any,<:Any,<:Any}) 
    if p.cachevtk[1]
        WriteVTK.vtk_save(get_pvd(p))
    end
end

function reset!(obj::PostProcessor)  
    obj.iter=0 
    obj.Λ = Vector{Float64}()
    is_vtk,filepath,pvd= obj.cachevtk
    isnothing(pvd) ? pvd =nothing : pvd = paraview_collection(filepath * "/Results", append=false)
    obj.cachevtk = (is_vtk, filepath, pvd)
end

function (obj::PostProcessor{<:Nothing,<:Any,<:Any})(Λ) end

function (obj::PostProcessor{<:StaticNonlinearModel,<:Any,<:Any})(Λ)
    obj.iter +=1
    push!(obj.Λ, Λ)
    obj.cache[1](obj, obj.cache[2]...)
end

function (obj::PostProcessor{<:StaticLinearModel,<:Any,<:Any})(Λ)
    obj.iter +=1
    push!(obj.Λ, Λ)
    obj.cache[1](obj, obj.cache[2]...)
end

function (obj::PostProcessor{<:DynamicNonlinearModel,<:Any,<:Any})(Λ)
    obj.iter +=1
    push!(obj.Λ, Λ)
    obj.cache[1](obj, obj.cache[2]...)
end

# Mechanical Physics

# mutable struct PostMechanical{A,B,C,D} <: AbstractPostProcessor
#     physicalmodel::A
#     flags::B
#     cache::C
#     cachevtk::D
#     function PostMechanical(model::Mechano, Ω, dΩ;
#         is_vtk=true,
#         filepath=datadir("sims", "Temp"),
#         strains::Bool=false,
#         cauchy::Bool=false
#     )

#         pvd = paraview_collection(filepath * "/Results", append=false)
#         flags = (; strains=strains, cauchy=cauchy)
#         cache = (Ω, dΩ)
#         cachevtk = (is_vtk, filepath, pvd)

#         # if strains
#         #     push!(cache, get_trial_space(CompModel_))
#         # end
#         A, B, C, D = typeof(model), typeof(flags), typeof(cache), typeof(cachevtk)
#         new{A,B,C,D}(model, flags, cache, cachevtk)
#     end
# end

# get_pvd(p::PostMechanical) = p.cachevtk[3]
# vtk_save(p::PostMechanical) =
#     if p.cachevtk[1]
#         WriteVTK.vtk_save(get_pvd(p))
#     end


# function (obj::PostMechanical)(state, Λ, Λ_, m)
#     println("PostMechanical")
#     Ω = get_triangulation(get_spaces(m)[1])
#     xh = FEFunction(get_trial_space(m), state)
#     pvd = obj.cachevtk[3]
#     filePath = obj.cachevtk[2]

#     if obj.cachevtk[1]
#         Λstring = replace(string(round(Λ, digits=2)), "." => "_")
#         pvd[Λ_] = createvtk(Ω,
#             filePath * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
#             cellfields=["u" => xh]
#         )
#     end
# end




# # ElectroMechanical Physics

# mutable struct PostElectroMechanical{A,B,C} <: AbstractPostProcessor
#     flags::A
#     cache::B
#     cachevtk::C
#     function PostElectroMechanical(model::ElectroMechano, Ω, dΩ;
#         is_vtk=true,
#         filepath=datadir("sims", "Temp"),
#         strains::Bool=false,
#         cauchy::Bool=false
#     )

#         pvd = paraview_collection(filepath * "/Results", append=false)
#         flags = (; strains=strains, cauchy=cauchy)
#         cache = (Ω, dΩ)
#         cachevtk = (is_vtk, filepath, pvd)

#         # if strains
#         #     push!(cache, get_trial_space(CompModel_))
#         # end
#         A, B, C = typeof(flags), typeof(cache), typeof(cachevtk)
#         new{A,B,C}(flags, cache, cachevtk)
#     end
# end

# get_pvd(p::PostElectroMechanical) = p.cachevtk[3]
# vtk_save(p::PostElectroMechanical) =
#     if p.cachevtk[1]
#         WriteVTK.vtk_save(get_pvd(p))
#     end


# function (obj::PostElectroMechanical)(state, Λ, Λ_, m)
#     println("PostElectroMechanical")
#     Ω = get_triangulation(get_spaces(m)[1])
#     xh = FEFunction(get_trial_space(m), state)
#     uh = xh[1]
#     φh = xh[2]
#     pvd = obj.cachevtk[3]
#     filePath = obj.cachevtk[2]

#     if obj.cachevtk[1] && (Λ_ % 20 == 0)
#         Λstring = replace(string(round(Λ, digits=2)), "." => "_")
#         pvd[Λ_] = createvtk(Ω,
#             filePath * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
#             cellfields=["u" => uh, "φ" => φh]
#         )
#     end
# end







# # ThermoElectroMechanical Physics

# mutable struct PostThermoElectroMechanical{A,B,C,D} <: AbstractPostProcessor
#     physicalmodel::A
#     flags::B
#     cache::C
#     cachevtk::D
#     function PostThermoElectroMechanical(model::ThermoElectroMechano, Ω, dΩ;
#         is_vtk=true,
#         filepath=datadir("sims", "Temp"),
#         interval=20,
#         strains::Bool=false,
#         cauchy::Bool=false,
#         entropy::Bool=false,
#         D0::Bool=false
#     )

#         pvd = paraview_collection(filepath * "/Results", append=false)
#         flags = (; strains=strains, cauchy=cauchy, entropy=entropy, D0=D0)
#         cache = (interval, Ω, dΩ)
#         cachevtk = (is_vtk, filepath, pvd)

#         # if strains
#         #     push!(cache, get_trial_space(CompModel_))
#         # end
#         A, B, C, D = typeof(model), typeof(flags), typeof(cache), typeof(cachevtk)
#         new{A,B,C,D}(model, flags, cache, cachevtk)
#     end
# end

# get_pvd(p::PostThermoElectroMechanical) = p.cachevtk[3]
# vtk_save(p::PostThermoElectroMechanical) =
#     if p.cachevtk[1]
#         WriteVTK.vtk_save(get_pvd(p))
#     end


# function (obj::PostThermoElectroMechanical)(state, Λ, Λ_, m)
#     println("PostThermoElectroMechanical")
#     Ω = obj.cache[2]
#     dΩ = obj.cache[3]
#     xh = FEFunction(get_trial_space(m), state)
#     uh = xh[1]
#     φh = xh[2]
#     θh = xh[3]
#     pvd = obj.cachevtk[3]
#     filePath = obj.cachevtk[2]
#     physmodel = obj.physicalmodel

#     cellfields=["u" => uh, "φ" => φh, "θ" => θh]
#     if obj.flags.cauchy
#         σ11h, σ12h, σ13h, σ22h, σ23h, σ33h, ph = Cauchy(physmodel, uh, φh, θh, Ω, dΩ, Λ)
#         push!(cellfields, "σ11" => σ11h, "σ12" => σ12h, "σ13" => σ13h, "σ22" => σ22h, "σ23" => σ23h, "σ33" => σ33h, "p" => ph)
#     end

#     if obj.flags.entropy
#         ηh= Entropy(physmodel, uh, φh, θh, Ω, dΩ, Λ)
#         push!(cellfields, "η" => ηh )
#     end

#     if obj.flags.D0
#         D0_1h,D0_2h,D0_3h= D0(physmodel, uh, φh, θh, Ω, dΩ, Λ)
#         push!(cellfields, "D0_1h" => D0_1h, "D0_2h" => D0_2h, "D0_3h" => D0_3h)
#     end

#     if obj.cachevtk[1] && (Λ_ % obj.cache[1] == 0)
#         Λstring = replace(string(round(Λ, digits=2)), "." => "_")
#         pvd[Λ_] = createvtk(Ω,
#             filePath * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
#             cellfields=cellfields
#         )

#     end
# end


function Cauchy(physmodel::ThermoElectroMechano, uh, φh, θh, Ω, dΩ, Λ=1.0)
    DΨ = physmodel(Λ)
    Kinematic_mec = Kinematics(Mechano)
    Kinematic_elec = Kinematics(Electro)
    F, _, _ = get_Kinematics(Kinematic_mec)
    E = get_Kinematics(Kinematic_elec)
    ∂Ψu = DΨ[2]
    refL2 = ReferenceFE(lagrangian, Float64, 0)
    ref = ReferenceFE(lagrangian, Float64, 1)
    VL2 = FESpace(Ω, refL2, conformity=:L2)
    V = FESpace(Ω, ref, conformity=:H1)
    n1 = VectorValue(1.0, 0.0, 0.0)
    n2 = VectorValue(0.0, 1.0, 0.0)
    n3 = VectorValue(0.0, 0.0, 1.0)
    σ11h = interpolate_everywhere(L2_Projection(n1 ⋅ ((∂Ψu ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)) * n1), dΩ, VL2), V)
    σ12h = interpolate_everywhere(L2_Projection(n1 ⋅ ((∂Ψu ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)) * n2), dΩ, VL2), V)
    σ13h = interpolate_everywhere(L2_Projection(n1 ⋅ ((∂Ψu ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)) * n3), dΩ, VL2), V)
    σ22h = interpolate_everywhere(L2_Projection(n2 ⋅ ((∂Ψu ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)) * n2), dΩ, VL2), V)
    σ23h = interpolate_everywhere(L2_Projection(n2 ⋅ ((∂Ψu ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)) * n3), dΩ, VL2), V)
    σ33h = interpolate_everywhere(L2_Projection(n3 ⋅ ((∂Ψu ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)) * n3), dΩ, VL2), V)
    ph = interpolate_everywhere(L2_Projection(tr ∘ (∂Ψu ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)), dΩ, VL2), V)
    return (σ11h, σ12h, σ13h, σ22h, σ23h, σ33h, ph)
end




function Entropy(physmodel::ThermoElectroMechano, uh, φh, θh, Ω, dΩ, Λ=1.0)
    DΨ = physmodel(Λ)
    Kinematic_mec = Kinematics(Mechano)
    Kinematic_elec = Kinematics(Electro)
    F, _, _ = get_Kinematics(Kinematic_mec)
    E = get_Kinematics(Kinematic_elec)
    η = DΨ[11]
    refL2 = ReferenceFE(lagrangian, Float64, 0)
    ref = ReferenceFE(lagrangian, Float64, 1)
    VL2 = FESpace(Ω, refL2, conformity=:L2)
    V = FESpace(Ω, ref, conformity=:H1)
    ηh = interpolate_everywhere(L2_Projection((η ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)), dΩ, VL2), V)
    return ηh
end

function D0(physmodel::ThermoElectroMechano, uh, φh, θh, Ω, dΩ, Λ=1.0)
    DΨ = physmodel(Λ)
    Kinematic_mec = Kinematics(Mechano)
    Kinematic_elec = Kinematics(Electro)
    F, _, _ = get_Kinematics(Kinematic_mec)
    E = get_Kinematics(Kinematic_elec)
    ∂ΨE = DΨ[3]
    refL2 = ReferenceFE(lagrangian, Float64, 0)
    ref = ReferenceFE(lagrangian, Float64, 1)
    VL2 = FESpace(Ω, refL2, conformity=:L2)
    V = FESpace(Ω, ref, conformity=:H1)
    n1 = VectorValue(-1.0, 0.0, 0.0)
    n2 = VectorValue(0.0, -1.0, 0.0)
    n3 = VectorValue(0.0, 0.0, -1.0)
    D0_1h = interpolate_everywhere(L2_Projection(n1 ⋅(∂ΨE ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)), dΩ, VL2), V)
    D0_2h = interpolate_everywhere(L2_Projection(n2 ⋅(∂ΨE ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)), dΩ, VL2), V)
    D0_3h = interpolate_everywhere(L2_Projection(n3 ⋅(∂ΨE ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)), dΩ, VL2), V)
    return (D0_1h,D0_2h,D0_3h)
end







function L2_Projection(Field, dΩ, VFE)
    a(Fieldh, v) = ∫(Fieldh * v) * dΩ
    l(v) = ∫(v * Field) * dΩ
    op = AffineFEOperator(a, l, VFE, VFE)
    return solve(op)
end

