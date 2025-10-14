
# =================
# Hyper-elasticity
# =================

# -----------------
# Stagered residual
# -----------------

function residual(physicalmodel::ElectroMechano, ::Type{Mechano}, (u, φ), v, dΩ, Λ=1.0)
    DΨ    = physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.Mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.Electro.Kinematic; Λ=Λ)
    ∂Ψu   = DΨ[2]
    ∫(∇(v)' ⊙ (∂Ψu ∘ (F∘∇(u)', E∘∇(φ))))dΩ
end

function residual(physicalmodel::ElectroMechano, ::Type{Electro}, (u, φ), vφ, dΩ, Λ=1.0)
    DΨ    = physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.Mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.Electro.Kinematic; Λ=Λ)
    ∂Ψφ   = DΨ[3]
    -1.0*∫(∇(vφ) ⋅ (∂Ψφ ∘ (F∘∇(u)', E∘∇(φ))))dΩ
end

# -----------------
# Stagered jacobian
# -----------------

function jacobian(physicalmodel::ElectroMechano, ::Type{Mechano}, (u, φ), du, v, dΩ, Λ=1.0)
    DΨ    = physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.Mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.Electro.Kinematic; Λ=Λ)
    ∂Ψuu  = DΨ[4]
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘∇(u)', E∘∇(φ))) ⊙ ∇(du)'))dΩ
end

function jacobian(physicalmodel::ElectroMechano, ::Type{Electro}, (u, φ), dφ, vφ, dΩ, Λ=1.0)
    DΨ    = physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.Mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.Electro.Kinematic; Λ=Λ)
    ∂Ψφφ  = DΨ[6]
    ∫(∇(vφ)' ⋅ ((∂Ψφφ ∘ (F∘∇(u)', E∘∇(φ))) ⋅ ∇(dφ)))dΩ
end

function jacobian(physicalmodel::ElectroMechano, ::Type{ElectroMechano}, (u, φ), (du, dφ), (v, vφ), dΩ, Λ=1.0)
    DΨ    = physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.Mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.Electro.Kinematic; Λ=Λ)
    ∂Ψφu  = DΨ[5]
    -1.0*∫(∇(dφ) ⋅ ((∂Ψφu ∘ (F∘∇(u)', E∘∇(φ))) ⊙ ∇(v)'))dΩ -
        ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (F∘∇(u)', E∘∇(φ))) ⊙ ∇(du)'))dΩ 
end

# -------------------
# Monolithic strategy
# -------------------

function residual(physicalmodel::ElectroMechano, (u, φ), (v, vφ), dΩ, Λ=1.0)
    residual(physicalmodel, Mechano, (u, φ), v, dΩ, Λ) +
    residual(physicalmodel, Electro, (u, φ), vφ, dΩ, Λ)
end


function jacobian(physicalmodel::ElectroMechano, (u, φ), (du, dφ), (v, vφ), dΩ, Λ=1.0)
    jacobian(physicalmodel, Mechano, (u, φ), du, v, dΩ, Λ)+
    jacobian(physicalmodel, Electro, (u, φ), dφ, vφ, dΩ, Λ)+
    jacobian(physicalmodel, ElectroMechano, (u, φ), (du, dφ), (v, vφ), dΩ, Λ)
end


# =================
# Visco-elasticity
# =================

# -----------------
# Stagered residual
# -----------------

function residual(physicalmodel::ViscoElectricModel, ::Type{Mechano}, (u, φ), v, dΩ, Λ, Δt, un, A)
    DΨ    = physicalmodel(Λ, Δt=Δt)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electro.Kinematic; Λ=Λ)
    ∂Ψu   = DΨ[2]
    ∫(∇(v)' ⊙ (∂Ψu ∘ (F∘∇(u)', F∘∇(un)', E∘∇(φ), A...)))dΩ
end

function residual(physicalmodel::ViscoElectricModel, ::Type{Electro}, (u, φ), vφ, dΩ, Λ, Δt, un, A)
    DΨ    = physicalmodel(Λ, Δt=Δt)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electro.Kinematic; Λ=Λ)
    ∂Ψφ   = DΨ[3]
    -1.0*∫(∇(vφ) ⋅ (∂Ψφ ∘ (F∘∇(u)', F∘∇(un)', E∘∇(φ), A...)))dΩ
end

# -----------------
# Stagered jacobian
# -----------------

function jacobian(physicalmodel::ViscoElectricModel, ::Type{Mechano}, (u, φ), du, v, dΩ, Λ, Δt, un, A)
    DΨ    = physicalmodel(Λ, Δt=Δt)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electro.Kinematic; Λ=Λ)
    ∂Ψuu  = DΨ[4]
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘∇(u)', F∘∇(un)', E∘∇(φ), A...)) ⊙ (∇(du)')))dΩ
end

function jacobian(physicalmodel::ViscoElectricModel, ::Type{Electro}, (u, φ), dφ, vφ, dΩ, Λ, Δt, un, A)
    DΨ    = physicalmodel(Λ, Δt=Δt)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electro.Kinematic; Λ=Λ)
    ∂Ψφφ  = DΨ[6]
    ∫(∇(vφ)' ⋅ ((∂Ψφφ ∘ (F∘∇(u)', F∘∇(un)', E∘∇(φ), A...)) ⋅ ∇(dφ)))dΩ
end

function jacobian(physicalmodel::ViscoElectricModel, ::Type{ElectroMechano}, (u, φ), (du, dφ), (v, vφ), dΩ, Λ, Δt, un, A)
    DΨ    = physicalmodel(Λ, Δt=Δt)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electro.Kinematic; Λ=Λ)
    ∂Ψφu  = DΨ[5]
    -1.0*∫(∇(dφ) ⋅ ((∂Ψφu ∘ (F∘∇(u)', F∘∇(un)', E∘∇(φ), A...)) ⊙ (∇(v)')))dΩ -
        ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (F∘∇(u)', F∘∇(un)', E∘∇(φ), A...)) ⊙ (∇(du)')))dΩ 
end

# -------------------
# Monolithic strategy
# -------------------

function residual(physicalmodel::ViscoElectricModel, (u, φ), (v, vφ), dΩ, Λ, Δt, un, A)
    residual(physicalmodel, Mechano, (u, φ), v, dΩ, Λ, Δt, un, A) +
    residual(physicalmodel, Electro, (u, φ), vφ, dΩ, Λ, Δt, un, A)
end

function jacobian(physicalmodel::ViscoElectricModel, (u, φ), (du, dφ), (v, vφ), dΩ, Λ, Δt, un, A)
    jacobian(physicalmodel, Mechano, (u, φ), du, v, dΩ, Λ, Δt, un, A) +
    jacobian(physicalmodel, Electro, (u, φ), dφ, vφ, dΩ, Λ, Δt, un, A) +
    jacobian(physicalmodel, ElectroMechano, (u, φ), (du, dφ), (v, vφ), dΩ, Λ, Δt, un, A)
end
