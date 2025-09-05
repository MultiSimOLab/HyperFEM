
abstract type KinematicModel end

struct KinematicDescription{Kind} end

get_Kinematics(::KinematicModel; Λ::Float64) = @abstractmethod

struct Kinematics{A,B} <: KinematicModel
    metrics::A

    function Kinematics(::Union{Type{Mechano},Type{Elasto}}; F::Function=(∇u) -> one(∇u) + ∇u)
        J(F) = det(F)
        H(F) = det(F) * inv(F)'
        metrics = (F, H, J)
        A = typeof(metrics)
        new{A,Mechano}(metrics)
    end

    function Kinematics(::Type{Visco}; F::Function=(∇u) -> one(∇u) + ∇u)
        C(F) = F' * F
        Ce(C,Uvα⁻¹) = Uvα⁻¹ * C * Uvα⁻¹
        metrics = (F, C, Ce)
        A = typeof(metrics)
        new{A,Visco}(metrics)
    end

    function Kinematics(::Type{Electro}; E::Function=(∇φ) -> -∇φ)
        metrics = (E)
        A = typeof(metrics)
        new{A,Electro}(metrics)
    end

    function Kinematics(::Type{Magneto}; H::Function=(∇φ) -> -∇φ)
        metrics = (H)
        A = typeof(metrics)
        new{A,Magneto}(metrics)
    end

end

get_Kinematics(obj::Kinematics; Λ=1.0) = obj.metrics



function getIsoInvariants(obj::Kinematics{<:Function,Mechano})
    F, H, J = obj.metrics
    I1(F) = tr(F' * F)
    I2(F) = tr(H(F)' * H(F))
    I3(F) = J(F)
    return (I1, I2, I3)
end
 

function getIsoInvariants(obj_m::Kinematics{<:Any,Mechano},obj_e::Kinematics{<:Any,Electro})
    F, H, J = obj_m.metrics
    E = obj_e.metrics
    I1(F) = tr(F' * F)
    I2(F) = tr(H(F)' * H(F))
    I3(F) = J(F)
    HE(F, E) = H(F) * E
    I4(F, E) = HE(F, E) ⋅ HE(F, E)
    I5(E) = E ⋅ E
    return (I1, I2, I3, I4, I5)
end
 
struct EvolutiveKinematics{A,B} <: KinematicModel
    metrics::B
    function EvolutiveKinematics(::Type{Mechano}; F::Function=(t) -> ((∇u) -> one(∇u) + ∇u))
        # F_(∇u) = one(∇u) + ∇u
        # F(t) = (∇u)->(Fmapping(t) ∘ F_)(∇u) 
        J(F) = det(F)
        H(F) = det(F) * inv(F)'
        metrics = (F, H, J)
        B = typeof(metrics)
        new{Mechano,B}(metrics)
    end

    function EvolutiveKinematics(::Type{Mechano}, δ::Float64; F::Function=(t) -> ((∇u) -> one(∇u) + ∇u))
        # F_(∇u) = one(∇u) + ∇u
        # F(t) = (∇u)->(Fmapping(t) ∘ F_)(∇u) 
        J_(F) = det(F)
        J(F) =  0.5*(det(F)+sqrt(det(F) ^2+δ^2))
        H(F) = J(F) * inv(F)'
        metrics = (F, H, J)
        B = typeof(metrics)
        new{Mechano,B}(metrics)
    end


    function EvolutiveKinematics(::Type{Electro}; E::Function=(t) -> ((∇φ) -> -∇φ))
        metrics = (E)
        B = typeof(metrics)
        new{Electro,B}(metrics)
    end
    function EvolutiveKinematics(::Type{Magneto}; H::Function=(t) -> ((∇φ) -> -∇φ))
        metrics = (H)
        B = typeof(metrics)
        new{Magneto,B}(metrics)
    end
end

get_Kinematics(obj::EvolutiveKinematics{Mechano,<:Any}; Λ=1.0) = (obj.metrics[1](Λ), obj.metrics[2], obj.metrics[3])
get_Kinematics(obj::EvolutiveKinematics{Electro,<:Any}; Λ=1.0) = obj.metrics(Λ)
get_Kinematics(obj::EvolutiveKinematics{Magneto,<:Any}; Λ=1.0) = obj.metrics(Λ)




