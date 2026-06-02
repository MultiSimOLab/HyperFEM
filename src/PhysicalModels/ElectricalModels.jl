
# ===================
# Electrical models
# ===================

struct IdealDielectric <: Electro
  ε::Real
  function IdealDielectric(; ε::Real)
    new(ε)
  end
end
