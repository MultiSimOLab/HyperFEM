
# ===================
# Electrical models
# ===================

struct IdealDielectric <: Electro
  ε::Float64
  function IdealDielectric(; ε::Float64)
    new(ε)
  end
end
