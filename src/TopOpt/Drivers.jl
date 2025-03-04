abstract type TopOpt end

get_state(::TopOpt) = @abstractmethod

function solve!(::TopOpt, ::Float64)
    @abstractmethod
end



#*******************************************************************************	
#    					 SIMP method
#*******************************************************************************	


struct SIMP{A,B,C,D,E} <: TopOpt
     Φ::A  # desing variables
     forward::B
     adjoint::C
     filter::D

    function SIMP(
        Φ::FEFunction, forward::ComputationalModel, adjointU::ComputationalModel, filter::ComputationalModel)
        spaces = (U, V, ∆U)
        x = zero_free_values(U)
        x⁻ = zero_free_values(U)
        _res = res(1.0)
        _jac = jac(1.0)
        op = get_algebraic_operator(FEOperator(_res, _jac, U, V, assem_U))
        nls_cache = instantiate_caches(x, nls, op)
        fwd_caches = (nls, nls_cache, x, x⁻, assem_U)

        A, B, C = typeof(res), typeof(jac), typeof(spaces)
        D, E = typeof(dirbc), typeof(fwd_caches)
        return new{A,B,C,D,E}(res, jac, spaces, dirbc, fwd_caches)
    end
end



