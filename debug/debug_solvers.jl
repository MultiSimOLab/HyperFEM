    # u = zero_free_values(U)
    # uh = FEFunction(U, u)

    # a(du, v) = jacobian(physmodel, uh, du, v, dΩ)
    # l(v) = residual(physmodel, uh, v, dΩ)+residual_Neumann(N_bc, v, dΓ, 1.0)

    # op = AffineFEOperator(a, l, U, V)
    # op_ = get_algebraic_operator(op)

    # A, b = get_matrix(op), get_vector(op)
    # ns = numerical_setup(symbolic_setup(ls, A), A)

    # Gridap.Algebra.solve!(u, ns, b)
