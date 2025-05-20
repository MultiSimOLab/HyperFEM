abstract type ComputationalModel end

get_state(::ComputationalModel) = @abstractmethod
get_measure(::ComputationalModel) = @abstractmethod
get_spaces(::ComputationalModel) = @abstractmethod
get_trial_space(m::ComputationalModel) = get_spaces(m)[1]
get_test_space(m::ComputationalModel) = get_spaces(m)[2]

function solve!(::ComputationalModel, ::Float64)
    @abstractmethod
end


#*******************************************************************************	
#    					 StaggeredModel
#*******************************************************************************	


struct StaggeredModel{A,B,C,D} <: ComputationalModel
    compmodels::A
    state⁺::B
    state⁻::C
    caches::D

    function StaggeredModel(CompModels::Tuple{Vararg{<:ComputationalModel}},
        state⁺::Tuple{Vararg{<:FEFunction}},
        state⁻::Tuple{Vararg{<:FEFunction}})

        @assert(length(CompModels) == length(state⁺) == length(state⁻))

        x⁺ = map((x) -> get_free_dof_values(x), state⁺)
        x⁻ = map((x) -> get_free_dof_values(x), state⁻)

        caches = (x⁺, x⁻)
        A, B, C, D = typeof(CompModels), typeof(state⁺), typeof(state⁻), typeof(caches)

        return new{A,B,C,D}(CompModels, state⁺, state⁻, caches)
    end

end


function solve!(m::StaggeredModel;
    stepping=(nsteps=20, maxbisec=15),
    kargsolve)

    x⁺, x⁻ = m.caches
    map((x) -> TrialFESpace!(x.spaces[1], x.dirichlet, 0.0), m.compmodels)
    map((x, y) -> TrialFESpace!(x.fe_space, y.dirichlet, 0.0), m.state⁻, m.compmodels)

    flagconv = 1 # convergence flag 0 (max bisections) 1 (max steps)
    ∆Λ = 1.0 / stepping[:nsteps]
    for time in 0:stepping[:nsteps]-1
        println("Time step: $time")
        stevol(Λ) = ∆Λ * (Λ + time)
        # map(x -> updateBC!(x.dirichlet, x.dirichlet.caches, [stevol for _ in 1:length(x.dirichlet.caches)]), m.compmodels)
        map(x -> updateBC!(x.dirichlet, stevol), m.compmodels)
        map((x) -> TrialFESpace!(x.spaces[1], x.dirichlet, 1.0), m.compmodels)
        _, flag = map((x, y) -> solve!(x; y...), m.compmodels, kargsolve)
        map((x, y) -> TrialFESpace!(x.fe_space, y.dirichlet, 1.0), m.state⁻, m.compmodels)
        map((x, y) -> x .= y, x⁻, x⁺)
    end
end

#*******************************************************************************	
#    					 StaticNonlinearModel
#*******************************************************************************	

struct StaticNonlinearModel{A,B,C,D,E} <: ComputationalModel
    res::A
    jac::B
    spaces::C
    dirichlet::D
    caches::E

    function StaticNonlinearModel(
        res::Function, jac::Function, U, V, dirbc;
        assem_U=SparseMatrixAssembler(U, V),
        nls::NonlinearSolver=NewtonSolver(LUSolver(); maxiter=10, rtol=1.e-8, verbose=true),
        xh::FEFunction=FEFunction(U, zero_free_values(U)))

        ∆U = TrialFESpace(U, dirbc, 0.0)
        TrialFESpace!(U, dirbc, 0.0)
        spaces = (U, V, ∆U)
        x = get_free_dof_values(xh)
        x⁻ = zero_free_values(U)
        _res = res(0.0)
        _jac = jac(0.0)
        op = get_algebraic_operator(FEOperator(_res, _jac, U, V, assem_U))
        nls_cache = instantiate_caches(x, nls, op)
        caches = (nls, nls_cache, x, x⁻, assem_U)

        A, B, C = typeof(res), typeof(jac), typeof(spaces)
        D, E = typeof(dirbc), typeof(caches)
        return new{A,B,C,D,E}(res, jac, spaces, dirbc, caches)
    end
end

# Getters
get_state(m::StaticNonlinearModel) = FEFunction(get_trial_space(m), m.caches[3])
get_measure(m::StaticNonlinearModel) = m.res.dΩ
get_spaces(m::StaticNonlinearModel) = m.spaces
get_assemblers(m::StaticNonlinearModel) = (m.caches[4])


# is_vtk=true,
#   filePath=datadir("sims", "Temp"),
#   vtk::WriteVTK.CollectionFile=paraview_collection(datadir("sims", "Temp") * "/Results", append=false)

function solve!(m::StaticNonlinearModel;
    stepping=(nsteps=20, maxbisec=15), RestartState::Bool=false ,ProjectDirichlet::Bool=true, 
    post=PostProcessor())

    reset!(post)
    flagconv = 1 # convergence flag 0 (max bisections) 1 (max steps)
    U, V = m.spaces
    TrialFESpace!(U, m.dirichlet, 0.0)
    nls, nls_cache, x, x⁻, assem_U = m.caches

    Λ = 0.0
    ∆Λ = 1.0 / stepping[:nsteps]
    nbisect = 0
    Λ_ = 0
    if RestartState
    x .*= 0.0
    end
    x⁻ .= x

    while Λ < 1.0
        Λ += ∆Λ
        Λ = min(1.0, Λ)
        if ProjectDirichlet
            dirichlet_preconditioning!(x, m, Λ, ∆Λ)
        end
        TrialFESpace!(U, m.dirichlet, Λ)
        res = m.res(Λ)
        jac = m.jac(Λ)
        op = get_algebraic_operator(FEOperator(res, jac, U, V, assem_U))
        Algebra.solve!(x, nls, op, nls_cache)
        r_abs = nls.log.residuals[nls.log.num_iters+1]
        r_rel = r_abs / nls.log.residuals[1]
        if !converged(nls.log.tols, nls.log.num_iters, r_abs, r_rel)
            @warn "Bisection performed!"
            x .= x⁻
            Λ -= ∆Λ
            ∆Λ = ∆Λ / 2
            nbisect += 1
            # @assert(nbisect <= stepping[:maxbisec], "Maximum number of bisections reached")
        else
            print_message(nls.log, "\nSTEP: $Λ_, Λ: $Λ\n")
            Λ_ += 1
            x⁻ .= x
            # Write to PVD
            post(Λ)
        end
        #  GC.gc()

        if nbisect > stepping[:maxbisec]
            @warn "Maximum number of bisections reached"
            flagconv = 0
            break
        end

    end

     vtk_save(post)

    return x, flagconv
end

function post_solve!(pvd, x, Λ, Λ_, m, filePath)

    Ω = get_triangulation(get_spaces(m)[1])
    xh = FEFunction(get_trial_space(m), x)

    Λstring = replace(string(round(Λ, digits=2)), "." => "_")
    pvd[Λ_] = createvtk(Ω,
        filePath * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
        cellfields=["u" => xh],
    )

    return pvd
end

function dirichlet_preconditioning!(x::Vector{Float64}, m::StaticNonlinearModel, Λ::Float64, ∆Λ::Float64)
    duh = get_dirichlet_preconditioner(m::StaticNonlinearModel, Λ::Float64, ∆Λ::Float64)
    x .+= get_free_dof_values(duh)
end

function get_dirichlet_preconditioner(m::StaticNonlinearModel, Λ::Float64, ∆Λ::Float64)
    _, V, ∆U = m.spaces
    uh = get_state(m)
    TrialFESpace!(∆U, m.dirichlet, ∆Λ)
    res = m.res(Λ - ∆Λ)
    jac = m.jac(Λ - ∆Λ)
    l(v) = -1.0 * res(uh, v)
    a(du, v) = jac(uh, du, v)
    op = AffineFEOperator(a, l, ∆U, V)
    ls = m.caches[1].ls
    duh = solve(ls, op)
    return duh
end
#*******************************************************************************	
#    					 DynamicNonlinearModel
#*******************************************************************************	


struct DynamicNonlinearModel{A,B,C,D,E,F} <: ComputationalModel
    res::A
    jac::B
    spaces::C
    dirichlet::D
    velocity::E
    caches::F

    function DynamicNonlinearModel(
        res::Function, jac::Function, U, Uold, V, dirbc, velocity::TimedependentCondition;
        assem_U=SparseMatrixAssembler(U, V),
        nls::NonlinearSolver=NewtonSolver(LUSolver(); maxiter=10, rtol=1.e-8, verbose=true),
        xh::FEFunction=FEFunction(U, zero_free_values(U)))

        spaces = (U, V, Uold)
        x = get_free_dof_values(xh)
        x⁻ = zero_free_values(Uold)
        uh⁻ = FEFunction(Uold, x⁻)
        vuh = velocity()
        _res = res(0.0, uh⁻, vuh)
        _jac = jac(0.0, uh⁻, vuh)
        op = get_algebraic_operator(FEOperator(_res, _jac, U, V, assem_U))
        nls_cache = instantiate_caches(x, nls, op)
        caches = (nls, nls_cache, x, x⁻, assem_U, uh⁻)
        A, B, C = typeof(res), typeof(jac), typeof(spaces)
        D, E, F = typeof(dirbc), typeof(velocity), typeof(caches)
        return new{A,B,C,D,E,F}(res, jac, spaces, dirbc, velocity, caches)
    end
end

# Getters
get_state(m::DynamicNonlinearModel) = FEFunction(get_trial_space(m), m.caches[3])
get_measure(m::DynamicNonlinearModel) = m.res.dΩ
get_spaces(m::DynamicNonlinearModel) = m.spaces
get_assemblers(m::DynamicNonlinearModel) = (m.caches[4])



function update_velocity!(vh, x, x⁻, Δt)
    vh_ = get_free_dof_values(vh)
    vh_ .*= -1.0
    vh_ .-= (2.0 / Δt) .* x⁻
    vh_ .+= (2.0 / Δt) .* x
    return vh
end


function solve!(m::DynamicNonlinearModel;
    stepping=(nsteps=10, Δt=0.1),
    post=PostProcessor())

    reset!(post)
    U, V, U⁻ = m.spaces
    t = 0.0
    Δt = stepping[:Δt]
    nsteps = stepping[:nsteps]
    nls, nls_cache, x, x⁻, assem_U, uh⁻ = m.caches
    itime = 0
    KE = zeros(Float64, nsteps)
    EE = zeros(Float64, nsteps)

    uh = FEFunction(U, x)

    for itime in 1:nsteps
        t += Δt
        vuh = m.velocity()
        _res = m.res(t, uh⁻, vuh)
        _jac = m.jac(t, uh⁻, vuh)
        TrialFESpace!(U, m.dirichlet, t)
        TrialFESpace!(U⁻, m.dirichlet, t - Δt)

        # @show(    ∑( ∫(uh⁻[1]⋅uh⁻[1])m.res.dΩ  )  )
        # @show(    ∑( ∫(uh⁻[2]⋅uh⁻[2])m.res.dΩ   ) )


        op = get_algebraic_operator(FEOperator(_res, _jac, U, V, assem_U))
        Algebra.solve!(x, nls, op, nls_cache)

        # @show(    ∑( ∫(uu[1]⋅uu[1])m.res.dΩ  )  )
        # @show(    ∑( ∫(uu[2]⋅uu[2])m.res.dΩ   ) )
        # @show(    ∑( ∫(uh⁻[1]⋅uh⁻[1])m.res.dΩ  )  )
        # @show(    ∑( ∫(uh⁻[2]⋅uh⁻[2])m.res.dΩ   ) )

        update_velocity!(m.velocity.vh, x, x⁻, Δt)
        x⁻ .= x
        print_message(nls.log, "\nStep: $itime, Time: $t\n")
        post(t)


        # # Kinetic Energy
        #  KE[itime] = ∑(∫(0.5  * (m.velocity.vh ⋅ m.velocity.vh))m.res.dΩ )
        # # Elastic Energy
        #  EE[itime] = ∑(∫(Ψ ∘ (∇(uh[1])',∇(uh[2])'))m.res.dΩ )
        GC.gc()

    end
    vtk_save(post)

    return x
end




#*******************************************************************************	
#    					 StaticLinearModel
#*******************************************************************************	


struct StaticLinearModel{A,B,C,D,E} <: ComputationalModel
    res::A
    jac::B
    spaces::C
    dirichlet::D
    caches::E

    function StaticLinearModel(
        res::Function, jac::Function, U, V, dirbc;
        assem_U=SparseMatrixAssembler(U, V),
        ls::LinearSolver=LUSolver(),
        xh::FEFunction=FEFunction(U, zero_free_values(U)))

        spaces = (U, V)
        op = AffineFEOperator(jac, res, U, V, assem_U)
        K, b = get_matrix(op), get_vector(op)
        x = get_free_dof_values(xh)
        # x = allocate_in_domain(K)
        fill!(x, zero(eltype(x)))
        ns = numerical_setup(symbolic_setup(ls, K), K)
        caches = (ns, K, b, x, assem_U)


        A, B, C = typeof(res), typeof(jac), typeof(spaces)
        D, E = typeof(dirbc), typeof(caches)
        return new{A,B,C,D,E}(res, jac, spaces, dirbc, caches)
    end



    function StaticLinearModel(
        res::Vector{<:Function}, jac::Function, U0, V0, dirbc;
        assem_U0=SparseMatrixAssembler(U0, V0),
        ls::LinearSolver=LUSolver())

        nblocks = length(res)
        U, V = repeat_spaces(nblocks, U0, V0)
        spaces = (U, V)
        ##  cache
        K = assemble_matrix(jac, assem_U0, U0, V0) # 1D
        b = allocate_in_range(K)
        fill!(b, zero(eltype(b))) # 1D
        x = repeated_allocate_in_domain(nblocks, K)
        fill!(x, zero(eltype(x))) # nD
        ns = numerical_setup(symbolic_setup(ls, K), K) # 1D
        caches = (ns, K, b, x, assem_U0)

        A, B, C = Vector{<:Function}, typeof(jac), typeof(spaces)
        D, E = typeof(dirbc), typeof(caches)
        return new{A,B,C,D,E}(res, jac, spaces, dirbc, caches)
    end



    function StaticLinearModel(
        jac::Function, U, V, dirbc;
        assem_U=SparseMatrixAssembler(U, V),
        ls::LinearSolver=LUSolver(),
        xh::FEFunction=FEFunction(U, zero_free_values(U)))

        spaces = (U, V)
        K = assemble_matrix(jac, assem_U, U, V)
        b = allocate_in_range(K)
        fill!(b, zero(eltype(b)))
        x = get_free_dof_values(xh)
        # x = allocate_in_domain(K)
        fill!(x, zero(eltype(x)))
        ns = numerical_setup(symbolic_setup(ls, K), K) # 1D
        caches = (ns, K, b, x, assem_U)

        A, B, C = Nothing, typeof(jac), typeof(spaces)
        D, E = typeof(dirbc), typeof(caches)
        return new{A,B,C,D,E}(nothing, jac, spaces, dirbc, caches)
    end



    function (m::StaticLinearModel)(x::AbstractVector{Float64}; Assembly=false)
        U, V = m.spaces
        jac = m.jac
        ns, K, b, _, assem_U = m.caches
        b .= x
        if Assembly
            assemble_matrix!(jac, K, assem_U, U, V)
        end
        numerical_setup!(ns, K)
        Algebra.solve!(x, ns, b)
        return x
    end

    function (m::StaticLinearModel)(xh::SingleFieldFEFunction; Measure=nothing,  Assembly=false, kwargs ...)
        x_ = get_free_dof_values(xh)
        _, V = m.spaces
        jac = m.jac
        ns, K, b, _, assem_U = m.caches
        assemble_vector!((v)->∫(xh*v)Measure, b, assem_U, V)
        if Assembly
            assemble_matrix!(jac, K, assem_U, U, V)
        end
        numerical_setup!(ns, K)
        Algebra.solve!(x_, ns, b)
        return x_
    end

 

end


function solve!(m::StaticLinearModel; Assembly=true, post=PostProcessor())
    
    reset!(post)
    U, V = m.spaces
    jac = m.jac
    res = m.res
    ns, K, b, x, assem_U = m.caches
    if Assembly
        assemble_matrix_and_vector!(jac, res, K, b, assem_U, U, V)
    else
        assemble_vector!(res, b, assem_U, V)
    end
    numerical_setup!(ns, K)
    Algebra.solve!(x, ns, b)
    post(1.0)
    vtk_save(post)

    return x
end



function solve!(m::StaticLinearModel, b::Vector{Float64}; Assembly=true, post=PostProcessor())
   
    reset!(post)
    U, V = m.spaces
    jac = m.jac
    ns, K, _, x, assem_U = m.caches
    if Assembly
        assemble_matrix!(jac, K, assem_U, U, V)
    end
    numerical_setup!(ns, K)
    Algebra.solve!(x, ns, b)
    post(1.0)
    vtk_save(post)
    return x
end



function solve!(m::StaticLinearModel{Vector{<:Function},<:Any,<:Any,<:Any,<:Any}; Assembly=true, post=PostProcessor())
    reset!(post)
    U, V = m.spaces
    U0 = U[1]
    V0 = V[1]
    jac = m.jac
    res = m.res
    ns, K, b, x, assem_U0 = m.caches

    if Assembly
        assemble_matrix!(jac, K, assem_U0, U0, V0)
    end

    numerical_setup!(ns, K)

    map(blocks(x), res) do xi, li
        assemble_vector!(li, b, assem_U0, V0)
        Algebra.solve!(xi, ns, b)
    end
    post(1.0)
    vtk_save(post)
    return x
end

# Getters
get_state(m::StaticLinearModel) = FEFunction(get_trial_space(m), m.caches[4])
get_measure(m::StaticLinearModel) = m.biform.dΩ
get_spaces(m::StaticLinearModel) = m.spaces
get_assemblers(m::StaticLinearModel) = (m.caches[5], m.plb_caches[2], m.adj_caches[4])


