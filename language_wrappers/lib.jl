module NMFMergeLib

using NMFMerge

struct ReturnValue
    niters::Int32
    converged::Bool
    objvalue::Float64
end

struct CMatrix{T}
    data::Ptr{T}
    rows::Int32
    cols::Int32
end

Base.@ccallable function nmfmerge_inplace(Wout::CMatrix{Float64}, Hout::CMatrix{Float64}, X::CMatrix{Float64}, ncomponents::Int32, tol::Float64, maxiter::Int32)::ReturnValue
    Wout.rows == X.rows || throw(ArgumentError("Wout and X must have the same number of rows"))
    Hout.cols == X.cols || throw(ArgumentError("Hout and X must have the same number of columns"))
    W, H, X = unsafe_wrap(Array, Wout.data, (Wout.rows, Wout.cols)),
              unsafe_wrap(Array, Hout.data, (Hout.rows, Hout.cols)),
              unsafe_wrap(Array, X.data, (X.rows, X.cols))
    result = nmfmerge(X, ncomponents; tol_final=tol, maxiter)
    W .= result.W
    H .= result.H
    return ReturnValue(result.niters, result.converged, result.objvalue)
end

end
