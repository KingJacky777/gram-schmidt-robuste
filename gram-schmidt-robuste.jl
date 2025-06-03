using LinearAlgebra

function gramschmidtrobust(A::Matrix)
    m, n = size(A)
    Q = Matrix{eltype(A)}(undef, m, 0)

    for i in 1:n
        v = A[:, i]
        for j in 1:size(Q, 2)
            v -= dot(Q[:, j], v) * Q[:, j]
        end

        norm_v = norm(v)
        if norm_v != 0
            q = v / norm_v
            Q = hcat(Q, q)
        end
    end

    return Q
end
