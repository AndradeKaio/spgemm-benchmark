using BenchmarkTools
using TensorMarket
using Finch
#using Galley


function spgemm_gustavson(A, B)
    z = fill_value(A) * fill_value(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w = Tensor(SparseByteMap(Element(z)))
    f_time = @elapsed @finch begin
        C .= 0
        for j in _
            w .= 0
            for k in _, i in _
                w[i] += A[i, k] * B[k, j]
            end
            for i in _
                C[i, j] = w[i]
            end
        end
    end
    return f_time, C
end


function main()
  A_name = ARGS[1]
  B_name = ARGS[2]
  opt = ARGS[3]
  

  A_COO = fread(A_name)
  B_COO = fread(B_name)

  A = Tensor(Dense(SparseList(Element(0.0))), A_COO)
  B = Tensor(Dense(SparseList(Element(0.0))), B_COO)

  N = A.lvl.shape[1]

  f_time = 0
  if opt == "0"
    f_time, C = spgemm_gustavson(A, B)
    #ftnswrite("result.tns", C)
  end
  
  println("$N,$N,$f_time,$A_name,$B_name")

end

main()

