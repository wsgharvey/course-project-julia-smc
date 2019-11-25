using MPI

function collect(item, comm)
    """
    given item by each proc, collects returns them all to proc 0.
    concatenates along first dimension
    """
    rank = MPI.Comm_rank(comm)
    n_proc = MPI.Comm_size(comm)
    per_proc = size(item, 1)
    if rank != 0
        MPI.Send(item, 0, rank+32, comm)
        return 0
    end
    s = [size(item)...]
    s[1] = s[1] * n_proc
    all_received = Array{Float64}(undef, (s...))
    item_received = Array{Float64}(undef, size(item))
    all_received[1:per_proc] = item
    for worker in 1:(n_proc-1)
        rreq = MPI.Irecv!(item_received, worker, worker+32, comm)  # awful
        MPI.Waitall!([rreq])
        all_received[1+(worker*per_proc):(worker+1)*per_proc] = item_received
    end
    return all_received
end
