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
    all_received[1:per_proc, :] = item
    for worker in 1:(n_proc-1)
        rreq = MPI.Irecv!(item_received, worker, worker+32, comm)  # awful
        MPI.Waitall!([rreq])
        all_received[1+(worker*per_proc):(worker+1)*per_proc, :] = item_received
    end
    return all_received
end

function collate(items, indices)
    s = [size(items)...]
    s[1] = size(indices, 1)
    collation = Array{Float64}(undef, (s...))
    for (i, index) in enumerate(indices)
        collation[i, :] = items[index, :]
    end
    return collation
end

function share_out(items, allocate_to, per_proc_shape, comm)
    """
    rank 0 shares out items between everyone
    - `items` can be None for all but rank 0
    - `allocate_to` is list of item indexes for each process
    """
    rank = MPI.Comm_rank(comm)
    n_proc = MPI.Comm_size(comm)
    n_items = size(allocate_to, 1)
    received = Array{Float64}(undef, per_proc_shape)
    if rank != 0
        rreq = MPI.Irecv!(received, 0, rank+32, comm)
        MPI.Waitall!([rreq])
        return received
    end
    per_proc = per_proc_shape[1]
    own_stuff = collate(items, allocate_to[1:per_proc])
    for worker in 1:(n_proc-1)
        to_send = collate(items, allocate_to[1+(worker*per_proc):(1+worker)*per_proc])
        MPI.Send(to_send, worker, worker+32, comm)
    end
    return own_stuff
end

function send_bool(mesg, comm)
    """
    send a bool from rank 0 to all others
    """
    rank = MPI.Comm_rank(comm)
    n_proc = MPI.Comm_size(comm)
    mesg_array = Array{Bool}(undef, 1)
    if rank != 0
        rreq = MPI.Irecv!(mesg_array, 0, rank+32, comm)
        MPI.Waitall!([rreq])
        return mesg_array[1]
    end
    mesg_array[1] = mesg
    for worker in 1:(n_proc-1)
        MPI.Send(mesg_array, worker, worker+32, comm)
    end
    return mesg
end
