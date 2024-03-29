using DrWatson
@quickactivate "Swarm" 


window_size = 60

using Flux, CSV, DataFrames, MLDataPattern, StatsBase, LegolasFlux, InterProcessCommunication, DataStructures, MultivariateStats,JLD, Sockets, OpenSoundControl

function load_model()
    encoder = Chain(
        # 60x900xb
        Conv((9,), 900 => 9000, relu; pad = SamePad()),
        MaxPool((2,)),
        # 30x9000xb
        Conv((5,), 9000 => 4500, relu; pad = SamePad()),
        MaxPool((2,)),
        # 15x4500xb
        Conv((5,), 4500 => 2250, relu; pad = SamePad()),
        # 15x2250xb
        MaxPool((3,)),
        Conv((3,), 2250 => 1000, relu; pad = SamePad()),
        Conv((3,), 1000 => 100, relu; pad = SamePad()),
        # 5x100xb
        Flux.flatten,
        Dense(500, 100)
    )
    decoder = Chain(
        Dense(100, 500),
        (x -> reshape(x, 5, 100, :)),
        # 5x100xb
        ConvTranspose((3,), 100 => 1000, relu; pad = SamePad()),
        ConvTranspose((3,), 1000 => 2250, relu; pad = SamePad()),
        Upsample((3,)),
        # 15x2250xb
        ConvTranspose((5,), 2250 => 4500, relu; pad = SamePad()),
        Upsample((2,)),
        # 30x4500xb
        ConvTranspose((5,), 4500 => 9000, relu; pad = SamePad()),
        Upsample((2,)),
        # 60x9000xb
        ConvTranspose((9,), 9000 => 900, relu; pad = SamePad()),
        # 60x900xb
    )
    model = Chain(encoder, decoder)
    model_row = read_model_row(projectdir("models","1d_300_model-116-8.0e-5.arrow"))
    load_weights!(model, model_row.weights)
    return model
end

# Open the array loaded in shared memory with the given id
function open_shared_data(id)
    fd = IPC._shm_open(id,IPC.O_RDONLY, IPC.S_IRUSR |IPC.S_IWUSR)
    if fd < 0
        error("Couldn't create fd for shared_data: $fd")
    end
    ptr = IPC._mmap(IPC.C_NULL, 3600, IPC.PROT_READ, IPC.MAP_SHARED, fd,0)
    m = SharedMemory{String}(ptr,3600,false,id)
    WrappedArray(m, Float32, (900))
end

# Open the mutex loaded in shared memory with the given id
function open_mutex(id)
    fd = IPC._shm_open(id,IPC.O_RDWR, IPC.S_IRWXO)
    if fd < 0
        error("Couldn't create fd for mutex: $fd")
    end
    ptr = IPC._mmap(IPC.C_NULL, 64, IPC.PROT_WRITE|IPC.PROT_READ, IPC.MAP_SHARED, fd,0)
    m = SharedMemory{String}(ptr,64,false,id)
    IPC.Mutex(m,0;shared=true)
end

function get_new_data(shared_array)
    copied_array = Array{Float32}(undef, 300,3)
    lock(mutex)
    copied_array = copy(shared_array)
    unlock(mutex)
    return copied_array
end

function load_new_data!(shared_array)
    data = get_new_data(shared_array)
    d = normalise(data)
    push!(buffer,d)
end

function normalise(M) 
    min = minimum(minimum(eachcol(M)))
    max = maximum(maximum(eachcol(M)))
    return (M .- min) ./ (max - min)
end

function load_pca_model()
    pca_data = load(projectdir("models", "PCA_data_10_dims.jld"))
    PCA{Float32}(pca_data["mean"],pca_data["proj"],pca_data["prinvars"],pca_data["tprinvar"],pca_data["tvar"])
end


model = load_model()
encoder = model[1] |> gpu
decoder = model[2] |> gpu
shared_array = open_shared_data("/position_data")
mutex = open_mutex("/mutex_lock")
buffer = CircularBuffer{Array{Float32}}(window_size)   # A circular buffer of the last $window_size frames
saved_model = load_pca_model()

#main loop - try to sync to 30fps like the simulation
sock1 = UDPSocket()
datas = Matrix{Float32}[]

function do_stuff(timer::Timer)
    load_new_data!(shared_array)
    if isfull(buffer)
        # create an encoding of the sliding window 
        i = Flux.unsqueeze(reshape(vcat(buffer...), (60,900)), 3) |> gpu
        encoding = encoder(i)
        loss = Flux.Losses.mse(decoder(encoding), i)
        # println(loss)
        # PCA that shit
        pca_data = MultivariateStats.transform(saved_model, encoding |> cpu)
        msg1 = OpenSoundControl.message("/flock/pca", "ffffffffff", pca_data...)
        push!(datas,pca_data)
        send(sock1,ip"127.0.0.1", 12345, msg1.data)
        send(sock1,ip"10.10.10.10", 12345, msg1.data)
    end
end

timer = Timer(do_stuff, 0, interval=1.0/30.0)




#close(timer)
#z = hcat(datas...)
#
#for i in 1:10
#    @show i
#    println(maximum(z[i,:]))
#    println(minimum(z[i,:]))
#end
