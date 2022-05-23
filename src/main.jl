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
    model_row = read_model_row(datadir("1d_300_model-112-8.0e-6.arrow"))
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

encoder = load_model()[1]
shared_array = open_shared_data("/position_data")
mutex = open_mutex("/mutex_lock")
buffer = CircularBuffer{Array{Float32}}(window_size)   # A circular buffer of the last $window_size frames


saved_model = load_pca_model()

#main loop - try to sync to 30fps like the simulation

function normalise(M) 
    min = minimum(minimum(eachcol(M)))
    max = maximum(maximum(eachcol(M)))
    return (M .- min) ./ (max - min)
end

sock1 = UDPSocket()

function do_stuff(timer::Timer)
    load_new_data!(shared_array)
    if isfull(buffer)
        # create an encoding of the sliding window 
        i = Flux.unsqueeze(reshape(vcat(buffer...), (60,900)), 3)
        encoding = encoder(i)
        # PCA that shit
        pca_data = MultivariateStats.transform(saved_model, encoding)
        msg1 = OpenSoundControl.message("/flock", "ffffff", pca_data...)
        println(pca_data)
        send(sock1,ip"127.0.0.1", 8000, msg1.data)
    end
end

for i in 1:60
    load_new_data!(shared_array)
end

t = Timer(do_stuff, 0, interval=0.01)

close(t)



do_stuff(Timer(1.0))

x = buffer.buffer

i = Flux.unsqueeze(reshape(vcat(x...), (60,900)),3)

@time encoder(i)