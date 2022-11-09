using JLD
using ProgressMeter
using Random

#stores a state with a default name in the data folder. Add an id to change the tail of the namestring
function rcmps_store(state::State;verbose=true, id="_")
    if id =="_"
        name_tail = randstring(10)
    else
        name_tail = id
    end
    name = string(state.model_param.name,"_g",state.model_param.coupling,"_D",state.D,id)
    filename=string("./data/",name,".jld")
    save(filename,"R",state.R,
        "K",state.K,"model",state.model_param.name,
        "coupling",state.model_param.coupling,
        "e0",state.e0,"grad",state.grad,
        "info",state.info,
        "D",state.D,"ρ0",state.ρ0)
    if verbose
        @info string("Rcmps stored in file ",filename)
    end
end

function rcmps_load(name)
    filename=string("./data/",name,".jld")
    state_dict=load(filename)
    R=state_dict["R"]
    K=state_dict["K"]
    state = State(R,K,ρ0=state_dict["ρ0"])
    state.e0=state_dict["e0"]
    state.grad=state_dict["grad"]
    state.model_param.coupling=state_dict["coupling"]
    state.model_param.name=state_dict["model"]
    state.info=state_dict["info"]
    state.D=state_dict["D"]
    return state
end

function update_stored_states()
    file_list=get_file_names()
    @showprogress for file in file_list
          filename=chop(file,tail=4)
          state=rcmps_load(filename)
          rcmps_store(state,filename)
    end
end

function get_file_names()
    files_or_dir=readdir("data")
    file_list=[]
    for file in files_or_dir
          if isdir(string("./data/",file))==false
                push!(file_list,file)
          end
    end
    return file_list
end

function get_states()
    file_list=get_file_names()
    states=[]
    for file in file_list
          filename=chop(file,tail=4)
          state=rcmps_load(filename)
          push!(states,state)
    end
    return states
end

function select_best(states)
    dict=Dict()
    for state in states
          D=state.D
          coupling_int=floor(Int64,state.model_param.coupling*1000)
          model=state.model_param.name
          key=string(model,"D",D,"g",coupling_int)
          if haskey(dict,key)
                e_best=dict[key].e0
                e0=state.e0
                if e0<e_best
                      dict[key]=state
                end
          else
                dict[key]=state
          end
    end
    return collect(values(dict))
end

function get_exact_state(states,model,coupling,D)
    states=select_best(states)
    result="empty"
    best_e0=1e10
    for state in states
        new_e0=state.e0
        if (state.model_param.name==model) & (abs(coupling-state.model_param.coupling)<1e-7) & (state.D==D) & (new_e0 ≤ best_e0)
            result=state
            best_e0=new_e0
        end
    end
    return result
end