"""
This module collects functions to solve Cahn-Hilliard equation with variable mobility supplemented with periodic boundary condition.
"""
#%% #=== package imports and PyTorch set up
#=== import PyTorch and set it to accurate data type
import torch as tc
"""OBSOLETE: This is done in CHPoser.py
#-== check the main device (gpu or cpu)
dev = 'cuda:0' if tc.cuda.is_available() else 'cpu'
print('Device for computations : {}'.format(dev), "(set by CHSolver.py)")
#-== set double precision as default (otherwise Pytorch uses single)
tc.set_default_dtype(tc.float64) 
print('Torch default data type: float64, complex128', "(set by CHSolver.py)")
"""
import matplotlib.pyplot as plt
from ppgd.ppgd_num_tools import ppgd, ini_fcm, lap, lapm
from CommonTools.fileio_tools import save, open_pickled_file


#%% #=== Global Constants
LOG_NO_ISSUE = {'res': 0., # fake residual for normal computations
                'iters': 0, # fake number of iterations for normal computations
                'abnormaly': 0, # 
                }
TO_LOG = ('res_out', 'iter_out', 'iter_tot', 'fft_count', 'time_consumed', 'iter_in', 'res_in', 'sz_in', 'egy_out')

#%% #=== Default parameters
FIXED_STCH_SETTINGS_KEYS = ('eps', 'N1', 'N2', 'beta','TOL_in', 'TOL_out', 'dv', 'dt')

#%%
def solve_ch(ch_eqn_data: dict, rep_par) -> tuple:
    """
    Solves the Cahn-Hilliard equation.
    """
    
    #=== unpack inputs
    KK = ch_eqn_data['KK']
    
    #=== some prelimiary tasks
    fig, ax = plt.subplots()
    #=== main routine



    #--= initialize data for the whole evolution
    ch_num_sol, ch_log, fixed_stCH_settings, num_sol_pre = init_ch_num_sol(ch_eqn_data, rep_par)

    #--= initialize the Fourier collocation method
    fcm_eigs2 = ini_fcm(ch_eqn_data['LL'], ch_eqn_data['NN']) # fcm_eigs2 is a collection of eigenvalues of differential operators

    #--= record initial state in the evolution.
    rec_evol(ch_eqn_data, ch_eqn_data['ic'], LOG_NO_ISSUE, num_sol_pre, ch_num_sol, ch_log, 0, rep_par, fig, ax) 

    #-== actually time evolution
    for kk in range(1, KK+1):
        #-== construct stationary CH equation based on time discretization
        stCH_data = get_stCH_data(ch_eqn_data, fixed_stCH_settings, num_sol_pre, kk)

        num_sol_cur, log_cur = solve_stCH(stCH_data, fcm_eigs2)
        
        #-== recording time evolution (done in place)
        rec_evol(ch_eqn_data, num_sol_cur, log_cur, num_sol_pre,ch_num_sol, ch_log, kk, rep_par, fig, ax) 

        #-== progress report
        print(f"  time marching {kk} made out of {KK}")

    #=== leave message
    print(f"Resolution: {ch_eqn_data['NN']}.")

    return ch_eqn_data, ch_num_sol, ch_log


def init_ch_num_sol(ch_eqn_data: dict, rep_par: dict) -> tuple:
    """Return an array to store the entire time evolution and computation log."""
    #=== unpack some data for readibility 
    KK = ch_eqn_data['KK'] # number of time grid points
    dev = ch_eqn_data['dev'] # device: cuda or cpu?
    NN = ch_eqn_data['NN']
    Nsaves = rep_par['Nsaves']

    #=== Construct fixed PPGD solver settings: copy from ch_eqn_data for the keys in FIXED_STCH_SETTINGS_KEYS
    fixed_stCH_settings = {key: value for key, value in ch_eqn_data.items() if key in FIXED_STCH_SETTINGS_KEYS}

    #=== Schedule data save
    schedule_save(rep_par, ch_eqn_data)

    #=== construct arrays for solution and list for log
    ch_log = []
    ch_num_sol = tc.zeros((Nsaves + 1, NN, NN), device=dev)
    num_sol_pre = [tc.zeros((NN, NN), device=dev), tc.zeros((NN, NN), device=dev)] # previous two solutions for implicit Euler time marching

    return ch_num_sol, ch_log, fixed_stCH_settings, num_sol_pre

def schedule_save(rep_par: dict, ch_eqn_data: dict) -> None:
    """Update data save schedule item in rep_par dict.
    """
    #--- unpack varilables for readibility
    KK = ch_eqn_data['KK']
    Nsaves = rep_par['Nsaves']
    dt = ch_eqn_data['dt']

    if rep_par['schedule_scheme'] == "periodic":
        rep_par['save_schedule'] = tc.linspace(0, KK, Nsaves + 1).int()
        rep_par['save_times'] = tc.linspace(0, KK, Nsaves + 1).int() * dt
        rep_par['last_save_ind'] = -1

def get_stCH_data(ch_eqn_data: dict, fixed_stCH_settings: dict, num_sol_pre: tc.Tensor, kk: int) -> dict:
    """Construct and return PPGD input data that is used for a stationary CH eqn."""
    #=== unpack some data for readibility
    dt = ch_eqn_data['dt']
    eps = ch_eqn_data['eps']

    #=== Construct grid functions for stationary CH eqn (See ppgd module for details)
    uu_s = num_sol_pre[1] # previous time iterate
    
    ff = 0. # forcing term ff does not appear in time dependent CH eqn.

    #--- work around for pickle to store ch_eqn_data: it cannot store lambda function. So, mobility is a string and use `eval` function when implement it.
    fn_mob = eval(ch_eqn_data['fn_mob'])
    MM = fn_mob(uu_s) # mobility grid function 
    
    #-== initial guess of PPGD method: use linear extrapolation using the last two iterates
    if kk == 1:
        uu0 = num_sol_pre[1] 
    elif kk > 1:
        uu0 = 2*num_sol_pre[1] - num_sol_pre[0]

    #=== Pack stationary CH data 
    stCH_data = {'ff' : ff, 
               'uu0' : uu0, 
               'uu_s' : uu_s, 
               'MM' : MM, 
               } 
    
    #-== set preconditioner
    set_precond(stCH_data, ch_eqn_data)
    
    #-== update fixed settings from ch_eqn_data
    stCH_data.update(fixed_stCH_settings)

    return stCH_data

def set_precond(stCH_data: dict, ch_eqn_data: dict):
    """Set preconditioner parameter."""
    #=== unpack some data for readibility
    dt = ch_eqn_data['dt']
    eps = ch_eqn_data['eps']

    #=== set preconditioner coefficients
    stCH_data['lamb'] = 1./dt # -2nd order
    stCH_data['gamma'] = 0. # 0th order
    stCH_data['eta'] = eps*eps # 2nd order

def solve_stCH(stCH_data: dict, fcm_eigs2: dict) -> dict:
    """Return numerical solution and computation log at the current time step."""
    # This work around is to avoid generating fcm_eigs2 over and over again
    stCH_data['fcm_eigs2'] = fcm_eigs2 

    #=== call PPGD solver for stationary CH eqn
    ppgd_out = ppgd(ppgd_in=stCH_data)

    #=== extract log and solution data  
    num_sol_cur = ppgd_out['num_sol']
    log_cur = {key: ppgd_out[key] for key in TO_LOG}
 
    return num_sol_cur, log_cur

def rec_evol(ch_eqn_data: dict, num_sol_cur: tc.Tensor, log_cur: dict, num_sol_pre: tc.Tensor, ch_num_sol: tc.Tensor, ch_log: list, kk: int, rep_par: dict, fig, ax) -> None:
    """Record numerical solution in the evolution array.
    
    Recording is in place by the way Python works. (No assignment is needed like `ch_num_sol = rec_evol(....)`)

    """
    #--- update recent time marching history (This must be done even when nothing is stored for proper time marching.)
    num_sol_pre[0] = num_sol_pre[1]
    num_sol_pre[1] = num_sol_cur
    
    if save_bool(kk, rep_par) == False:
        return

    t_ind = rep_par['last_save_ind'] + 1

    ch_num_sol[t_ind, :, :] = num_sol_cur
    update_log(log_cur, ch_log, kk)    
    data = {'ch_log': ch_log,
            'ch_num_sol': ch_num_sol,
            'ch_eqn_data': ch_eqn_data,
            'rep_par': rep_par
            }
    #--- overwrite the file with the updated data
    save(data, rep_par['filename'], overwrite=True)

    rep_par['last_save_ind'] = t_ind

    #--- plot current time
    plot_cur_state(fig, ax, ch_eqn_data, num_sol_cur, kk)

def compute_rhs(gfns: tc.Tensor, ch_eqn_data: dict):
    r"""
    Return right hand side term of CH eqn.

    See the #discretization of ReadMe.md

    INPUT
        gfns: A collection of two or more grid functions (N-by-N). Consecutive grid functions play the roles of u^k and u^{k+1}
        ch_eqn_data: The main dict data of information on the CH equation considered.
    OUTPUT
        rhss: A collection of right hand side functions. This collection has one less grid functions than the input gfns because two consecutive grid functions yield one right hand side function.
    """
    pass

def plot_cur_state(fig, ax, ch_eqn_data, num_sol_cur, kk):
    """Plot current state of the evolution."""
    #=== unpack some data for readibility
    KK = ch_eqn_data['KK']
    dt = ch_eqn_data['dt']
    TT = ch_eqn_data['TT']
    LL = ch_eqn_data['LL']
    Nsaves = 20
    jmp = int(KK/Nsaves) 
    
    #=== From the second drawing, use the AxesImage (return of imshow) from the first drawing
    try:
        img = ch_eqn_data['img']
    except:
        pass

    if num_sol_cur.device != 'cpu': num_sol_cp = num_sol_cur.cpu()
    
    if kk == 0:
        img = ax.imshow(num_sol_cp, cmap='jet', interpolation='spline36', extent=(0, LL, 0, LL))
        fig.colorbar(img)
    else:
        img.set_data(num_sol_cp)
        img.set_clim(tc.min(num_sol_cp), tc.max(num_sol_cp))    

    ax.set_title(f"t = {kk*dt:.2f} (T = {TT})")
    plt.draw()
    plt.pause(0.001)

    ch_eqn_data['img'] = img

def plot_still_shot(fig, ax, ch_eqn_data, num_sol_cur, kk):
    """Plot a certain state of the evolution.
    
    This is similar to `plot_cur_state`. But the latter only updates the data after plotting the initial condition (hence not appropriate for plotting from the saved data) while the former redraw the phase variable.
    """
    #=== unpack some data for readibility
    KK = ch_eqn_data['KK']
    dt = ch_eqn_data['dt']
    TT = ch_eqn_data['TT']
    LL = ch_eqn_data['LL']
    Nsaves = 20
    jmp = int(KK/Nsaves) 
    
    #=== drawing
    if num_sol_cur.device != 'cpu': num_sol_cp = num_sol_cur.cpu()
    
    img = ax.imshow(num_sol_cp, cmap='jet', interpolation='spline36', extent=(0, LL, 0, LL))
    fig.colorbar(img)
    ax.set_title(f"t = {kk*dt:.2f} (T = {TT})")
    plt.draw()
    plt.pause(0.001)


def update_log(log_cur: dict, ch_log: list, kk: int) -> None:
    """Update current computation log to evolution log.
    """
    if kk == 0:
        return

    ch_log.append(log_cur)


def save_bool(kk: int, rep_par: dict) -> bool:
    """Return a boolean of whether to save data."""
    return (kk in rep_par['save_schedule']) and rep_par['save_data']


def embedded_arr(small_arr: tc.Tensor, big_arr:tc.Tensor, padding_const=tc.nan) -> tc.Tensor:
    """Return an array that embed a small array to a big array when their sizes are not compatible by torch.Tensor.expand.

    This is a helper function (i.e., special case) of torch.nn.functional.pad, which features more various implementations.
    
    For example, the small array (top) is embeded to the big array (bottom)
    (small_arr)
    tensor([[1., 1., 1., 1.], 
            [1., 1., 1., 1.]])
    (big_arr)
    tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])
    (embedded array)
    tensor([[1., 1., 1., 1., nan],
            [1., 1., 1., 1., nan],
            [nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, nan]])
    
    `embed_arr` function do not pad beginning of the each axis, hence the even-indexed entries of the tuple `pad` are all 0. Odd-indexed entries of `pad` are the diferrence between the big and small array for each axis.
    Following the convention of torch.nn.functional.pad, the tuple `pad` list the number of padding in reverse order. For example, the above example is a consequence of pad = (0, 1, 0, 3). This means, along the last axis, no padding at the beginning and 1 padding at the end will be made. And along the second most axis, no padding at the beginning and 3 padding at the end will be made. 
    """
    ndim = small_arr.ndim

    if ndim != big_arr.ndim:
        IndexError("Array Embedding Error: two arrays must have the same ndim.")
    
    #--- compute the depth to pad for each axis (in the reverse order following the convention of torch.nn.functional.pad)
    pad = [big_arr.size()[i] - small_arr.size()[i] for i in range(ndim-1, -1, -1)]
    #--- insert 0 in between so that no padding is done at the beginning of each axis.
    for i in range(ndim-1, -1, -1):
        pad.insert(i, 0)

    return tc.nn.functional.pad(small_arr, tuple(pad), 'constant', tc.nan)

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    #=== set parameters
