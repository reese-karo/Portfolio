"""Construct examples of Cahn-Hilliard equations."""

#%% Package imports
#=== import PyTorch and set it to accurate data type
import torch as tc
#-== check the main device (gpu or cpu)
dev = 'cuda:0' if tc.cuda.is_available() else 'cpu'
print('Device for computations : {}'.format(dev), "(set by CHPoser.py)")
#-== set double precision as default (otherwise Pytorch uses single)
tc.set_default_dtype(tc.float64) 
print('Torch default data type: float64, complex128', "(set by CHPoser.py)")

#%% Global constants and functions
num_par_default = { 
    'NN': 2**3, # resolution of spatial discretization
    'time_dis': "implicit Euler", # time discretization scheme
    'KK': 1, # number of time steps
    'beta' : 1., # step size for outer loop
    'N1' : 1000, # maximum number of inner iterations
    'N2' : 1000, # maximum number of outer iterations
    'lamb' : 1., # preconditioner paramter lambda 
    'gamma' : 0., # preconditioner parameter gamma
    'TOL_out' : 1e-6, # tolerance of outer loop
    'TOL_in' : 1e-6, # tolerance of inner loop
    'dev': dev # computing device ('cpu' or 'cuda:0')
}

pde_par_default = { 
    'eps': 1e-2, # epsilon (thickness of interfaces)
    'TT': 1.,
    'LL' : 1., # width of the domain (currently, only square domain)
}


#--- function for exponential blob at (x0, y0)
fn_blob = lambda xx, yy, x0, y0, LL: tc.exp( tc.cos( (2*tc.pi/LL) * (xx-x0)) \
        + tc.cos( (2*tc.pi/LL) * (yy-y0) ) )

#--- function for regularized mobility
fn_reg_quad_mob = lambda uu, del0: tc.sqrt((1-uu*uu)*(1-uu*uu) + del0*del0) 

#%% Definition of main functions
def pose_ch(pde_par: dict, num_par: dict) -> tuple:
    """Return complete settings for fully discrete CH eqn.
    """
    #=== Generate spatial grid domain
    xx, yy, hh, dv = gen_grid_dom(num_par, pde_par)

    #--= set up time marching grid domain
    tt, dt = gen_time_grid(num_par, pde_par)

    ch_eqn_data = {}

    #=== generate array data
    #-== construct initial condition
    ch_eqn_data["ic"] = gen_ic(pde_par, xx, yy) # initial condition grid function 

    #--= update default values
    tmp = {key: value for key, value in num_par_default.items() if key not in num_par}
    num_par.update(tmp)

    tmp2 = {key: value for key, value in pde_par_default.items() if key not in pde_par}
    pde_par.update(tmp2)

    #=== pack the output
    num_par['xx'] = xx
    num_par['yy'] = yy

    ch_eqn_data['hh'] = hh # spatial grid spacing
    ch_eqn_data['dv'] = dv # volume element
    ch_eqn_data['tt'] = tt # time grid
    ch_eqn_data['dt'] = dt

    tmp = {key: value for key, value in num_par.items() if key not in ch_eqn_data}
    ch_eqn_data.update(tmp)

    tmp = {key: value for key, value in pde_par.items() if key not in ch_eqn_data}
    ch_eqn_data.update(tmp)
    
    return ch_eqn_data

def trim_log (ch_log: dict):
    """Trim nan's and reorganize computation log data.
    """
    #=== Trim data of the shape (KK, N1)
    # iter_out_max = tc.max(arr, axix=1)
    pass

def gen_grid_dom(num_par: dict, pde_par: dict) -> tuple:
    """Return grid domain (xx, yy), grid spacing (hh), and volume element (dv).
    
    This is an extension of `tc.meshgrid`

    Input
    - num_par: numerical parameters of CH eqn.
    - pde_par: PDE parameters of CH eqn.
    OUTPUT
    - xx, yy, hh, dv 
    """
    LL = pde_par['LL']
    NN = num_par['NN']
    xx = tc.linspace(0, LL * ((NN-1)/NN) , NN, device=dev) 
    yy = tc.linspace(0, LL * ((NN-1)/NN) , NN, device=dev) 
    xx, yy = tc.meshgrid(xx, yy, indexing='ij')

    hh = LL/NN
    dv = hh*hh

    return xx, yy, hh, dv

def gen_time_grid(num_par: dict, pde_par: dict) -> tc.Tensor:
    """Return time grid domain.
    
    This extends 1D grid generator. 

    Input
    - num_par: numerical parameters of CH eqn.
    - pde_par: PDE parameters of CH eqn.
    OUTPUT
    - time grid
    """
    try:
        start = pde_par['init time']
    except KeyError:
        start = 0.
    
    end = pde_par['TT']
    Nnodes = num_par['KK'] + 1
    dt = (end - start) / num_par['KK']

    return tc.linspace(start, end, Nnodes, device=dev), dt


def noisy_gfn(size: tuple, avg_height: float, fluc: float, seed=None) -> tc.Tensor:
    """Return a grid function with random fluctuation.

    INPUT
    - size: size of the uniform grid domain (size of array)
    - avg_height: average height of the fluctuation
    - fluc: amplification of random fluctuation (i.e. absolute value of maximum fluctuation)

    OUTPUT
    - a grid function of
    """
    if seed:
        tc.manual_seed(seed)

    return avg_height + (2*fluc)*(tc.rand(size, device=dev) - 0.5)

def gen_blob_gfn(xx: tc.Tensor, yy: tc.Tensor, x0:float, y0:float, LL:float) -> tc.Tensor:
    """return a grid function of an exponential blob at (x0, y0).

    The resulting function is periodic smooth (i.e., derivatives of any order are all periodic).

    INPUT
    - xx, yy: grid domain (usually obtain from torch.meshgrid)
    - x0, y0: the point of blob's peak. 
    - LL: length of the domain

    OUPUT
    - a grid function of an exponential blob at (x0, y0).
    """
    return tc.exp( tc.cos( (2*tc.pi/LL) * (xx-x0)) \
        + tc.cos( (2*tc.pi/LL) * (yy-y0) ) )

def gen_ic(pde_par: dict, xx: tc.Tensor, yy: tc.Tensor) -> tc.Tensor:
    """Return an example initial condition for CH eqn.
    
    If ic_example is:
    "random fluctuation": par must follow the format of `noisy_gfn`.
    "exponential blob": par must follow the format of `blob_gfn`.
    """
    ic_example = pde_par['ic_par']['type']

    if ic_example == "random fluctuation":
        avg_height, fluc = pde_par['ic_par']['avg_height'], pde_par['ic_par']['fluc']
        size = xx.size()
        try:
            seed = pde_par['ic_par']['seed']
        except:
            seed = None

        ic_gfn = noisy_gfn(size, avg_height, fluc, seed)
    
    elif ic_example == "exponential blob":
        #--- gather parameters: peak point of blob and domain length
        x0, y0, LL = pde_par['ic_par']['x0'], pde_par['ic_par']['y0'], pde_par['LL']
        
        par = {'xx': xx, 'yy': yy, 'x0': x0, 'y0': y0, 'LL': LL}
        
        #--- generate and normalize IC so that -1 <= IC <= 1
        ic_gfn = gen_blob_gfn(**par)
        ic_gfn = normalize_gfn(ic_gfn)
    
    if tc.max(tc.abs(ic_gfn)) > 1.:
        raise("I.C. Error: initial condition is not physical. It must be b/w -1 and 1")

    return ic_gfn

def normalize_gfn(gfn: tc.Tensor) -> tc.Tensor:
    """Return a normalized grid function.
    
    The result grid function satisfies -1 <= gfn <= 1.
    """

    #--- make mean zero
    gfn = gfn - tc.mean(gfn) 

    #--- make maximum in absolute value one
    gfn = gfn/tc.max(tc.abs(gfn)) 

    return gfn

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm

    LL = 1.
    NN = 256
    xx = tc.linspace(0, LL * ((NN-1)/NN) , NN, device=dev) # grid preparation
    yy = tc.linspace(0, LL * ((NN-1)/NN) , NN, device=dev) # grid preparation
    xx, yy = tc.meshgrid(xx, yy, indexing='ij') # grid preparation
    x0, y0 = -0.7, 0.3
    

    # par = {'xx': xx, 'yy': yy, 'x0': x0, 'y0': y0, 'LL': LL}
    # gfn = gen_ic("exponential blob", par)
    # fig, ax = plt.subplots(1, 1, figsize=(5,5), subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(xx.cpu(), yy.cpu(), gfn.cpu(), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    par = {'size': (NN, NN), 'avg_height': 1., 'fluc': 0.2}
    gfn = gen_ic("random fluctuation", par)
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.imshow(gfn.cpu(), cmap=cm.coolwarm)

    plt.show()
    print(gfn)
    print(tc.mean(gfn), tc.min(gfn))