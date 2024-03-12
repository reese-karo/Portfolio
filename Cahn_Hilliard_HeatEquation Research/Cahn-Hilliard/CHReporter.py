"""This file collects reporting tools of CH eqn."""

#%% package/module imports
import torch as tc
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from cycler import cycler
import pickle
from CommonTools import fileio_tools as fio
import warnings
  
#%% Constants and matplotlib global set up
from CommonTools.vis_tools import  CONST_rcParams, colormarker10_cycler
TIME_LOG_KEYS = ('iter_out', 'iter_in', 'res_out', 'egy_out')
YLABELS = {'iter_out': "Outer iterations taken",
            'iter_in': "Inner iterations taken (average)",
            'res_out': "Outer residual (last)",
            'egy_out': "Free energy",
            }

plt.rc('font', family='serif')
plt.rcParams.update(CONST_rcParams)

#%% Main functions
def report(ch_eqn_data: dict, ch_num_sol: tc.Tensor, ch_log: list, rep_par: dict) -> None:
    """Console report, plot, and save computation result."""

    #=== console report
    console_report(ch_eqn_data, ch_num_sol, ch_log, rep_par)

    #=== plot
    #-== initial and final states plot
    plot_init_final(ch_eqn_data, ch_num_sol, ch_log, rep_par)

    #-== computation log plot
    plot_log(ch_eqn_data, ch_num_sol, ch_log, rep_par)

def plot_max_mean(ch_eqn_data, ch_num_sol, ch_log, rep_par):
    """Plot maximum phase variable (in absolute value) and mean value.
    
    OUTPUT
        figure object of the plot.    
    """
    # unpack some variables from input: number of times when still shots of evolution are saved 
    KK_saved = rep_par['last_save_ind']

    #--- set axes parameter (title)
    axes_par = {'title': "Features of numerical solution"}

    #=== Maximum 
    #--- Construct an array of maximum phase variable along times of evolution (--> yaxis on the left)
    yys = [tc.max(tc.abs(ch_num_sol[i])) for i in range(KK_saved)]
    yys = tc.tensor(yys)
    
    #--- Create figure and set some axes parameters
    fig, ax = plt.subplots(1,1, layout='constrained')
    axes_par.update({'ylabel': "Maximum phase variable"})
    plot_seq(ax, yys, axes_par=axes_par)


    #=== Mean value (practically 'mass')
    #--- Construct an array of mean values along times of evolution (--> yaxis on the right)
    yys = [tc.mean(ch_num_sol[i]) for i in range(KK_saved)]
    yys = tc.tensor(yys)

    #--- set some axes parameters and plot
    axes_par = {'ylabel': "Mass"}
    color = "r"
    Line2D_par = {'marker': "s", 'color': color}
    tick_par = {'labelcolor': color}
    ax2 = ax.twinx()
    plot_seq(ax2, yys, axes_par=axes_par, Line2D_par=Line2D_par, tick_par=tick_par)

    return fig


def plot_log(ch_eqn_data, ch_num_sol, ch_log, rep_par):
    """Plot computation log of CH equation."""
    KK_saved = rep_par['last_save_ind']
    fig_list = []
    # axes_par = {'title': "Features of numerical solution"}

    # #--- Maximum phase variable
    # yys = [tc.max(tc.abs(ch_num_sol[i])) for i in range(KK_saved)]
    # yys = tc.tensor(yys)
    
    # fig, ax = plt.subplots(1,1, layout='constrained')
    # axes_par.update({'ylabel': "Maximum phase variable"})
    # plot_seq(ax, yys, axes_par=axes_par)

    # #--- Mass
    # yys = [tc.mean(ch_num_sol[i]) for i in range(KK_saved)]
    # yys = tc.tensor(yys)

    # axes_par = {'ylabel': "Mass"}
    # color = "r"
    # Line2D_par = {'marker': "s", 'color': color}
    # tick_par = {'labelcolor': color}
    # ax2 = ax.twinx()
    # plot_seq(ax2, yys, axes_par=axes_par, Line2D_par=Line2D_par, tick_par=tick_par)
    fig = plot_max_mean(ch_eqn_data, ch_num_sol, ch_log, rep_par)
    
    fig_list.append(fig)

    plt.show()

    #--- Outer iterations, average inner iterations, free energy, last residuals (outer)
    for key in TIME_LOG_KEYS:
        if 'average' in YLABELS[key]:
            yys = [tc.nanmean((ch_log[i
            ][key].float())) for i in range(KK_saved)]
        elif 'last' in YLABELS[key]:
            yys = [ch_log[i][key][-1] for i in range(KK_saved)]
        else:
            yys = [ch_log[i][key] for i in range(KK_saved)]
        yys = tc.tensor(yys)

        axes_par = {'title': "Log data along evolution",
                    'ylabel': YLABELS[key],
                    }
        Line2D_par = {'marker': "o", 'color': "b"}
        fig, ax = plt.subplots(1,1, layout='constrained')
        plot_seq(ax, yys, axes_par=axes_par, Line2D_par=Line2D_par)
        
        fig_list.append(fig)

        plt.show()    

    return fig_list

def report_saved_data(filename: str):
    """Report saved data."""
    if not filename.endswith('.pkl'):
        warnings.warn("file name does not seem to be made by pickle: report of saved data may not work.")

    file = fio.open_pickled_file(filename)

    ch_log = file['ch_log']
    ch_eqn_data = file['ch_eqn_data']
    ch_num_sol = file['ch_num_sol']  
    rep_par = file['rep_par']

    report(ch_eqn_data, ch_num_sol, ch_log, rep_par)

    print("    Saved data reported from:", filename)


def plot_seq(ax: mpl.figure.Axes, yys, xx=None, Line2D_par = {}, axes_par = {}, tick_par = {}):
    """
    A helper function to plot sequences in a single axes.

    INPUT
        ax: Axes object where the sequence data is ploted.
        yys
    """
    if isinstance(yys, tc.Tensor):
        ndim = yys.ndim
        if ndim > 2:
            IndexError("Tensor Size Error: yys.ndim must be <= 2.")
        elif ndim == 1:
            yys = [yys] # pass to `list` case
        else:
            Nrows, Ncols = tuple(yys.size())
            max_cols = Ncols
    
    if isinstance(yys, list):
        Nrows = len(yys)
        Ncols = []
        for yy in yys:
            if yy.ndim >1:
                IndexError("Tensor Size Error: yys[i].ndim must be <= 1.")
            else:
                Ncols.append(yy.size()[0])
        max_cols = max(Ncols)

    if xx is None:
        xx = tc.arange(max_cols)

    if xx.device != 'cpu': xx = xx.cpu()

    if max_cols <= 50:
        ax.set_prop_cycle(colormarker10_cycler)

    for i in range(Nrows):
        if yys[i].device != 'cpu': yys[i] = yys[i].cpu()
        Line2D = ax.plot(xx, yys[i], **Line2D_par)
    
    #--- Put title, x[y]lable, x[y]scale, legend,
    set_ax_properties(ax, axes_par)

    #--- Set ticks
    set_ticks(ax, tick_par)

    return Line2D


def set_ax_properties(ax, axes_par):
    """Helper function that updates various properties.

    This is an extension of various methods of Axes object.
    INPUT
        ax: Axes object where the parameters are applied.
        axes_par: dictionary of various parameters. The keys must follow the official names of Axes methods and the values must follow the official option arguments for those methods. This is because this function uses `eval(script)` where `script = ax.set_[key]([value])`. For example, to change the title of an Axes, we usually use `ax.set_title('Title text')`. 'title' and 'Title text' are plugged into key and value respectively. So, the key must be 'title' and the value must 'Title text'.
            """
    for key, value in axes_par.items():
        if value and key == 'legend':
            eval(f"ax.{key}({value})")
        elif value:
            eval(f"ax.set_{key}('{value}')")

def set_ticks(ax, tick_par: dict):
    '''Helper function to set up ticks all at once.

    This is an extension of matplotlib.axes.Axes.tick_params()
    INPUT
        ax: AxesSubplot object
        tick_par: dictionary of parameters
    OUPUT
        no output: new AxesSubplot object setting applies without returing and assigning
    
    Note:
    See the Matplotlib documentation for what can be done using tick_param
    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
    '''
    #--- mark ticks at only integer values
    for key, value in tick_par.items():
        if value:
            script = "ax.tick_params(axis='y'," + key + "='" + value + "')"
            eval(script)

def console_report(ch_eqn_data: dict, ch_num_sol: tc.Tensor, ch_log: list, rep_par: dict) -> None:
    """Console report summary of computation."""

    #=== description message
    rep_txt = []
    
    #=== corresponding data to report
    rep_data = []

    for _ in range(len(rep_txt)):
        print(rep_txt, rep_data)


def plot_init_final(ch_eqn_data: dict, ch_num_sol: tc.Tensor, ch_log: list, rep_par: dict) -> None:
    """Plot inticial state and final state."""
    #=== copy some data for readability
    dt = ch_eqn_data['dt'] # time step
    tt = ch_eqn_data['tt'] # time grid
    KK_saved = rep_par['last_save_ind'] # number of time grid points
    save_times = rep_par['save_times']
    
    #=== convert to cpu: grid domain and the functions to plot
    xx, yy, fn = ch_eqn_data['xx'], ch_eqn_data['yy'], ch_num_sol

    if xx.is_cuda:
        xx = xx.cpu()
        yy = yy.cpu()
        fn = fn.cpu()
    
    #=== collect time index to plot (: can be modified by user)
    ind_to_plot = (0, KK_saved)
    #-== collect actual evolution time for title of the plots (the two lines are automatized)
    ind_for_slicing = list(ind_to_plot)
    time_to_plot = save_times[ind_for_slicing]

    #=== plot the result at the time grid `ind_to_plot`
    fig, ax = plt.subplots(1, 2, figsize=(5,10), subplot_kw={"projection": "3d"})
    
    for kk in range(len(ind_to_plot)):
        surf = ax[kk].plot_surface(xx, yy, fn[ind_to_plot[kk],:,:], cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax[kk].set_title(f"t = {time_to_plot[kk]}")
    
    plt.show()

'''OBSOLETE: data saving is not done during time marching
def save_num_data(ch_eqn_data: dict, ch_num_sol: tc.Tensor, ch_log: list, rep_par: dict) -> None:
    """Save numerical solution, computation log, etc."""
    try:
        data_to_save = rep_par['data_to_save']
    except:
        data_to_save = []

    try:
        prefix = rep_par['file_prefix']
    except:
        prefix = ""

    #-== collect the data using builtins 'eval'
    data = {}
    for item in data_to_save:
        data[item]= eval(item)
    
    time_stamp = fio.get_filename('auto.pkl') # returns a time stamp with the extension 'ext' if 'auto.ext' is passed
    filename = prefix + str(time_stamp) # convert to string: output of `get_filename` is a Pathlib object
    with open(filename, 'wb') as fstream:
        pickle.dump(data, fstream)

    print("   numerical data is stored:", filename)
'''