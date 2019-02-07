import seaborn as sns
import numpy as np
import os
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D

from IPython.display import HTML
import matplotlib.pyplot as plt
import worm_util
from datetime import datetime

def xkcd_colors():
    color_names = ["windows blue",
                   "red",
                   "amber",
                   "faded green",
                   "dusty purple",
                   "orange",
                   "clay",
                   "pink",
                   "greyish",
                   "mint",
                   "light cyan",
                   "steel blue",
                   "forest green",
                   "pastel purple",
                   "salmon",
                   "dark brown"]

    colors = sns.xkcd_palette(color_names)
    return colors

def remove_frame(ax_array,all_off=False):
    for ax in np.ravel(ax_array):
        if not all_off:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
        if all_off:
            ax.set_axis_off()


def savefig(fig, title, save_path='../figures/'):
    ''' Formats title and automatically saves in directory
    '''
    fig.savefig(os.path.join(save_path, '{}.jpg'.format(title)), bbox_inches='tight', transparent=True,dpi=800
)
    
    
    
    
    
def scatter_animation_2D(X):
    fig, ax = plt.subplots()
    num_timesteps = X.shape[0]

    ax.set_xlim((np.min(X[:,0]), np.max(X[:,0])))
    ax.set_ylim((np.min(X[:,1]), np.max(X[:,1])))
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')

    def update_plot(i, data, scatter):
        past_points = min(i,5)
        scatter.set_offsets(X[i-past_points:i,:])
        c = [(0,0,0,(i+1)/past_points) for i in range(0,past_points)]

        scatter.set_color(c)

        return scatter,

    scatter = plt.scatter([], [], s=100)

    ani = animation.FuncAnimation(fig, update_plot, frames=num_timesteps,
                                  fargs=(X, scatter))
    return ani

def scatter_animation_3D(Y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3D')
#     plt.close()
    num_timesteps = Y.shape[0]

    ax.set_xlim((np.min(Y[:,0]), np.max(Y[:,0])))
    ax.set_ylim((np.min(Y[:,1]), np.max(Y[:,1])))
    ax.set_zlim((np.min(Y[:,2]), np.max(Y[:,2])))
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')

    def update_plot(i, data, scatter):
        past_points = min(i,5)
        scatter._offsets3d = (Y[i-past_points:i,0], Y[i-past_points:i,1],Y[i-past_points:i,2])
        c = [(0,0,0,(i+1)/past_points) for i in range(0,past_points)]
        scatter._facecolor3d = c
        scatter._edgecolor3d = c
        scatter.set_color(c)

        return scatter,

    scatter = ax.scatter([], [], [],s=100,depthshade=False)

    ani = animation.FuncAnimation(fig, update_plot, frames=50,
                                  fargs=(Y, scatter))
    return ani

def scatter_animation_2D_and_3D(X, Y):
    fig = plt.figure(figsize=(10,5))
    X_ax = fig.add_subplot(121,)
    Y_ax = fig.add_subplot(122, projection='3d')

    num_timesteps = Y.shape[0]

    X_ax.set_xlim((np.min(X[:,0]), np.max(X[:,0])))
    X_ax.set_ylim((np.min(X[:,1]), np.max(X[:,1])))
    X_ax.set_xlabel('Latent dim 1')
    X_ax.set_ylabel('Latent dim 2')


    Y_ax.set_xlim((np.min(Y[:,0]), np.max(Y[:,0])))
    Y_ax.set_ylim((np.min(Y[:,1]), np.max(Y[:,1])))
    Y_ax.set_zlim((np.min(Y[:,2]), np.max(Y[:,2])))
    Y_ax.set_xlabel('Obs dim 1')
    Y_ax.set_ylabel('Obs dim 2')
    Y_ax.set_zlabel('Obs dim 3')

    def update_plot(i, X, Y, X_scatter, Y_scatter):
        past_points = min(i,5)

        X_scatter.set_offsets(X[i-past_points:i,:])    
        Y_scatter._offsets3d = (Y[i-past_points:i,0], Y[i-past_points:i,1],Y[i-past_points:i,2])

        c = [(0,0,0,(i+1)/past_points) for i in range(0,past_points)]
        X_scatter.set_color(c)
        Y_scatter._facecolor3d = c
        Y_scatter._edgecolor3d = c

        return X_scatter, Y_scatter,

    X_scatter = X_ax.scatter([], [], s=100)
    Y_scatter = Y_ax.scatter([], [], [],s=100,depthshade=False)

    ani = animation.FuncAnimation(fig, update_plot, frames=num_timesteps,
                                  fargs=(X, Y, X_scatter, Y_scatter))

    return ani

def plot_vector_field(*args, color='black'):
    num_plots = len(args)
    fig, ax = plt.subplots(1,num_plots,figsize=(4*num_plots, 4))
    ax = np.atleast_1d(ax)
    xlims = [-2, 2]
    ylims = [-2, 2]
    X1, X2 = np.meshgrid(np.linspace(xlims[0], xlims[1], 10), np.linspace(ylims[0], ylims[1], 10))
    points = np.stack((X1, X2))
    for i, A in enumerate(args):
        AX = np.einsum('ij,jkl->ikl', A, points)
        Q = ax[i].quiver(X1, X2, AX[0] - X1, AX[1] - X2, units='width', color=color)      
        ax[i].set_xlim(xlims)
        ax[i].set_ylim(ylims)
    remove_frame(ax)
    
    
def plot_latents_over_time(X):
    d_latent = X.shape[1]
    fig, ax = plt.subplots(d_latent, 1, figsize= (10,4),sharex=True)
    ax[0].set_title("Latent states over time")
    for i in range(d_latent):
        ax[i].plot(X[:,i])
        ax[i].set_ylabel('Latent dim {}'.format(i + 1))

    ax[-1].set_xlabel('Time')
    remove_frame(ax)
    plt.tight_layout()
    
    
def plot_observations_over_time(Y):
    d_observation = Y.shape[1]
    fig, ax = plt.subplots(d_observation, 1, figsize= (10,4),sharex=True)
    ax[0].set_title("Observations over time")
    for i in range(d_observation):
        ax[i].plot(Y[:,i])
        ax[i].set_ylabel('Obs. dim {}'.format(i + 1))
    ax[-1].set_xlabel('Time')
    remove_frame(ax)
    plt.tight_layout()
    
    
def plot_crawling_worm(coordinates,fps,colors=np.array([])):
    fig, ax = plt.subplots()

    plt.close()
    T = coordinates.shape[0]
    ax.set_xlim(( np.min(coordinates[:,:,0]), np.max(coordinates[:,:,0])))
    ax.set_ylim(( np.min(coordinates[:,:,1]), np.max(coordinates[:,:,1])))
    line, = ax.plot([], [], lw=2)
    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        x = coordinates[i,:,0]
        y = coordinates[i,:,1]
        line.set_data(x, y)
        if colors.size > 0:
            line.set_c(colors[i])
        else:
            line.set_c('green')
        return (line,)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=T, interval=1000/np.round(fps))

    return anim

def plot_eigen_worms(eigen_worms):
    fig, ax = plt.subplots(1,eigen_worms.shape[1], figsize=(2*eigen_worms.shape[1],2),sharex=True, sharey=True)
    for i in np.arange(eigen_worms.shape[1]):
        eigen_skeleton = worm_util.angles_to_skeleton(eigen_worms[:,i])
        ax[i].plot(eigen_skeleton[:,0],eigen_skeleton[:,1],label=i, color=xkcd_colors()[i])
        ax[i].spines['left'].set_visible(False)
        ax[i].axes.get_yaxis().set_visible(False)
    remove_frame(ax)
    
def plot_eigen_projections(eigen_projections, fps):
    fig, ax = plt.subplots(1,1)
    colors = xkcd_colors()
    T = 3000
    for i in np.arange(eigen_projections.shape[1]):
        x = np.linspace(0, T*fps/1000, T)
        ax.plot(x, eigen_projections[:T,i] - 14*i, color = colors[i])
    ax.set_yticks([])
    ax.set_xlabel('Time (s)')
    remove_frame(ax)

    
def plot_discrete_states(z, max_time):
    plt.subplot(212)
    xkcd = np.array(xkcd_colors())
    z = xkcd[z,:]
    plt.imshow(z[None,:], aspect="auto",extent=[0, max_time, 0, 1])
    plt.xlim(0, max_time)
    plt.ylabel("$z_{\\mathrm{inferred}}$")
    plt.yticks([])
    plt.xlabel("time (s)")

def plot_discrete_state_behavioral_histograms(hmm_z, all_modes):
    K = np.unique(hmm_z).shape[0]
    fig, ax = plt.subplots(K,1, figsize=(3, K*2), sharey=True) 
    xkcd = xkcd_colors()
    for i, state in enumerate(np.unique(hmm_z)):
        state_times = hmm_z == state
        modes = all_modes[state_times]
        mode_counts = []
        for mode in [-1, 0, 1]:
            mode_counts.append(np.sum(modes == mode)*100/modes.shape[0])
        ax[i].bar([-1, 0, 1], mode_counts, color=xkcd[i])  
        ax[i].set_xticks([-1, 0, 1])
        ax[i].set_title('Discrete state {}'.format(i + 1))
        ax[i].set_ylabel('%')
        ax[i].set_xticklabels([])

    ax[-1].set_xticklabels(['Backwards', 'Paused', 'Forwards'])

    remove_frame(ax)
    plt.tight_layout()