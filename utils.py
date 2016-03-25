import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def draw_2d(Z, title='Heat map'):
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(Z, cmap=cm.jet)
    ax.set_title(title)

def draw_3d(Z, title='Heat map', stride=5, figsize=(15, 6)):
    X, Y = np.meshgrid(np.arange(0.0, Z.shape[1], 1.0), np.arange(0.0, Z.shape[0], 1.0))
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride, cmap=cm.jet, linewidth=0, antialiased=True, alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(15, 10)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def compare_2d(hm, hm_approx, figsize=(15, 4), title_left='Heat map', title_right='Approximation'):
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    fig.subplots_adjust(right=0.8)

    im = ax[0].imshow(hm, cmap=cm.jet, vmin=np.min(hm), vmax=np.max(hm))
    ax[0].set_title(title_left)
    ax[1].imshow(hm_approx, cmap=cm.jet, vmin=np.min(hm_approx), vmax=np.max(hm_approx))
    ax[1].set_title(title_right)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

def compare_3d(hm, hm_approx, stride=10, figsize=(18, 4), title_left='Heat map', title_right='Approximation'):
    Z = hm.T
    Z2 = hm_approx.T
    X, Y = np.meshgrid(np.arange(0.0, Z.shape[1], 1.0), np.arange(0.0, Z.shape[0], 1.0))

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(right=0.8)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride, cmap=cm.jet, linewidth=0, antialiased=True, alpha=0.7)
    ax.set_zlim3d(0, max(Z.flatten()))
    ax.set_title(title_left)
    ax.view_init(15, 10)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(X, Y, Z2, rstride=stride, cstride=stride, cmap=cm.jet, linewidth=0, antialiased=True, alpha=0.7)
    ax.set_zlim3d(0, max(Z.flatten()))
    ax.set_title(title_right)
    ax.view_init(15, 10)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    fig.colorbar(surf, cax=cbar_ax)

    plt.show()

def err(hm, hm_approx):
    approx_error = hm_approx - hm
    title = 'Error distribution\nmean:{:.4f}, std:{:.4f}'.format(approx_error.mean(), approx_error.std())

    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    fig.subplots_adjust(right=0.8)

    sns.distplot(approx_error.flatten(), ax=ax[0])
    ax[0].set_title(title)
    im = ax[1].imshow(approx_error, cmap='RdBu', vmin=np.min(approx_error), vmax=np.max(approx_error))
    ax[1].set_title('Error')

    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

    sns.jointplot(hm, hm_approx, kind="reg")
    plt.show()
