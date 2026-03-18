import torch
import geoopt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import imageio
from pygifsicle import optimize


class COLORS:
    SHINY_GREEN = "#bfffbf"
    SHINY_BLUE = "#a9e7ff"
    MAT_RED = "#ffc7c7"
    MAT_YELLOW = "#ffffbd"
    NEON_PINK = "#ff3ea0"
    BACKGROUND_BLUE = "#1e0c45"
    TEXT_COLOR = "#ffffff"


def setup_plot(manifold, lo=None, width=7, height=7, grid_line_width=0.3, with_background=True):

    # define figure parameters
    rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    rcParams["text.usetex"] = True
    rcParams['figure.figsize'] = width, height
    sns.set_style("white")

    # create figure
    fig = plt.figure()

    # determine manifold properties
    K = manifold.k
    R = manifold.radius

    # add circle
    circle = plt.Circle((0, 0), R, fill=with_background, color=COLORS.BACKGROUND_BLUE)
    plt.gca().add_artist(circle)
    if K > 0:
        circle_border = plt.Circle((0, 0), R, fill=False, color="gray",
                                   linewidth=2.0)
        plt.gca().add_artist(circle_border)


    # add background color
    if not(K < 0) and with_background:
        plt.gca().set_facecolor(COLORS.BACKGROUND_BLUE)

    # set up plot axes and aspect ratio
    if lo==None:
        if K < 0:
            lo = -R - 0.1
        else:
            lo = -2 * R - 0.1
    hi = -lo
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect("equal")

    # add grid of geodesics
    add_geodesic_grid(plt.gca(), manifold, line_width=grid_line_width)

    return fig, plt, (lo, hi)


def add_geodesic_grid(ax: plt.Axes, manifold: geoopt.Stereographic, line_width=0.1):

    # define geodesic grid parameters
    N_EVALS_PER_GEODESIC = 10000
    STYLE = "--"
    COLOR = "gray"
    LINE_WIDTH = line_width

    # get manifold properties
    K = manifold.k.item()
    R = manifold.radius.item()

    # get maximal numerical distance to origin on manifold
    if K < 0:
        # create point on R
        r = torch.tensor((R, 0.0), dtype=manifold.dtype)
        # project point on R into valid range (epsilon border)
        r = manifold.projx(r)
        # determine distance from origin
        max_dist_0 = manifold.dist0(r).item()
    else:
        max_dist_0 = np.pi * R
    # adjust line interval for spherical geometry
    circumference = 2*np.pi*R

    # determine reasonable number of geodesics
    # choose the grid interval size always as if we'd be in spherical
    # geometry, such that the grid interpolates smoothly and evenly
    # divides the sphere circumference
    n_geodesics_per_circumference = 4 * 6  # multiple of 4!
    n_geodesics_per_quadrant = n_geodesics_per_circumference // 2
    grid_interval_size = circumference / n_geodesics_per_circumference
    if K < 0:
        n_geodesics_per_quadrant = int(max_dist_0 / grid_interval_size)

    # create time evaluation array for geodesics
    if K < 0:
        min_t = -1.2*max_dist_0
    else:
        min_t = -circumference/2.0
    t = torch.linspace(min_t, -min_t, N_EVALS_PER_GEODESIC)[:, None]

    # define a function to plot the geodesics
    def plot_geodesic(gv):
        ax.plot(*gv.t().numpy(), STYLE, color=COLOR, linewidth=LINE_WIDTH)

    # define geodesic directions
    u_x = torch.tensor((0.0, 1.0))
    u_y = torch.tensor((1.0, 0.0))

    # add origin x/y-crosshair
    o = torch.tensor((0.0, 0.0))
    if K < 0:
        x_geodesic = manifold.geodesic_unit(t, o, u_x)
        y_geodesic = manifold.geodesic_unit(t, o, u_y)
        plot_geodesic(x_geodesic)
        plot_geodesic(y_geodesic)
    else:
        # add the crosshair manually for the sproj of sphere
        # because the lines tend to get thicker if plotted
        # as done for K<0
        ax.axvline(0, linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)
        ax.axhline(0, linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)

    # add geodesics per quadrant
    for i in range(1, n_geodesics_per_quadrant):
        i = torch.as_tensor(float(i))
        # determine start of geodesic on x/y-crosshair
        x = manifold.geodesic_unit(i*grid_interval_size, o, u_y)
        y = manifold.geodesic_unit(i*grid_interval_size, o, u_x)

        # compute point on geodesics
        x_geodesic = manifold.geodesic_unit(t, x, u_x)
        y_geodesic = manifold.geodesic_unit(t, y, u_y)

        # plot geodesics
        plot_geodesic(x_geodesic)
        plot_geodesic(y_geodesic)
        if K < 0:
            plot_geodesic(-x_geodesic)
            plot_geodesic(-y_geodesic)


def add_K_box(plt, K):
    props = dict(pad=10.0, facecolor='white', edgecolor='black', linewidth=0.5)
    plt.gca().text(0.05, 0.95, f"$\kappa={K:1.3f}$",
                   transform=plt.gca().transAxes,
                   verticalalignment='top', bbox=props)


def get_interpolation_Ks(num=200):
    # S-curve going through zero
    x = np.linspace(-1.0, 1.7, num=num)
    Ks = (x**3).tolist()
    Ks = [K for K in Ks if abs(K) > 0.001]
    return Ks


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, tmp_file, dpi=90):
    fig.savefig(tmp_file, dpi=dpi)
    img = imageio.imread(tmp_file)
    return img


def save_img_sequence_as_boomerang_gif(imgs, out_filename, fps=24):

    # invert image sequence
    l2 = list(imgs)
    l2.reverse()
    boomerang = imgs + l2

    # save gif
    imageio.mimsave(out_filename, boomerang, fps=fps)

    # optimize gif file size
    optimize(out_filename)


def save_img_sequence_as_gif(imgs, out_filename, fps=24):

    # save gif
    imageio.mimsave(out_filename, imgs, fps=fps)

    # optimize gif file size
    optimize(out_filename)


def add_3D_geodesic_grid(ax: plt.Axes, manifold: geoopt.Stereographic, line_width=0.1):

    # define geodesic grid parameters
    N_EVALS_PER_GEODESIC = 1000
    STYLE = "--"
    COLOR = "gray"
    LINE_WIDTH = line_width

    # get manifold properties
    K = manifold.k.item()
    R = manifold.radius.item()

    # get maximal numerical distance to origin on manifold
    if K < 0:
        # create point on R
        r = torch.tensor((R, 0.0, 0.0), dtype=manifold.dtype)
        # project point on R into valid range (epsilon border)
        r = manifold.projx(r)
        # determine distance from origin
        max_dist_0 = manifold.dist0(r).item()
    else:
        max_dist_0 = np.pi * R
    # adjust line interval for spherical geometry
    circumference = 2*np.pi*R

    # determine reasonable number of geodesics
    # choose the grid interval size always as if we'd be in spherical
    # geometry, such that the grid interpolates smoothly and evenly
    # divides the sphere circumference
    n_geodesics_per_circumference = 4 * 6  # multiple of 4!
    n_geodesics_per_quadrant = n_geodesics_per_circumference // 2
    grid_interval_size = circumference / n_geodesics_per_circumference
    if K < 0:
        n_geodesics_per_quadrant = int(max_dist_0 / grid_interval_size)

    # create time evaluation array for geodesics
    if K < 0:
        min_t = -1.2*max_dist_0
    else:
        min_t = -circumference/2.0
    t = torch.linspace(min_t, -min_t, N_EVALS_PER_GEODESIC)[:, None]

    # define a function to plot the geodesics
    def plot_geodesic(gv):
        ax.plot(*gv.t().numpy(), STYLE, color=COLOR, linewidth=LINE_WIDTH)

    # define geodesic directions
    u_x = torch.tensor((0.0, 1.0, 0.0))
    u_y = torch.tensor((1.0, 0.0, 0.0))
    u_z = torch.tensor((0.0, 0.0, 1.0))

    # add origin x/y/z-crosshair
    o = torch.tensor((0.0, 0.0, 0.0))
    if K < 0:
        x_geodesic = manifold.geodesic_unit(t, o, u_x)
        y_geodesic = manifold.geodesic_unit(t, o, u_y)
        z_geodesic = manifold.geodesic_unit(t, o, u_z)
        plot_geodesic(x_geodesic)
        plot_geodesic(y_geodesic)
        plot_geodesic(z_geodesic)
    else:
        # add the crosshair manually for the sproj of sphere
        # because the lines tend to get thicker if plotted
        # as done for K<0
        ax.plot([-max_dist_0, max_dist_0], [0, 0], [0, 0], linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)
        ax.plot([0, 0], [-max_dist_0, max_dist_0], [0, 0], linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)
        ax.plot([0, 0], [0, 0], [-max_dist_0, max_dist_0], linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)

    # add xy, xz, yz planes
    for i in range(1, n_geodesics_per_quadrant):
        i = torch.as_tensor(float(i))
        # determine start of geodesic on x/y/z-crosshair
        dx = manifold.geodesic_unit(i*grid_interval_size, o, u_x)
        dy = manifold.geodesic_unit(i*grid_interval_size, o, u_y)
        dz = manifold.geodesic_unit(i*grid_interval_size, o, u_z)

        # compute point on geodesics
        xy_geodesic = manifold.geodesic_unit(t, dy, u_x)
        yx_geodesic = manifold.geodesic_unit(t, dx, u_y)
        xz_geodesic = manifold.geodesic_unit(t, dx, u_z)
        zx_geodesic = manifold.geodesic_unit(t, dz, u_x)
        yz_geodesic = manifold.geodesic_unit(t, dy, u_z)
        zy_geodesic = manifold.geodesic_unit(t, dz, u_y)

        # plot geodesics
        plot_geodesic(xy_geodesic)
        plot_geodesic(yx_geodesic)
        plot_geodesic(xz_geodesic)
        plot_geodesic(zx_geodesic)
        plot_geodesic(yz_geodesic)
        plot_geodesic(zy_geodesic)
        if K < 0:
            plot_geodesic(-xy_geodesic)
            plot_geodesic(-yx_geodesic)
            plot_geodesic(-xz_geodesic)
            plot_geodesic(-zx_geodesic)
            plot_geodesic(-yz_geodesic)
            plot_geodesic(-zy_geodesic)
            
    """
    # add geodesics per quadrant
    for i in range(1, n_geodesics_per_quadrant):
        for j in range(1, 2):
            i = torch.as_tensor(float(i))
            j = torch.as_tensor(float(j))
            # determine start of geodesic on x/y/z-crosshair
            dx = manifold.geodesic_unit(i*grid_interval_size, o, u_x)
            dy = manifold.geodesic_unit(i*grid_interval_size, o, u_y)
            dz = manifold.geodesic_unit(i*grid_interval_size, o, u_z)

            # determine start of geodesic on xy/xz/yz-plane
            dxy = manifold.geodesic_unit(j*grid_interval_size, dx, u_y)
            dxz = manifold.geodesic_unit(j*grid_interval_size, dx, u_z)
            dyx = manifold.geodesic_unit(j*grid_interval_size, dy, u_x)
            dyz = manifold.geodesic_unit(j*grid_interval_size, dy, u_z)
            dzx = manifold.geodesic_unit(j*grid_interval_size, dz, u_x)
            dzy = manifold.geodesic_unit(j*grid_interval_size, dz, u_y)

            # compute point on geodesics
            xy_geodesic = manifold.geodesic_unit(t, dxy, u_z)
            yx_geodesic = manifold.geodesic_unit(t, dyx, u_z)
            xz_geodesic = manifold.geodesic_unit(t, dxz, u_y)
            zx_geodesic = manifold.geodesic_unit(t, dzx, u_y)
            yz_geodesic = manifold.geodesic_unit(t, dyz, u_x)
            zy_geodesic = manifold.geodesic_unit(t, dzy, u_x)

            # plot geodesics
            plot_geodesic(xy_geodesic)
            plot_geodesic(yx_geodesic)
            plot_geodesic(xz_geodesic)
            plot_geodesic(zx_geodesic)
            plot_geodesic(yz_geodesic)
            plot_geodesic(zy_geodesic)
            if K < 0:
                plot_geodesic(-xy_geodesic)
                plot_geodesic(-yx_geodesic)
                plot_geodesic(-xz_geodesic)
                plot_geodesic(-zx_geodesic)
                plot_geodesic(-yz_geodesic)
                plot_geodesic(-zy_geodesic)"""


def setup_plot_3D(manifold, lo=None, width=7, height=7, grid_line_width=0.3, with_background=True):

    # define figure parameters
    rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    rcParams["text.usetex"] = True
    rcParams['figure.figsize'] = width, height
    sns.set_style("white")

    # create figure
    fig = plt.figure()
    plt.style.use('dark_background')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_visible(False)

    # determine manifold properties
    K = manifold.k
    R = manifold.radius

    # Sphere parameters
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(np.size(u)), np.cos(v))

    # add sphere
    ax.plot_surface(
        x, y, z,
        color=COLORS.BACKGROUND_BLUE,
        alpha=0.4,
        edgecolor='none'
    )
    if K > 0:
        ax.plot_surface(
            x, y, z,
            color='gray',
            alpha=0.2,
            edgecolor='none',
            linewidth=2.0
        )

    # add background color
    if not(K < 0) and with_background:
        plt.gca().set_facecolor(COLORS.BACKGROUND_BLUE)

    # set up plot axes and aspect ratio
    if lo==None:
        if K < 0:
            lo = -R - 0.1
        else:
            lo = -2 * R - 0.1
    hi = -lo
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_zlim(lo, hi)
    plt.gca().set_aspect("equal")

    # add grid of geodesics
    add_3D_geodesic_grid(plt.gca(), manifold, line_width=grid_line_width)

    return fig, plt, (lo, hi)