# %% Imports
# Built-in packages
from collections.abc import Sequence
import pathlib
from pathlib import Path
import re
import warnings

# Third-party packages
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.typing import ArrayLike
from scipy import signal


# %%Global plot settings
rcparams_figure = {
    'figure.figsize': (16, 8),
    'figure.dpi': 144.0,
    'figure.edgecolor': 'white',
    'figure.facecolor': 'white',
    'figure.subplot.top': .94,    # Distance between suptitle and subplots
    'figure.constrained_layout.use': True
    }

rcparams_axes = {
    'axes.labelsize': 28,
    'axes.titlesize': 33,
    'axes.labelpad': 15,
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'None',
    'axes.titlepad': 6.0,
    'axes.titlecolor': 'auto',
    'axes.titlelocation': 'center',
    'axes.prop_cycle': mpl.cycler('color', ['k', 'k', 'k', 'k', 'k', 'k'])
                      + mpl.cycler('linestyle',
                                   ['-', '--', '-.', ':',
                                    (0, (4, 2, 1, 2, 1, 2)),
                                    (0, (5, 2, 5, 2, 1, 2))])
    }

rcparams_line = {
    'lines.linestyle': "-",
    'lines.linewidth': 1.2,
    'lines.marker': "None",
    'lines.markersize': 10,
    'scatter.marker': "+",
    'lines.color': "k",
    'lines.markeredgecolor': "auto",
    'lines.markeredgewidth': 1.0,
    'lines.markerfacecolor': "auto",
    'lines.markersize': 10.0
    }

rcparams_ticks = {
    # Direction and visibility
    'xtick.direction': 'in',       # {"in", "out", "inout"}
    'ytick.direction': 'in',
    "xtick.top": True,
    "ytick.right": True,
    'xtick.alignment': 'center',

    # Tick length (points)
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.minor.size": 5,
    "ytick.minor.size": 5,

    # Tick width (points)
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.75,
    "ytick.minor.width": 0.75,

    # Padding between ticks and labels (points)
    "xtick.major.pad": 10,
    "ytick.major.pad": 10,
    "xtick.minor.pad": 5,
    "ytick.minor.pad": 5,

    # Label font properties
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "xtick.labelcolor": "inherit",
    "ytick.labelcolor": "inherit",
    "xtick.alignment": "center",
    "ytick.alignment": "center",
}

rcparams_legend = {
    'legend.borderaxespad': 0.5,
    'legend.borderpad': 0.4,
    'legend.columnspacing': 2.0,
    'legend.edgecolor': '0.8',
    'legend.facecolor': 'inherit',
    'legend.fancybox': True,
    'legend.fontsize': 22,
    'legend.framealpha': 0.8,
    'legend.frameon': True,
    'legend.handleheight': 0.7,
    'legend.handlelength': 2.0,
    'legend.handletextpad': 0.8,
    'legend.labelcolor': 'None',
    'legend.labelspacing': 0.5,
    'legend.loc': 'best',
    'legend.markerscale': 1.0,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'legend.shadow': False,
    'legend.title_fontsize': None,
    }

rcparams_text = {
    'text.color': 'black',
    'font.size': 22
    }

rcparams_export = {
    'savefig.bbox': "tight",
    'savefig.dpi': 300,
    'savefig.edgecolor': 'auto',
    'savefig.facecolor': 'auto',
    'savefig.format': 'pdf',
    'savefig.orientation': 'portrait',
    'savefig.transparent': True,
    'savefig.pad_inches': .1
    }

_grid_style = {
    "major": {
        "axis": "both",
        "lw": 1.5,
        "color": "0.5",
        "alpha": .6,
        "zorder": 1
        },
    "minor": {
        "axis": "both",
        "lw": 1,
        "color": "0.65",
        "alpha": .6,
        "zorder": 1
        },
    }

_latex_text_profile = {
    'text.usetex': True,  # Activate latex rendering
    'font.family': 'serif',  # LaTeX default font family
    "pgf.texsystem": "pdflatex",  # Use pdflatex for generating PDFs
    "pgf.rcfonts": False,  # Ignore Matplotlib's default font settings
    'text.latex.preamble':
        "\n".join([r'\usepackage{amsmath}',  # Optional, for math symbols
                   r'\usepackage{siunitx}']),
    "pgf.preamble": "\n".join([  # pgf plots will use this preamble
                               r"\usepackage[utf8]{inputenc}",
                               r"\usepackage[T1]{fontenc}",
                               r"\usepackage{amsmath}",
                               r"\usepackage[detect-all,locale=DE]{siunitx}",
                               ])
    }

_default_text_profile = {
    "font.family": "serif",
    "font.sans-serif": ["Times New Roman"],
    'mathtext.fontset': 'cm'
    }

mss = {".": dict(marker=".", ms=15, mec="k"),
       "+": dict(marker="+", ms=15, mec="k"),
       "x": dict(marker="x", ms=15, mec="k"),
       "d": dict(marker="d", ms=8, mec="k"),
       "1": dict(marker="1", ms=15, mec="k"),
       "v": dict(marker="v", ms=8, mec="k"),
       "default": dict(marker="+", ms=10, mec="k")
       }

_scale_factor = .65

#%% SIunitx mappings

siunitx_units_mapping = {
    "A": r"\ampere",
    "cd": r"\candela",
    "K": r"\kelvin",
    "Bq": r"\becquerel",
    "°C": r"\degreeCelsius",
    "C": r"\coulomb",
    "F": r"\farad",
    "Gy": r"\gray",
    "Hz": r"\hertz",
    "H": r"\henry",
    "J": r"\joule",
    "lm": r"\lumen",
    "kat": r"\katal",
    "lx": r"\lux",
    "N": r"\newton",
    "Ω": r"\ohm",
    "Ohm": r"\ohm",
    "Pa": r"\pascal",
    "rad": r"\radian",
    "S": r"\siemens",
    "Sv": r"\sievert",
    "sr": r"\steradian",
    "T": r"\tesla",
    "V": r"\volt",
    "W": r"\watt",
    "Wb": r"\weber",
    "au": r"\astronomicalunit",
    "B": r"\bel",
    "Da": r"\dalton",
    "d": r"\day",
    "dB": r"\decibel",
    "deg": r"\degree",
    "°": r"\degree",
    "%": r"\percent",
    "eV": r"\v",
    "ha": r"\hectare",
    "h": r"\hour",
    "L": r"\litre",
    "'": r"\arcminute",
    "min": r"\minute",
    "\"": r"\arcsecond",
    "Np": r"\neper",
    "t": r"\tonne",
    "fg": r"\fg",
    "pg": r"\pg",
    "ng": r"\ng",
    "ug": r"\ug",
    "mg": r"\mg",
    "g": r"\g",
    "kg": r"\kg",
    "pm": r"\pm",
    "nm": r"\nm",
    "μm": r"\um",
    "mm": r"\mm",
    "cm": r"\cm",
    "dm": r"\dm",
    "m": r"\m",
    "km": r"\km",
    "as": r"\as",
    "fs": r"\fs",
    "ps": r"\ps",
    "ns": r"\ns",
    "μs": r"\us",
    "ms": r"\ms",
    "s": r"\s",
    "fmol": r"\fmol",
    "pmol": r"\pmol",
    "nmol": r"\nmol",
    "μmol": r"\umol",
    "mmol": r"\mmol",
    "mol": r"\mol",
    "kmol": r"\kmol",
}

siunitx_prefixes_mapping = {
    "q": r"\quecto",
    "r": r"\ronto",
    "y": r"\yocto",
    "z": r"\zepto",
    "a": r"\atto",
    "f": r"\femto",
    "p": r"\pico",
    "n": r"\nano",
    "μ": r"\micro",
    r"\mu": r"\micro",
    "m": r"\milli",
    "c": r"\centi",
    "d": r"\deci",
    "da": r"\deca",
    "h": r"\hecto",
    "k": r"\kilo",
    "M": r"\mega",
    "G": r"\giga",
    "T": r"\tera",
    "P": r"\peta",
    "E": r"\exa",
    "Z": r"\zetta",
    "Y": r"\yotta",
    "R": r"\ronna",
    "Q": r"\quetta",
    }


# %% Plot functions
def plot_line(x, y, ax=None,
              profile="fullsize", scale=1, latex=False,
              plt_labels=None, ax_labels=None, ax_units=None,
              colors=None, cmap=None, alpha=None,
              linestyles=None, linewidths=None, markers=None,
              ax_lims=None, margins=True, autoscale_y=True, overflow=True,
              ax_ticks=None, ax_tick_lbls=None,
              ax_ticks_minor=None, ax_tick_lbls_minor=None,
              ax_show_minor_ticks=True, ax_show_grid=True,
              ax_show_grid_minor=False,
              exp_fld=None, fname=None, ftype=".svg", savefig=True,
              return_obj=False):

    # Prepare plot data
    x, y = _prepare_xy_line(x, y)

    x, y, ax_lims = _adjust_value_range(x, y, ax_lims=ax_lims, margins=margins,
                                        autoscale_y=autoscale_y,
                                        overflow=overflow)

    # Prepare style settings
    n_lines = max(x.shape[0], y.shape[0])
    plt_labels, axis_labels, col, alpha, ls, lw, markers = \
        _resolve_style_line(n_lines=n_lines, plt_labels=plt_labels,
                            ax_labels=ax_labels, ax_units=ax_units,
                            latex=latex, colors=colors, cmap=cmap, alpha=alpha,
                            linestyles=linestyles, linewidths=linewidths,
                            markers=markers)

    # Plot
    rc_profile = _prepare_rcparams(latex=latex, profile=profile,
                                  scale=scale)

    with mpl.rc_context(rc_profile):
        # Create figure
        if ax is None:
            fig, ax = plt.subplots()
        elif isinstance(ax, mpl.axes._axes.Axes):
            fig = ax.figure
        else:
            raise TypeError("Axis must be a matplotlib axes object or None")

        # Plot lines
        for i in range(x.shape[0]):
            if ls is not None:
                ax.plot(x[i, :], y[i, :], label=plt_labels[i], **markers[i],
                        lw=lw[i], ls=ls[i], c=col[i], alpha=alpha[i], zorder=2)
            else:
                ax.plot(x[i, :], y[i, :], label=plt_labels[i], **markers[i],
                        lw=lw[i], c=col[i], alpha=alpha[i], zorder=2)

        _format_axes_line(ax=ax, ax_labels=ax_labels, ax_lims=ax_lims,
                          ax_ticks=ax_ticks, ax_tick_lbls=ax_tick_lbls,
                          ax_ticks_minor=ax_ticks_minor,
                          ax_tick_lbls_minor=ax_tick_lbls_minor,
                          ax_show_minor_ticks=ax_show_minor_ticks,
                          ax_show_grid=ax_show_grid,
                          ax_show_grid_minor=ax_show_grid_minor)

    # Export figure to file
    if savefig is True:
        if fname is None or (isinstance(fname, str) and len(fname) == 0):
            if any(ax_lbl_i is None for ax_lbl_i in axis_labels):
                raise ValueError("Export filename could not be determined. "
                                 "Either explicit filename via parameter "
                                 "'fname' or axis labels required")
            lbls = [lbl.replace("$", "").replace("\\", "") for lbl in ax_labels]
            fname = f"{lbls[1]}_vs_{lbls[0]}"

        # Check filetype
        if isinstance(ftype, str):
            if not ftype.startswith("."):
                ftype = "." + ftype

            if ftype not in (".png", ".jpg", ".svg", ".pdf", ".pgf"):
                raise ValueError("Unsupported export file type. Supported "
                                 "types are png, jpg, svg, pdf and pgf")
        else:
            raise TypeError("ftype must be str")

        # Check export folder
        if exp_fld is None:
            exp_fld = Path.cwd()
        if isinstance(exp_fld, (str, pathlib.PurePath)):
            if not Path(exp_fld).is_dir():
                raise OSError("Export folder does not exist at "
                              + str(Path(exp_fld).resolve()))
        else:
            raise TypeError("Export folder must be a str or a pathlib.Path "
                            "object")

        # Export figure
        fpath = Path(exp_fld, fname + ftype)
        fig.savefig(fpath)
    else:
        fpath = None

    if return_obj:
        return fig, ax, fpath
    else:
        plt.close(fig)

def _prepare_xy_line(x, y):
    # Convert x and y to numpy
    x = np.asarray(x)
    y = np.asarray(y)

    # Unify dimensions of x & y
    if x.ndim > 1 or y.ndim > 1:
        if x.ndim == 1:
            x = np.tile(x.flatten(), (y.shape[0], 1))
        if y.ndim == 1:
            y = np.tile(y.flatten(), (x.shape[0], 1))
    else:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

    return x, y

def _resolve_style_line(n_lines, plt_labels=None, ax_labels=None, ax_units=None,
                       latex=None, colors=None, cmap=None, alpha=None,
                       linestyles=None, linewidths=None, markers=None):
    # Check plot labels
    plt_labels = _check_style_variable(var=plt_labels, name="plt_labels",
                                       req_type=str, n_lines=n_lines)
    if plt_labels[0] is None:
        plt_labels = ["var"+str(i) for i in range(n_lines)]

    # Format axis labels & units
    if ax_labels is None:
        axis_labels = [None]*n_lines
    elif isinstance(ax_labels, (Sequence, np.ndarray)):
        if not len(ax_labels) == 2:
            raise ValueError("Invalid number of axis labels. Must be length 2")
        if isinstance(ax_units, (Sequence, np.ndarray)):
            if not len(ax_labels) == 2:
                raise ValueError("Invalid number of axis units. "
                                 + "Must be length 2")
            if latex is True:
                axis_labels = [latex_notation(ax_labels[i], ax_units[i])
                               for i in range(2)]
            else:
                axis_labels = [ax_labels[i] + " [" + ax_units[i] + r"]"
                               for i in range(2)]
        else:
            if latex is True:
                axis_labels = [ensure_math(ax_labels[i]) for i in range(2)]
            else:
                axis_labels = [ax_labels[i] for i in range(2)]
    else:
        raise TypeError("Invalid input for linestyles. Must be a Sequence of "
                        "Strings")

    # Prepare colors
    col = None
    if cmap is not None:
        if isinstance(cmap, str):
            try:
                col = mpl.colormaps[cmap](np.linspace(0, 1, n_lines))
            except KeyError as e:
                warnings.warn(str(e) + ". Proceeding with default colors")
        elif isinstance(cmap, (ListedColormap, LinearSegmentedColormap)):
            col = cmap(np.linspace(0, 1, n_lines))
        elif isinstance(cmap, (tuple, list)):
            cmap = LinearSegmentedColormap.from_list("BEM_cmap", cmap)
            col = cmap(np.linspace(0, 1, n_lines))
    elif isinstance(colors, (str)) and colors:
        col = colors
        col = [col]*n_lines
    elif isinstance(colors, (list, tuple)) and colors:
        if len(colors) == 3 \
            and all(isinstance(c_i, (int, np.integer)) for c_i in colors)\
                and all(c_i >= 0 and c_i <= 1 for c_i in colors):
            col = mpl.colors.to_hex(colors)
            col = [col]*n_lines
        elif len(colors) == n_lines:
            for c in colors:
                if not mpl.colors.is_color_like(c):
                    warnings.warn("Color input contained invalid colors. "
                                  + "Proceeding with default colors")
                    col = list("k")*n_lines  # Default color
                    break
            col = colors
        else:
            warnings.warn("Incompatible shape of colors. Must be a single "
                          + "color or a list-like with the same length as "
                          + "the number of lines to plot. Proceeding with "
                          + "default colors")
    else:
        col = list("k")*n_lines  # Default color

    # Check Alpha values
    alpha = _check_style_variable(var=alpha, name="alpha",
                                  req_type=(int, float, np.number),
                                  n_lines=n_lines)
    if not all((alpha_i is None or (alpha_i >= 0 and alpha_i <= 1))
               for alpha_i in alpha):
        raise ValueError("Alpha values must lie within 0 to 1")

    # Check Linestyles
    ls = _check_style_variable(var=linestyles, name="linestyles", req_type=str,
                               n_lines=n_lines)

    # Check Linewidths
    lw = _check_style_variable(var=linewidths, name="linewidths",
                               req_type=(int, float, np.number),
                               n_lines=n_lines)

    if markers is None:
        markers = [{"marker": None}]*n_lines
    elif isinstance(markers, str) and len(markers) > 0:
        if markers in mss.keys():
            markers = [mss[markers]]*n_lines
        else:
            markers = [dict(marker=markers)]*n_lines
    elif isinstance(markers, (Sequence, np.ndarray)):
        if len(markers) == 0:
            markers = [None]*n_lines
        elif len(markers) == n_lines:
            if not all(isinstance(m_i, str) for m_i in markers):
                raise TypeError("Invalid element type for markers parameter."
                                "Must be a Sequence of Strings")

            markers = []
            for i in range(n_lines):
                if markers[i] in mss.keys():
                    markers.append(mss[markers[i]])
                else:
                    markers.append(dict(marker=markers[i]))
        else:
            raise ValueError("Invalid number of markers for number of input "
                             + "dimensions")
    else:
        raise TypeError("Invalid input for markers. Must be either a single"
                        "String or a Sequence of Strings")

    return plt_labels, axis_labels, col, alpha, ls, lw, markers


def _check_style_variable(var, name, req_type, n_lines, fill_value=None):
    # Prepare type name string
    name_mapping = {str: "String",
                    int: "Integer",
                    np.integer: "Integer",
                    float: "Float",
                    (int, float, np.number): "numerical value"
                    }

    if req_type in name_mapping.keys():
        type_name = name_mapping.get(req_type)
    elif isinstance(req_type, tuple):
        type_name = [repr(req_type_i) for req_type_i in req_type]
        type_name = " / ".join(type_name)
    else:
        type_name = repr(req_type)

    # Check if variable is valid
    if var is None:
        var = [fill_value]*n_lines
    elif isinstance(var, req_type) and len(var) > 0:
        var = [var]*n_lines
    elif isinstance(var, (Sequence, np.ndarray)):
        if len(var) == n_lines:
            if not all(isinstance(var_i, req_type) for var_i in var):
                raise TypeError("Invalid element type for " + name
                                + " parameter. Must be a Sequence of")

            var = var
        else:
            raise ValueError("Invalid number of " + name + " for "
                             + "number of input dimensions")
    else:
        raise TypeError("Invalid input for " + name + ". Must be either a "
                        "single " + type_name + "or a Sequence")

    return var


def _adjust_value_range(x, y, ax_lims=None, margins=True, autoscale_y=True,
                       overflow=True):
    """
    Adjusts the value range and axis limits of 2d line plot data.

    Parameters
    ----------
    ax_lims : {None, Sequence}, optional
        Axis limits for the x- and y-axis. Must be either None or a 2-element
        Sequence in which each element is a 2-element Sequence consisting of
        the lower & upper axis limit.\n
        The first element pertains to the x-axis, the second to the y-axis.\n
        If no axis limits should be enforced, None can be given. This also
        applies to the elements of the sequence if limits should only be
        specified for one oxis.\n
        The default is None.
    margins : {bool, Sequence}, optional
        Selection whether margins around the data should be displayed. \n
        Can either be specified globally as a single boolean, or individually
        for the x- and y-axis by providing a sequence with two boolean
        values.\n
        Note that this also adjusts the axis limits if they were set too large
        for the actual data.\n
        The default is True.
    autoscale_y : bool, optional
        Selection whether the y-axis should be scaled to match the plotted data
        within the x-axis limits. Thus only relevant, if x-axis limits are
        specified and no y-axis limits are given (If y-axis limits are
        specified, they overwrite this parameters).\n
        The default is True.
    overflow : {bool, Sequence}, optional
        Selection whether overflow of the plotted values into the margins are
        allowed. This applies in the case that axis limits are specified and
        margins is set to true for at least one axis.\n
        Can either be specified globally as a single boolean, or individually
        for the x- and y-axis by providing a sequence with two boolean
        values.\n
        The default is True.

    Raises
    ------
    TypeError
        In case of wrong input types.

    Returns
    -------
    XXXX Write return values

    """
    # Prepare x & y vakzes
    x, y = _prepare_xy_line(x, y)

    # Check axis limits
    ax_lims = _check_axis_variable(ax_lims, name="axis limits", sort=True,
                                   req_len=2)

    # Check entries for margins selection
    if isinstance(margins, bool):
        margins = (margins, margins)
    elif not isinstance(margins, (Sequence, np.ndarray)) or \
            not all(isinstance(m, bool) for m in margins):
        raise TypeError("margins must be boolean or a sequence of "
                        "booleans")

    # Check entry for autoscale_y
    if not isinstance(autoscale_y, bool):
        raise TypeError("autoscale_y must be boolean")

    # Check entries for overflow
    if any(ax_lims) and any(margins):
        if isinstance(overflow, bool):
            overflow = (overflow, overflow)
        elif not isinstance(overflow, (Sequence, np.ndarray)) or \
                not all(isinstance(o, bool) for o in overflow):
            raise TypeError("overflow must be boolean or a sequence of "
                            "booleans")

    # Copy axis limits and convert elements to lists to enable item assignment
    ax_lims_adjusted = [lim_i if lim_i is None else list(lim_i)
                        for lim_i in ax_lims]

    # Get data ranges
    data_lims = np.empty((2, x.shape[0], 2))
    data_lims[0, :, 0] = np.nanmin(x, axis=1)
    data_lims[0, :, 1] = np.nanmax(x, axis=1)
    data_lims[1, :, 0] = np.nanmin(y, axis=1)
    data_lims[1, :, 1] = np.nanmax(y, axis=1)

    data_lims_global = np.array([np.min(data_lims[:, :, 0], axis=1),
                                 np.max(data_lims[:, :, 1], axis=1)]).T

    # Applay autoscale for y-axis
    # Note: Manually specified y-axis limits are prioritized over autoscaling
    if autoscale_y and ax_lims[1] is None:
        if ax_lims[0] is None:
            warnings.warn("y-axis autoscaling not possible without x-axis "
                          "limits. Resuming without y-axis autoscaling.")
        else:
            ax_lims_adjusted[1] = [np.min(y[x >= ax_lims[0][0]]),
                                   np.max(y[x <= ax_lims[0][1]])]

    # Loop over axes
    data = np.stack((x, y), axis=0).astype(float)
    for i in range(2):
        if ax_lims_adjusted[i] is None:
            ax_lims_adjusted[i] = data_lims_global[i, :]

        # Remove overflow
        if not overflow[i]:
            if ax_lims_adjusted[i][0] > data_lims_global[i, 0] \
                    or ax_lims_adjusted[i][1] < data_lims_global[i, 1]:
                # Limit lies within the data => Adjust data ranges to
                # prevent overflow into the margins
                for j in range(x.shape[0]):
                    data[i, j,
                         ((data[i, j, :] < ax_lims_adjusted[i][0])
                          | (data[i, j, :] > ax_lims_adjusted[i][1]))
                         ] = np.nan

        # Adjust axis limits to fit margins
        if margins[i]:
            # Adjust axis limits to enable/disable margins
            margin = abs(data_lims_global[i, 1]-data_lims_global[i, 0])*.05

            if ax_lims_adjusted[i][0] >= data_lims_global[i, 0]:
                # Limit lies within the data
                ax_lims_adjusted[i][0] -= margin
            else:
                # Limit outside of value range
                ax_lims_adjusted[i][0] = data_lims_global[i, 0] - margin

            if ax_lims_adjusted[i][1] <= data_lims_global[i, 1]:
                # Limit lies within the data
                ax_lims_adjusted[i][1] += margin
            else:
                # Limit outside of value range
                ax_lims_adjusted[i][1] = data_lims_global[i, 1] + margin

    x, y = data  # Unpack combined data again

    return x, y, ax_lims_adjusted


def _format_axes_line(ax, ax_labels=None, ax_lims=None,
                      ax_ticks=None, ax_tick_lbls=None,
                      ax_ticks_minor=None, ax_tick_lbls_minor=None,
                      ax_show_minor_ticks=True, ax_show_grid=True,
                      ax_show_grid_minor=False,
                      profile="fullsize", scale=1):
    # Set axis labels
    if isinstance(ax_labels[0], str):
        ax.set_xlabel(ax_labels[0])
    if isinstance(ax_labels[1], str):
        ax.set_ylabel(ax_labels[1])

    # Format ticks
    _format_ticks(ax=ax, which="major", ticks=ax_ticks, labels=ax_tick_lbls,
                  limits=ax_lims)
    if ax_show_minor_ticks:
        _format_ticks(ax=ax, which="minor", ticks=ax_ticks_minor,
                      labels=ax_tick_lbls_minor, limits=ax_lims)
        ax.minorticks_on()
    else:
        ax.minorticks_off()

    # Format grid
    if not isinstance(not ax_show_grid, bool):
        raise TypeError("ax_show_grid must be boolean")
    if not isinstance(not ax_show_grid_minor, bool):
        raise TypeError("ax_show_grid_minor must be boolean")

    if ax_show_grid is True:
        ax.grid(visible=True, which="major",
                **_scale_dict_params(_grid_style["major"], profile=profile,
                                     scale=scale))
        if ax_show_grid_minor is True:
            ax.grid(visible=True, which="minor",
                    **_scale_dict_params(_grid_style["minor"], profile=profile,
                                         scale=scale))
        else:
            ax.grid(visible=False, which="minor", axis="both")
    else:
        ax.grid(visible=False, which="both", axis="both")

    # Set axis limits
    ax_lims = _check_axis_variable(ax_lims, name="axis limits", sort=True,
                                  req_len=2)
    if ax_lims[0] is not None:
        ax.set_xlim(ax_lims[0])
    if ax_lims[1] is not None:
        ax.set_ylim(ax_lims[1])

def _format_ticks(ax, which="major", ticks=None, labels=None, limits=None):
    # Check axis
    if not isinstance(ax, mpl.axes._axes.Axes):
        raise TypeError("Parameter 'axis' must be a matplotlib axes object")

    # Check parameter "which"
    if not isinstance(which, str):
        raise TypeError("Parameter 'which' must be a str")
    if which not in ("major", "minor"):
        raise ValueError("Parameter 'which' must be a 'major' or 'minor'")

    # Check axis limits
    limits = _check_axis_variable(limits, name="axis limits", sort=True,
                                 req_len=2)

    # Check axis ticks
    ticks = _check_axis_variable(ticks, name=which + " axis ticks")
    if all(ticks_i is None for ticks_i in ticks):
        return

    # Check tick labels
    labels = _check_axis_variable(labels, name=which + " axis labels")
    for i in range(2):
        if ticks[i] is not None and labels[i] is not None:
            if not len(labels[i]) == len(ticks[i]):
                raise ValueError("Invalid number for" + which + " axis tick "
                                 "labels of axis " + str(i)
                                 + ". Must be match number of tick positions")

    # Adjust ticks and labels based on axis limits
    for i in range(2):
        if limits[i] is not None:
            idx_valid = (ticks[i] <= limits[i][1]) & (ticks[i] >= limits[i][0])
            ticks[i] = ticks[i][idx_valid]

            if labels[i] is not None:
                labels[i] = labels[i][idx_valid]

    # Set axis ticks
    minor = True if which == "minor" else False
    if ticks[0] is not None:
        ax.set_xticks(ticks=ticks[0], labels=labels[0], minor=minor)
    if ticks[1] is not None:
        ax.set_yticks(ticks=ticks[1], labels=labels[1], minor=minor)


def _check_axis_variable(var, name, sort=False, req_len=None):
    if not isinstance(sort, bool):
        raise TypeError("sort must be a boolean")

    if not (req_len is None or isinstance(req_len, (int, np.integer))):
        raise TypeError("req_len must be an integer or None")

    if var is None:
        var = [None, None]
    elif isinstance(var, (Sequence, np.ndarray)):
        if not len(var) == 2:
            raise ValueError("Invalid number of elements for " + name + ". "
                             "Must be a Sequence with two elements")

        var = list(var)
        for i in range(2):
            if var[i] is None:
                continue
            if req_len is not None and not len(var[i]) == req_len:
                raise ValueError("Invalid number of elements for axis limits "
                                 "of axis " + str(i) + ". "
                                 "Must be a Sequence with two elements")
            var[i] = _validate_arraylike_numeric(var[i], name=name, ndim=1)

            if sort:
                var[i] = np.sort(var[i])
    else:
        raise TypeError(name + " must be a Sequence or None")

    return var


def _prepare_rcparams(latex=False, profile="fullsize", scale=1):
    if not isinstance(profile, str):
        raise TypeError("profile must be a String")

    if not isinstance(latex, bool):
        raise TypeError("latex must be a bool or None")

    rc_profile = rcparams_figure | rcparams_axes | rcparams_line \
        | rcparams_ticks | rcparams_legend | rcparams_text | rcparams_export

    # Scale plot appearance proportional to chosen scale
    rc_profile = _scale_dict_params(rc_profile, profile=profile, scale=scale)

    if latex:
        rc_profile.update(_latex_text_profile)
    else:
        rc_profile.update(_default_text_profile)

    return rc_profile

def _scale_dict_params(param_dict, profile="fullsize", scale=1):
    param_dict_scaled = param_dict.copy()

    if profile != "fullsize":
        if profile == "halfsize":
            scale = 2
        else:
            if not _validate_numeric(scale, allow_neg=False,
                                     allow_zero=False):
                raise ValueError("scale factor must be a scalar "
                                 "non-negative positive numeric value")

        if profile in ("partsize", "halfsize"):
            scale = (scale-1)*_scale_factor + 1  # Empirical scale

        param_dict_scaled = {key: (val * scale
                                   if (key.endswith(("size", "pad", "width",
                                                     "spacing", "length",
                                                     "height", "pad_inches"))
                                       and isinstance(val,
                                                      (int, float, np.number)))
                                   else val)
                             for key, val in param_dict_scaled.items()
                             }
    return param_dict_scaled

# %% Utility functions


def get_ax_size(fig, ax):
    """Calculates the axes size in pixels

    Parameters:
        fig (matplotlibe figure):
            Figure for which to calculate the axes size
        ax (matplotlibe axes):
            Axes for which to calculate the axes size

    Returns:
        width (float):
            Width of the figure in pixels
        height (float):
            height of the figure in pixels
    """
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height


def calc_text_pos(ax_lims, ax_size, base_pos, offset=-80):
    """Calculate the position of Text based on an offset in pixels.

    Parameters:
        ax_lims (array-like):
            Lower and upper limits of the axis
        ax_size (float):
            Size of the axis in pixels
        base_pos (float):
            Base position of the text in the unit of the axis ticks
        offset (float - optional):
            Desired offset of the text from the base_pos in pixels
            (default: -80)

    Returns:
        pos (float):
            Adjusted position of the text in the unit of the axis ticks
    """

    val_len = ax_lims[1]-ax_lims[0]
    pixel_pos = (base_pos-ax_lims[0])/val_len*ax_size+offset
    if pixel_pos <100:
        offset = -offset*4
        pixel_pos = (base_pos-ax_lims[0])/val_len*ax_size+offset
    elif ax_size-pixel_pos <100:
        offset = -offset*4
        pixel_pos = (base_pos-ax_lims[0])/val_len*ax_size+offset

    return (pixel_pos)/ax_size*val_len+ ax_lims[0]


def convert_to_siunitx(unit: str, brackets=True) -> str:
    """
    Convert a string containing a unit into a siunitx unit command

    Args:
        unit (str): Input string containing the unit.

    Returns:
        str: Unit converted to siunitx command.

    """
    # Check if input is valid
    if not isinstance(unit, str) or not unit:
        return ""

    # Replace round brackets for exponents by curly brackets
    unit = re.sub(r"(\^\(([^)]*)\))",
                  lambda match: r"^{" + match.group(2) + "}",
                  unit)

    # Replace round brackets for subscript by curly brackets
    unit = re.sub(r"(_\(([^)]*)\))",
                  lambda match: r"_{" + match.group(2) + "}",
                  unit)

    # Find all units in the string
    units = set(re.findall(r"[a-zA-Z°\"']+", unit))
    if not units:
        return ""

    # Replace units by the corresponding siunitx command
    for u in units:
        if u in siunitx_units_mapping.keys():
            unit = unit.replace(u, siunitx_units_mapping[u])
        elif len(u)>=2 and u[0] in siunitx_prefixes_mapping.keys() \
            and u[1:] in siunitx_units_mapping.keys():
                unit.replace(u,
                             siunitx_prefixes_mapping[u[0]]
                             + siunitx_units_mapping[u[1:]])

    if brackets:
        return r"$\:\left[\unit{" + unit + r"}\right]$"
    else:
        return r"$\:\unit{" + unit + r"}$"


def ensure_math(text):
    # Check if input is valid
    if not isinstance(text, str) or not text:
        return ""

    # Place curly brackets around subscripts
    text = re.sub(r"(_(\S*)\s)",
                  lambda match: r"_{" + match.group(2) + "}",
                  text)
    # Place curly brackets around subscripts
    text = re.sub(r"(_(\S*\Z))",
                  lambda match: r"_{" + match.group(2) + "}",
                  text)

    # Replace round brackets for subscript by curly brackets
    text = re.sub(r"(_\(([^)]*)\))",
                  lambda match: r"_{" + match.group(2) + "}",
                  text)

    # Place curly brackets around exponents
    text = re.sub(r"(\^(\S*)\s)",
                  lambda match: r"^{" + match.group(2) + "}",
                  text)

    # Replace round brackets for subscript by curly brackets
    text = re.sub(r"(\^(\S*\Z))",
                  lambda match: r"^{" + match.group(2) + "}",
                  text)

    # Replace round brackets for subscript by curly brackets
    text = re.sub(r"(\^\(([^)]*)\))",
                  lambda match: r"^{" + match.group(2) + "}",
                  text)

    #Replace white spaces with respective latex math command
    text = text.replace(" ", r"\:")

    # Ensure inline math mode
    text = "$" + text.replace("$", "") + "$"

    return text


def latex_notation(lbl="", unit="", brackets=True):
    return ensure_math(lbl) + convert_to_siunitx(unit, brackets=brackets)


def round_sig_digits(val, sig_figs=2):
    """
    Round a value to a specified number of significant digits

    Args:
        val (Union[int | float]): Value to round.
        sig_figs (int, optional): Number of significant digits to round to.
            Defaults to 2.

    Returns:
        val (Union[int | float]): Rounded value.

    """
    if val == 0:
        return 0
    return round(val, -int(np.floor(np.log10(abs(val)))) + (sig_figs - 1))

def _validate_arraylike_numeric(arr: ArrayLike, name: str = "",
                                ndim: int | None = None,
                                allow_neg: bool = True,
                                allow_zero: bool = True):
    """
    Validate if a variabe is array-like and contains only numeric values.

    Parameters
    ----------
    arr : ArrayLike
        The variable to check.
    name : str, optional
        Name of the variable (used for exception texts). If name='', the
        default name 'arr' is used.\n
        The default is "".
    ndim : int or None, optional
        Number of ndimension that arr is supposed to have.
        If None is specified, the ndimension is not checked.\n
        The default is None.
    allow_neg : bool, optional
        Selection whether negative values are allowed in the array.\n
        The default is True.
    allow_zero : bool, optional
        Selection whether zeros are allowed in the array.\n
        The default is True.

    Raises
    ------
    TypeError
        If name is not a string.\n
        If arr contains non-numeric values.
    ValueError
        If ndim is not a positive integer or None.\n
        If arr does not have the number of ndimensions specified in ndim.\n
        If arr is empty.\n
        If arr contains negative values (only if allow_neg=False).\n
        If arr contains zeros (only if allow_zero=False).

    Returns
    -------
    arr : np.ndarray
        The input array-like as a numpy array.

    """
    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    if not name:
        name = "Array"

    arr = np.asarray(arr)
    if ndim is not None:
        if not isinstance(ndim, int) \
                or not _validate_numeric(ndim,
                                         allow_neg=True, allow_zero=True):
            raise ValueError("ndim must be None or a positive integer.")

        if arr.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}D.")

    if not np.issubdtype(arr.dtype, np.number) or not np.isrealobj(arr):
        raise TypeError(f"{name} must contain only numeric values.")

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not allow_neg and np.any(arr < 0):
        raise ValueError(f"{name} must contain only positive values.")

    if not allow_zero and np.any(arr == 0):
        raise ValueError(f"{name} must contain only non-zero values.")

    return arr

def _validate_numeric(val: int | float | np.number,
                      allow_neg: bool = True,
                      allow_zero: bool = True):
    """
    Validate if a variable is a scalar numeric value.

    Parameters
    ----------
    val : int | float | np.number
        Variable to check.
    allow_neg : bool, optional
        Selection whether the variable is allowed to be negative.\n
        The default is True.
    allow_zero : bool, optional
        Selection whether the variable is allowed to be zero.\n
        The default is True.

    Returns
    -------
    bool
        True if the variable is a scalar numeric value (and non-zero if
        allow_zero=False and positive if allow_neg=False), else False.

    """
    if not isinstance(val, (int, float, np.number)) or np.isnan(val) \
            or np.isinf(val):
        return False
    elif not allow_neg and val < 0:
        return False
    elif not allow_zero and val == 0:
        return False
    else:
        return True
