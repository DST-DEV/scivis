"""
Module for matplotlib figure and axes formatting

Includes functionalities for line styling, axes formatting and value range
adjustments.
"""

# Built-in packages
from collections.abc import Sequence
import warnings

# Third-party packages
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# User-defined packages
from scivis import rcparams, utils
from scivis import latex_formatting as ltx


def _prepare_xy_line(x, y):
    """
    Convert the x- & y-data into 2-d arrays and assert matching array sizes.
    Higher-dimensional arrays (ndim>2) are converted to 2d if all dimensions
    except for 2 have length 1.

    Parameters
    ----------
    x : ArrayLike
        x-axis values.
    y : ArrayLike
        y-axis values.

    Returns
    -------
    x : np.ndarray
        x-axis values as a 2d array with the same shape as y.
    y : np.ndarray
        y-axis values as a 2d array with the same shape as x.

    """
    # Convert x and y to numpy
    x = utils._validate_arraylike_numeric(x)
    y = utils._validate_arraylike_numeric(y)

    # Unify dimensions of x & y
    if x.ndim == 1 and y.ndim == 1:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
    else:
        if x.ndim > 2 or y.ndim > 2:
            shape_x = [s_i for s_i in x.shape if s_i > 1]
            shape_y = [s_i for s_i in y.shape if s_i > 1]
            if len(shape_x) <= 2 and len(shape_y) <= 2:
                x = x.reshape(shape_x)
                y = y.reshape(shape_y)
            else:
                raise ValueError("Input arrays could not be broadcasted into "
                                 "2d arrays")

        if x.ndim == 1:
            x = np.tile(x.flatten(), (y.shape[0], 1))
        if y.ndim == 1:
            y = np.tile(y.flatten(), (x.shape[0], 1))

    return x, y


def _resolve_style_line(n_lines, plt_labels=None, show_legend=True,
                        ax_labels=None, ax_units=None,
                        latex=False, colors=None, cmap=None, alpha=None,
                        linestyles=None, linewidths=None, markers=None):
    """
    Prepare the style variables for a line plot with n_lines number of lines.

    Parameters
    ----------
    n_lines : int
        Number of lines which are plotted.
    plt_labels : None | Sequence of str, optional
        Labels for each of the lines. The default is None.
    show_legend : bool, optional
        Selection whether a legend should be displayed. If no labels are
        specified via plt_labels, default names "var_<i>" are assigned for each
        line.\n
        The default is True.
    ax_labels : None | Sequence of str, optional
        Axis labels. Must be either None or a list of two Nones / strings.\n
        The default is None.
    ax_units : None | Sequence of str, optional
        Axis units. Must be either None or a list of two Nones / strings.\n
    latex : bool, optional
        Selection whether to format use latex text interpretation.\n
        The default is False.
    colors : None | str | sequence, optional
        Line colors. Can be specified either as a single color which is applied
        globally to all lines, or as a sequence with one color for each line.
        Accepts any valid matplotlib color format.\n
        If None is given, the default colors from the rcParams are used.
        The default is None.
    cmap : None | str | ListedColormap | LinearSegmentedColormap | tuple | list, optional
        Colormap to apply to the lines. The following inputs are allowed:\n
        - None: Use default colors from rcParams\n
        - str: A valid matplotlib colormap name\n
        - ListedColormap / LinearSegmentedColormap: a colormap object
        - tuple / list: a list of colors from which to create a linear
          colormap\n
        This overwrites the color parameter.\n
        The default is None.
    alpha : None | int | float | sequence of {int, float, numpy.number}, optional
        Transparency values. Can be specified either as a scalar global value
        or individually as a sequence with one element for each line.\n
        The default is None.
    linestyles : None | str | sequence of str, optional
        Linestyles. Can be specified either as a scalar global value
        or individually as a sequence with one element for each line.\n
        The default is None.
    linewidths : None | int | float | sequence of {int, float, numpy.number}, optional
        Linewidths. Can be specified either as a scalar global value
        or individually as a sequence with one element for each line.\n
        The default is None.
    markers : None | Sequence of str, optional
        Linewidths. Can be specified either as a scalar global value
        or individually as a sequence with one element for each line.\n
        The default is None.

    Returns
    -------
    plt_labels : sequence of {None, str}
        Plot labels (1 per line).
    axis_labels : sequence of {None, str}
        Axis labels (1 per line).
    col : sequence of {None, str, int}
        Line colors (1 per line).
    alpha : sequence of {None, int, float, numpy.number}
        Transparency values (1 per line).
    ls : sequence of {None, str}
        Linestyles (1 per line).
    lw : sequence of {None, int, float, numpy.number}
        Linewidths (1 per line).
    markers : sequence of {None, str}
        Markers (1 per line).

    """
    if not isinstance(latex, bool):
        raise TypeError("latex must be a boolean value.")

    # Check plot labels
    if isinstance(show_legend, bool):
        if show_legend:
            plt_labels = _check_style_variable(var=plt_labels,
                                               name="plt_labels",
                                               req_type=str, n_lines=n_lines)
            if plt_labels[0] is None:
                plt_labels = ["var"+str(i) for i in range(n_lines)]
        else:
            plt_labels = [None]*n_lines
    else:
        raise TypeError("show_legend must be a boolean value.")

    # Format axis labels & units
    if ax_labels is None:
        axis_labels = [None, None]
    elif isinstance(ax_labels, (Sequence, np.ndarray)):
        if not len(ax_labels) == 2:
            raise ValueError("Invalid number of axis labels. Must be length 2")

        if not all(isinstance(lbl_i, str) or lbl_i is None
                   for lbl_i in ax_labels):
            raise TypeError("Invalid element type for ax_labels parameter. "
                            "Must be a Sequence of str or None.")

        if isinstance(ax_units, (Sequence, np.ndarray)):
            if not len(ax_units) == 2:
                raise ValueError("Invalid number of axis units. "
                                 + "Must be length 2.")

            if not all(isinstance(unit_i, str) or unit_i is None
                       for unit_i in ax_units):
                raise TypeError("Invalid element type for ax_units parameter. "
                                "Must be a Sequence of str or None.")

            if latex is True:
                axis_labels = [ltx.latex_notation(ax_labels[i], ax_units[i])
                               if ax_labels[i] is not None else None
                               for i in range(2)]
            else:
                axis_labels = [None, None]
                for i in range(2):
                    if isinstance(ax_labels[i], str) and ax_labels[i]:
                        if isinstance(ax_units[i], str) and ax_units[i]:
                            ax_units[i] = " [" + ax_units[i] + r"]"
                        else:
                            ax_units[i] = ""
                        axis_labels[i] = ax_labels[i] + ax_units[i]

        else:
            if latex is True:
                axis_labels = [ltx.ensure_math(ax_labels[i]) for i in range(2)]
            else:
                axis_labels = [ax_labels[i]
                               if ax_labels[i] is not None else None
                               for i in range(2)]
    else:
        raise TypeError("Invalid input for linestyles. Must be a Sequence of "
                        "Strings.")

    # Prepare colors
    col = None
    if cmap is not None:
        if isinstance(cmap, str):
            if cmap not in mpl.colormaps.keys():
                raise ValueError("Unknown colormap. Please use a valid "
                                 "maptlotlib colormap")
            else:
                col = mpl.colormaps[cmap](np.linspace(0, 1, n_lines))
        elif isinstance(cmap, (ListedColormap, LinearSegmentedColormap)):
            col = cmap(np.linspace(0, 1, n_lines))
        elif isinstance(cmap, (tuple, list)):
            cmap = LinearSegmentedColormap.from_list("scivis_cmap", cmap)
            col = cmap(np.linspace(0, 1, n_lines))
        else:
            raise TypeError("cmap must be a string, a ListedColormap / "
                            "LinearSegmentedColormap object, or a list/tuple "
                            "of colors")
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
                                  + "Proceeding with default colors.")
                    col = list("k")*n_lines  # Default color
                    break
            col = colors
        else:
            warnings.warn("Incompatible shape of colors. Must be a single "
                          + "color or a list-like with the same length as "
                          + "the number of lines to plot. Proceeding with "
                          + "default colors.")
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
        if markers in rcparams.mss.keys():
            markers = [rcparams.mss[markers]]*n_lines
        else:
            markers = [dict(marker=markers)]*n_lines
    elif isinstance(markers, (Sequence, np.ndarray)):
        if len(markers) == 0:
            markers = [None]*n_lines
        elif len(markers) == n_lines:
            if not all(isinstance(m_i, str) for m_i in markers):
                raise TypeError("Invalid element type for markers parameter."
                                "Must be a Sequence of Strings.")

            markers = []
            for i in range(n_lines):
                if markers[i] in rcparams.mss.keys():
                    markers.append(rcparams.mss[markers[i]])
                else:
                    markers.append(dict(marker=markers[i]))
        else:
            raise ValueError("Invalid number of markers for number of input "
                             + "dimensions.")
    else:
        raise TypeError("Invalid input for markers. Must be either a single"
                        "String or a Sequence of Strings.")

    return plt_labels, axis_labels, col, alpha, ls, lw, markers


def _check_style_variable(var, name, req_type, n_lines, fill_value=None):
    """
    Check if a style variable is either a scalar value of None or the required
    type or a sequence of None or the required type.
    If the variable is None, it is converted to a list with length n_lines
    filled with the fill_value.

    Parameters
    ----------
    var : scalar | Sequence
        Variable to check.
    name : str
        Name of the variable.
    req_type : type
        Required data type for the variable.
    n_lines : int
        Number of lines which are plotted. The variable needs to have this
        length if it isn't scalar
    fill_value : None | scalar, optional
        Value to fill var with in case it is None. The default is None.

    Raises
    ------
    TypeError
        If var is neither None, nor req_type nor a sequence of req_type.
    ValueError
        If var is a sequence and its number of elements is not n_lines.

    Returns
    -------
    var : list
        List of length n_lines with either the original values from val, or the
        fill_value.

    """
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
    elif isinstance(var, req_type):
        var = [var]*n_lines
    elif isinstance(var, (Sequence, np.ndarray)) and len(var) > 0:
        if len(var) == n_lines:
            if not all(isinstance(var_i, req_type) for var_i in var):
                raise TypeError("Invalid element type for " + name
                                + " parameter. Must be a Sequence of "
                                + type_name + ".")

            var = list(var)
        else:
            raise ValueError("Invalid number of " + name + " for "
                             + "number of input dimensions.")
    else:
        raise TypeError("Invalid input for " + name + ". Must be either a "
                        "single " + type_name + "or a Sequence.")

    return var


def _adjust_value_range(x, y, ax_lims=None, margins=True, autoscale_y=True,
                        overflow=False):
    """
    Adjusts the value range and axis limits of 2d line plot data.

    Parameters
    ----------
    ax_lims : None | Sequence, optional
        Axis limits for the x- and y-axis. Must be either None or a 2-element
        Sequence in which each element is a 2-element Sequence consisting of
        the lower & upper axis limit.\n
        The first element pertains to the x-axis, the second to the y-axis.\n
        If no axis limits should be enforced, None can be given. This also
        applies to the elements of the sequence if limits should only be
        specified for one oxis.\n
        The default is None.
    margins : bool | Sequence, optional
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
    overflow : bool | Sequence, optional
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
    x : np.ndarray
        Filtered x-axis values.
    y : np.ndarray
        Filtered y-axis values.
    ax_lims_adjusted : list
        Adjusted axis limits.

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
                        "booleans.")

    # Check entry for autoscale_y
    if not isinstance(autoscale_y, bool):
        raise TypeError("autoscale_y must be boolean")

    # Check entries for overflow
    if any(margins):
        if isinstance(overflow, bool):
            overflow = (overflow, overflow)
        elif not isinstance(overflow, (Sequence, np.ndarray)) or \
                not all(isinstance(o, bool) for o in overflow):
            raise TypeError("overflow must be boolean or a sequence of "
                            "booleans.")

    # Copy axis limits and convert elements to lists to enable item assignment
    ax_lims_adjusted = [lim_i if lim_i is None else list(lim_i)
                        for lim_i in ax_lims]

    # Get data ranges
    data_range = np.empty((2, x.shape[0], 2))
    data_range[0, :, 0] = np.nanmin(x, axis=1)
    data_range[0, :, 1] = np.nanmax(x, axis=1)
    data_range[1, :, 0] = np.nanmin(y, axis=1)
    data_range[1, :, 1] = np.nanmax(y, axis=1)

    data_range_global = list(np.array([np.min(data_range[:, :, 0], axis=1),
                                       np.max(data_range[:, :, 1], axis=1)]).T)
    data_range_global = [list(data_range_i)
                         for data_range_i in data_range_global]

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
            ax_lims_adjusted[i] = data_range_global[i][:]

        # Adjust axis limits to fit margins
        if margins[i]:
            # Adjust axis limits to enable/disable margins
            margin = abs(data_range_global[i][1]-data_range_global[i][0])*.025
            ax_lims_wo_margins = ax_lims_adjusted[i].copy()

            if ax_lims_adjusted[i][0] >= data_range_global[i][0]:
                # Limit lies within the data
                ax_lims_adjusted[i][0] -= margin
            else:
                # Limit outside of value range
                ax_lims_adjusted[i][0] = data_range_global[i][0] - margin

            if ax_lims_adjusted[i][1] <= data_range_global[i][1]:
                # Limit lies within the data
                ax_lims_adjusted[i][1] += margin
            else:
                # Limit outside of value range
                ax_lims_adjusted[i][1] = data_range_global[i][1] + margin

            if overflow[i]:
                # Enforce data limits until axis limits +- margins
                data[i, :] = utils.replace_outside_nan(data[i, :],
                                                       *ax_lims_adjusted[i])
            else:
                # Enforce data limits until axis limits
                data[i, :] = utils.replace_outside_nan(data[i, :],
                                                       *ax_lims_wo_margins)
        else:
            # Remove overflow
            if not overflow[i]:
                data[i, :] = utils.replace_outside_nan(data[i, :],
                                                       *ax_lims_adjusted[i])

    x, y = data  # Unpack combined data again

    return x, y, ax_lims_adjusted


def _format_axes_line(ax, ax_labels=None, ax_lims=None,
                      ax_ticks=None, ax_tick_lbls=None,
                      ax_ticks_minor=None, ax_tick_lbls_minor=None,
                      ax_show_minor_ticks=True, ax_show_grid=True,
                      ax_show_grid_minor=False,
                      profile="fullsize", scale=1):
    """


    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Axes to format.
    ax_labels : None | Sequence of str, optional
        Axis labels. Must be either None or a list of two Nones / strings.\n
        The default is None.
    ax_units : None | Sequence of str, optional
        Axis units. Must be either None or a list of two Nones / strings.\n
    ax_ticks : None | sequence {int, float, np.number}, optional
        Major tick mark positions. The default is None.
    ax_tick_lbls : None | sequence {int, float, np.number, str}, optional
        Major tick labels. The default is None.
    ax_ticks_minor : None | sequence {int, float, np.number}, optional
        Minor tick mark positions. The default is None.
    ax_tick_lbls_minor : None | sequence {int, float, np.number, str}, optional
        Minor tick labels. The default is None.
    ax_show_minor_ticks : bool, optional
        Selection whether to show major ticks.\n
        The default is True.
    ax_show_grid : bool, optional
        Selection whether to show major gird lines.\n
        The default is True.
    ax_show_grid_minor : bool, optional
        Selection whether to show minor gird lines. If ax_show_grid is False,
        the minor grid lines are automatically deactivated as well.\n
        The default is False.
    profile : String, optional
        Profile settings for the scaling.\n
        - "fullsize": Optimized for using the figure in full size (i.e.
          width = text width) on A4 paper in portrait\n
        - "halfsize": Optimized for using the figure in half size (i.e.
          width = 0.5 * text width) on A4 paper in portrait\n
        - "partsize": Optimized for using the figure in partial size (i.e.
          width = factor * text width) on A4 paper in portrait.
          The parameter 'scale' signifies the scale of the figure on the page\n
        - "custom_scale": Custom scaling factor for the rcParams\n
        The default is "fullsize".
    scale : int | float | np.number, optional
        Scaling factor of font sizes & padding. Only applied if profile '
        partsize' or 'custom_scale' is selected. \n
        The default is 1.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    None.

    """
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
        raise TypeError("ax_show_grid must be boolean.")
    if not isinstance(not ax_show_grid_minor, bool):
        raise TypeError("ax_show_grid_minor must be boolean.")

    if ax_show_grid is True:
        ax.grid(visible=True, which="major",
                **rcparams._scale_dict_params(rcparams.grid_style["major"],
                                              profile=profile,
                                              scale=scale))
        if ax_show_grid_minor is True:
            ax.grid(visible=True, which="minor",
                    **rcparams._scale_dict_params(rcparams.grid_style["minor"],
                                                  profile=profile,
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
    """
    Formats the ticks of a selected axis and applies axis limits to the axis.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Axes to format.
    which : {"major", "minor"}, optional
        Selectio whether major or minor ticks should be formatted.\n
        The default is "major".
    ticks : None | sequence {int, float, np.number}, optional
        Tick mark positions. The default is None.
    labels : None | sequence {int, float, np.number, str}, optional
        Major tick labels. The default is None.
    limits : None | sequence {int, float, np.number}, optional
        Axis limits. The default is None.

    Raises
    ------
    TypeError
        If parameter 'which' is not a boolean.
    ValueError
        If parameter 'which' is not 'major' or 'minor'\n
        If number of elements for tick positions and tick marks don't match.

    Returns
    -------
    None.

    """
    # Check axis
    if not isinstance(ax, mpl.axes._axes.Axes):
        raise TypeError("Parameter 'axis' must be a matplotlib axes object.")

    # Check parameter "which"
    if not isinstance(which, str):
        raise TypeError("Parameter 'which' must be a str.")
    if which not in ("major", "minor"):
        raise ValueError("Parameter 'which' must be a 'major' or 'minor'.")

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
                                 + ". Must be match number of tick positions.")

    # Adjust ticks and labels based on axis limits
    for i in range(2):
        if limits[i] is not None and ticks[i] is not None:
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
    """
    Check if an axis variable is either None or a sequence with two elements
    which each must be either None or a sequence of the required length.
    Optionally, the two sequences can be sorted in ascending order.

    Parameters
    ----------
    var : scalar | Sequence
        Variable to check.
    name : str
        Name of the variable.
    sort : boole, optional
        Selection whether the sequences within the variable should be sorted
        in ascending order.\n
        The default is False.
    req_len : int | np.integer, optional
        Required length of the elements of var.\n
        The default is None.

    Raises
    ------
    TypeError
        If sort is not a boolean.\n
        If req_len is not an integer.\n
        If var is neither None nor a Sequence
    ValueError
        If var is a Sequence with not exactly 2 elements.\n
        If var is a Sequence and one of its element does not have the length
        specified by req_len

    Returns
    -------
    var : list
        List with two elements which each are either None or a sequence of the
        required length.

    """
    if not isinstance(sort, bool):
        raise TypeError("sort must be a boolean")

    if not (req_len is None or isinstance(req_len, (int, np.integer))):
        raise TypeError("req_len must be an integer or None")

    if var is None:
        var = [None, None]
    elif isinstance(var, (Sequence, np.ndarray)):
        if not len(var) == 2:
            raise ValueError("Invalid number of elements for " + name + ". "
                             + "Must be a Sequence with two elements.")

        var = list(var)
        for i in range(2):
            if var[i] is None:
                continue
            if req_len is not None and not len(var[i]) == req_len:
                raise ValueError("Invalid number of elements for " + name
                                 + " of axis " + str(i) + ". "
                                 + "Must be a Sequence with two elements.")
            var[i] = utils._validate_arraylike_numeric(var[i], name=name,
                                                       ndim=1)

            if sort:
                var[i] = np.sort(var[i])
    else:
        raise TypeError(name + " must be a Sequence or None.")

    return var


def _apply_rcparams_to_axes(ax):
    """
    Apply the scivis rcParams to an existing axes.\n
    Note: Not all axes parameters can be changed after its creation. The
    appearance might thus differ to axes created with the scivis rcParams.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Matplotlib axes object to apply the rcparams to.

    Raises
    ------
    TypeError
        If ax is not a matplotlib axes object.

    Returns
    -------
    None.

    """

    if not isinstance(ax, mpl.axes._axes.Axes):
        raise TypeError("Axis must be a matplotlib axes object.")

    # --- Axes properties ---
    ax.set_facecolor(rcparams.rcparams_axes['axes.facecolor'])
    for spine in ax.spines.values():
        spine.set_linewidth(rcparams.rcparams_axes['axes.linewidth'])
        spine.set_edgecolor(rcparams.rcparams_axes['axes.edgecolor'])

    ax.xaxis.label.set_size(rcparams.rcparams_axes['axes.labelsize'])
    ax.yaxis.label.set_size(rcparams.rcparams_axes['axes.labelsize'])
    ax.title.set_fontsize(rcparams.rcparams_axes['axes.titlesize'])
    ax.title.set_color(rcparams.rcparams_axes['axes.titlecolor']
                       if rcparams.rcparams_axes['axes.titlecolor'] != "auto"
                       else rcparams.rcparams_text['text.color'])
    ax.title.set_position((0.5, 1.0))
    ax.set_prop_cycle(rcparams.rcparams_axes['axes.prop_cycle'])

    # --- Tick parameters ---
    ax.tick_params(
        axis='both',
        which='major',
        direction=rcparams.rcparams_ticks['xtick.direction'],
        length=rcparams.rcparams_ticks['xtick.major.size'],
        width=rcparams.rcparams_ticks['xtick.major.width'],
        pad=rcparams.rcparams_ticks['xtick.major.pad'],
        top=rcparams.rcparams_ticks['xtick.top'],
        right=rcparams.rcparams_ticks['ytick.right'],
        labelsize=rcparams.rcparams_ticks['xtick.labelsize']
    )
    ax.tick_params(
        axis='both',
        which='minor',
        direction=rcparams.rcparams_ticks['xtick.direction'],
        length=rcparams.rcparams_ticks['xtick.minor.size'],
        width=rcparams.rcparams_ticks['xtick.minor.width'],
        pad=rcparams.rcparams_ticks['xtick.minor.pad']
    )

    # --- Grid ---
    ax.grid(visible=True, which="major", **rcparams.grid_style["major"])
    ax.grid(visible=True, which="minor", **rcparams.grid_style["minor"])

    # --- Legend ---
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(rcparams.rcparams_legend['legend.frameon'])
        leg.get_frame().set_alpha(
            rcparams.rcparams_legend['legend.framealpha'])
        leg.get_frame().set_edgecolor(
            rcparams.rcparams_legend['legend.edgecolor'])
        leg.get_frame().set_facecolor(
            rcparams.rcparams_legend['legend.facecolor']
            if rcparams.rcparams_legend['legend.facecolor'] != 'inherit'
            else ax.get_facecolor()
        )
        for text in leg.get_texts():
            text.set_fontsize(rcparams.rcparams_legend['legend.fontsize'])
            if rcparams.rcparams_legend['legend.labelcolor'] != "None":
                text.set_color(rcparams.rcparams_legend['legend.labelcolor'])


def _apply_rcparams_to_figure(fig):
    """
    Apply the scivis rcParams to an existing figure.\n
    Note: Not all figure parameters can be changed after its creation. The
    appearance might thus differ to a figure created with the scivis rcParams.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib figure to apply the rcparams to.

    Raises
    ------
    TypeError
        If ax is not a matplotlib axes object.

    Returns
    -------
    None.

    """

    if not isinstance(fig, mpl.figure.Figure):
        raise TypeError("fig must be a matplotlib figure object.")

    # Size and DPI
    if 'figure.figsize' in rcparams.rcparams_figure:
        fig.set_size_inches(*rcparams.rcparams_figure['figure.figsize'],
                            forward=True)
    if 'figure.dpi' in rcparams.rcparams_figure:
        fig.set_dpi(rcparams.rcparams_figure['figure.dpi'])

    # Face/edge colors
    fig.patch.set_facecolor(rcparams.rcparams_figure.get(
        'figure.facecolor', 'white'))
    fig.patch.set_edgecolor(rcparams.rcparams_figure.get(
        'figure.edgecolor', 'white'))

    # Constrained layout
    fig.set_constrained_layout(rcparams.rcparams_figure.get(
        'figure.constrained_layout.use', False))

    # Subplot spacing
    fig.subplots_adjust(top=rcparams.rcparams_figure.get(
        'figure.subplot.top', 0.94))
