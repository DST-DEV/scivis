"""
Scivis is a highly customizeable and at the same time low effort visualisation
package for scientific purposes. It offers default plot settings for common
plots in scientific reports as well as full customization if necessary. All of
this is combined into a few comprehensive API commands
"""

# Built-in packages
import pathlib
from pathlib import Path

# Third-party packages
import matplotlib as mpl
import matplotlib.pyplot as plt

# User-defined packages
import scivis.formatting as scifrmt
from scivis import rcparams


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
              exp_fld=None, fname=None, ftype=".svg", savefig=False,
              return_obj=False):
    """
    Plot one or multiple lines.

    Parameters
    ----------
    x : ArrayLike
        x-axis data. Must either be a vector, a 2d array or a higher
        dimensional array in which all but 2 dimensions have length 1.\n
        In the case of a 2d array, each row is intepreted as a separate line.
        The length of the vector / the number of elements on the second axis
        must match the y-axis values.
    y : ArrayLike
        y-axis data. Must either be a vector, a 2d array or a higher
        dimensional array in which all but 2 dimensions have length 1.\n
        In the case of a 2d array, each row is intepreted as a separate line.
        The length of the vector / the number of elements on the second axis
        must match the x-axis values.
    ax : matplotlib.axes._axes.Axes, optional
        Axes to plot the data on. If None is given, a new figure is created.\n
        The default is None.
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
    latex : bool, optional
        Selection whether to format use latex text interpretation.\n
        The default is False.
    plt_labels : None | Sequence of str, optional
        Labels for each of the lines. The default is None.
    ax_labels : None | Sequence of str, optional
        Axis labels. Must be either None or a list of two Nones / strings.\n
        The default is None.
    ax_units : None | Sequence of str, optional
        Axis units. Must be either None or a list of two Nones / strings.\n
    colors : None | str | sequence, optional
        Line colors. Can be specified either as a single color which is applied
        globally to all lines, or as a sequence with one color for each line.
        Accepts any valid matplotlib color format.
        The default is None.
    cmap : None | str, optional
        Colormap to apply to the lines. This overwrites the color parameter.\n
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
    exp_fld : None | str | pathlib.Path, optional
        Export folder to save the figure into. If None is given, the current
        working directory is used.\n
        The default is None.
    fname : str, optional
        Filename for saving the figure. If None is given, the filename is
        generated from the axis labels.\n
        The default is None.
    ftype : str, optional
        Filetype to use for exporting the figure. Must be a valid image file
        type that is supported by matplotlibs figure.savefig.\n
        The default is ".svg".
    savefig : bool, optional
        Whether the figure should be exported as an image file. \n
        The default is False.
    return_obj : bool, optional
        Whether the figure & axes object and the export path should be
        returned.\n
        The default is False.

    Raises
    ------
    TypeError
        If ax is neither None nor a matplotlib.axes._axes.Axes object.\n
        If exp_fld is neither None, nor a string nor a pathlib.Path object.\n
        If fname is neither None nor a string.\n
        If ftype is not a string.
    ValueError
        If the fname is None or an empty string and no axis label was specified
        for either the x- or y-axis. These labels are needed for the filename
        generation.
    OSError
        If the folder specified in exp_fld does not exist.

    Returns
    -------
    fig : None | matplotlib.figure.Figure
        Figure object or None if return_obj is set to False.
    ax : None | matplotlib.axes._axes.Axes
        Axes object or None if return_obj is set to False.
    fpath : None | pathlib.Path
        Export file path. If savefig=False or return_obj=False, then None is
        returned.

    """

    # Prepare plot data
    x, y = scifrmt._prepare_xy_line(x, y)

    x, y, ax_lims = scifrmt._adjust_value_range(x, y, ax_lims=ax_lims,
                                                margins=margins,
                                                autoscale_y=autoscale_y,
                                                overflow=overflow)

    # Prepare style settings
    n_lines = max(x.shape[0], y.shape[0])
    plt_labels, axis_labels, col, alpha, ls, lw, markers = \
        scifrmt._resolve_style_line(
            n_lines=n_lines, plt_labels=plt_labels,
            ax_labels=ax_labels, ax_units=ax_units,
            latex=latex, colors=colors, cmap=cmap, alpha=alpha,
            linestyles=linestyles, linewidths=linewidths, markers=markers)

    # Plot
    rc_profile = rcparams._prepare_rcparams(latex=latex, profile=profile,
                                            scale=scale)

    with mpl.rc_context(rc_profile):
        if latex:
            # Note: These parameters are also part of the latex_text_profile
            # dict from the scivis.rcparams module. However, matplotlib
            # apparently handles them differntly when directly assigning them
            # compared to them being part of the rc_profile for the
            # mpl.rc_context.
            mpl.rcParams['text.latex.preamble'] = \
                "\n".join([r'\usepackage{amsmath}',  # Optional, for math symbols
                           r'\usepackage{siunitx}'])
            mpl.rcParams.update({"pgf.preamble": "\n".join([
                    r"\usepackage[utf8]{inputenc}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{amsmath}",
                    r"\usepackage[detect-all,locale=DE]{siunitx}",
                    ])})

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

        scifrmt._format_axes_line(
            ax=ax, ax_labels=axis_labels, ax_lims=ax_lims,
            ax_ticks=ax_ticks, ax_tick_lbls=ax_tick_lbls,
            ax_ticks_minor=ax_ticks_minor,
            ax_tick_lbls_minor=ax_tick_lbls_minor,
            ax_show_minor_ticks=ax_show_minor_ticks,
            ax_show_grid=ax_show_grid, ax_show_grid_minor=ax_show_grid_minor)

    # Export figure to file
    if savefig is True:
        if fname is None or (isinstance(fname, str) and len(fname) == 0):
            if any(ax_lbl_i is None for ax_lbl_i in ax_labels):
                raise ValueError("Export filename could not be determined. "
                                 "Either explicit filename via parameter "
                                 "'fname' or axis labels required")
            lbls = [lbl.replace("$", "").replace("\\", "")
                    for lbl in ax_labels]
            fname = f"{lbls[1]}_vs_{lbls[0]}"
        elif not isinstance(fname, str):
            raise TypeError("fname must be either None or str")

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
        return None, None, None
        plt.close(fig)
