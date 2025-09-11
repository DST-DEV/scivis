import numpy as np
from matplotlib import pyplot as plt
import warnings

x = np.tile(np.arange(20), (3, 1))
y = np.empty_like(x)
y[0,:] = x[0,:]**2
y[1,:] = 0.6*(x[0,:]-2)**2
y[2,:] = 0.4*(x[0,:]+1)**2.5

ax_lims=[[3, 15], None]

fig, ax = plt.subplots()
plt.plot(x.T, y.T)
ax.set_xlim(ax_lims[0])
ax.set_ylim(ax_lims[1])

def _filter_axis_vals(x, y, ax_lims=None, margins=True, autoscale_y=True,
                       overflow=True):

    if isinstance(margins, bool):
        margins = [margins, margins]

    if isinstance(overflow, bool):
        overflow = [overflow, overflow]

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
            ax_lims_adjusted[1] = [np.min(y[x>=ax_lims[0][0]]),
                                   np.max(y[x<=ax_lims[0][1]])]

    # Loop over axes
    data = np.stack((x, y), axis=0).astype(float)
    for i in range (2):
        if ax_lims_adjusted[i] is None:
            ax_lims_adjusted[i] = data_lims_global[i,:]

        # Remove overflow
        if not overflow[i]:
            if ax_lims_adjusted[i][0] > data_lims_global[i, 0] \
                    or ax_lims_adjusted[i][1] < data_lims_global[i, 1]:
                # Limit lies within the data => Adjust data ranges to
                # prevent overflow into the margins
                # todo this is only adjusting x, not y for iteration step 2
                for j in range(x.shape[0]):
                    data[i, j,
                         ((data[i, j, :]<ax_lims_adjusted[i][0])
                          | (data[i, j, :]>ax_lims_adjusted[i][1]))
                         ] = np.nan

        # Adjust axis limits to fit margins
        if margins[i]:
            # Adjust axis limits to enable/disable margins
            margin = abs(data_lims_global[i, 1]-data_lims_global[i, 0])*.05

            if ax_lims_adjusted[i][0]>=data_lims_global[i,0]:
                # Limit lies within the data
                ax_lims_adjusted[i][0] -= margin
            else:
                # Limit outside of value range
                ax_lims_adjusted[i][0] = data_lims_global[i,0] - margin

            if ax_lims_adjusted[i][1]<=data_lims_global[i,1]:
                # Limit lies within the data
                ax_lims_adjusted[i][1] += margin
            else:
                # Limit outside of value range
                ax_lims_adjusted[i][1] = data_lims_global[i, 1] + margin
        # else:  # No margins
        #     ax_lims_adjusted[i] = data_lims_global[i,:]

    x, y = data  # Unpack combined data again

    return x, y, ax_lims_adjusted

x1, y1, lims = _filter_axis_vals(x, y, ax_lims=ax_lims,
                                 margins=False,
                                 autoscale_y=True,
                                 overflow=False)

fig, ax = plt.subplots()
plt.plot(x1.T, y1.T)
ax.set_xlim(lims[0])
ax.set_ylim(lims[1])
