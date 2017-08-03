# Anthony Ho <anthony.ho@energy.ca.gov>
# Last update 8/1/2017
"""
Python module for plotting utility data
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


colors = sns.color_palette('Paired', 12)


# Handy function for setting commonly used plot modifiers
def setproperties(fig=None, ax=None, figsize=None,
                  suptitle=None, title=None,
                  legend=None, legendloc=1, legendwidth=2.5, legendbox=None,
                  xlabel=None, ylabel=None, xlim=None, ylim=None,
                  scix=False, sciy=False,
                  scilimitsx=(-3, 3), scilimitsy=(-3, 3),
                  logx=False, logy=False, majorgrid=None, minorgrid=None,
                  borderwidth=2.5, tight=True, pad=1.6,
                  fontsize=None, legendfontsize=20, tickfontsize=20,
                  labelfontsize=20, titlefontsize=18, suptitlefontsize=20,
                  xticklabelrot=None, yticklabelrot=None,
                  equal=False, symmetric=False):
    """Convenient tool to set properties of a plot in a single command"""
    # Get figure and axis handles
    if not fig:
        fig = plt.gcf()
    if not ax:
        ax = plt.gca()

    # Set background color to white
    fig.patch.set_facecolor('w')

    # Define figure size if provided
    if figsize:
        fig.set_size_inches(figsize, forward=True)

    # Set titles if provided
    if suptitle is not None:
        if fontsize is None:
            fig.suptitle(suptitle, fontsize=suptitlefontsize, y=0.99)
        else:
            fig.suptitle(suptitle, fontsize=fontsize, y=0.99)
    if title is not None:
        ax.set_title(title, y=1.02)
    # Show legend if requested
    if legend:
        legend = plt.legend(loc=legendloc, numpoints=1,
                            fontsize=legendfontsize, frameon=legendbox)
        legend.get_frame().set_linewidth(legendwidth)
    # Set x and y labels if provided
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # Set x and y limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # Apply scientific notation to x and y tick marks if requested
    if scix:
        ax.ticklabel_format(axis='x', style='sci', scilimits=scilimitsx)
    if sciy:
        ax.ticklabel_format(axis='y', style='sci', scilimits=scilimitsy)
    # Change axis to log scale if requested
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    # Set major and minor grid in plot
    if majorgrid is not None:
        ax.grid(b=majorgrid, which='major')
    if minorgrid is not None:
        ax.grid(b=minorgrid, which='minor')

    # Rotate x and y tick labels if requested
    if xticklabelrot is not None:
        xticklabels = ax.get_xticklabels()
        for ticklabel in xticklabels:
            ticklabel.set_rotation(xticklabelrot)
    if yticklabelrot is not None:
        yticklabels = ax.get_yticklabels()
        for ticklabel in yticklabels:
            ticklabel.set_rotation(yticklabelrot)

    # Set borderwidth (not visible if using seaborn default theme)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(borderwidth)

    # Set individual fontsizes if fontsize is not specified
    if fontsize is None:
        plt.setp(ax.get_xticklabels(), fontsize=tickfontsize)
        plt.setp(ax.get_yticklabels(), fontsize=tickfontsize)
        ax.xaxis.label.set_fontsize(labelfontsize)
        ax.yaxis.label.set_fontsize(labelfontsize)
        ax.title.set_fontsize(titlefontsize)
    # Set all fontsizes to fontsize if fontsize is specified
    else:
        plt.setp(ax.get_xticklabels(), fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)
        ax.title.set_fontsize(fontsize)

    # Set tight figure and padding
    if tight:
        fig.tight_layout(pad=pad)

    # Set equal aspect
    if equal:
        ax.set_aspect('equal', adjustable='box')

    # Set symmetric axis limits
    if symmetric:
        xlim_abs = max(abs(i) for i in ax.get_xlim())
        ylim_abs = max(abs(i) for i in ax.get_ylim())
        ax.set_xlim((-xlim_abs, xlim_abs))
        ax.set_ylim((-ylim_abs, ylim_abs))


def plot_heatmap_type_cz(df, types, cz, value,
                         func='mean', q=0.95,
                         figsize=(6, 3), cbar_label=None):
    # Extract rows from the specified property types
    ind = pd.Series([False] * len(df))
    for type_tuple in types:
        property_type = type_tuple[0]
        secondary_type = type_tuple[1]
        in_type = ((df[('cis', 'PropertyType')] == property_type) &
                   (df[('cis', 'Secondary Type')] == secondary_type))
        ind = ind | in_type
    # Extract rows from the specified climate zones
    cz = [str(i) for i in cz]
    ind = ind & df[('cis', 'cz')].isin(cz)
    # Extract rows
    data = df.loc[ind,
                  [('cis', 'cz'),
                   ('cis', 'PropertyType'),
                   ('cis', 'Secondary Type'),
                   value]
                  ]
    # Process df for output
    data.columns = data.columns.droplevel()
    data['Property type'] = data['PropertyType']+' - '+data['Secondary Type']
    data = data.drop('PropertyType', axis=1)
    data = data.drop('Secondary Type', axis=1)
    data['cz'] = data['cz'].astype(int)

    # Compute
    data_grouped = data.groupby(['Property type', 'cz'])
    if func == 'mean':
        summary = data_grouped.mean().reset_index()
    elif func == 'percentile':
        summary = data_grouped.quantile(q).reset_index()

    # Reshape
    summary = summary.pivot(index='Property type',
                            columns='cz',
                            values=value[1])

    # Plot
    fig = plt.figure(figsize=figsize)
    ax_hm = fig.add_axes([0.125, 0.125, 0.62, 0.755])
    ax_cb = fig.add_axes([0.76, 0.125, 0.05, 0.755])
    # Define cbar label
    if cbar_label is None:
        if func == 'mean':
            cbar_label = 'Average annual EUI\n(kBtu/sq. ft.)'
        elif func == 'percentile':
            text = '{:d}th percentile annual EUI\n(kBtu/sq. ft.)'
            cbar_label = text.format(int(q * 100))

    sns.heatmap(summary,
                ax=ax_hm, cbar_ax=ax_cb, cbar_kws={'label': cbar_label})
    setproperties(ax=ax_hm,
                  xlabel='Climate zone', ylabel='Property type',
                  tickfontsize=16, labelfontsize=16, tight=False)
    setproperties(ax=ax_cb,
                  fontsize=16, tight=False)

    return fig, ax_hm, ax_cb


def plot_box_cz(df, cz, value, min_counts=10, types=None,
                figsize=None):
    # Extract rows from the specified property types
    if types:
        ind = pd.Series([False] * len(df))
        for type_tuple in types:
            property_type = type_tuple[0]
            secondary_type = type_tuple[1]
            in_type = ((df[('cis', 'PropertyType')] == property_type) &
                       (df[('cis', 'Secondary Type')] == secondary_type))
            ind = ind | in_type
    else:
        ind = pd.Series([True] * len(df))
    # Extract rows from the specified climate zone
    cz = str(cz)
    ind = ind & (df[('cis', 'cz')] == cz)
    data = df.loc[ind,
                  [('cis', 'PropertyType'),
                   ('cis', 'Secondary Type'),
                   value]
                  ]
    # Process df for output
    data.columns = data.columns.droplevel()
    data['Property type'] = data['PropertyType']+' - '+data['Secondary Type']
    data = data.drop('PropertyType', axis=1)
    data = data.drop('Secondary Type', axis=1)

    # Cut off
    if min_counts:
        counts = data.groupby('Property type').size()
        types_pf = counts[counts > min_counts].index
    # Select types within min_counts
    data = data[data['Property type'].isin(types_pf)]

    # Get sorted order
    types_order = data.groupby('Property type').median().sort_values(by=value[1]).index

    # Plot
    if figsize is None:
        height = len(types_pf)  # fix bug when min_counts = 0
        figsize = (8, height * 0.6)
    fig = plt.figure(figsize=figsize)
    ax = sns.boxplot(x=value[1], y='Property type', data=data,
                     order=types_order,
                     orient='h', color=colors[8])

    setproperties(ax=ax,
                  xlabel='Average annual EUI from 2009-2015\n(kBtu/sq. ft.)',
                  ylabel='Property type',
                  title='Climate zone '+cz,
                  tickfontsize=16, labelfontsize=16, tight=False)

    return fig, ax
