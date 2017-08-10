# Anthony Ho <anthony.ho@energy.ca.gov>
# Last update 8/1/2017
"""
Python module for plotting utility data
"""


import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


colors = sns.color_palette('Paired', 12)


# Handy function for setting commonly used plot modifiers
def setproperties(fig=None, ax=None, figsize=None,
                  suptitle=None, title=None,
                  legend=None, legendloc=1, legendwidth=2.5, legendbox=None,
                  legend_bbox_to_anchor=None,
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
                            fontsize=legendfontsize, frameon=legendbox,
                            bbox_to_anchor=legend_bbox_to_anchor)
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


def plot_num_bldg_vs_time(df, field, list_iou=None,
                          figsize=(8, 6), **kwargs):
    fig = plt.figure(figsize=figsize)
    # Plot by IOUs
    if list_iou:
        for i, iou in enumerate(list_iou):
            ind = df[('cis', 'iou')].str.contains(iou)
            num_bldg = df[field][ind].notnull().sum()
            x = pd.to_datetime(num_bldg.index)
            plt.plot(x, num_bldg,
                     linewidth=3, color=colors[i * 2 + 1],
                     label=iou.upper())
    # Plot all
    num_bldg = df[field].notnull().sum()
    x = pd.to_datetime(num_bldg.index)
    plt.plot(x, num_bldg,
             linewidth=4, color='k',
             label='All')

    setproperties(xlabel='Year',
                  ylabel='Number of buildings\nwith billing data',
                  legend=True, legend_bbox_to_anchor=(1, 1), legendloc=2,
                  fontsize=20, legendfontsize=16, **kwargs)
    return fig, plt.gca()


# Plot heatmap with building types as rows and climate zone as columns
def plot_heatmap_type_cz(df, list_types, list_cz, value,
                         func='mean', q=0.95,
                         cmap=None, cbar_label=None,
                         figsize=(7, 6)):
    # Extract rows from the specified property types and climate zones
    list_cz = [str(cz) for cz in list_cz]
    ind = (df[('cis', 'building_type')].isin(list_types) &
           df[('cis', 'cz')].isin(list_cz))

    # Extract rows and columns
    data = df.loc[ind, [('cis', 'cz'), ('cis', 'building_type'), value]]
    # Process df for output
    data.columns = data.columns.droplevel()
    data['cz'] = data['cz'].astype(int)

    # Compute
    data_grouped = data.groupby(['building_type', 'cz'])
    if func == 'mean':
        summary = data_grouped.mean().reset_index()
    elif func == 'percentile':
        summary = data_grouped.quantile(q).reset_index()

    # Reshape
    summary = summary.pivot(index='building_type',
                            columns='cz',
                            values=value[1])
    summary = summary.reindex(index=list_types)

    # Plot
    fig = plt.figure(figsize=figsize)
    ax_hm = fig.add_axes([0.125, 0.125, 0.62, 0.755])
    ax_cb = fig.add_axes([0.76, 0.125, 0.05, 0.755])
    # Define cbar label
    if cbar_label is None:
        if 'fit' in value[1]:
            label_suffix = 'change in annual EUI\n(kBtu/ft2/year)'
        elif 'avg' in value[1]:
            label_suffix = 'average annual EUI\n(kBtu/ft2)'
        else:
            label_suffix = ''
        if func == 'mean':
            label_prefix = 'Mean of '
        elif func == 'percentile':
            label_prefix = '{:d}th percentile of '.format(int(q * 100))
        else:
            label_prefix = ''
        cbar_label = label_prefix + label_suffix
    # Define colormap
    if cmap is None:
        if 'fit' in value[1]:
            cmap = 'RdYlBu_r'
        elif 'avg' in value[1]:
            cmap = 'YlOrRd'

    sns.heatmap(summary,
                ax=ax_hm, cmap=cmap,
                cbar_ax=ax_cb, cbar_kws={'label': cbar_label})
    setproperties(ax=ax_hm,
                  xlabel='Climate zone', ylabel='Building type',
                  tickfontsize=16, labelfontsize=16, tight=False)
    setproperties(ax=ax_cb,
                  fontsize=16, tight=False)

    return fig, ax_hm, ax_cb


def plot_box(df, by, selection, value, min_sample_size=5,
             figsize=None, xlim=None, xlabel=None):
    # Extract rows from the specified property types and climate zones
    selection = str(selection)
    ind = (df[('cis', by)] == selection)
    # Extract rows and columns
    data = df.loc[ind, [('cis', 'cz'), ('cis', 'building_type'), value]]
    # Process df for output
    data.columns = data.columns.droplevel()
    data = data.dropna()
    data['cz'] = data['cz'].astype(int)

    # Define variables
    if by == 'cz':
        y = 'building_type'
        ylabel = 'Building type'
        title = 'Climate zone ' + selection
    elif by == 'building_type':
        y = 'cz'
        ylabel = 'Climate zone'
        title = selection

    # Cut off
    sample_size = data.groupby(y).size()
    ind_pf = sample_size[sample_size > min_sample_size].index
    # Select types within min_counts
    data = data[data[y].isin(ind_pf)]

    # Get sorted order
    if by == 'cz':
        median = data.groupby(y).median()
        order = median.sort_values(by=value[1]).index
    else:
        order = None

    # Define xlabel
    if xlabel is None:
        if 'fit' in value[1]:
            xlabel = 'Change in annual EUI from 2009-2015\n(kBtu/ft2/year)'
        elif 'avg' in value[1]:
            xlabel = 'Average annual EUI from 2009-2015 \n(kBtu/ft2)'

    # Plot
    if figsize is None:
        height = len(ind_pf)
        figsize = (8, height * 0.4)
    fig = plt.figure(figsize=figsize)
    ax = sns.boxplot(x=value[1], y=y, data=data,
                     order=order,
                     orient='h', color=colors[8])

    setproperties(ax=ax,
                  xlabel=xlabel, ylabel=ylabel, title=title,
                  xlim=xlim, tickfontsize=16, labelfontsize=16, tight=False)

    return fig, ax


def plot_building_avg_monthly(bills, info, figsize=(6, 5)):
    if isinstance(info, dict):
        address = info['address'].upper()
        city = info['city'].upper()
        zipcode = str(info['zip'])[0:5]
        ind = ((bills[('cis', 'address')] == address) &
               (bills[('cis', 'city')] == city) &
               (bills[('cis', 'zip')] == zipcode))
        full_addr = ',' .join([info['address'], info['city'], zipcode])  # to fix
    else:
        ind = (bills[('cis', 'PropertyID')] == str(info))
        full_addr = str(info)   # to fix

    building = bills[ind]
    property_type = building[('cis', 'PropertyType')].iloc[0]
    secondary_type = building[('cis', 'Secondary Type')].iloc[0]
    cz = building[('cis', 'cz')].iloc[0]

    group_ind = ((bills[('cis', 'PropertyType')] == property_type) &
                 (bills[('cis', 'Secondary Type')] == secondary_type) &
                 (bills[('cis', 'cz')] == cz))
    group = bills[group_ind]

    bldg_trace = building['EUI_tot_mo_avg_2009_2015'].iloc[0]
    group_traces = group['EUI_tot_mo_avg_2009_2015']
    group_mean_trace = group['EUI_tot_mo_avg_2009_2015'].mean()

    months = [int(mo) for mo in bldg_trace.index]

    fig = plt.figure(figsize=figsize)
    for (i, row) in group_traces.iterrows():
        plt.plot(months, row, color='0.7')
    plt.plot(months, bldg_trace, color='r', linewidth=3)
    plt.plot(months, group_mean_trace, color='b', linewidth=3)
    plt.xticks(months)

    setproperties(xlabel='Month',
                  ylabel='Average monthly EUI\nfrom 2009-2015\n(kBtu/sq. ft.)',
                  title='Building: ' + full_addr + '\nType = ' + property_type + ' - ' + secondary_type + ', CZ = ' + str(cz),
                  tickfontsize=16, labelfontsize=16)

    return fig


#def plot_


def _parse_building_info(bills, info):
    if isinstance(info, dict):
        address = info['address'].upper()
        city = info['city'].upper()
        zipcode = str(info['zip'])[0:5]
        ind = ((bills[('cis', 'address')] == address) &
               (bills[('cis', 'city')] == city) &
               (bills[('cis', 'zip')] == zipcode))
        full_addr = ', ' .join([info['address'], info['city'], zipcode])  # to fix cases
    else:
        ind = (bills[('cis', 'PropertyID')] == str(info))
        full_addr = str(info)                                             # to fix for actual address
    building = bills[ind]
    building_type = building[('cis', 'building_type')].iloc[0]
    cz = str(building[('cis', 'cz')].iloc[0])
    return building, full_addr, building_type, cz


def plot_bldg_hist(bills, info, value, histrange=None,
                   figsize=(6, 5), xlabel=None):
    # Parse building info
    building, full_addr, building_type, cz = _parse_building_info(bills, info)
    # Get group
    group_ind = ((bills[('cis', 'building_type')] == building_type) &
                 (bills[('cis', 'cz')] == cz))
    group = bills[group_ind]

    building_eui = building[value].iloc[0]
    group_eui = group[value]
    group_eui = group_eui[group_eui.notnull()]
    group_eui_mean = group_eui.mean()
    percentile = stats.percentileofscore(group_eui, building_eui)

    # Define xlabel and title
    if xlabel is None:
        if 'fit' in value[1]:
            xlabel = 'Change in annual EUI from 2009-2015\n(kBtu/ft2/year)'
        elif 'avg' in value[1]:
            xlabel = 'Average annual EUI from 2009-2015 \n(kBtu/ft2)'
    title = full_addr + '\nType = ' + building_type + ', CZ = ' + cz

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    # num_bins = min(20, int(np.ceil(len(group_eui) / 3)))                  # to fix
    ax = sns.distplot(group_eui,
                      hist_kws={'range': histrange},
                      kde_kws={'clip': histrange})
    ylim = ax.get_ylim()
    ax.plot([building_eui, building_eui], ylim, color='r', linewidth=2)
    ax.plot([group_eui_mean, group_eui_mean], ylim, color='b', linewidth=2)
    ax.text(building_eui, ylim[1] * 1.05, '{:.1f}%'.format(percentile),
            ha="center", fontsize=16)

    setproperties(xlabel=xlabel, ylabel='Density', title=title,
                  ylim=(ylim[0], ylim[1] * 1.15),
                  tickfontsize=18, labelfontsize=18)

    return fig, ax


def plot_eui_vs_age(df, cz, figsize=(8, 7)):
    data = df[[('cis', 'Year Built'),
               ('cis', 'Year Renovated'),
               ('cis', 'PropertyType'),
               ('cis', 'cz'),
               ('summary', 'EUI_tot_avg_2009_2015')]]
    data = data[data[('cis', 'cz')] == str(cz)]
    data.columns = data.columns.droplevel()
    data['Year'] = data[['Year Built', 'Year Renovated']].max(axis=1)
    data = data.drop('Year Built', axis=1)
    data = data.drop('Year Renovated', axis=1)
    data = data[data['Year'].notnull()]
    data['Year'] = data['Year'].astype(int)

    fig = plt.figure(figsize=figsize)
    for key, grp in data.groupby('PropertyType'):
        plt.plot(grp['Year'], grp['EUI_tot_avg_2009_2015'], 'o', label = key, alpha=0.75)
    plt.legend(loc = 'best')
    
    setproperties(xlabel='Year built / last renovated',
                  ylabel='Average annual EUI from 2009-2015\n(kBtu/sq. ft.)',
                  title='CZ = ' + str(cz),
                  tickfontsize=16, labelfontsize=16)

    return
