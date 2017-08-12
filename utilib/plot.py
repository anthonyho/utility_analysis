# Anthony Ho <anthony.ho@energy.ca.gov>
# Last update 8/11/2017
"""
Python module for plotting customer-level energy consumption data
"""


import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Default colors for plotting
colors = sns.color_palette('Paired', 12)


# Set default translation dictionary for abbr
terms = {'elec': 'electric',
         'gas': 'gas',
         'tot': 'total'}


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
    """Plot number of building with consumption data over time"""
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
    # Set miscell properties
    setproperties(xlabel='Year',
                  ylabel='Number of buildings\nwith billing data',
                  legend=True, legend_bbox_to_anchor=(1, 1), legendloc=2,
                  fontsize=20, legendfontsize=16, **kwargs)
    return fig, plt.gca()


def get_group(df, building_type=None, cz=None, other=None):
    """Function to extract group of specific building type and/or climate
    zone and/or other attributes"""
    ind = pd.Series(True, index=df.index)
    if building_type is not None:
        if isinstance(building_type, str):
            building_type = [building_type]
        ind = ind & (df[('cis', 'building_type')].isin(building_type))
    if cz is not None:
        if np.issubdtype(type(cz), np.integer) or isinstance(cz, str):
            cz = [cz]
        cz = [str(item) for item in cz]
        ind = ind & (df[('cis', 'cz')].isin(cz))
    if other is not None:
        for key in other:
            if np.issubdtype(type(other[key]),
                             np.integer) or isinstance(other[key], str):
                value = [other[key]]
            else:
                value = other[key]
            value = [str(item) for item in value]
            ind = ind & df[key].isin(value)
    return df[ind]


def _parse_building_info(bills, info):
    """Function for parsing building info from either address or propertyID"""
    # Identify building from address or property ID
    if isinstance(info, dict):
        address = info['address'].upper()
        city = info['city'].upper()
        zipcode = str(info['zip'])[0:5]
        ind = ((bills[('cis', 'address')] == address) &
               (bills[('cis', 'city')] == city) &
               (bills[('cis', 'zip')] == zipcode))
        building = bills[ind]
    else:
        ind = (bills[('cis', 'PropertyID')] == str(info))
        building = bills[ind]
        address = building['cis']['address'].iloc[0]
        city = building['cis']['city'].iloc[0]
        zipcode = building['cis']['zip'].iloc[0]
    # Define full address
    full_addr = ', ' .join([address.title(), city.title(), zipcode])
    # Get building type and climate zone
    building_type = building[('cis', 'building_type')].iloc[0]
    cz = str(building[('cis', 'cz')].iloc[0])
    return building, full_addr, building_type, cz


def plot_heatmap_type_cz(df, list_types, list_cz, value,
                         func='mean', q=0.95,
                         cmap=None, cbar_label=None,
                         figsize=(7, 6)):
    """Plot heatmap of value grouped with building types as rows and climate
    zone as columns"""
    # Extract rows from the specified building types and climate zones
    group = get_group(df, building_type=list_types, cz=list_cz)
    # Extract relevant and columns
    data = group.loc[:, [('cis', 'cz'), ('cis', 'building_type'), value]]
    # Process df for next steps
    data.columns = data.columns.droplevel()
    data['cz'] = data['cz'].astype(int)
    # Compute according to the function specified
    data_grouped = data.groupby(['building_type', 'cz'])
    if func == 'mean':
        summary = data_grouped.mean().reset_index()
    elif func == 'percentile':
        summary = data_grouped.quantile(q).reset_index()
    else:
        summary = data_grouped.apply(func).reset_index()
    # Reshape to wide form for heatmap
    summary = summary.pivot(index='building_type',
                            columns='cz',
                            values=value[1])
    summary = summary.reindex(index=list_types)

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

    # Plot
    fig = plt.figure(figsize=figsize)
    ax_hm = fig.add_axes([0.125, 0.125, 0.62, 0.755])
    ax_cb = fig.add_axes([0.76, 0.125, 0.05, 0.755])
    sns.heatmap(summary,
                ax=ax_hm, cmap=cmap,
                cbar_ax=ax_cb, cbar_kws={'label': cbar_label})
    # Set miscell properties
    setproperties(ax=ax_hm,
                  xlabel='Climate zone', ylabel='Building type',
                  tickfontsize=16, labelfontsize=16, tight=False)
    setproperties(ax=ax_cb,
                  fontsize=16, tight=False)

    return fig, ax_hm, ax_cb


def plot_box(df, by, selection, value, min_sample_size=5,
             figsize=None, xlim=None, xlabel=None):
    """Plot boxplot of value for a particular climate zone or building type"""
    # Extract rows from the specified building types and climate zones
    group = get_group(df, other={('cis', by): selection})
    # Extract relevant rows and columns
    data = group.loc[:, [('cis', 'cz'), ('cis', 'building_type'), value]]
    # Process df for next steps
    data.columns = data.columns.droplevel()
    data = data.dropna()
    data['cz'] = data['cz'].astype(int)

    # Define variable and labels
    if by == 'cz':
        y = 'building_type'
        ylabel = 'Building type'
        title = 'Climate zone ' + str(selection)
    elif by == 'building_type':
        y = 'cz'
        ylabel = 'Climate zone'
        title = selection

    # Identify building types/climate zones with minimum sample size
    sample_size = data.groupby(y).size()
    ind_pf = sample_size[sample_size > min_sample_size].index
    # Select building types/climate zones with minimum sample size
    data = data[data[y].isin(ind_pf)]

    # Get sorted order of building types
    if by == 'cz':
        median = data.groupby(y).median()
        order = median.sort_values(by=value[1]).index
    else:
        order = None

    # Define label
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
    # Set miscell properties
    setproperties(ax=ax,
                  xlabel=xlabel, ylabel=ylabel, title=title,
                  xlim=xlim, tickfontsize=16, labelfontsize=16, tight=False)

    return fig, ax


def plot_bldg_hist(bills, info, value, histrange=None,
                   figsize=(6, 5), xlabel=None):
    """Plot histogram of value with line indicating the value of current
    building"""
    # Parse building info
    building, full_addr, building_type, cz = _parse_building_info(bills, info)
    # Extract rows from the specified building types and climate zones
    group = get_group(bills, building_type=building_type, cz=cz)
    # Get values
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

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    # num_bins = min(20, int(np.ceil(len(group_eui) / 3)))             # to fix
    ax = sns.distplot(group_eui,
                      hist_kws={'range': histrange},
                      kde_kws={'clip': histrange})
    ylim = ax.get_ylim()
    ax.plot([building_eui, building_eui], ylim, color='r', linewidth=2,
            label='Current building')
    ax.plot([group_eui_mean, group_eui_mean], ylim, color='b', linewidth=2,
            label='Group average')
    ax.text(building_eui, ylim[1] * 1.05, '{:.1f}%'.format(percentile),
            ha="center", fontsize=16)
    # Set miscell properties
    setproperties(xlabel=xlabel, ylabel='Density', title=title,
                  ylim=(ylim[0], ylim[1] * 1.15),
                  legend=True, legend_bbox_to_anchor=(1, 1), legendloc=2,
                  tickfontsize=18, labelfontsize=18, legendfontsize=16)

    return fig, ax


def plot_bldg_avg_monthly_fuel(bills, info, fuel='all', year_range=None,
                               figsize=(6, 5), ylabel=None):
    """Plot the average monthly EUI of a building by specified fuel types"""
    # Parse building info
    building, full_addr, building_type, cz = _parse_building_info(bills, info)
    # Define fuel types
    if isinstance(fuel, list):
        list_fuel = fuel
    elif fuel == 'all':
        list_fuel = ['gas', 'elec', 'tot']
    else:
        list_fuel = [fuel]
    # Define label and title
    if ylabel is None:
        ylabel = 'Average monthly EUI\nfrom 2009-2015\n(kBtu/sq. ft.)'
    title = full_addr + '\nType = ' + building_type + ', CZ = ' + cz

    # Plot
    fig = plt.figure(figsize=figsize)
    for fuel in list_fuel:
        _plot_bldg_avg_monthly_fuel_single(building, fuel, year_range)
    # Set miscell properties
    setproperties(xlabel='Month', ylabel=ylabel, title=title, xlim=(1, 12),
                  legend=True, legend_bbox_to_anchor=(1, 1), legendloc=2,
                  tickfontsize=16, labelfontsize=16, legendfontsize=16)

    return fig, plt.gca()


def _plot_bldg_avg_monthly_fuel_single(building, fuel, year_range=None):
    """Plot the average monthly EUI of a building by a specified fuel type"""
    # Extract yearly trace of the building and transform to multi-index df
    field = 'EUI_' + fuel
    bldg_all_trace = building[field].copy()
    list_yr_mo = [tuple(yr_mo.split('-')) for yr_mo in bldg_all_trace.columns]
    bldg_all_trace.columns = pd.MultiIndex.from_tuples(list_yr_mo)
    # Extract the average monthly trace of the building
    if year_range:
        start_year = int(year_range[0])
        end_year = int(year_range[1])
        list_year = [str(year) for year in range(start_year, end_year + 1)]
        field_prefix = str(start_year) + '_' + str(end_year)
        field_avg_mo = field + '_mo_avg_' + field_prefix
    else:
        field_avg_mo = field + '_mo_avg'
    bldg_mean_trace = building[field_avg_mo].iloc[0]

    # Define label and colors
    months = [int(mo) for mo in bldg_mean_trace.index]
    min_alpha = 0.2
    if fuel == 'tot':
        color_i = 0
    elif fuel == 'elec':
        color_i = 2
    elif fuel == 'gas':
        color_i = 4

    # Plot
    for i, year in enumerate(list_year):
        curr_yr_trace = bldg_all_trace[year].iloc[0]
        alpha = (1 - min_alpha) / len(list_year) * i + min_alpha
        plt.plot(months, curr_yr_trace,
                 color=colors[color_i], alpha=alpha, linewidth=2,
                 label='_nolegend_')
    plt.plot(months, bldg_mean_trace,
             color=colors[color_i + 1], linewidth=4, label=terms[fuel].title())
    plt.xticks(months)


def plot_bldg_avg_monthly_group(bills, info, fuel='tot', year_range=None,
                                figsize=(6, 5), ylabel=None):
    """Plot the average monthly EUI of a building by a specified fuel type
    compared against all other buildings in group"""
    # Parse building info
    building, full_addr, building_type, cz = _parse_building_info(bills, info)
    # Get group
    group = get_group(bills, building_type=building_type, cz=cz)

    # Define field name from fuel type and year range
    if year_range:
        start_year = str(year_range[0])
        end_year = str(year_range[1])
        field_prefix = '_' + str(start_year) + '_' + str(end_year)
    else:
        field_prefix = ''
    field_mean = ('summary', 'EUI_' + fuel + '_avg' + field_prefix)
    field_avg_mo = 'EUI_' + fuel + '_mo_avg' + field_prefix
    # Access data
    building_eui = building[field_mean].iloc[0]
    group_eui = group[field_mean]
    group_eui = group_eui[group_eui.notnull()]
    percentile = stats.percentileofscore(group_eui, building_eui)

    bldg_trace = building[field_avg_mo].iloc[0]
    group_traces = group[field_avg_mo]
    group_mean_trace = group[field_avg_mo].mean()

    # Define labels and title
    months = [int(mo) for mo in bldg_trace.index]
    if ylabel is None:
        ylabel = 'Average monthly EUI\nfrom 2009-2015\n(kBtu/sq. ft.)'
    title = full_addr + '\nType = ' + building_type + ', CZ = ' + cz

    # Plot
    fig = plt.figure(figsize=figsize)
    for (i, row) in group_traces.iterrows():
        plt.plot(months, row, color='0.9', label='_nolegend_')
    plt.plot(months, bldg_trace, color='r', linewidth=3,
             label='Current building ' + terms[fuel])
    plt.plot(months, group_mean_trace, color='b', linewidth=3,
             label='Group average ' + terms[fuel])
    plt.xticks(months)
    ax = plt.gca()
    ax.text(12.2, bldg_trace.iloc[-1], '{:.1f}%'.format(percentile),
            va="center", fontsize=16)
    # Set miscel properties
    setproperties(xlabel='Month', ylabel=ylabel, title=title, xlim=(1, 12),
                  legend=True, legend_bbox_to_anchor=(1, 1), legendloc=2,
                  tickfontsize=16, labelfontsize=16, legendfontsize=16)

    return fig, ax


def plot_bldg_full_timetrace(bills, info, fuel='all',
                             figsize=(8, 5), ylabel=None):
    """Plot the average monthly EUI of a building by specified fuel types"""
    # Parse building info
    building, full_addr, building_type, cz = _parse_building_info(bills, info)
    # Define fuel types
    if isinstance(fuel, list):
        list_fuel = fuel
    elif fuel == 'all':
        list_fuel = ['gas', 'elec', 'tot']
    else:
        list_fuel = [fuel]
    # Define label and title
    if ylabel is None:
        ylabel = 'Monthly EUI (kBtu/sq. ft.)'
    title = full_addr + '\nType = ' + building_type + ', CZ = ' + cz

    # Plot
    fig = plt.figure(figsize=figsize)
    for fuel in list_fuel:
        # Define colors
        if fuel == 'tot':
            color_i = 0
        elif fuel == 'elec':
            color_i = 2
        elif fuel == 'gas':
            color_i = 4
        # Extract data
        field = 'EUI_' + fuel
        trace = bills[field].iloc[0]
        yr_mo = pd.to_datetime(trace.index)
        plt.plot(yr_mo, trace,
                 color=colors[color_i + 1], linewidth=4,
                 label=terms[fuel].title())
    # Set miscell properties
    setproperties(xlabel='Year', ylabel=ylabel, title=title,
                  legend=True, legend_bbox_to_anchor=(1, 1), legendloc=2,
                  tickfontsize=16, labelfontsize=16, legendfontsize=16)

    return fig, plt.gca()


def plot_eui_vs_age(df, cz, figsize=(8, 7)):                           # to fix
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
        plt.plot(grp['Year'], grp['EUI_tot_avg_2009_2015'],
                 'o', label=key, alpha=0.75)
    plt.legend(loc='best')

    setproperties(xlabel='Year built / last renovated',
                  ylabel='Average annual EUI from 2009-2015\n(kBtu/sq. ft.)',
                  title='CZ = ' + str(cz),
                  tickfontsize=16, labelfontsize=16)

    return fig, plt.gca()
