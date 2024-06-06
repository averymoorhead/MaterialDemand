import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.lines import Line2D
import math

from selenium import webdriver                                      
driver = webdriver.Chrome()

# %% user inputs and import data

input_dir = r'Inputs\\'
r_outputs_dir = r'Outputs\\'
fig_outputs_dir = r'Figures\\'

input_usgs = f'{input_dir}risk_charts_inputs.xlsx'

input_usgs_import_shares = pd.read_excel(input_usgs, sheet_name = 'import_shares')
input_usgs_import_dependency = pd.read_excel(input_usgs, sheet_name = 'import_dependency').set_index('material')
input_usgs_production = pd.read_excel(input_usgs, sheet_name = 'production', index_col = 0)
input_usgs_reserves = pd.read_excel(input_usgs, sheet_name = 'reserves', index_col = 0)
input_crc = pd.read_excel(input_usgs, sheet_name = 'crc').iloc[:, 0:2]
input_aggregate = pd.read_excel(input_usgs, sheet_name = 'aggregate').replace('-', float('nan'))

materials_demand = pd.read_csv(f'{r_outputs_dir}materials_demand.csv').replace({'Ref' : 'REF'})
repeat_materials_demand = pd.read_csv(f'{r_outputs_dir}repeat_materials_demand.csv')
spur_dist_sum = pd.read_csv(f'{r_outputs_dir}spur_dist_sum.csv')

rare_earths = ['Dysprosium', 'Neodymium', 'Praseodymium', 'Terbium', 'Yttrium']
    
# %% format input df from multi-model (mm) materials demand projection script

mat_dem_mm_group = materials_demand.groupby(
    ['year', 'scenario']).sum(numeric_only = True)

mat_dem_mm_group = mat_dem_mm_group.filter(like='mean', axis=1)
mat_dem_mm_group.columns = mat_dem_mm_group.columns.str.replace('_mean','', regex = True)

mat_dem_mm_group = mat_dem_mm_group.loc[:, (mat_dem_mm_group != 0).any(axis=0)] # Drop unused materials

mat_dem_mm_group_stack = pd.DataFrame()

for col in mat_dem_mm_group.columns:
    data = mat_dem_mm_group[[col]].reset_index().rename(columns = {col : 'value'})
    data['material'] = col
    mat_dem_mm_group_stack = pd.concat([mat_dem_mm_group_stack, data])
    
mat_dem_mm_group_stack_rare = mat_dem_mm_group_stack.loc[mat_dem_mm_group_stack['material'].isin(rare_earths)]
mat_dem_mm_group_stack_rare = mat_dem_mm_group_stack_rare.groupby(['year', 'scenario']).sum('value').reset_index()
mat_dem_mm_group_stack_rare['material'] = 'Rare Earths'
mat_dem_mm_group_stack = pd.concat([mat_dem_mm_group_stack, mat_dem_mm_group_stack_rare])

# %% format input df from REPEAT materials demand projection script

mat_dem_repeat_group = repeat_materials_demand.groupby(
    ['year', 'scenario']).sum(numeric_only = True)

mat_dem_repeat_group = mat_dem_repeat_group.filter(like='mean', axis=1)
mat_dem_repeat_group.columns = mat_dem_repeat_group.columns.str.replace('_mean','', regex = True)

mat_dem_repeat_group = mat_dem_repeat_group.loc[:, (mat_dem_repeat_group != 0).any(axis=0)] # Drop unused materials

mat_dem_repeat_group_stack = pd.DataFrame()

for col in mat_dem_repeat_group.columns:
    data = mat_dem_repeat_group[[col]].reset_index().rename(columns = {col : 'value'})
    data['material'] = col
    mat_dem_repeat_group_stack = pd.concat([mat_dem_repeat_group_stack, data])
    
mat_dem_repeat_group_stack_rare = mat_dem_repeat_group_stack.loc[mat_dem_repeat_group_stack['material'].isin(rare_earths)]
mat_dem_repeat_group_stack_rare = mat_dem_repeat_group_stack_rare.groupby(['year', 'scenario']).sum('value').reset_index()
mat_dem_repeat_group_stack_rare['material'] = 'Rare Earths'
mat_dem_repeat_group_stack = pd.concat([mat_dem_repeat_group_stack, mat_dem_repeat_group_stack_rare])

# %% format input df from REPEAT spur line materials demand projection script

mat_dem_repeat_spur = spur_dist_sum.groupby(
    ['year', 'scenario']).sum(numeric_only = True).rename(columns = {'Cu' : 'Copper'}).drop(columns = {'totalspurdistance'})

mat_dem_repeat_spur_stack = pd.DataFrame()

for col in mat_dem_repeat_spur.columns:
    data = mat_dem_repeat_spur[[col]].reset_index().rename(columns = {col : 'value'})
    data['material'] = col
    data['value'] = data['value'] / 1000 # Conversion to Thousand Metric ton
    mat_dem_repeat_spur_stack = pd.concat([mat_dem_repeat_spur_stack, data])
    
mat_dem_repeat_spur_stack = pd.concat([mat_dem_repeat_group_stack, mat_dem_repeat_spur_stack])

mat_dem_repeat_spur_stack = mat_dem_repeat_spur_stack.groupby(['year', 'scenario', 'material']).sum('value').reset_index()

# %% format input dfs for import shares/dependency

import_dependency = input_usgs_import_dependency.replace({'E' : 0})
import_dependency['avg'] = import_dependency.mean(axis=1)

import_shares = pd.merge(left = input_usgs_import_shares, right = input_crc[['country', 'crc']], 
                         left_on = 'country', right_on = 'country', how = 'left').replace(np.nan, 'Undefined')

import_shares['crc'] = np.where(import_shares['country'] == 'China', 'China', import_shares['crc'])
    
import_shares_crc = import_shares.groupby(['material', 'crc']).sum('share').reset_index()

# %% format reserves df

reserves = input_usgs_reserves[1:].replace('-', np.nan).dropna(axis=1)
reserves.columns = reserves.columns.str.replace('.2','', regex = True)

reserves_stack = pd.DataFrame(columns = ['material', 'share'])

for mat in reserves.columns:
    data = pd.DataFrame(reserves[mat]).rename(columns = {mat:'share'})
    data['material'] = mat
    reserves_stack = pd.concat([reserves_stack, data])
    
reserves_stack = reserves_stack.reset_index(drop = False).rename(columns = {'index':'country'})
reserves_stack = pd.merge(left = reserves_stack, right = input_crc[['country', 'crc']], 
                         left_on = 'country', right_on = 'country', how = 'left').replace(np.nan, 'Undefined')

reserves_stack['crc'] = np.where(reserves_stack['country'] == 'United States', 
                                 'United States', reserves_stack['crc'])

reserves_stack['crc'] = np.where(reserves_stack['country'] == 'China', 'China', reserves_stack['crc'])
    
reserves_crc = reserves_stack.groupby(['material', 'crc']
                                      ).sum().reset_index().set_index('material')

reserves_crc_group = pd.DataFrame(reserves_stack.groupby(['material'])['share'].sum())

reserves_crc['share'] = reserves_crc['share'] / reserves_crc_group['share'] * 100
reserves_crc = reserves_crc.reset_index()

# %% format production df
production = input_usgs_production[2:].replace('-', 0)

production_stack = pd.DataFrame(columns = ['material', 'share'])

for mat in production[production.columns[::2]]:
    data = pd.DataFrame(production[mat] + production[f'{mat}.1']).rename(columns = {0 : 'share'})
    data['share'], data['material'] = data['share'] / 2, mat #average of 2021 and 2022 values

    if data['share'].sum() >0: #Filters out entries with no production data
        production_stack = pd.concat([production_stack, data])
    
production_stack = production_stack.reset_index(drop = False).rename(columns = {'index':'country'})    
production_stack = pd.merge(left = production_stack, right = input_crc[['country', 'crc']], 
                         left_on = 'country', right_on = 'country', how = 'left').replace(np.nan, 'Undefined')

production_stack['crc'] = np.where(production_stack['country'] == 'United States', 
                                 'United States', production_stack['crc'])


production_stack['crc'] = np.where(production_stack['country'] == 'China', 'China', production_stack['crc'])
    
production_crc = production_stack.groupby(['material', 'crc']).sum().reset_index().set_index('material')
production_crc_group = pd.DataFrame(production_stack.groupby(['material'])['share'].sum())

production_crc['share'] = production_crc['share'] / production_crc_group['share'] * 100
production_crc = production_crc.reset_index()

# %% add average values to US specific USGS data

input_aggregate_avg = pd.DataFrame(columns = input_aggregate.columns)

for mat in input_aggregate['material'].unique():
    data = input_aggregate.loc[input_aggregate['material'] == mat]
    data = data.groupby(['material'], dropna = False).mean().reset_index()
    data['year'] = 'avg'
    input_aggregate_avg = pd.concat([input_aggregate_avg, data])

input_aggregate_avg = input_aggregate_avg.reset_index(drop = True)

# %% add all crc categories to dfs

def crc_fill(data): # function to fill all crc categories
    
    crc_cat = ['OECD',1,2,3,4,5,6,7,'China', 'Undefined', 'United States']
    df = pd.MultiIndex.from_product([list(data['material'].unique()), crc_cat], names= ['material', 'crc'])
    data = data.set_index(['material', 'crc']).reindex(df, fill_value = 0).reset_index()
    return data
    
import_shares_crc = crc_fill(import_shares_crc)
reserves_crc = crc_fill(reserves_crc)
production_crc = crc_fill(production_crc)

# %% format df for combination plot multi-model material demand projections and CRC

mat_dem_mm_crc_group = pd.merge(import_shares_crc, import_dependency[['avg']], left_on = 'material', right_index = True)
mat_dem_mm_crc_group['share'] = mat_dem_mm_crc_group['share'] * ((mat_dem_mm_crc_group['avg']) / 100)
mat_dem_mm_crc_group['share'] = np.where(mat_dem_mm_crc_group['crc'] == 'United States', 
                                 100 - mat_dem_mm_crc_group['avg'], mat_dem_mm_crc_group['share'])

mat_dem_mm_crc_group = pd.merge(mat_dem_mm_crc_group, mat_dem_mm_group_stack, on = 'material')
mat_dem_mm_crc_group['value'] = mat_dem_mm_crc_group['value'] * (mat_dem_mm_crc_group['share'] / 100)

# %% format df for combination plot REPEAT material demand projections and CRC

mat_dem_repeat_crc_group = pd.merge(import_shares_crc, import_dependency[['avg']], left_on = 'material', right_index = True)
mat_dem_repeat_crc_group['share'] = mat_dem_repeat_crc_group['share'] * ((mat_dem_repeat_crc_group['avg']) / 100)
mat_dem_repeat_crc_group['share'] = np.where(mat_dem_repeat_crc_group['crc'] == 'United States', 
                                 100 - mat_dem_repeat_crc_group['avg'], mat_dem_repeat_crc_group['share'])

mat_dem_repeat_crc_group = pd.merge(mat_dem_repeat_crc_group, mat_dem_repeat_group_stack, on = 'material')
mat_dem_repeat_crc_group['value'] = mat_dem_repeat_crc_group['value'] * (mat_dem_repeat_crc_group['share'] / 100)

# %% format df for combination plot REPEAT + spur lines material demand projections and CRC

mat_dem_repeat_spur_crc_group = pd.merge(import_shares_crc, import_dependency[['avg']], left_on = 'material', right_index = True)
mat_dem_repeat_spur_crc_group['share'] = mat_dem_repeat_spur_crc_group['share'] * ((mat_dem_repeat_spur_crc_group['avg']) / 100)
mat_dem_repeat_spur_crc_group['share'] = np.where(mat_dem_repeat_spur_crc_group['crc'] == 'United States', 
                                 100 - mat_dem_repeat_spur_crc_group['avg'], mat_dem_repeat_spur_crc_group['share'])

mat_dem_repeat_spur_crc_group = pd.merge(mat_dem_repeat_spur_crc_group, mat_dem_repeat_spur_stack, on = 'material')
mat_dem_repeat_spur_crc_group['value'] = mat_dem_repeat_spur_crc_group['value'] * (mat_dem_repeat_spur_crc_group['share'] / 100)

# %% crc plots (SI - no material demand projections included)

category_names = ['OECD', 1, 2, 3, 4, 5, 6, 7, 'China', 'Undefined']
category_names_us = ['OECD', 1, 2, 3, 4, 5, 6, 7, 'China', 'Undefined', 'United States']

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["green","green", "yellow","orange", "red", "brown", "grey", "purple"])

import_shares_plot_dict = dict()
for material in import_shares_crc['material'].unique():
    import_shares_plot_dict[material] = import_shares_crc.loc[import_shares_crc[
        'material'] == material, 'share'].values.T.tolist()

reserves_plot_dict = dict()
for material in reserves_crc['material'].unique():
    reserves_plot_dict[material] = reserves_crc.loc[reserves_crc[
        'material'] == material, 'share'].values.T.tolist()
    
production_plot_dict = dict()
for material in production_crc['material'].unique():
    production_plot_dict[material] = production_crc.loc[production_crc[
        'material'] == material, 'share'].values.T.tolist()

def crc_plot(data, cat_names, cmap):

    labels = list(data.keys())
    df = np.array(list(data.values()))
    df_cum = df.cumsum(axis=1)
    category_colors = plt.get_cmap(cmap)(
        np.linspace(0.15, 0.95, df.shape[1]))

    fig, ax = plt.subplots(figsize=(12, 11))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(df, axis=1).max())
    ax.margins(y=0.01)

    for i, (colname, color) in enumerate(zip(cat_names, category_colors)):
        widths = df[:, i]
        starts = df_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if c > 1:
                ax.text(x, y, str(int(c)), ha='center', va='center',
                        color='black')
    ax.legend(ncol=len(cat_names), bbox_to_anchor=(0.47,1.05, 0, 0),
              loc='upper center', frameon = False)

    return fig

crc_plot(import_shares_plot_dict , category_names, cmap).savefig(
    f'{fig_outputs_dir}figSI_import_shares_crc.png', dpi = 1000, bbox_inches='tight')

crc_plot(reserves_plot_dict , category_names_us, cmap).savefig(
    f'{fig_outputs_dir}figSI_reserves_shares_crc.png', dpi = 1000, bbox_inches='tight')

crc_plot(production_plot_dict , category_names_us, cmap).savefig(
    f'{fig_outputs_dir}figSI_production_shares_crc.png', dpi = 1000, bbox_inches='tight')

# %% net import plot

data = import_dependency.sort_values(by=['avg'])
data_error = data.iloc[:, 0:5].sub(data['avg'], axis = 0)
data_error['min'], data_error['max'] = data_error.min(axis=1), data_error.max(axis=1)
data = pd.merge(data, data_error[['min', 'max']], left_index = True, right_index = True)

plt.figure(figsize = (12, 11))
plt.barh(data.index, data['avg'], align='center',
         alpha=0.9, color = 'red', xerr = (data['min'] * -1, data['max']))
 
plt.barh(data.index, (100 - data['avg']) * -1, align='center', 
         alpha=0.6, color = 'green', xerr = (data['min'] * -1, data['max']))

plt.grid(linewidth = 0.1)
plt.xlabel('%')

plt.margins(y=0.01, x = 0.01)

plt.savefig(f'{fig_outputs_dir}figSI_net_import_shares.png', dpi = 1000, bbox_inches='tight')
    
# %% combination plot material demand projections and CRC (production/consumption/net imports).

# 1 Multi-model log scale
# 2 Multi-model linear
# 3 REPEAT log scale
# 4 REPEAT linear
# 5 REPEAT + Spur lines log scale
# 6 REPEAT + Spur lines linear

col_dict = {'United States' : (0.50196078, 0.16535179, 0.50196078, 1. ),
            'OECD' : (0.04313725, 0.52344483, 0.        , 1.),
            1 : (0.59215686, 0.79687812, 0.        , 1),
            2 : (1.        , 0.94048443, 0.        , 1),
            3 : (1.        , 0.7467128 , 0.        , 1),
            4 : (1.        , 0.4567474 , 0.        , 1.),
            5 : (1.        , 0.10149942, 0.        , 1.        ),
            6 : (0.85190311, 0.06911188, 0.06911188, 1.        ),
            7 : (0.65813149, 0.15953864, 0.15953864, 1.        ),
            'China' : (0.56796617, 0.34854287, 0.34854287, 1.        ),
            'Undefined' : (0.50196078, 0.45471742, 0.50196078, 1.        ),
            }

input_dfs = {'multi_model' : mat_dem_mm_crc_group, 'repeat' : mat_dem_repeat_crc_group, 
             'repeat_spur_lines' : mat_dem_repeat_spur_crc_group}

ytypes = ['log',
          'linear'
          ]

for output_name, input_df in input_dfs.items():
    for ytype in ytypes:
        
        data = input_df.copy()
        data['stack'], data['value'] = data['year'].astype(str) + '-' + data['scenario'], data['value']
        
        cols = 3
        entries = len(data['material'].unique())
        rows = int(np.ceil(entries / cols))
        
        fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(12, 12), sharex = True)
        axs = axs.ravel()
        
        n=0
        
        for mat in data['material'].unique():
            for scen in ['IRA', 'REF']:
                bottom = 0
                            
                if scen == 'IRA':
                    offset = -0.47
                    alpha = 1
                else:
                    offset = 0.47
                    alpha = 0.5
                
                plot_loc = data.loc[(data['material'] == mat) & (data['scenario'] == scen)
                                    ]
                plot_loc = pd.pivot_table(plot_loc, index = ['year'], columns = 'crc', 
                                          values = 'value')
                
                plot_loc = plot_loc[list(col_dict.keys())]
                
                
                axs[n].set_title(mat)
                axs[n].set_yscale(ytype)
                
                for col in plot_loc.columns:
                    if scen == 'IRA':
                        label = col
                        
                    else:
                        if col == 'United States':
                            #label = 'REF'
                            label = ''
                        else:
                            label = ''
                        
                    axs[n].bar(x = plot_loc.index + offset, height = plot_loc[col], 
                            bottom = bottom, color = col_dict.get(col), alpha = alpha, label = label)
     
                    bottom = bottom + plot_loc[col]
                    
            prod = input_aggregate_avg['production'].loc[(input_aggregate_avg['material'] == mat) & (
                input_aggregate_avg['year'] == 'avg')].iloc[0]
            
            cons = input_aggregate_avg['consumption'].loc[(input_aggregate_avg['material'] == mat) & (
                input_aggregate_avg['year'] == 'avg')].iloc[0]
            
            trade = input_aggregate_avg['net_import'].loc[(input_aggregate_avg['material'] == mat) & (
                input_aggregate_avg['year'] == 'avg')].iloc[0]
    
            if math.isnan(prod) == False:    
                axs[n].axhline(y = prod, color = 'red')
                axs[n].axhline(y = cons, color = 'blue')
                axs[n].axhline(y = trade, color = 'green')
            n = n + 1
        
        handles, labels = axs[0].get_legend_handles_labels()
        prod_leg = Line2D([0], [0], color= 'red', linewidth=1.5, linestyle='-')
        cons_leg = Line2D([0], [0], color= 'blue', linewidth=1.5, linestyle='-')
        trade_leg = Line2D([0], [0], color= 'green', linewidth=1.5, linestyle='-')
        
        handles.append(prod_leg), handles.append(cons_leg), handles.append(trade_leg)
        labels.append('Production'), labels.append('Consumption'), labels.append('Net Import')
        
        fig.legend(handles, labels, loc='lower center', ncols = 5, bbox_to_anchor = [0.18,0.045,1,1], frameon=False)
        fig.text(0, 0.5, 'Thousand Metric ton/year', va='center', rotation='vertical')
        plt.tight_layout()
        
        # adds x axis labels/ticks for the shared x axis. Haven't figured out to
        plt.xticks(np.arange(2025, 2040, 5))
    
        for ax in axs.flat[n:]:
            ax.remove()
            
        plt.savefig(f'{fig_outputs_dir}fig3_{output_name}_{ytype}.png', dpi = 1000)
                
# %% combination plot material demand projections and CRC (US and Global economic reserves).

# 1 Multi-model linear
# 2 REPEAT linear
# 3 REPEAT + Spur lines linear

col_dict = {'United States' : (0.50196078, 0.16535179, 0.50196078, 1. ),
            'OECD' : (0.04313725, 0.52344483, 0.        , 1.),
            1 : (0.59215686, 0.79687812, 0.        , 1),
            2 : (1.        , 0.94048443, 0.        , 1),
            3 : (1.        , 0.7467128 , 0.        , 1),
            4 : (1.        , 0.4567474 , 0.        , 1.),
            5 : (1.        , 0.10149942, 0.        , 1.        ),
            6 : (0.85190311, 0.06911188, 0.06911188, 1.        ),
            7 : (0.65813149, 0.15953864, 0.15953864, 1.        ),
            'China' : (0.56796617, 0.34854287, 0.34854287, 1.        ),
            'Undefined' : (0.50196078, 0.45471742, 0.50196078, 1.        ),
            }

input_dfs = {'multi_model' : mat_dem_mm_group_stack, 'repeat' : mat_dem_repeat_group_stack, 
             'repeat_spur_lines' : mat_dem_repeat_spur_stack}

ytypes = ['log', 
          'linear'
          ]

for output_name, input_df in input_dfs.items():
    for ytype in ytypes:
        
        data = input_df.copy()
        data2 = reserves_stack.copy()
        data2['share'] = pd.to_numeric(data2['share'])
        data2 = data2.groupby(['material', 'crc']).sum('share').reset_index()
        data2 = crc_fill(data2)
        
        data_us = data2.loc[data2['crc'] == 'United States']
        data_other = data2.loc[data2['crc'] != 'United States']
        
        cols = 7
        entries = len(data2['material'].unique())
        rows = int(np.ceil(entries / cols))
        
        fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(12, 6), sharex = True)
        axs = axs.ravel()
        
        n=0
        
        for mat in data2['material'].unique():

            bottom = 0

            plot_loc = data.loc[(data['material'] == mat)].reset_index(drop = True)
            plot_loc_us = data_us.loc[(data_us['material'] == mat)].reset_index(drop = True)
            plot_loc_other = data_other.loc[(data_other['material'] == mat)]
            plot_loc_other = pd.pivot_table(plot_loc_other, columns = 'crc', 
                                      values = 'share').reset_index(drop = True)
            plot_loc_other = plot_loc_other[list(col_dict.keys())[1:]]
            
            axs[n].set_title(mat)
            axs[n].set_yscale(ytype)
            
            dem = plot_loc.value.max()
            
            for col in plot_loc_other.columns:
                
                if type(col) is float:
                    label = str(col)[0]
                else:
                    label = col
                         
                axs[n].bar(x = plot_loc_other.index + -0.47, height = int(plot_loc_other[col] / dem), 
                        bottom = bottom, color = col_dict.get(col), label = label)
 
                bottom = bottom + int(plot_loc_other[col] / dem)
                
            axs[n].bar(x = plot_loc_us.index + 0.47, height = int(plot_loc_us['share'] / dem), 
                       color = col_dict.get('United States'), label = 'United States')
            axs[n].bar_label(axs[n].containers[-1], label_type ='edge')
            
            if ytype == 'linear':
                
                axs[n].yaxis.get_major_formatter().set_scientific(False)

            n = n + 1
            
        handles, labels = axs[0].get_legend_handles_labels()
        
        fig.legend(handles, labels, loc='lower center', ncols = 1, bbox_to_anchor = [0.42,0.06,1,1], frameon=False)
        fig.text(-0.01, 0.5, 'Economic reserves/Max annual material demand', va='center', rotation='vertical')
        
        plt.xticks([])
        plt.tight_layout()
        
        for ax in axs.flat[n:]:
            ax.remove()
            
        fig.savefig(f'{fig_outputs_dir}fig4_{output_name}_{ytype}.png', dpi = 1000, bbox_inches='tight')