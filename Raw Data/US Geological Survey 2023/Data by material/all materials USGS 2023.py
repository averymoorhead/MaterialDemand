# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 20:24:12 2023

@author: mbrinkerink
"""

import pandas as pd
import glob
import csv

# %%

file_name = 'USGS_2023_IRA materials calculations_NEW.xlsx'

#Creates a list of all .csv files that match the search_string criteria. Leave at '' if all csv's in folder are required.
csv_end_string = ''
all_files = [file for file in glob.glob('*.{}'.format('csv')) if file.endswith(csv_end_string)]

writer = pd.ExcelWriter(file_name , engine = 'xlsxwriter')

for file in all_files:
    csv = pd.read_csv(file)
    filestrip = file.replace('mcs2023-', '').replace('_salient.csv', '')
    csv.to_excel(writer, sheet_name=filestrip, startcol = -1)

writer.close()