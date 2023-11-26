rm(list = ls())
setwd("C:/Users/avery/Downloads/Material demand data for Avery")

library(readxl)
library(data.table)
library(dplyr)
library(tidyverse)
library(triangle)
library(stringr)
library(knitr)

technologies = c("offshorewind", "solar", "wind")
materials = c("Steel", "Glass", "Cu", "Concrete", "Aluminum")
scenarios = c("ira_mid", "baseline")
Spur_Dist_Delta = read_excel("spur_line_state_distribution_delta_y.xlsx", range= cell_cols(3:17))

Spur_Dist_Delta <- subset(Spur_Dist_Delta, policy %in% c("baseline", "ira_mid"))
Spur_Dist_Delta <- Spur_Dist_Delta[!is.na(Spur_Dist_Delta$dist_km_Difference),]

Spur_Dist_Sum <- Spur_Dist_Delta %>% group_by(policy,tech) %>% summarise(totalspurdistance = sum(dist_km_Difference)) #summing  the different model results 


Trans_Dist_Delta = read_excel("transmission_cost_distribution_results_delta_y.xlsx")
Trans_Dist_Delta <- Trans_Dist_Delta %>% select(run_name, dist_km_Difference)
Trans_Dist_Delta <- subset(Trans_Dist_Delta, run_name %in% c("baseline", "ira_mid"))

Trans_Dist_Delta <- Trans_Dist_Delta %>% group_by(run_name) %>% summarise(totaltransdistance = sum(dist_km_Difference)) #averaging the different model results 

Transmission_Intensity = read_excel("Transmission material intensity.xlsx", range = cell_rows(1:6))

Transmission_Intensity <- Transmission_Intensity %>% select(Material, Value)

      