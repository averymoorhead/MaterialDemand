rm(list = ls())

setwd("C:/Users/avery/OneDrive/Desktop/MaterialDemand/Raw Data")

library(readxl)
library(data.table)
library(dplyr)
library(tidyverse)
library(triangle)
library(stringr)
library(knitr)

#material intensity
transmission_material_intensity = read_excel("Transmission Material intensity.xlsx", range = cell_cols(1:4))
transmission_material_intensity <- transmission_material_intensity[1:5, ]

#spur lines
Spur_Dist_Delta = read_excel("spur_line_state_distribution_delta_y.xlsx", range= cell_cols(3:17))
Spur_Dist_Delta <- subset(Spur_Dist_Delta, policy %in% c("baseline", "ira_mid"))
Spur_Dist_Delta <- Spur_Dist_Delta[!is.na(Spur_Dist_Delta$dist_km_Difference),]

Spur_Dist_Sum <- Spur_Dist_Delta %>%
  group_by(policy, tech, year) %>%
  summarise(totalspurdistance = sum(dist_km_Difference)) %>%
  filter(year %in% c("2026", "2030", "2035")) %>% 
  mutate(year = ifelse(year == "2026", "2025", as.character(year))) %>% # Rename 2026 values to 2025
  mutate(totalspurdistance = totalspurdistance / 2) #as each year is reported in 2 year increments 

#multiplying by material intensity 
Spur_Dist_Sum <- Spur_Dist_Sum %>%
  mutate(Cu = totalspurdistance * transmission_material_intensity$Value[transmission_material_intensity$Material == "Cu"])%>% 
  mutate(Glass = totalspurdistance * transmission_material_intensity$Value[transmission_material_intensity$Material == "Glass"]) %>% 
  mutate(Concrete = totalspurdistance * transmission_material_intensity$Value[transmission_material_intensity$Material == "Concrete"]) %>%
  mutate(Aluminum = totalspurdistance * transmission_material_intensity$Value[transmission_material_intensity$Material == "Aluminum"]) %>%
  mutate(Steel = totalspurdistance * transmission_material_intensity$Value[transmission_material_intensity$Material == "Steel"])

#currently only including the spur lines as there are high levels of uncertainty with the tranmission lines 
# Trans_Dist_Delta = read_excel("transmission_cost_distribution_results_delta_y.xlsx")
# Trans_Dist_Delta <- Trans_Dist_Delta %>% select(run_name, dist_km_Difference)
# Trans_Dist_Delta <- subset(Trans_Dist_Delta, run_name %in% c("baseline", "ira_mid"))
# Trans_Dist_Delta <- Trans_Dist_Delta %>% group_by(run_name) %>% summarise(totaltransdistance = sum(dist_km_Difference)) #averaging the different model results 


#saving to inputs folder
folder_path <- "C:/Users/avery/OneDrive/Desktop/MaterialDemand/Inputs"
file_path1 <- file.path(folder_path, "spur_dist_sum.csv")
# Write each dataframe to its respective CSV file in the specified folder
write.csv(Spur_Dist_Sum, file = file_path1, row.names = FALSE)

      