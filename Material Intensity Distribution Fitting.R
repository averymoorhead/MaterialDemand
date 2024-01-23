rm(list = ls())

setwd("C:/Users/avery/OneDrive/Desktop/MaterialDemand/Inputs")

library(gamlss)
library(gamlss.dist)
library(gamlss.add)
library(readxl)

completedata <- read.csv("intensity_data.csv")
completedata <- completedata[completedata$technology %in% c("utility-scale solar pv", "onshore wind"), ]
#reducing the number of combos right now so it runs. 

#finding the unique combinations. 
unique_combinations <- unique(completedata[, c("Material", "technology")])

#looping throught each unique combination and running fitDist on it. 
for (i in 1:nrow(unique_combinations)) {
  material_tech <- unique_combinations[i, ]
  
  # data processing
  subset <- completedata[completedata$Material == material_tech$Material & completedata$technology == material_tech$technology, ]
  subset <- subset[["value"]]
  
  # here I am just removing the output for from complete data as it is very long 
  sink("sink.txt")
  
  tryCatch({
    fit <- suppressWarnings(fitDist(subset, k = 2, type = "realplus", method = "best", trace = TRUE, try.gamlss = TRUE))
  }, error = function(e) {
  })
    sink()
  
  # Display fit$fits for each iteration in the console - this has the different distributions ranked 
  cat("Material:", material_tech$Material, ", Technology:", material_tech$technology, "\n")
  print(fit$fits)
  cat("\n-----------------------------\n")
}
