setwd('D:/masmovil')

library(plotly)
library(readr)
library(data.table)
library(dplyr)
library(tidyverse)

# Gain Summary, AP/CP Stats
diffgroup_data <- as.data.table(read.csv('./data/data_for_gain_calculation/main_diffgroup_data_churn.csv'))
agentwise_data <- read.csv('./data/data_for_gain_calculation/agentwise_data_churn.csv')
callerwise_data <- read.csv('./data/data_for_gain_calculation/callerwise_data_churn.csv')


# Functions for AP and CP correction calculations
calculateAPCorrection <- function(agentwise_data,ncalls.matrix.agentwise,aq_100){
  tmp <- merge(ncalls.matrix.agentwise,agentwise_data, by='id_agente')
  return((sum(tmp$ncalls.x*tmp$churn_rate)/sum(tmp$ncalls.x)-aq_100)/aq_100)
}
calculateCPCorrection <- function(callerwise_data,ncalls.matrix.callerwise,cq_100){
  tmp <- merge(ncalls.matrix.callerwise,callerwise_data, by='callgroup')
  return((sum(tmp$ncalls.x*tmp$churn_rate)/sum(tmp$ncalls.x)-cq_100)/cq_100)
}

calculateGainByDiffgroup <- function(main_diffgroup_data,agentwise_data,callerwise_data,on_off_preference='all'){
  if(on_off_preference=='off'){
    diffgroup_data <- main_diffgroup_data[on_off==0]
  } else {
    diffgroup_data <- main_diffgroup_data
  }
  diffgroups <- 0:100
  cr_100 <- sum(diffgroup_data$Churn.30)/dim(diffgroup_data)[1]
  aq_100 <- sum(agentwise_data$nchurn)/sum(agentwise_data$ncalls)
  cq_100 <- sum(callerwise_data$nchurn)/sum(callerwise_data$ncalls)
  gains <- c()
  ap_corrections <- c()
  cp_corrections <- c()
  adjusted_gains <- c()
  for(diffgroup in diffgroups){
    tmp <- diffgroup_data[max_diffgroup_raw<=diffgroup]
    cr <- tmp %>% summarise(cr= sum(Churn.30)/n())
    
    ncalls.matrix.agentwise <- tmp %>% group_by(id_agente) %>% summarise(ncalls = n())
    ncalls.matrix.callerwise <- tmp %>% group_by(callgroup) %>% summarise(ncalls = n())
    
    gain <- (cr-cr_100)[[1]]
    ap_correction <- calculateAPCorrection(agentwise_data,ncalls.matrix.agentwise,aq_100)
    cp_correction <- calculateCPCorrection(callerwise_data,ncalls.matrix.callerwise,cq_100)
    adjusted_gain <- gain - ap_correction - cp_correction
    
    gains <- c(gains,gain)
    ap_corrections <- c(ap_corrections, ap_correction)
    cp_corrections <- c(cp_corrections, cp_correction)
    adjusted_gains <- c(adjusted_gains, adjusted_gain)
  }
  
  gain_by_diffgroup_data <- data.table(diffgroup=diffgroups, gain=gains, ap_correction=ap_corrections, cp_correction=cp_corrections, adjusted_gain=adjusted_gains)
  return(gain_by_diffgroup_data)
}

gain_by_diffgroup_all <- calculateGainByDiffgroup(diffgroup_data,agentwise_data,callerwise_data,on_off_preference = 'all')

write.csv(gain_by_diffgroup_all,'./data/data_for_gain_calculation/gain_by_diffgroup_churn.csv',row.names = FALSE)