FrictionandMismatch = function(dtCumulativeData, group){
  # Data is a data.table
  # group is a list that is grouped by
  nrowDa             = nrow(dtCumulativeData)
  # print(dtCumulativeData[1:3, ])
  if(nrow(dtCumulativeData) > 1){
    cum_localEmptybr    = dtCumulativeData[-nrowDa, .(local_empty_cdg)] + dtCumulativeData[-nrowDa, .(local_empty_premier)] + 
      dtCumulativeData[-nrowDa, .(local_empty_smrt)] + dtCumulativeData[-nrowDa, .(local_empty_transcab)]
    cum_foreignPickupbr = dtCumulativeData[-1, .(foreign_pickup_phobooking_cdg)] + dtCumulativeData[-1, .(foreign_pickup_tpbbooking_cdg)] +
      dtCumulativeData[-1, .(foreign_pickup_nonbooking_cdg)] + 
      dtCumulativeData[-1, .(foreign_pickup_phobooking_premier)] + dtCumulativeData[-1, .(foreign_pickup_tpbbooking_premier)] +
      dtCumulativeData[-1, .(foreign_pickup_nonbooking_premier)] + 
      dtCumulativeData[-1, .(foreign_pickup_phobooking_smrt)] + dtCumulativeData[-1, .(foreign_pickup_tpbbooking_smrt)] +
      dtCumulativeData[-1, .(foreign_pickup_nonbooking_smrt)] + 
      dtCumulativeData[-1, .(foreign_pickup_phobooking_transcab)] + dtCumulativeData[-1, .(foreign_pickup_tpbbooking_transcab)] +
      dtCumulativeData[-1, .(foreign_pickup_nonbooking_transcab)]
    
    ## friction_t variable by min{cle_t-1, cfp_t}
    friction           = pmin(cum_localEmptybr, cum_foreignPickupbr)
    
    ## mismatch_t variable by min{cll_t-1, cfp_t - friction_t}
    cum_leftEmptybr    = dtCumulativeData[-nrowDa, .(left_empty_cdg)] + dtCumulativeData[-nrowDa, .(left_empty_premier)] + 
      dtCumulativeData[-nrowDa, .(left_empty_smrt)] + dtCumulativeData[-nrowDa, .(left_empty_transcab)]
    mismatch           = pmin(cum_leftEmptybr, cum_foreignPickupbr - friction)
    
    efficient          = cum_foreignPickupbr - friction - mismatch
    names(friction)    = "friction"
    names(mismatch)    = "mismatch"
    names(efficient)   = "efficient"
    
    # total pickup
    dtCumulativeData   = as.data.frame(dtCumulativeData)
    total_pickups      = rowSums(dtCumulativeData[-1, grepl("pickup", colnames(dtCumulativeData))]) 
    friction_pct       = friction/total_pickups
    mismatch_pct       = mismatch/total_pickups
    efficient_pct      = efficient/total_pickups
    
    # print(class(friction_pct))
    names(friction_pct)    = "friction_pct"
    names(mismatch_pct)    = "mismatch_pct"
    names(efficient_pct)   = "efficient_pct"
    
    # Avg. fare
    avg.fare = dtCumulativeData[-1, grepl("pickup", colnames(dtCumulativeData))|grepl("fare", colnames(dtCumulativeData))]
    avg.fare[, c(FALSE, TRUE)] = avg.fare[, c(FALSE, TRUE)]/avg.fare[, c(TRUE, FALSE)]
    avg.fare[is.na(avg.fare)]  = 0
    
    dtFrictionandMismatch = cbind(dtCumulativeData[-1, c("Date", "time")]
                                  , avg.fare
                                  , dtCumulativeData[-1, grepl("empty", colnames(dtCumulativeData))|grepl("break", colnames(dtCumulativeData))]
                                  , total_pickups
                                  , friction
                                  , friction_pct
                                  , mismatch
                                  , mismatch_pct
                                  , efficient
                                  , efficient_pct)
    # print(friction)
    return(dtFrictionandMismatch)
  }
  
}