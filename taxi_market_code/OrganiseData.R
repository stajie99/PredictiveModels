OrganiseData = function(dt_original){
  #{Add an index column; Remove unused columns from the orginal data}
  
  dt_original[, uniqueInd := 1:nrow(dt_original)]
  dt_silent = dt_original[, .(uniqueInd, idtrip, idvehicle, iddriver, halfhour, xdropoff, ydropoff, xpickup, ypickup)]
  save(dt_silent, file = "dt_silent.RData")
  
  therest = c("uniqueInd", "idvehicle", "iddriver", setdiff(names(dt_original), names(dt_silent)))
  rm(dt_silent)
  return(dt_original[, ..therest])

}



