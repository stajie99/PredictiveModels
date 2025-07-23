PrepareForStage1 = function(dtTaxiTrips, lengthInterval){
  idUnique            = unique(dtTaxiTrips$id_composite_key)
  # Dates, time
  dateRange           = range(dtTaxiTrips[, Date])
  dates               = seq(dateRange[1], dateRange[2], by=1)
  
  nIntervals          = 24*60/lengthInterval # 288 intervals per day
  dtTaxiTrips[,     `:=`(t_start = period_to_seconds(hms(start_time))%/%(lengthInterval*60) + 1
                         , t_end = period_to_seconds(hms(end_time))%/%(lengthInterval*60) + 1)]
  
  ## transform to universal time
  Date0               = dateRange[1]
  DateN               = dateRange[2]
  dtTaxiTrips[,     `:=`(t_startUniq = t_start + as.numeric(Date - Date0)*nIntervals
                         , t_endUniq = t_end + as.numeric(Date + (t_start > t_end) - Date0)*nIntervals)]
  
  timeConsecutive     = data.table(uniq_time = 1 : (length(dates)*nIntervals))
  
  
  return(list(idUnique, Date0, DateN, nIntervals, dtTaxiTrips, timeConsecutive))
}
