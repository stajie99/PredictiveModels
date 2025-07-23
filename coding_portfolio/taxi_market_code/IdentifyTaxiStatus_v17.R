IdentifyTaxiStatus= function(x, timeConsecutive, Date0, DateN, nIntervals, id){
  # 72 (72 = 3 pickup/empty/break * 2 local/foreign * 3 booking types * 4 companies) status variables indicating 
  # taxis' status per area per 5-minute interval starts here.
  # Generate left_empty, local_empty, local_pickup, foreign_pickup, foreign_empty, ...
  # Depend on dropoff time and start time, dropoff area and pickarea
  
  # Vectorization, not for loops
  # February 11, 2020, by Jiejie
  TimeConsecutive = copy(timeConsecutive)
  
  # 0. Time difference binary variables
  islesseq60mins = as.numeric(x$time_diff[-1] <= 60)
  islgthan60mins = 1 - islesseq60mins
  
  # 1. Local variables.
  nrowx                 = nrow(x)
  islocaltmp              = x[-nrowx, .(pareadropoff)] == x[-1, .(pareapickup)]
  islocal                 = islocaltmp * islesseq60mins
  islocal_break           = islocaltmp * islgthan60mins
  
  #    Foreign variables.
  isforeign               = (1-islocaltmp) * islesseq60mins
  isleft_break            = (1-islocaltmp) * islgthan60mins
  rm(islocaltmp)
  
  # 2. Pick up variable. 
  # Each time point at t_start implies a pickup for pickuparea, either local or foreign.
  dt_pickupTimeAreaFare          = x[-1, .(area = pareapickup, uniq_time = t_startUniq, fare = totalfare)] 
  # if(dt_pickupTimeAreaFare[area == "", .N]>0){print(dt_pickupTimeAreaFare[area == "", .N])}
  # 3. Booking variable
  isphobooking = as.numeric(x$feebooking[-1] > 0)
  istpbbooking = x$tpbbooking[-1]
  ispho_and_tpb= isphobooking + istpbbooking
  
  isphobooking = isphobooking - (ispho_and_tpb == 2)
  isnonbooking = as.numeric(ispho_and_tpb == 0)
  # a = istpbbooking + isphobooking + isnonbooking
  # if(any(a != 1)){print(a)}
  
  
  # 5. Empty (break) variable. 
  # The time points between t_end[i] and t_start[i+1] indicate empty/break for dropoffarea, either left_empty/break or local empty/break.
  
  # TIME used 2s
  # ptm = proc.time()
  timeEmpty_EndandStart = cbind(x[-nrowx, .(t_endUniq)], x[-1, .(t_startUniq)])
  
  timeticksBtwn = function(et){
    t1 = et[1]+1
    t2 = et[2]-1
    if(t1<=t2){y = t1:t2}else{y = c()}
    
    return(y)
  }
  timeticksEmpty_list   = apply(timeEmpty_EndandStart, 1,timeticksBtwn)
  # timeticksEmpty_list   = lapply(timeEmpty_EndandStart, timeticksBtwn)
  
  if(is.matrix(timeticksEmpty_list)){
    timeticksEmpty_list = lapply(seq_len(ncol(timeticksEmpty_list)), function(i) timeticksEmpty_list[,i])
  }
  lengthsTimeTicksEmpty= lengths(timeticksEmpty_list, use.names = FALSE)
  
  # Local_empty
  if(length(lengthsTimeTicksEmpty) > 0){
    islocal_empty      = rep(islocal, lengthsTimeTicksEmpty)
    islocal_break      = rep(islocal_break, lengthsTimeTicksEmpty)
    
    isleft_empty       = rep(isforeign, lengthsTimeTicksEmpty)
    isleft_break       = rep(isleft_break, lengthsTimeTicksEmpty)
    
    vec                = rep(1:(nrowx - 1), lengthsTimeTicksEmpty)
    isEmptyBreak       = data.table(area = x$pareadropoff[vec], uniq_time = unlist(timeticksEmpty_list),
                                  islocal_empty = islocal_empty, isleft_empty = isleft_empty,
                                  islocal_break = islocal_break, isleft_break = isleft_break)
    
    rm(islocal_empty, islocal_break, isleft_empty, isleft_break, vec, timeticksEmpty_list)
  }
  
  
  ###  
  ind1 = dt_pickupTimeAreaFare$uniq_time
  
  TimeConsecutive[ind1, `:=`(area = dt_pickupTimeAreaFare$area, 
                                  islocal_pickup_phobooking = islocal * isphobooking, fare_localpk_pb = islocal * isphobooking * dt_pickupTimeAreaFare$fare,
                                  islocal_pickup_tpbbooking = islocal * istpbbooking, fare_localpk_tb = islocal * istpbbooking * dt_pickupTimeAreaFare$fare,
                                  islocal_pickup_nonbooking = islocal * isnonbooking, fare_localpk_nb = islocal * isnonbooking * dt_pickupTimeAreaFare$fare,
                                  isforeign_pickup_phobooking = isforeign * isphobooking, fare_foreignpk_pb = isforeign * isphobooking * dt_pickupTimeAreaFare$fare,
                                  isforeign_pickup_tpbbooking = isforeign * istpbbooking, fare_foreignpk_tp = isforeign * istpbbooking * dt_pickupTimeAreaFare$fare,
                                  isforeign_pickup_nonbooking = isforeign * isnonbooking, fare_foreignpk_nb = isforeign * isnonbooking * dt_pickupTimeAreaFare$fare)]
  

  rm(islocal, isforeign, isphobooking, istpbbooking, isnonbooking, dt_pickupTimeAreaFare, ind1)
  
  if(length(lengthsTimeTicksEmpty) > 0){
    ind2 = isEmptyBreak$uniq_time
    
    TimeConsecutive[ind2, `:=`(area = isEmptyBreak$area, 
                                    islocal_empty = isEmptyBreak$islocal_empty, isleft_empty  = isEmptyBreak$isleft_empty,
                                    islocal_break = isEmptyBreak$islocal_break, isleft_break = isEmptyBreak$isleft_break)]
    
    
    
  }else{
    TimeConsecutive[, `:=`(area = NA, islocal_empty = 0, isleft_empty  = 0,
                                islocal_break = 0, isleft_break = 0)]
  }
  
  
  TimeConsecutive[, `:=`(time = rep(1:nIntervals, DateN-Date0+1)
                              , Date = rep(seq(Date0, DateN, by = 1), each = nIntervals)
                              , composite_id = rep(id, nrow(TimeConsecutive)))]
  
  # TimeConsecutive[, isforeign_empty :=0]
  
  columnsnd = c("composite_id", "Date", "area", "time", "uniq_time",
                "islocal_pickup_phobooking", "fare_localpk_pb",
                "islocal_pickup_tpbbooking", "fare_localpk_tb",
                "islocal_pickup_nonbooking", "fare_localpk_nb",
                "isforeign_pickup_phobooking", "fare_foreignpk_pb",
                "isforeign_pickup_tpbbooking", "fare_foreignpk_tp",
                "isforeign_pickup_nonbooking", "fare_foreignpk_nb",
                "islocal_empty", "isleft_empty",
                "islocal_break", "isleft_break")
  
  # setcolorder(TimeConsecutive, columnsnd)
  
  Data_taxi5types = TimeConsecutive[, ..columnsnd]

  ConvertNAto0(Data_taxi5types, 6:ncol(Data_taxi5types))
  
  # print(paste(nrow(x)- sum(x$time_diff>60, na.rm = TRUE)
  #             , sum(
  #               colSums(
  #                 Data_taxi5types[, .(islocal_pickup_phobooking, islocal_pickup_tpbbooking
  #                                   , islocal_pickup_nonbooking
  #                                   , isforeign_pickup_phobooking, isforeign_pickup_tpbbooking
  #                                   , isforeign_pickup_nonbooking)], na.rm = TRUE)), sep = "|"))
  # checked
  # [1] "129|128"
  # [1] "1|0"
  # [1] "17|16"
  # [1] "26|25"
  # 
  names(Data_taxi5types)[6:ncol(Data_taxi5types)] = paste(names(Data_taxi5types)[6:ncol(Data_taxi5types)], as.character(x$company[1]), sep = "_")
  
  rm(list=setdiff(ls(), "Data_taxi5types"))
  # x = Data_taxi5types[is.na(area), 6:ncol(Data_taxi5types)]
  # print(range(x))
  
  # write.csv(Data_taxi5types, file = paste0("C:/Users/bizzjie/Downloads/taxi machao 20171017/tables/taxi5types",id[1,1],"-", id[1,2],"20190923.csv", sep=""))
  return(Data_taxi5types)
}


