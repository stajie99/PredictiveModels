ConvertNAto0 = function(DT, colset = seq_len(ncol(DT))) {
  # either of the following for loops
  
  # # by name :
  # for (j in names(DT))
  #   set(DT,which(is.na(DT[[j]])),j,0)
  
  # or by number (slightly faster than by name) :
  # for (j in seq_len(ncol(DT)))
  for (j in colset)
    set(DT,which(is.na(DT[[j]])),j,0)
}