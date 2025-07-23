AccumulateTaxiStatusTables = function(files, m1, m2){
  # 1. File 1:m1
  ptm = proc.time()
  taxiStatusCum = rbindlist(lapply(files[1:m1], fread), fill = TRUE)
  print(paste("Binding File 1-", m1, sep = ""))
  print(proc.time() - ptm)
  
  ptm = proc.time()
  ConvertNAto0(taxiStatusCum)
  
  
  taxiStatusCumulative1 = taxiStatusCum[, lapply(.SD, sum), keyby=.(area, Date, time)]
  print(paste("Accumulating File 1-", m1, sep = ""))
  print(proc.time() - ptm)
  rm(taxiStatusCum)
  
  # print((m1+1):m2)
  # print((m2+1):length(files))
  
  # 2. File m1:m2
  taxiStatusCum = rbindlist(lapply(files[(m1+1):m2], fread), fill = TRUE)
  ConvertNAto0(taxiStatusCum)
  taxiStatusCumulative2 = taxiStatusCum[, lapply(.SD, sum), keyby=.(area, Date, time)]
  rm(taxiStatusCum)
  
  # 3. File m2:length(files)
  taxiStatusCum = rbindlist(lapply(files[(m2+1):length(files)], fread), fill = TRUE)
  ConvertNAto0(taxiStatusCum)
  taxiStatusCumulative3 = taxiStatusCum[, lapply(.SD, sum), keyby=.(area, Date, time)]
  rm(taxiStatusCum)
  # 4. Sum three parts
  taxiStatusCum = rbind(taxiStatusCumulative1, taxiStatusCumulative2, taxiStatusCumulative3)
  ConvertNAto0(taxiStatusCum)
  dtTaxiStatusCumulative = taxiStatusCum[, lapply(.SD, sum), keyby=.(area, Date, time)]
  
  rm(taxiStatusCum, taxiStatusCumulative1, taxiStatusCumulative2, taxiStatusCumulative3)
  return(dtTaxiStatusCumulative)
}


