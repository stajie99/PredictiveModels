# reads in the data and saves them to a list
# returns the list variable as input.

import pandas as pd
from datetime import datetime

SPLINE_ORDER                = 4
PRESICION_FG_ALG            = 0.0001
NUMBER_PRINCIPAL_COMPONENTS = 3
FONT_SIZE                   = 7

COUNTRY_NAMES               = ['US', 'UK', 'EU', 'JP']
MATURITIES                  = [1/12,2/12,0.25,4/12,5/12,0.5,1,2,3,4,5,6,7,8,9,10,12,15,20,25,30]

# create knot sequence


# import data for US
df = pd.read_excel("Data\\df_irs.xlsx", sheet_name= "data")
print(df.date[0])

# 3D surface plot of yield curves for each economy
