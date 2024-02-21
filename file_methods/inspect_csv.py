import pandas as pd 

bfdf = pd.read_csv(r"p:/archivedprojects/11205237-grade/MODELS/Rijn/HBV/KNMI14_output/Referentie/220001231HBV.csv", 
            parse_dates=True, index_col=0)