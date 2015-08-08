#
# Reading Data from Excel Spreadsheet Files
# A_pyt/m_Excel_read.py
#
import pandas as pd
import matplotlib.pyplot as plt

# Open Excel Spreadsheet and Read Date
DAX = pd.read_excel('A_pyt/DAX_data.xlsx', 'sheet1',
                    index_col=0, parse_dates=True)

# Print 10 Most Current Daily Data Sets
print DAX.ix[-10:].to_string()

# Plot Close Levels for Whole Data Set
DAX['Close'].plot(label='DAX Index', grid=True)
plt.legend(loc=0)
