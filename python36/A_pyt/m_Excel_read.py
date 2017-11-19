#
# Reading Data from Excel Spreadsheet Files
# A_pyt/m_Excel_read.py
#
import pandas as pd
import matplotlib.pyplot as plt

# Open Excel Spreadsheet and Read Date
DAX = pd.read_excel('A_pyt/DAX_data.xlsx', index_col=0)

# Print 10 Most Current Daily Data Sets
print(DAX.iloc[-10:].to_string())

# Plot Close Levels for Whole Data Set
DAX['Close'].plot(label='DAX Index', figsize=(10, 6))
plt.legend(loc=0)
