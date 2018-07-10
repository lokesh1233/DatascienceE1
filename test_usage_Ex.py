# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import datetime
sns.set()
 
df = pd.read_csv('test_usage_ext_1.csv')
df = df.drop(['User','VIEW_NAME', 'BUTTON_NAME', 'LANGUAGE', 'PLATFORM', 'USERAGENT', 'VENDOR', 'COLORDEPTH', "HEIGHT", "ORIENTATION", "PIXELDEPTH", "WIDTH", "HASH", "HREF", "OS", "ZSYSTEM", "FIELD1", "FIELD2", "FIELD4", "PATH"], axis=1)
df['values'] = 1
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='ignore', yearfirst=True, format="%Y%m%d%H%M%S")

df_Pivot = df.pivot_table(index="Timestamp", columns='APPLICATION_NAME', values="values", aggfunc = np.mean)
df_Pivot.fillna(0, inplace=True)
df_Pivot = df_Pivot.resample('60T').sum()

# df['date1'] = pd.to_datetime()
# 
# df['date2'] = pd.to_datetime(df['date2'])
# 
# df1 = pd.DataFrame(index=pd.date_range(df.date1.min(), df.date2.max()), columns = ['score1sum', 'score2sum'])
# 
# df1[['score1sum','score2sum']] = df1.apply(lambda x: df.loc[(df.date1 <= x.name) & 
#                                                             (x.name <= df.date2),
#                                                             ['score1','score2']].sum(), axis=1)
# df1.rename_axis('usedate').reset_index()


#df['Timestamp'] = df['Timestamp'].dt.date

df_Pivot.plot(figsize=(20,10), linewidth=5, fontsize=20, colormap='gist_rainbow')
plt.xlabel('hour', fontsize=20);
print()