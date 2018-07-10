import pandas as pd
import numpy as np
 
df = pd.read_csv('test_usage.csv')
#df_unique = df.drop_duplicates(subset=['User', 'APPLICATION_NAME', 'FIELD3'], keep='first', inplace=False)

groupby_allVal = df.groupby(['User', 'APPLICATION_NAME', 'FIELD3']).count()
df_unique = groupby_allVal.add_suffix('_Count').reset_index()
print(df_unique)
columns = df_unique.columns

def Sorting(lst):
    lst.sort(key=len, reverse=True)
    return lst

for col in columns:
    update_Val = df_unique[col]
    if update_Val.dtype != 'int64':
        update_Val = update_Val.str.replace('(',' ')
        update_Val = update_Val.str.replace(')','')
        col_unique = update_Val.unique().tolist()
        col_unique = Sorting(col_unique)
        idx = 0
        for uni in col_unique:
            update_Val = update_Val.str.replace(uni, str(idx))
            idx = idx + 1
            df_unique.update(update_Val)
    df_unique.update(update_Val)
# 
# for key, value in groupby_allVal.count().items():
#     print('')
#     groupby_allVal[key]
#     


np.save('outfile',np.array(df_unique) )

print('')





#===============    ================================================================
# img = imread('68.jpg')
# img_tinted = img * [1, 1, 0.9]
# #print(img.dtype, img.shape) 
# # Show the original image
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# 
# 
# plt.subplot(1,2,2)
# plt.imshow(img_tinted)
# 
# #plt.imshow(np.uint8(img_tinted))
# 
# plt.show()
#===============================================================================