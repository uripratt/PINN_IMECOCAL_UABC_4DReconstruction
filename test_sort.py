import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('Cl_Imec98_12.xlsx')
df['Profile_ID'] = df['Linea'].astype(str) + '_' + df['Estacion'].astype(str) + '_' + df['Año'].astype(str) + '_' + df['mes'].astype(str) + '_' + df['Dia'].astype(str)
df['Station_ID'] = df['Linea'].astype(str) + '_' + df['Estacion'].astype(str)

st_df = df[df['Station_ID'] == '97_50']
st_df = st_df.sort_values(by=['Profile_ID', 'Profundidad'])

plt.figure(figsize=(6, 8))
# sort=False tells seaborn to not sort the data by the first variable (x), which makes the line follow the DataFrame's order (sorted by Depth)
sns.lineplot(data=st_df, x='Clorofila', y='Profundidad', units='Profile_ID', estimator=None, color='red', alpha=0.5, sort=False)
plt.gca().invert_yaxis()
plt.savefig('test_97_50.png')
