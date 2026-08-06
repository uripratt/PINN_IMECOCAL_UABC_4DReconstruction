import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('Cl_Imec98_12.xlsx')
df['Profile_ID'] = df['Linea'].astype(str) + '_' + df['Estacion'].astype(str) + '_' + df['Año'].astype(str) + '_' + df['mes'].astype(str) + '_' + df['Dia'].astype(str)
df['Station_ID'] = df['Linea'].astype(str) + '_' + df['Estacion'].astype(str)
df = df.sort_values(by=['Profile_ID', 'Profundidad'])

# Bin depths to 10m intervals for a cleaner mean
bins = range(0, 250, 10)
labels = range(5, 245, 10)
df['Profundidad_Bin'] = pd.cut(df['Profundidad'], bins=bins, labels=labels, include_lowest=True)

st = '100.0_30.0' # Example station
st_df = df[df['Station_ID'] == st]

plt.figure()
for pid in st_df['Profile_ID'].unique():
    p_df = st_df[st_df['Profile_ID'] == pid]
    plt.plot(p_df['Clorofila'], p_df['Profundidad'], color='gray', alpha=0.5)

mean_profile = st_df.groupby('Profundidad_Bin')['Clorofila'].mean().reset_index()
plt.plot(mean_profile['Clorofila'], mean_profile['Profundidad_Bin'], color='black', linewidth=3)
plt.gca().invert_yaxis()
plt.savefig('test_plot_binned.png')
