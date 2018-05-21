import pandas as pd
df = pd.read_excel(sys.argv[1], sheet_name=sys.argv[2], skiprows=[0])
print(df.columns)
df = df[['Area', 'Distance from Origin', 'Distance to Image Border XY',
       'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Intensity Max',
       'Intensity Mean', 'Intensity Median', 'Intensity Min',
       'Intensity StdDev', 'Intensity Sum', 'Number of Vertices', 'Position X',
       'Position Y', 'Position Z', 'Sphericity', 'Volume']]
print(df[:20])
