import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


filePath='../data/test_data/winequality-white.csv'
df = pd.read_csv(filePath, sep=';')
x = df.drop(columns=['quality']).copy()
y = df['quality']

#print(df.head()[0])
#print(y)

for col in df.columns:
	if col!='quality':
		plt.title("quality v. "+col)
		fig = plt.figure(figsize = (10,6))
		ax=sns.barplot(x = 'quality', y = col, data = df)
		ax.set_title("quality v. "+col)
		plt.show()
