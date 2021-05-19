#%%
import pandas as pd

df=pd.read_csv("C:/Users/mauac/Desktop/STT5100/data_ass_auto.csv",delim_whitespace=True)
X=df.drop('Montant_paye')
df.head(5)
# %%
df.shape
df.describe()
df.columns

#### Train_Test_split, Cross Validation

# %%


X=df.drop('Montant_paye',axis=1)
Y=df['Montant_paye']

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test , Y_train, Y_test =train_test_split(X, Y,test_size=0.5)

model=LinearRegression()


model.fit(X_train,Y_train)

print(model.score(X_test,Y_test))

predictions=model.predict(X)

predictions[:5]

# %%