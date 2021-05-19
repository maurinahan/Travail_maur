# %%
import matplotlib.pyplot as plt
a='Hello World'

print('Hello World')
plt.plot([1, 2, 3, 4])

## Listes

A=[1,3,5,5]
A[0]
B=['Dublin','Cgt']
A.append(50)
A.insert(2,4)
A.extend(B)
len(A)
for index,valeur in enumerate(A):
    print(index,valeur)

C=enumerate(A)
 

    def fibonacci(n):
        A=[]
        a,b=0,1
        while a<n:
            A.append(a)
            a, b=b,a+b
        return A

fibonacci(10)

dic={
    "positif":[],
    "negatif":[]
}
listes=[-2,-6,5,7,-34,6]
listes_pos=[]
listes_neg=[]
for values in listes:
    if values >0 :
        listes_pos.append(values)
    else:
       listes_neg.append(values) 

dic={
    "positif":listes_pos,
    "negatif":listes_neg
}

print(dic)

liste=[i**2 for i in range(10)]
liste

[values for values in range(0,10,2)]
# %%
prenoms=["Marc","Mathieu","Jean"]
age=[23,10,45]
{prenoms:age for prenoms,age in zip(age,prenoms)}

# %%

### Numpy
import numpy as np

A=np.array([1,2,4])
A.shape

B=np.zeros((3,2))
B.ndim
B.shape
#%%
C=np.ones((3,3))
print(C)
C.ndim
C.shape

# %%
D=np.full((2,3),4)
print(D)

np.linspace(0,10,20)
# %%
np.arange(10,34,3)
# %%
def initializtion(m,n):
    A=np.random.randn(m,n)
    B=np.ones((m,1))
    C=np.concatenate((A,B),axis=1)
    return C

initializtion(3,2)
# %%
B=np.array([[1,2,3],[3,5,7],[4,7,9]])
B.sum(axis=1)
B.sort(axis=0)
B.diagonal()
np.exp(B)
np.corrcoef(B)[:,1:]
B.std(axis=0)

# %%
np.corrcoef(B)[:,1:]
np.linalg.det(B)
np.linalg.inv(B)

# %%
np.random.seed(0)
A=np.random.randint(0,100,[10,5])
A
(A-A.mean(axis=0))/A.std(axis=0)

# %%
B+2
# %%
B+np.ones((1,3))
# %%
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(0,2,10)
y=x**2
plt.figure()
plt.plot(x,y)
plt.plot(x,x**3)
plt.show()
# %%
dataset={f"experience{i}": np.random.randn(100) for i in range(4)}
print(dataset)

[k for k in dataset.keys()]

def graphique (dataset):
    i=0
    for k in dataset.keys():
         i+=1
         lignes=len(dataset.keys())
         plt.figure()
         plt.subplot(lignes,1,i)
         plt.plot(np.arange(0,len(dataset[k])),dataset[k])
         plt.title(f"experience_{i}")
         plt.show()
    
graphique(dataset)     
 

# %%
len(dataset[list(dataset.keys())[0]])

# %%
