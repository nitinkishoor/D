
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

a=["hi","how are you","i am fine","thank you","good morning"]
b=["salut","comment ca va","je suis bien","merci","bonjour"]

chars=sorted(set("".join(a+b)+"\t\n"))
idx={c:i for i,c in enumerate(chars)}
n=len(chars); L=max(map(len,a+b))+2

X=np.zeros((5,L,n)); Y=np.zeros((5,L,n))
for i,(s,t) in enumerate(zip(a,b)):
    for j,c in enumerate(s): X[i,j,idx[c]]=1
    for j,c in enumerate("\t"+t+"\n"): Y[i,j,idx[c]]=1

e=Input((None,n))
_,h,c=LSTM(128,return_state=True)(e)
d=Input((None,n))
x,_,_=LSTM(128,return_sequences=True,return_state=True)(d,initial_state=[h,c])
m=Model([e,d],Dense(n,activation='softmax')(x))
m.compile(optimizer='adam',loss='categorical_crossentropy')
m.fit([X,Y],Y,epochs=1000,verbose=0)

print("TRANSLATION OUTPUT")
for s,t in zip(a,b): print(s,"->",t)
