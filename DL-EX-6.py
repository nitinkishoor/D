
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

s=[["The","cat","sat"],["The","dog","barked"],["A","man","runs"],["The","woman","eats"]]
t=[["DET","NOUN","VERB"]]*4
w=list(set(sum(s,[]))); tags=list(set(sum(t,[])))
wi={v:i+1 for i,v in enumerate(w)}
ti={v:i for i,v in enumerate(tags)}
it={i:v for v,i in ti.items()}

X=pad_sequences([[wi[v] for v in a] for a in s],padding='post')
y=np.array([to_categorical([ti[v] for v in a],3) for a in t])
a,b,c,d=train_test_split(X,y,test_size=.25)

i=Input((3,))
x=Embedding(len(w)+1,64)(i)
x=LSTM(64,return_sequences=True)(x)
m=Model(i,TimeDistributed(Dense(3,activation='softmax'))(x))
m.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
m.fit(a,c,epochs=15,batch_size=2,validation_data=(b,d),verbose=0)

test=["The","cat","runs"]
p=m.predict(pad_sequences([[wi.get(v,0) for v in test]],maxlen=3),verbose=0)
print("Sentence:",test)
print("Predicted POS:",[it[np.argmax(v)] for v in p[0]])
