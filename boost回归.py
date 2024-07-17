import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import  GradientBoostingRegressor

np.random.seed(1)
X = np.random.uniform(0,1,500)
y = np.sin(2*np.pi*X)+np.random.normal(0,0.1,500)
data = pd.DataFrame(X,columns=['X'])
data['y']=y
w = np.linspace(0,1,100)

sns.scatterplot(x = 'X',y = 'y',s= 20,data=data,alpha = 0.3)
plt.plot(w,np.sin(2*np.pi*w))
plt.show()

for i,m in zip([1,2,3,4],[1,10,100,1000]):
    model = GradientBoostingRegressor(n_estimators=m,max_depth=1,learning_rate=1,random_state=123)
    model.fit(X.reshape(-1,1),y)
    pred = model.predict(w.reshape(-1,1))
    plt.subplot(2,2,i)
    plt.plot(w,np.sin(2*np.pi*w),'k',linewidth =1)
    plt.plot(w,pred,'b')
    plt.text(0.65,0.8,f'M={m}')
plt.subplots_adjust(wspace=0.4,hspace=0.4)
plt.show()








