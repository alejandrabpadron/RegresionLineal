#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


x = 2.6 + 7 * np.random.randn(20000)


# In[3]:


res = 0 + 0.9 * np.random.randn(20000)


# In[4]:


y_pred = 5 + 0.3 * x


# In[5]:


y_act = 5 + 0.3 * x + res


# In[6]:


x_list = x.tolist()
y_pred_list = y_pred.tolist()
y_act_list = y_act.tolist()


# In[7]:


data = pd.DataFrame(
    {
        "x":x_list,
        "y_actual":y_act_list,
        "y_prediccion":y_pred_list
    }
)


# In[8]:


data.head()


# In[9]:


data.corr()


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


y_mean = [np.mean(y_act) for i in range(1, len(x_list) + 1)]


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data["x"],data["y_prediccion"])
plt.plot(data["x"], data["y_actual"], "ro")
plt.plot(data["x"],y_mean, "g")
plt.title("Valor Actual vs Predicción")


# In[13]:


y_m = np.mean(y_act)
data["SSR"]=(data["y_prediccion"]-y_m)**2
data["SSD"]=(data["y_prediccion"]-data["y_actual"])**2
data["SST"]=(data["y_actual"]-y_m)**2


# In[14]:


data.head()


# In[15]:


SSR = sum(data["SSR"])
SSD = sum(data["SSD"])
SST = sum(data["SST"])


# In[16]:


SSR


# In[17]:


SSD


# In[18]:


SST


# In[19]:


R2=(SSR/SST)*100


# In[20]:


R2


# In[21]:


x_mean = np.mean(data["x"])
y_mean = np.mean(data["y_actual"])
x_mean, y_mean


# In[ ]:


data["beta_n"] = (data["x"]-x_mean)*(data["y_actual"]-y_mean)
data["beta_d"] = (data["x"]-x_mean)**2


# In[ ]:


beta = sum(data["beta_n"])/sum(data["beta_d"])


# In[ ]:


alpha = y_mean - beta * x_mean


# In[ ]:


alpha, beta


# In[ ]:


data["y_model"] = alpha + beta * data["x"]


# In[ ]:


data.head()


# In[ ]:


data.corr()


# In[ ]:


SSR = sum((data["y_model"]-y_mean)**2)
SSD = sum((data["y_model"]-data["y_actual"])**2)
SST = sum((data["y_actual"]-y_mean)**2)


# In[ ]:


SSR,SSD,SST


# In[ ]:


R2=(SSR/SST)*100
R2


# In[ ]:


y_mean = [np.mean(y_act) for i in range(1, len(x_list) + 1)]

get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data["x"],data["y_prediccion"])
plt.plot(data["x"], data["y_actual"], "ro")
plt.plot(data["x"],y_mean, "g")
plt.plot(data["x"], data["y_model"])
plt.title("Valor Actual vs Predicción")

