#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
module_path = os.path.abspath(os.path.join('./helper'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
print(" Including modules from ", module_path)

