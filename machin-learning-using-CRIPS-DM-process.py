#!/usr/bin/env python
# coding: utf-8

# # Projet de machine learning en utilsant le processus CRISP-DM
# Ajouter les sources :
# - https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb
# - Smartphone and Smartwatch-Based Biometrics Using Activities of Daily Living. IEEE Access, 7:133190-133202, Sept. 2019.
# 
# ## Compréhension des affaires
# *Il s'agira d'expliquer les objectifs du projet et de détailler le plan du projet*

# ## Compréhension des données
# ### Mise en place

# In[4]:


# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


# ### Récupération des données

# In[7]:


import zipfile
import urllib

DOWNLOAD_ROOT = "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/"
WISDM_PATH = os.path.join("datasets", "wisdm")
WISDM_URL = DOWNLOAD_ROOT + "wisdm-dataset.zip"

def fetch_wisdm_data(wisdm_url=WISDM_URL, wisdm_path=WISDM_PATH):
    if not os.path.isdir(wisdm_path):
        os.makedirs(wisdm_path)
    zip_path = os.path.join(wisdm_path, "wisdm-dataset.zip")
    if not os.path.isfile(zip_path):
        urllib.request.urlretrieve(wisdm_url, zip_path)
    uncompressed_path = os.path.join(wisdm_path, "wisdm-dataset")
    if not os.path.isdir(uncompressed_path):
        with zipfile.ZipFile(zip_path,"r") as zip_ref:
            zip_ref.extractall(wisdm_path)


# In[8]:


fetch_wisdm_data()


# ### Description des données

# Les données sont décrites dans le fichier WISDM-dataset-description.pdf du jeu de données précédemment téléchargé.
# Ce jeu de données est composé de données provenant de 51 personnes à qui on a demandé de réaliser 18 tâches de 3 minutes. Chacun des sujets à une smartwatch attaché au poignet de leurs mains dominantes et un smartphone dans leurs poches. Les données collectés proviennent des girocscopes et des accéléromètres de la montre et du téléphone.
# 
# Un praitraitement des données à déja était fait dans le jeu de données. Il en résulte un jeu de données etiquetées sur les exemples au lieu de données étiquetées sur une série temporelle. Le traitement pour faire cela consiste à utiliser une fonction porte de 10s sur chaques séries temporelles de données.

# In[19]:


# library for reading arff file
# You can install it via 'pip install liac-arff'
# https://pypi.org/project/liac-arff/ for more informations
import arff

def load_wisdm_data(wisdm_path=WISDM_PATH):
    arffPathExample = os.path.join(WISDM_PATH,'wisdm-dataset','arff_files','phone','accel','data_1600_accel_phone.arff')
    data = arff.load(open(arffPathExample, 'r'))
    return data


# In[29]:


wisdmExample = load_wisdm_data()
import pandas as pd
attributesName = [wisdmExample['attributes'][i][0] for i in range (len(wisdmExample['attributes']))]
dataFrame = pd.DataFrame(data = wisdmExample['data'], columns=attributesName)
dataFrame.head()


# In[30]:


dataFrame.info()


# In[50]:


dataFrame.describe()


# In[59]:


dataFrame.hist(bins = 50, figsize=(20,30))
#save_fig("attribute_histogram_plots")
plt.show()


# In[ ]:




