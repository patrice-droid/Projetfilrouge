#!/usr/bin/env python
# coding: utf-8





import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from bokeh.plotting import figure, show
from bokeh.io import output_notebook


# In[2]:





# In[3]:


#Copies de sauvegarde

# Chemins des fichiers originaux
chemin_category_tree = '/Users/Patrice/Desktop/Data/ProjetDA/Dataset/category_tree.csv'
chemin_events = '/Users/Patrice/Desktop/Data/ProjetDA/Dataset/events.csv'
chemin_item_properties_part1 = '/Users/Patrice/Desktop/Data/ProjetDA/Dataset/item_properties_part1.csv'
chemin_item_properties_part2 = '/Users/Patrice/Desktop/Data/ProjetDA/Dataset/item_properties_part2.csv'

# Lire les fichiers originaux
df_cat = pd.read_csv('/Users/Patrice/Desktop/Data/ProjetDA/Dataset/category_tree.csv')
df_events = pd.read_csv('/Users/Patrice/Desktop/Data/ProjetDA/Dataset/events.csv')
df_it_prop1 = pd.read_csv('/Users/Patrice/Desktop/Data/ProjetDA/Dataset/item_properties_part1.csv')
df_it_prop2 = pd.read_csv('/Users/Patrice/Desktop/Data/ProjetDA/Dataset/item_properties_part2.csv')

# Enregistrer les copies sous de nouveaux noms
df_cat.to_csv('/Users/Patrice/Desktop/Data/ProjetDA/Dataset/category_tree.csv', index=False)
df_events.to_csv('/Users/Patrice/Desktop/Data/ProjetDA/Dataset/events.csv', index=False)
df_it_prop1.to_csv('/Users/Patrice/Desktop/Data/ProjetDA/Dataset/item_properties_part1.csv', index=False)
df_it_prop2.to_csv('/Users/Patrice/Desktop/Data/ProjetDA/Dataset/item_properties_part2.csv', index=False)


# In[4]:


#


# In[5]:


# Analyse descriptive, traitement et exploration initiale


# In[6]:


#1er fichier sur 4


# In[7]:


copied_path = '/Users/Patrice/Desktop/Data/ProjetDA/Dataset/category_tree.csv' #remove ‘content/’ from path then use
df_cat = pd.read_csv(copied_path)


# In[8]:


#Exploration initiale et Analyse descriptive de df_cat


# In[9]:


print(df_cat.head(10))
print(df_cat.tail(10))
print(df_cat.info())
df_cat.describe()


# In[10]:


#Le code suivant va d'abord identifier les valeurs manquantes dans df_cat,
#puis il va calculer des statistiques descriptives telles que le nombre de catégories uniques et de parents uniques,
#et la fréquence des différents types d'événements (comme les vues, les clics, etc.).
#Enfin, il visualisera ces informations à l'aide d'un histogramme en barres et le nbr de doublons.


# In[11]:


# Analyse des valeurs manquantes
missing_values = df_cat.isnull().sum()
print("Valeurs manquantes",missing_values)

# Statistiques descriptives
unique_categories = df_cat['categoryid'].nunique()
unique_parents = df_cat['parentid'].nunique()
top_parent_categories = df_cat['parentid'].value_counts().head(10)

print("Nombre de catégorie uniques",unique_categories)
print("Nombre de parents uniques",unique_parents)
print("Fréquences catégorie parents",top_parent_categories)


# Identification des doublons df_cat
duplicates_events = df_cat.duplicated().sum()
print("Nombre de doublons dans df_events:", duplicates_events)


# In[12]:


#


# In[13]:


#2ème fichier sur 4


# In[14]:


copied_path = '/Users/Patrice/Desktop/Data/ProjetDA/Dataset/events.csv' 
df_events = pd.read_csv(copied_path)


# In[15]:


#Exploration initiale et Analyse descriptive de df_events


# In[16]:


print(df_events.head(10))
print(df_events.tail(10))
print(df_events.info())
df_events.describe()


# In[17]:


#Le code suivant va d'abord identifier les valeurs manquantes dans df_events,
#puis il va calculer des statistiques descriptives telles que le nombre de visiteurs uniques et d'articles uniques,
#et la fréquence des différents types d'événements (comme les vues, les clics, etc.).
#Enfin, il visualisera ces informations à l'aide d'un histogramme en barres et le nbr de doublons


# In[18]:


# Analyse des valeurs manquantes pour df_events
missing_values_events = df_events.isnull().sum()
print("Valeurs manquantes dans df_events", missing_values_events)

# Statistiques descriptives pour df_events
unique_visitors = df_events['visitorid'].nunique()
unique_items = df_events['itemid'].nunique()
events_count = df_events['event'].value_counts()
top_items_viewed = df_events[df_events['event'] == 'view']['itemid'].value_counts().head(10)

print("Nombre de visiteurs uniques", unique_visitors)
print("Nombre d'articles uniques", unique_items)
print("Fréquences des événements", events_count)
print("Top 10 des articles les plus consultés", top_items_viewed)


# Identification des doublons df_events
duplicates_events = df_events.duplicated().sum()
print("Nombre de doublons dans df_events:", duplicates_events)


# In[19]:


#


# In[20]:


#3ème fichier sur 4


# In[21]:


copied_path = '/Users/Patrice/Desktop/Data/ProjetDA/Dataset/item_properties_part1.csv' 
df_it_prop1 = pd.read_csv(copied_path)


# In[22]:


#Exploration initiale et Analyse descriptive de df_it_prop1


# In[23]:


print(df_it_prop1.head(10))
print(df_it_prop1.tail(10))
print(df_it_prop1.info())
df_it_prop1.describe()


# In[24]:


#Ce script réalise une analyse similaire à celle que nous avons fait pour df_cat et df_events.
#Il commence par identifier les valeurs manquantes dans df_it_prop1,
#puis calcule des statistiques descriptives comme le nombre d'articles uniques et de propriétés uniques,
#et la fréquence des propriétés les plus courantes.
#Enfin, il visualise ces données à l'aide d'un histogramme en barres,
#mettant en avant les 10 propriétés les plus fréquentes puis le nbr de doublons


# In[25]:


# Analyse des valeurs manquantes pour df_it_prop1
missing_values_it_prop1 = df_it_prop1.isnull().sum()
print("Valeurs manquantes dans df_it_prop1", missing_values_it_prop1)

# Statistiques descriptives pour df_it_prop1
unique_items_prop1 = df_it_prop1['itemid'].nunique()
unique_properties = df_it_prop1['property'].nunique()
top_properties = df_it_prop1['property'].value_counts().head(10)

print("Nombre d'articles uniques dans df_it_prop1", unique_items_prop1)
print("Nombre de propriétés uniques", unique_properties)
print("Top 10 des propriétés les plus fréquentes", top_properties)


# Identification des doublons df_it_prop1
duplicates_events = df_it_prop1.duplicated().sum()
print("Nombre de doublons dans df_it_prop1:", duplicates_events)


# In[26]:


#


# In[27]:


#4ème fichier sur 4


# In[28]:


copied_path = '/Users/Patrice/Desktop/Data/ProjetDA/Dataset/item_properties_part2.csv' #remove ‘content/’ from path then use
df_it_prop2 = pd.read_csv(copied_path)


# In[29]:


#Exploration initiale et Analyse descriptive de df_it_prop2


# In[30]:


print(df_it_prop2.head(10))
print(df_it_prop2.tail(10))
print(df_it_prop2.info())
df_it_prop2.describe()


# In[31]:


#Ce script réalise une analyse similaire à celle que nous avons fait pour df_cat et df_events.
#Il commence par identifier les valeurs manquantes dans df_it_prop2,
#puis calcule des statistiques descriptives comme le nombre d'articles uniques et de propriétés uniques,
#et la fréquence des propriétés les plus courantes.
#Enfin, il visualise ces données à l'aide d'un histogramme en barres,
#mettant en avant les 10 propriétés les plus fréquentes puis le nbr de doublons


# In[32]:


# Analyse des valeurs manquantes pour df_it_prop1
missing_values_it_prop2 = df_it_prop2.isnull().sum()
print("Valeurs manquantes dans df_it_prop2", missing_values_it_prop2)

# Statistiques descriptives pour df_it_prop1
unique_items_prop2 = df_it_prop2['itemid'].nunique()
unique_properties = df_it_prop2['property'].nunique()
top_properties = df_it_prop2['property'].value_counts().head(10)

print("Nombre d'articles uniques dans df_it_prop2", unique_items_prop2)
print("Nombre de propriétés uniques", unique_properties)
print("Top 10 des propriétés les plus fréquentes", top_properties)


# Identification des doublons df_it_prop2
duplicates_events = df_it_prop2.duplicated().sum()
print("Nombre de doublons dans df_it_prop2:", duplicates_events)


# In[33]:


#


# In[34]:


# Obtenir le nombre de lignes pour chaque table
nb_lignes_category_tree = df_cat.shape[0]
nb_lignes_events = df_events.shape[0]
nb_lignes_item_properties_part1 = df_it_prop1.shape[0]
nb_lignes_item_properties_part2 = df_it_prop2.shape[0]

print("Nombre de lignes dans category_tree:", nb_lignes_category_tree)
print("Nombre de lignes dans events:", nb_lignes_events)
print("Nombre de lignes dans item_properties_part1:", nb_lignes_item_properties_part1)
print("Nombre de lignes dans item_properties_part2:", nb_lignes_item_properties_part2)


# In[35]:


#


# In[36]:


#A ce stade de l'analyse, nous allons traiter des Nan et du format des variables pour df_cat


# In[37]:


#1. Gérer les Valeurs Manquantes (NaN)
#Dans df_cat, il semble que les valeurs NaN soient présentes uniquement dans la colonne parentid.
#L'option choisie pour les gérer :

#Remplacer les NaN par une valeur spécifique car ces valeurs NaN ont une signification dans le contexte,
#indiquant des catégories de niveau supérieur sans parent.
#Nous les remplaçons par la valeur -1 :


# In[38]:


df_cat['parentid'].fillna(-1, inplace=True)


# In[39]:


#2. Vérifier et Modifier le Format des Variables
#Pour le format des variables,chaque colonne doit être dans le format approprié pour l'analyse.
#categoryid et parentid devraient être des entiers (int), car ils représentent des identifiants.


# In[40]:


# Vérifier les types de données
print(df_cat.dtypes)

# Convertir 'parentid' en entier (remplacé par -1 précédemment)
df_cat['parentid'] = df_cat['parentid'].astype(int)

print()

# Vérifier à nouveau les types de données
print(df_cat.dtypes)


# In[41]:


#Conséquences pour l'Analyse du remplacement des Nan par -1 dans df_cat['parentid'] :

#Analyse de la Hiérarchie : Avec cette modification, nous pouvons maintenant analyser plus facilement
#la structure hiérarchique de vos catégories.
#Les catégories avec parentid égal à -1 sont les catégories de niveau supérieur.


# Nous gardons à l'esprit que le -1 est une valeur artificielle,introduite pour une meilleure gestion des données.
# Cela doit est pris en compte dans toutes les analyses futures qui impliquent la colonne parentid.



# In[42]:


# Exploration des Relations entre Catégories:

# Nous analysons la structure hiérarchique des catégories
# en examinant les relations parent-enfant entre les categoryid et parentid :


# In[43]:


# Identifier les catégories de niveau supérieur (sans parent)
top_level_categories = df_cat[df_cat['parentid'] == -1]

# Identifier les sous-catégories pour chaque catégorie de niveau supérieur
sub_categories = df_cat[df_cat['parentid'] != -1]


# In[44]:


# Compter le nombre de sous-catégories pour chaque catégorie parent
sub_category_counts = sub_categories['parentid'].value_counts()

print(sub_category_counts)


# In[45]:


#Interprétation des Résultats

#Catégories avec Beaucoup de Sous-Catégories : Les catégories avec le plus grand nombre de sous-catégories
#(comme la '250' ou la '362'ayant 31 et 22 sous-catégories, etc.) peuvent être des catégories générales ou très populaires.
#Elles pourraient représenter des segments de produits vastes ou diversifiés.


#Catégories avec Peu ou une sous-catégorie : ces catégories pourraient être plus spécialisées ou moins fréquentes.


# In[46]:


#


# In[47]:


#A ce stade de l'analyse, nous allons traiter des Nan et du format des variables pour df_event


# In[48]:


#1. Gérer les Valeurs Manquantes (NaN)
#Dans df_events, les valeurs NaN sont présentes uniquement dans la colonne transactionid.
#L'option choisie pour les gérer :

#Remplacer les NaN par une valeur spécifique car ces valeurs NaN ont une signification dans le contexte,
#et indiquent des événements qui ne sont pas des transactions (des vues ou des clics),
#Nous les remplaçons par la valeur -1 pour indiquer l'absence de transaction.


# In[49]:


df_events['transactionid'].fillna(-1, inplace=True)


# In[50]:


#2. Vérifier et Modifier le Format des Variables
#Pour le format des variables,chaque colonne doit être dans le format approprié pour l'analyse.
#timestamp doit être converti en un format datetime pour une analyse temporelle plus aisée.
#visitorid, itemid, et transactionid devraient être des entiers.


# In[51]:


# Convertir 'timestamp' en datetime
df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit='ms')

# Créer la Colonne Mois

df_events['mois'] = df_events['timestamp'].dt.month

# Créer la Colonne Jour de la Semaine

df_events['jour_semaine'] = df_events['timestamp'].dt.day_name()

# Créer la Colonne Week-End

# weekday renvoie un nombre (où lundi = 0 et dimanche = 6),
# donc les jours de week-end (samedi et dimanche) sont ceux où ce nombre est supérieur à 4.

df_events['week_end'] = df_events['timestamp'].dt.weekday.apply(lambda x: 1 if x > 4 else 0)

# Créer la Colonne Jour du Mois

df_events['jour_mois'] = df_events['timestamp'].dt.day

# Créer la Colonne Heure

df_events['heure'] = df_events['timestamp'].dt.hour


# 'visitorid', 'itemid', et 'transactionid' sont des entiers
df_events['visitorid'] = df_events['visitorid'].astype(int)
df_events['itemid'] = df_events['itemid'].astype(int)
df_events['transactionid'] = df_events['transactionid'].astype(int)

# Vérifier les types de données après modification
print(df_events.dtypes)

# Afficher les premières lignes pour vérifier
print(df_events.head(10))

print(df_events.shape)


# In[52]:


#


# In[53]:


#A ce stade de l'analyse, nous allons traiter des Nan et du format des variables pour df_it_prop1


# In[54]:


#1. Gérer les Valeurs Manquantes (NaN)
#Aucun Nan présent


# In[55]:


#2. Vérifier et Modifier le Format des Variables
#Pour le format des variables,chaque colonne doit être dans le format approprié pour l'analyse.
#timestamp devrait être converti en format datetime pour une analyse temporelle.
#itemid doit être un entier.


# In[56]:


# Convertir 'timestamp' en datetime
df_it_prop1['timestamp'] = pd.to_datetime(df_it_prop1['timestamp'], unit='ms')

# Créer la Colonne Mois

df_it_prop1['mois'] = df_it_prop1['timestamp'].dt.month

# Créer la Colonne Jour de la Semaine

df_it_prop1['jour_semaine'] = df_it_prop1['timestamp'].dt.day_name()

# Créer la Colonne Week-End

# weekday renvoie un nombre (où lundi = 0 et dimanche = 6),
# donc les jours de week-end (samedi et dimanche) sont ceux où ce nombre est supérieur à 4.

df_it_prop1['week_end'] = df_it_prop1['timestamp'].dt.weekday.apply(lambda x: 1 if x > 4 else 0)

# Créer la Colonne Jour du Mois

df_it_prop1['jour_mois'] = df_it_prop1['timestamp'].dt.day

# Créer la Colonne Heure

df_it_prop1['heure'] = df_it_prop1['timestamp'].dt.hour


# 'itemid' est un entier
df_it_prop1['itemid'] = df_it_prop1['itemid'].astype(int)

# Vérifier les types de données après modification
print(df_it_prop1.dtypes)


# In[57]:


#


# In[58]:


#A ce stade de l'analyse, nous allons traiter des Nan et du format des variables pour df_it_prop2


# In[59]:


#1. Gérer les Valeurs Manquantes (NaN)
#Aucun Nan présent


# In[60]:


#2. Vérifier et Modifier le Format des Variables
#Pour le format des variables,chaque colonne doit être dans le format approprié pour l'analyse.
#timestamp devrait être converti en format datetime pour une analyse temporelle.
#itemid doit être un entier.


# In[61]:


# Convertir 'timestamp' en datetime
df_it_prop2['timestamp'] = pd.to_datetime(df_it_prop2['timestamp'], unit='ms')

# Créer la Colonne Mois

df_it_prop2['mois'] = df_it_prop2['timestamp'].dt.month

# Créer la Colonne Jour de la Semaine

df_it_prop2['jour_semaine'] = df_it_prop2['timestamp'].dt.day_name()

# Créer la Colonne Week-End

# weekday renvoie un nombre (où lundi = 0 et dimanche = 6),
# donc les jours de week-end (samedi et dimanche) sont ceux où ce nombre est supérieur à 4.

df_it_prop2['week_end'] = df_it_prop2['timestamp'].dt.weekday.apply(lambda x: 1 if x > 4 else 0)

# Créer la Colonne Jour du Mois

df_it_prop2['jour_mois'] = df_it_prop2['timestamp'].dt.day

# Créer la Colonne Heure

df_it_prop2['heure'] = df_it_prop2['timestamp'].dt.hour


# Assurer que 'itemid' est un entier
df_it_prop2['itemid'] = df_it_prop2['itemid'].astype(int)

# Vérifier les types de données après modification
print(df_it_prop2.dtypes)


# In[62]:


#


# In[63]:


#Nous avons réalisé les étapes de nettoyage, d'information et d'uniformisation pour ces 4 df.

#Maintenant, nous allons envisager les fusions pertinentes ou pas entre ces quatre dataframes


# In[64]:


# Q1 :Concatener les jeux de données item_1 et item_2 dans un nouveau dataframe item


# In[65]:


# Concaténation des dataframes df_it_prop1 et df_it_prop2
item = pd.concat([df_it_prop1, df_it_prop2], ignore_index=True)

#Dans ce code, ignore_index=True est utilisé pour réinitialiser l'index dans le nouveau dataframe item.
#Cela garantit que les indices des deux dataframes originaux ne se chevauchent pas dans le dataframe concaténé.


# In[66]:


#Vérifications
print(item.head())

print(item.shape)

#La taille du nouveau dataframe est affiché pour s'assurer qu'il contient le nombre de lignes attendu,
#qui devrait être la somme des lignes des deux dataframes originaux


# In[67]:


#nbre d'itemid dans item
liste_itemid_item=item.itemid.unique()
print(liste_itemid_item)
print("Nbre itemid dans item")
print(len(liste_itemid_item))


# In[68]:


# Q2: Supprimer les itemid qui sont présents dans le nouveau dataframe item
#(résultant de la concaténation de item_properties_part1 et item_properties_part2) mais pas dans df_events

# Nous effectuons une opération de filtrage :

# Filtrer les 'itemid' dans 'item' qui sont également présents dans 'df_events'
item_filtered = item[item['itemid'].isin(df_events['itemid'])]

# Cette ligne de code crée un nouveau dataframe item_filtered qui ne contient que les lignes de item
# où itemid est également présent dans df_events.




# Avant de procéder à la suppression des itemid dans df_events, nous vérifions s'il y a des itemid dans
#df_events qui ne sont pas présents dans item :

print("Nombre unique d'itemid dans df_events:", df_events['itemid'].nunique())
print("Nombre unique d'itemid dans item_filtered:", item_filtered['itemid'].nunique())



# In[69]:


#La différence dans le nombre d'itemid uniques entre df_events et item_filtered indique
#que certains itemid dans df_events ne se trouvent pas dans le nouveau dataframe item.
#C'est normal et attendu, étant donné que nous avons filtré item pour ne garder que
#les itemid présents dans df_events.

#Le fait que item_filtered contienne moins d'itemid uniques que df_events (76470 contre 90424)
#suggère que item ne couvre pas tous les itemid de df_events.
#Cela peut être dû à diverses raisons, telles que des enregistrements manquants ou
#des différences dans la période de collecte des données entre les deux ensembles de données.


# In[70]:


# Q3 : créer deux nouveaux DataFrames item_availability et item_categoryid à partir du dataframe item_filtered

# Pour créer deux nouveaux DataFrames item_availability et item_categoryid à partir du dataframe item_filtered,
# Nous devons filtrer les données en fonction de la valeur de la colonne property :


# In[71]:


# Créer item_availability
#Ce DataFrame contiendra les lignes où property est égal à "available" :

#item_availability = item_filtered[item_filtered['property'] == 'available']

item_availability = item_filtered[item_filtered['property'] == 'available'].copy()

# Nous utilisons la méthode .copy() pour créer explicitement une copie indépendante du DataFrame
# lors de sa création initiale : une copie indépendante du DataFrame élimine le SettingWithCopyWarning
# lors du tri plus tard.


# In[72]:


#Créer item_categoryid
#Ce DataFrame contiendra les lignes où property est égal à "categoryid" :

item_categoryid = item_filtered[item_filtered['property'] == 'categoryid'].copy()


# In[73]:


# Vérification et exploration :

# Vérifier les premières lignes et la structure des nouveaux DataFrames
print(item_availability.head())
print(item_categoryid.head())

print("Nombre de lignes dans item_availability:", item_availability.shape[0])
print("Nombre de lignes dans item_categoryid:", item_categoryid.shape[0])


# In[74]:


#Q4 : Faire une première jointure merge_asof entre df et item_availability (dans un dataframe nommé merged_1 par ex)



# In[75]:


# Pour effectuer une jointure de type merge_asof entre df_events et item_availability, nous devons d'abord
# nous assurer que les deux DataFrames ont une colonne en commun sur laquelle la jointure peut être effectuée.
# Typiquement, pour merge_asof, cette colonne est souvent une colonne de temps.


# In[79]:


# Étape 1: Préparation des DataFrames
# On s'assure que les colonnes timestamp sont triées dans les deux DataFrames :

df_events.sort_values(by='timestamp', inplace=True)
item_availability.sort_values(by='timestamp', inplace=True)


# In[80]:


#Étape 2: Jointure avec merge_asof

merged_1 = pd.merge_asof(df_events, item_availability, on='timestamp', by='itemid', direction="nearest")

#Le paramètre by='itemid' assure que la jointure est également basée sur la correspondance des
#itemid entre les deux DataFrames et le paramètre direction="nearest" permet l'ajustement des
#timestamps pour éviter des NaN


# In[81]:


#Étape 3: Vérification du résultat

print(merged_1.head())

print(merged_1.dtypes)

print(merged_1.shape)


# In[82]:


#Q5 : Deuxième jointure merge_asof entre merged_1 et item_category (dans un dataframe nommé merged_2 par ex)

# Étape 1: Préparer item_categoryid
# nous vérifions que item_categoryid est trié par timestamp et que la colonne itemid est présente pour la jointure:

item_categoryid.sort_values(by='timestamp', inplace=True)


# In[83]:


# Étape 2: Effectuer la jointure merge_asof
# Cette jointure se fait entre merged_1 et item_categoryid, en utilisant à nouveau timestamp comme clé de jointure
# et en s'assurant que les itemid correspondent :

merged_2 = pd.merge_asof(merged_1, item_categoryid, on='timestamp', by='itemid', direction="nearest", suffixes=('_x', '_y'))


# In[84]:


# Étape 3: Vérification du résultat

print(merged_2.head())

print(merged_2.dtypes)

print(merged_2.shape)


# In[85]:


# Q6 : Renommer les variables value_x et value_y trouvés après le merge par des noms plus informatifs
#(soit available et category_id) et supprimer les variables property_x et property_y.


# In[86]:


# Étape 1: Renommer les colonnes

merged_2.rename(columns={'value_x': 'available', 'value_y': 'category_id'}, inplace=True)


# In[87]:


# Étape 2: Supprimer les colonnes inutiles

merged_2.drop(columns=['property_x', 'property_y'], inplace=True)


# In[88]:


# Étape 3: Vérifier le résultat

print(merged_2.head())

print(merged_2.dtypes)

print(merged_2.shape)


# In[89]:


# Q7 : Faire une dernière jointure à gauche entre merged_2 et le csv category_tree.
#Ici on fait une jointure à gauche pour ne pas perdre les informations sur les produits qui n'ont pas
#de category dans le dataframe.


# In[90]:


# Dans cette jointure, nous alignons les données en fonction de la colonne category_id dans merged_2
# et de la colonne categoryid dans df_cat donc nous devons nous assurer que les types de données des
# colonnes utilisées pour la jointure sont identiques.

# Étape 1:  Préparation pour la jointure
# on s'assure que les colonnes à joindre (category_id dans merged_2 et categoryid dans df_cat) sont du même type

merged_2['category_id'] = merged_2['category_id'].astype('float')
df_cat['categoryid'] = df_cat['categoryid'].astype('float')

print(df_cat.shape)
print(merged_2.shape)


# In[91]:


# Étape 2: Effectuer la jointure à gauche :

final_merged = pd.merge(merged_2, df_cat, how='left', left_on='category_id', right_on='categoryid')


# In[92]:


# Étape 3: Vérification du résultat :

print(final_merged.head())

print(final_merged.shape)


# In[93]:


# Q8 : Générer 3 nouvelles variables "addtocart", "transaction", "view" à partir de la variable 'event'


# In[94]:


# Pour générer trois nouvelles variables (colonnes) addtocart, transaction, view à partir de la
#variable event dans final_merged, nous pouvons utiliser une technique appelée "one-hot encoding".
#Cela implique de créer une nouvelle colonne pour chaque valeur unique dans event, où chaque
#colonne contiendra des valeurs binaires (0 ou 1) indiquant la présence de cet événement.


# Étape 1: Étape 1: Créer les nouvelles colonnes
# Nous créons trois nouvelles colonnes, chacune correspondant à une valeur différente de event
#('addtocart', 'transaction', 'view').
# Chaque colonne aura une valeur de 1 là où event correspond à cette valeur, et 0 sinon.

#Étape 1: Créer les nouvelles colonnes

final_merged['addtocart'] = (final_merged['event'] == 'addtocart').astype(int)
final_merged['transaction'] = (final_merged['event'] == 'transaction').astype(int)
final_merged['view'] = (final_merged['event'] == 'view').astype(int)


# Étape 2: Vérification des nouvelles colonnes
print(final_merged[['event', 'addtocart', 'transaction', 'view']].head())



print(final_merged.dtypes)


# In[95]:


print(final_merged.columns)


# In[96]:


# Q9 : Faire un groupby item_id et appliquer une agrégation sur les variables
# "addtocart" -> somme
# "transaction" -> somme
# "view" -> somme
# "available" -> garder que la dernière valeur
# "category_id" -> garder la première valeur
# "parentid" -> garder la première valeur
# "mois"-> garder le nombre de mois unique
# "visitorid"-> garder le nombre de visiteurs uniques qui ont consulté le produit


# In[97]:


# D'après l'index des colonnes et les types de données, il semble que nous ayons des colonnes en double
#dans final_merged, notamment addtocart, transaction, et view.
#Cela peutt être une source d'erreur  lors de l'agrégation.


# In[98]:


# Étape de préparation: Supprimer les colonnes en double
# Avant d'effectuer l'agrégation, nous supprimons les colonnes en double :

# Garder une seule instance de chaque colonne en double
cols_to_keep = {
    'addtocart': 'addtocart',
    'transaction': 'transaction',
    'view': 'view'
}
final_merged = final_merged.loc[:, ~final_merged.columns.duplicated()].rename(columns=cols_to_keep)


# In[99]:


#  Pour effectuer un groupement par itemid dans le DataFrame final_merged et appliquer les agrégations
#spécifiées sur différentes colonnes, nous pouvons utiliser la méthode groupby de Pandas en combinaison
#avec .agg() :


# Étape 1: Groupement et Agrégation

aggregated_data = final_merged.groupby('itemid').agg({
    'addtocart': 'sum',                   # Somme des valeurs pour addtocart
    'transaction': 'sum',                 # Somme des valeurs pour transaction
    'view': 'sum',                        # Somme des valeurs pour view
    'available': 'last',                  # Dernière valeur pour available
    'category_id': 'first',               # Première valeur pour category_id
    'parentid': 'first',                  # Première valeur pour parentid
    'mois': pd.Series.nunique,            # Nombre de mois uniques
    'visitorid': pd.Series.nunique        # Nombre de visiteurs uniques
}).reset_index()


# In[100]:


# Étape 2: Vérification des résultats

print(aggregated_data.head(20))
print(aggregated_data.shape)
aggregated_data.isna().sum()


# In[101]:


# Identification des doublons aggregated_data
duplicates = aggregated_data.duplicated().sum()
print("Nombre de doublons dans aggregated_data:", duplicates)


# In[102]:


#elimination des Nan
aggregated_data.dropna(subset=['available', 'category_id', 'parentid'], inplace=True)


print(aggregated_data.shape)


# In[103]:


#Datavisualisation


# In[104]:


#Pour nous aider à l'analyse et à la détection de tendances, voici quelques visualisations puis
#nous passerons à l'analyse.


# In[105]:


# Etape 1. Histogrammes et Barplots
# le but visualiser la distribution des actions (comme addtocart, transaction, view) pour différents itemid.


# In[106]:


# Histogramme pour 'addtocart'
fig = px.histogram(aggregated_data, x='addtocart', title="Distribution of Add to Cart Events")
fig.show()


# In[107]:


# Histogramme pour 'transaction'
fig = px.histogram(aggregated_data, x='transaction', title="Distribution of transaction Events")
fig.show()


# In[108]:


# Histogramme pour 'view'
fig = px.histogram(aggregated_data, x='view', title="Distribution of view / Events")
fig.show()


# In[109]:


# Etape 2. Scatter Plot
# Pour analyser la relation entre view et addtocart.


# In[110]:


sns.scatterplot(data=aggregated_data, x='view', y='addtocart')
plt.title("View vs Add to Cart")
plt.xlabel("View")
plt.ylabel("Add to Cart")
plt.show()


# In[111]:


# Pour analyser la relation entre view et transaction.
sns.scatterplot(data=aggregated_data, x='view', y='transaction')
plt.title("View vs transaction")
plt.xlabel("View")
plt.ylabel("transaction")
plt.show()


# In[112]:


# Etape 3. Heatmap pour Corrélation entre Variables
# Nous cherchons à visualiser les corrélations entre différentes variables telles que addtocart, transaction, view
# et visitorid.
# Cela peut aider à identifier des relations intéressantes entre les actions des utilisateurs.


# In[113]:


# Calcul de la matrice de corrélation
corr = aggregated_data[['addtocart', 'transaction', 'view', 'visitorid']].corr()

# Heatmap avec Seaborn
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[114]:


# Nous identifions une relation intéressante entre view et visitorid


# In[115]:


# Etape 4. Pair Plot
# Nous cherchons à visualiser les relations bivariées entre plusieurs paires de variables


# In[116]:


# Pair Plot avec Seaborn
sns.pairplot(aggregated_data[['addtocart', 'transaction', 'view', 'visitorid']])
plt.show()


# In[117]:


# Etape 5. Bubble Chart
# Nous cherchons à visualiser visualiser trois dimensions de données :
# view et addtocart sur les axes X et Y et la taille du bubble pour représenter visitorid.


# In[118]:


# Bubble chart avec Plotly
fig = px.scatter(aggregated_data, x="view", y="addtocart", size="visitorid", color="category_id", title="Bubble Chart: Views vs AddToCart with Visitor Size")
fig.show()


# In[119]:


# Etape 6. Etat des stocks, quelle est la proportion de produits disponible/non disponible? =>sns.countplot available
# sur la période considérée, nous avons un peu plus de produits indisponibles en comparaison aux produits disponibles,
# Une influence la non réalisation des transactions?


# Création du graphique
sns.countplot(x='available', data=aggregated_data, palette='Set2')

# Ajout du titre et des étiquettes d'axe
plt.title("Disponibilité des Produits", fontsize=16)
plt.xlabel("Statut de Disponibilité", fontsize=12)
plt.ylabel("Nombre de Produits", fontsize=12)

# Améliorer la légende
# S'available' est un indicateur binaire (comme 0 et 1):
plt.xticks([0, 1], ['Indisponible', 'Disponible'])

# Afficher les valeurs au-dessus des barres
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='gray', xytext=(0, 5), textcoords='offset points')

# Afficher le graphique
plt.show()


# In[120]:


# Etape 7 Répartitions des interactions par mois

import plotly.graph_objects as go

values = aggregated_data.mois.value_counts()

# Création du Pie Chart
fig = go.Figure(go.Pie(
    values=values.values,
    labels=values.index,
    pull=[0.1 if i == values.values.argmax() else 0 for i in range(len(values))],  # Mettre en évidence la part la plus grande
    hoverinfo='label+percent',  # Afficher le pourcentage et le label au survol
    textinfo='value'  # Afficher la valeur sur les parts
))

# Personnalisation du layout
fig.update_layout(
    title_text='Répartition des Interactions par Mois',  # Titre du graphique
    title_x=0.5,  # Centrer le titre
    legend_title='Mois',  # Titre pour la légende
    legend=dict(
        traceorder='normal',
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        )
    )
)

fig.show()

# la période de mai à juin semble être la période qui a le plus d'evènements


# In[121]:


#Etape 8 : Vente par catégorie

fig = px.box(aggregated_data, x='category_id', y='addtocart', color='category_id', title="Addtocart by Category")
fig.update_layout(
    xaxis_title="ID de la Catégorie",
    yaxis_title="Nombre d'ajouts au panier",
    legend_title="ID de la Catégorie",
    font=dict(family="Arial, sans-serif", size=12, color="RebeccaPurple"),
    title_font_size=20,
    legend=dict(yanchor="middle", y=0.5, xanchor="right", x=1.05),
    template='plotly_dark',
    bargap=0.2,
    xaxis_tickangle=-45,
    xaxis=dict(tickfont=dict(size=10)),
    xaxis_rangeslider_visible=True
)
fig.update_traces(boxpoints='outliers')
fig.show()


# In[122]:


#créer une nouvelle colonne target dans votre DataFrame aggregated_data, qui indiquera 0 s'il n'y a pas de
#transactions et 1 s'il y a au moins une transaction pour chaque enregistrement
#Nous allons utiliser une approche conditionnelle :


# In[123]:


aggregated_data['target'] = (aggregated_data['transaction'] > 0).astype(int)


# In[124]:


#Exploration initiale et Analyse descriptive de aggregated_data

print(aggregated_data.head(10))
print(aggregated_data.tail(10))
print(aggregated_data.info())
aggregated_data.describe()


# In[125]:


#La nouvelle colonne target a été correctement créée dans aggregated_data. Cette colonne contient des valeurs
#binaires (0 ou 1), où :

#0 indique qu'il n'y a pas eu de transactions pour cet itemid
#1 indique qu'il y a eu au moins une transaction pour cet itemid
# La structure du df montre maintenant 10 colonnes, y compris la nouvelle colonne target.
#La colonne target est correctement typée en tant qu'entier (int64),
#et elle a une valeur non nulle pour chaque enregistrement dans le DataFrame,


# In[126]:


#Pour utiliser la colonne available dans des modèles de machine learning, nous devons convertir ces catégories
#en valeurs numériques. L'utilisation de l'encodage One-Hot est ce qui est conseillé.

available_dummies = pd.get_dummies(aggregated_data['available'], prefix='available')
aggregated_data = pd.concat([aggregated_data, available_dummies], axis=1)


# In[127]:


#Nouvelle exploration initiale et Analyse descriptive de aggregated_data

print(aggregated_data.head(10))
print(aggregated_data.tail(10))
print(aggregated_data.info())
aggregated_data.describe()


# In[128]:


#Nous pouvons envisager de supprimer la colonne originale available pour éviter la redondance dans les données :
#aggregated_data.drop('available', axis=1, inplace=True)


# In[129]:


#Pour créer deux DataFrames distincts à partir de aggregated_data, un pour la régression et un autre
#pour la classification, nous devons d'abord déterminer quels ensembles de caractéristiques (features)
#seront utilisés pour chaque tâche comme le choix de la variable cible (target) .


# In[130]:


#DataFrame pour la Régression :

# Nous souhaitons prédire 'view' car il est crucial de choisir une variable cible qui varie de manière continue :
#features_for_regression = ['itemid', 'addtocart', 'transaction', 'available_0', 'available_1', 'category_id', 'parentid', 'mois', 'visitorid']
#df_regression = aggregated_data[features_for_regression]


# Inclure 'view' en tant que colonne cible dans df_regression
features_for_regression = ['itemid', 'addtocart', 'transaction', 'available_0', 'available_1', 'category_id', 'parentid', 'mois', 'visitorid']
target_for_regression = ['view']

# S'assurer d'inclure la variable cible dans le DataFrame
df_regression = aggregated_data[features_for_regression + target_for_regression]


# In[131]:


#DataFrame pour la Classification :

# Nous souhaitons prédire si une transaction a eu lieu (la colonne target crée). Nous incluons toutes les
# autres variables caractéristiques sauf celles utilisées spécifiquement pour la régression.


# La variable cible est 'target'
features_for_classification = ['itemid', 'addtocart', 'view', 'available_0', 'available_1', 'category_id', 'parentid', 'mois', 'visitorid', 'target']
df_classification = aggregated_data[features_for_classification]





# In[132]:


#Question 1 : Entrainer 2 modéles sur la régression :


