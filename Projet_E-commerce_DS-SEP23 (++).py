# LIBRARY IMPORT

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import plotly.graph_objects as go
import plotly.express as px
import io

# Import CSV
df_it_prop1 = pd.read_csv("item_properties_part1.csv")
df_it_prop2 = pd.read_csv("item_properties_part2.csv" )
df = pd.read_csv("events.csv")
tree = pd.read_csv("category_tree.csv" )

st.title("Projet E-commerce DS - SEP23")
st.sidebar.title("Sommaire")
pages=["Sommaire", "I- Exploration des données", "II-Transformation et Préprocessing", "III- Datavisualisation", "IV- Modèles ML supervisés et non supervisés" , "V- Interprétation et conclusion"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
  st.write("### Sommaire")

if page == pages[1] : 
# Exploration des datasets

  st.write("## I- Exploration des données")
  st.write("### Importation et affichage des 5 premières lignes de chaque DataFrame")
  st.write("##### Fichier 1: item_properties_part1 : 2520260 lignes, 4 colonnes")
  st.dataframe(df_it_prop1.head(5))

  st.write("Infos du DataFrame")
  buffer = io.StringIO()
  df_it_prop1.info(buf = buffer)
  s = buffer.getvalue()
  st.text(s)
  st.markdown("""---""")
  st.write("##### Fichier 2: item_properties_part2 : 2 115 992 lignes, 4 colonnes")
  st.dataframe(df_it_prop2.head(5))

  st.write("Infos du DataFrame")
  buffer = io.StringIO()
  df_it_prop2.info(buf = buffer)
  s = buffer.getvalue()
  st.text(s)

  st.markdown("""---""")
  st.write("##### Fichier 3: events : 275 610 lignes, 5 colonnes")
  st.dataframe(df.head(5))

  st.write("Infos du DataFrame")
  buffer = io.StringIO()
  df.info(buf = buffer)
  s = buffer.getvalue()
  st.text(s)

  st.markdown("""---""")
  st.write("##### Fichier 4: category : 1669 lignes, 2 colonnes")
  st.dataframe(tree.head(5))

  st.write("Infos du DataFrame")
  buffer = io.StringIO()
  tree.info(buf = buffer)
  s = buffer.getvalue()
  st.text(s)

if page == pages[2] : 
  st.write("## II-Transformation et Préprocessing")
  st.write("### Concaténation des 2 fichiers qui ont la même structure en un df nommée item")
  item = pd.concat([df_it_prop1,df_it_prop2], axis = 0)
  st.write("#### Nettoyage : vérification des valeurs manquantes et doublons")

  st.write("Pourcentage de valeurs manquantes dans les variables de item")
  st.write(pd.DataFrame(index = item.columns, columns = ['%_valeurs_manquantes'], data = (item.isna().sum().values / len(item)*100)))
  st.write("Doublons : ",item.duplicated().sum())

  st.markdown("""---""")
  #nbre d'itemid dans item
  liste_itemid_item = item.itemid.unique()
  #Liste d'item et nbre d'itemid dans df
  liste_itemid_df = df['itemid'].unique()

  st.write("#### Filtrage de df avec les itemid commun à item et df")
  df = df.loc[df['itemid'].isin(liste_itemid_item)]
  df = df.reset_index(drop = True)
  st.write("Le nombre d'iteme uniques commun au DataFrame item & df : ",len(df))
  st.write("Seules les informations sur la disponibilité ou non du produit ainsi que sa catégorie peuvent être utiles dans item")
  #création de 2 df item_ availability et item_categ
  item_availability=item.loc[item.property == 'available']
  item_category=item.loc[item.property == 'categoryid']

  item_category = item_category.reset_index(drop = True)
  item_availability = item_availability.reset_index(drop = True)
  st.write("Utilisation de 'merge_asof' pour récuprer les informations sur l'évenement et les propriétés des produits au moment le plus précis où l'evement s'est déroulé")
  st.write("REVOIR FORMULATION AU DESSUS")

  #fusion de df avec item_availability pour récupérer les infos sur la disponibilité de nos produits  
  item_availability.itemid = item_availability.itemid.astype('int64')

  merged_1 = pd.merge_asof(df.sort_values('timestamp'),item_availability.sort_values('timestamp'),by = 'itemid', on = 'timestamp',direction = 'nearest')
  merged_1.head()

  #création merged_2 pour récupérer à présent les categories de certains produits
  item_category.itemid = item_category.itemid.astype('int64')

  merged_2 = pd.merge_asof(merged_1.sort_values('timestamp'),item_category.sort_values('timestamp'),by = 'itemid', on = 'timestamp',direction = 'nearest')
  merged_2.head()

  st.write("Le df nommé merged_2 comporte ainsi toutes les informations récupérées du df item et du df(event)")
  st.markdown("""---""")
  st.write("#### Néttoyage du DataFrame merged_2 :")

  st.write("Suppression des colonnes property et transactionid")
  merged_2 = merged_2.drop(['property_x','property_y','transactionid'], axis=1)

  st.write("Renommage des colonnes pour plus de clarté")
  merged_2 = merged_2.rename(columns = {'value_x': "available", "value_y": "categoryid"})

  merged_2.categoryid = merged_2.categoryid.astype(int)

  buffer = io.StringIO()
  merged_2.info(buf = buffer)
  s = buffer.getvalue()
  st.text(s)
  
  st.markdown("""---""")
  st.write("#### Dernière fusion du df merged_2 avec ce 4ième fichier pour récupérer les parentid")
  #nous allons à présent récuperer les informations sur les parentid correspondant à nos catégories dans tree
  df_final = merged_2.merge(tree, how = 'left', on = 'categoryid')

  st.write("1- Création de 3 nouvelles colonnes issues de la colonne event pour avoir une colonne par vus/Ajout au panier/transaction")
  #création de 3 nouvelles variables à partir de la colonne event
  df_final = df_final.join(pd.get_dummies(data = df_final['event']))

  st.write("2- Conversion de timestamp en datetime")
  # conversion en datetime de timestamp
  df_final['timestamp'] = pd.to_datetime(df_final['timestamp'], unit = 'ms')

  st.write("3- Création d'une colonne month")
  # Créer la Colonne Mois
  df_final['month'] = df_final['timestamp'].dt.month

  st.write("4- Vérification et derniers nettoyages")
  st.write(df_final.head())
  st.write(df_final.shape)

  st.write("Vérification NaN, duplicated, types col")
  st.write(pd.DataFrame(index = df_final.columns, columns = ['%_valeurs_manquantes'], data = (df_final.isna().sum().values / len(item)*100)))
  st.write("Doublons : ",df_final.duplicated().sum())
  st.write("Types des variables : ")
  st.write(df_final.dtypes)

  st.write("Remplacement des fillna de parentid par 9999 qui correspond à Other")
  df_final['parentid'].fillna(9999, inplace=True)

  st.write("Conversion des variables")
  df_final[['available','parentid','addtocart','transaction','view']] = df_final[['available','parentid','addtocart','transaction','view']].astype(int)
  
  st.write("Aperçu des doublons")
  duplicates_df_final = df_final[df_final.duplicated(keep = False)]
  st.write(duplicates_df_final)
  st.write("Suppression doublons car ils sont identiques")
  df_final = df_final.drop_duplicates()
  st.write("Suppression des colonnes event et timestamp qui ne sont plus utiles")
  df_final = df_final.drop(['event','timestamp'], axis = 1)

  st.markdown("""---""")
  st.write("#### Agrégation des données grâce au groupby sur les variables, application des fonctions d'agrégation différentes en fonction des variables : ")
  
  st.write("addtocart   : somme")
  st.write("transaction : somme")
  st.write("view        : somme")
  st.write("available   : garder la dernière valeur")
  st.write("categoryid  : garder la première valeur")
  st.write("parentid    : garder la première valeur")
  st.write("month       : garder le nombre de mois unique")
  st.write("visitorid   : garder le nombre de visiteurs uniques qui ont consulté le produit")

  #agregation des données
  dictag = {'addtocart':'sum','transaction':'sum','view':'sum','available':'last','categoryid':'first','parentid':'first','month':'nunique','visitorid':'nunique' }
  df_final_ag = df_final.groupby('itemid').agg(dictag)
  
  st.markdown("""---""")
  st.write("#### Aperçu du df_final qui servira pour les parties Visualisation et Modèles de Machine Learning")
  st.write(df_final_ag.head())
  st.write(df_final_ag.shape)
  st.write(df_final_ag.describe())

if page == pages[3] : 
  st.write("### III- Datavisualisation")

if page == pages[4] : 
  st.write("### IV- Modèles ML supervisés et non supervisé")

#DataFrame pour la Régression :

# Nous souhaitons prédire 'view' car il est crucial de choisir une variable cible qui varie de manière continue :
#features_for_regression = ['itemid', 'addtocart', 'transaction', 'available_0', 'available_1', 'category_id', 'parentid', 'mois', 'visitorid']
#df_regression = aggregated_data[features_for_regression]


# Inclure 'view' en tant que colonne cible dans df_regression
features_for_regression = ['itemid', 'addtocart', 'transaction', 'available_0', 'available_1', 'category_id', 'parentid', 'mois', 'visitorid']
target_for_regression = ['view']

# S'assurer d'inclure la variable cible dans le DataFrame
df_regression = aggregated_data[features_for_regression + target_for_regression]

#Pour prédire la variable view dans une tâche de régression, nous utilisons la régression linéaire

#1 Préparation des données

from sklearn.model_selection import train_test_split

# Séparer les caractéristiques (X) et la cible (y)
X = df_regression.drop('view', axis=1)
y = df_regression['view']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#2. Normalisation des caractéristiques
#La normalisation est particulièrement utile pour la régression linéaire et peut également bénéficier à la forêt aléatoire
#dans certains cas.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#3. Construction et Entraînement du Modèle
#Régression Linéaire

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

#3. Construction et Entraînement des Modèles
#Régression Linéaire

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

#4 Évaluation des Modèles
# Nous utilisons des métriques différentes pour évaluer les performances du modèle

#MSE : Le MSE mesure la moyenne des carrés des erreurs, c'est-à-dire la différence entre les valeurs observées
# et celles prédites par le modèle.

# RMSE (Root Mean Squared Error) : Le RMSE est simplement la racine carrée du MSE, ce qui rend l'erreur
#plus interprétable car elle est dans les mêmes unités que la variable cible.

#MAE (Mean Absolute Error) : Le MAE mesure la moyenne des erreurs absolues. Contrairement au MSE, le
# MAE n'élève pas les erreurs au carré, ce qui donne une meilleure idée des erreurs en termes réels.

# R² Score (Coefficient of Determination) : Le R² Score mesure à quel point les variations de la variable cible
# peuvent être expliquées par les caractéristiques du modèle. Un score de 1 indique que le modèle
# explique parfaitement toute la variabilité, tandis qu'un score de 0 indiquerait que le modèle n'explique pas
# du tout la variabilité.


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Prédictions
y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

# Évaluation pour la régression linéaire
print("Régression Linéaire:")
print("MSE:", mean_squared_error(y_test, y_pred_linear))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_linear)))
print("MAE:", mean_absolute_error(y_test, y_pred_linear))
print("R2 Score:", r2_score(y_test, y_pred_linear))

#5 Interpretation

#Les résultats montrent des performances intéressantes du modèle sur la tâche de prédiction du nombre de vues (`view`),


### Régression Linéaire

#MSE et RMSE sont relativement bas, ce qui indique une bonne précision du modèle.
#MAE est également faible, suggérant que les erreurs moyennes de prédiction sont minimes.
#R2 Scoretrès proche de 1 (0.989) montre que le modèle de régression linéaire explique une grande partie de la variance des données,ce qui est excellent.

#Visualition de la régression linéaire
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (10,10))
plt.scatter(y_pred_linear, y_test, c='green')
plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color = 'red')
plt.xlabel("prediction")
plt.ylabel("vrai valeur")
plt.title('Régression Linéaire pour les view')
plt.show()

#######################
# Isolation Forest 

#Notre jeux de donnée est très déséquilibré avec un classe majoritaire ( 0) très dominante et une deuxième classe très minoritaire
#donc nous allons utiliser l'Isolation Forest 
#Ce modèle est conçu pour identifier les observations qui s'écartent de la norme, ce qui peut correspondre à notre
#classe minoritaire dans un contexte déséquilibré

#Nous allons d'abord entraîner les modèles sur une partie des données (ensemble d'entraînement) puis évaluer leur performance sur une
#autre partie (ensemble de test), et enfin évaluer leur performance sur l'intégralité des données pour comparer.

# 1 :

# Séparation en Ensembles d'Entraînement et de Test

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Préparation des données
X = df_classification.drop('target', axis=1)
y = df_classification['target']

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2 :

# Isolation Forest
#Entraînement et Évaluation sur l'Ensemble de Test

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score

# Entraînement d'Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso_forest.fit(X_train_scaled)

# Prédiction sur l'ensemble de test
y_pred_test_iso = iso_forest.predict(X_test_scaled)
y_pred_test_iso = np.where(y_pred_test_iso == 1, 0, 1)  # Conversion

# Évaluation sur l'ensemble de test
print("Isolation Forest Performance sur le Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred_test_iso))
print(classification_report(y_test, y_pred_test_iso))


#Évaluation sur l'Intégralité des Données

X_scaled_full = scaler.transform(X)
y_pred_full_iso = iso_forest.predict(X_scaled_full)
# Convertir les prédictions pour correspondre aux étiquettes cibles (1 pour anomalies, 0 pour normal)
y_pred_full_iso = np.where(y_pred_full_iso == 1, 0, 1)

print("Isolation Forest Performance sur l'Intégralité des Données:")
print("Accuracy:", accuracy_score(y, y_pred_full_iso))
print(classification_report(y, y_pred_full_iso))

# 3 :

#Matrice de Confusion
#La matrice de confusion pour chaque modèle sur l'ensemble de test peut nous fournir des insights visuels sur leur performance,
#en particulier la répartition des vrais positifs, faux positifs, vrais négatifs, et faux négatifs.

# Isolation Forest

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calcul de la matrice de confusion pour Isolation Forest
cm_iso = confusion_matrix(y_test, y_pred_test_iso)

# Tracé de la matrice de confusion
plt.figure(figsize=(7, 5))
sns.heatmap(cm_iso, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Matrice de Confusion pour Isolation Forest')
plt.xlabel('Prédit')
plt.ylabel('Vrai')
plt.show()


#Calcul des moyennes géométriques  Isolation Forest

from imblearn.metrics import geometric_mean_score

# Pour Isolation Forest
gmean_iso = geometric_mean_score(y_test, y_pred_test_iso)

print(f"Moyenne Géométrique pour Isolation Forest: {gmean_iso}")

#Isolation Forest: 0.727
#Avec une moyenne géométrique d'environ 0.727, Isolation Forest montre une capacité raisonnable à identifier
#correctement les deux classes, mais cette valeur suggère aussi qu'il y a de la place pour l'amélioration.
#Un score parfait serait de 1, ce qui indiquerait que le modèle a un taux de vrais positifs de 100%
#pour toutes les classes.
#En pratique, une valeur supérieure à 0.7 est souvent considérée comme bonne dans des contextes
#où les classes sont déséquilibrées.
#Cela indique que le modèle a un équilibre assez bon entre la sensibilité (capacité à détecter les anomalies)
#et la spécificité (capacité à identifier les observations normales).





if page == pages[5] : 
  st.write("### V- Interprétation et conclusion")
