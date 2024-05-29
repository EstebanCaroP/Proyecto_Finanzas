# Este apartado será para disponer de todas las funciones requeridas para el proyecto de recursos humanos

# ------------------------------- Librerias necesarias ------------------------------- 

# Librerias necesarias 
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer # Para imputación
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler # Escalar variables 
from sklearn.feature_selection import RFE
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# Selecciona modelos 
def sel_variables(modelos, X, y, threshold):
    
    var_names_ac = np.array([])
    for modelo in modelos:
        #modelo=modelos[i]
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit = True, threshold = threshold)
        var_names = modelo.feature_names_in_[sel.get_support()]
        var_names_ac = np.append(var_names_ac, var_names)
        var_names_ac = np.unique(var_names_ac)
    
    return var_names_ac


# Validación del rendimiento de los modelos 
def medir_modelos(modelos, scoring, X, y, cv):

    metric_modelos = pd.DataFrame()
    for modelo in modelos:
        scores = cross_val_score(modelo, X, y, scoring = scoring, cv = cv )
        pdscores = pd.DataFrame(scores)
        metric_modelos = pd.concat([metric_modelos,pdscores], axis = 1)
    
    metric_modelos.columns = ["logistic_r","rf_classifier","sgd_classifier","xgboost_classifier"]
    return metric_modelos