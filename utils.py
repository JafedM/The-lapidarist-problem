import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error as mse, r2_score as r2, mean_absolute_error as mae


#----Cleaning data
def remove_outliers(df, col_name, thres=1.5):
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    uplim = Q3 + (thres*IQR)
    lwlim = Q1 - (thres*IQR)

    outliers = df[ (df[col_name] < lwlim) | (df[col_name] > uplim) ]
    print('{} had {} outliers.'.format(col_name, len(outliers)))
    df = df.drop(outliers.index)

    return df



#---Variables analysis
def get_boxplot(df):
    df.plot(kind='box',figsize=(15,8),subplots=True,layout=(4,4))
    plt.show()
    pass

def get_coor_matrix(df):
    corr = df.corr(numeric_only=True)
    return corr.style.background_gradient(cmap='coolwarm')

def plot_histograms(df):
    cat_cols = df.select_dtypes(include=[object])
    
    with sns.axes_style("darkgrid"):
        for col in cat_cols.columns:
            df[col].hist(bins=len(df[col].value_counts()), color='skyblue')
            plt.title(col)
            plt.grid(False)
            plt.show()

def compare_price(df, col, val1, val2, obj='price'):
    val1_data = df[df[col]==val1]
    val2_data = df[df[col]==val2]

    print('{} resume: \n {}'.format(val1, val1_data[obj].describe()))
   
    print('\n{} resume: \n {}'.format(val2, val2_data[obj].describe()))
    pass
    

#----Train models
def train_and_eval(model, parameters, x_train, x_valid, y_train, y_valid, model_name=''):
    clf = GridSearchCV(model, parameters, n_jobs=-1)
    clf.fit(x_train, y_train.reshape((1,-1))[0])

    y_pred = clf.predict(x_valid)
    
    print(\
        'Evaluation metrics for {}. \n   RMSE:{}, MAE:{}, R2:{} \n'.format(model_name, np.sqrt(mse(y_valid,y_pred)), mae(y_valid,y_pred), r2(y_valid,y_pred))
        )
    print("The best parameters across ALL searched params:\n", clf.best_params_)

def train_svm(x_train, x_valid, y_train, y_valid, model_name=''):
    model = SVR()
   
    model.fit(x_train, y_train)

    y_pred = model.predict(x_valid)
    
    print(\
        'Evaluation metrics for {}. \n   RMSE:{}, MAE:{}, R2:{} \n'.format(model_name, np.sqrt(mse(y_valid,y_pred)), mae(y_valid,y_pred), r2(y_valid,y_pred))
        )
    
    pass