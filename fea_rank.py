import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import KFold


class FeaImportance():
    def __init__(self,data_path,seed,test_size,top_k,fea_nums):
        self.data_path = data_path
        self.seed = seed
        self.test_size = test_size
        self.top_k = top_k
        self.fea_nums = fea_nums
        

    def data_pre(self):
        df = pd.read_excel(self.data_path,skiprows=0)
        self.x = df.iloc[:,:-1]
        self.y = df.iloc[:,-1]
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size=self.test_size,random_state=self.seed)

    def rf_fit(self):
        model = RandomForestRegressor(n_estimators=100,n_jobs=-1,random_state=self.seed)
        kf = KFold(n_splits=5,random_state=self.seed,shuffle=True)
        r_scorek_train,maek_train,rmsek_train,r_scorek_test,maek_test,rmsek_test =  ([] for i in range(6))
        for train_index,test_index in kf.split(self.x.values,self.y.values):
            X_train, X_test = self.x.values[train_index], self.x.values[test_index]
            y_train, y_test = self.y.values[train_index], self.y.values[test_index]

            model.fit(X_train, y_train)

            y_train_predicted = model.predict(X_train)
            y_test_predicted = model.predict(X_test)

            rmse_train = round(math.sqrt(mean_squared_error(y_train, y_train_predicted)),4)
            rmse_test = round(math.sqrt(mean_squared_error(y_test, y_test_predicted)),4)

            mae_train = round(mean_absolute_error(y_train, y_train_predicted),4)
            mae_test = round(mean_absolute_error(y_test, y_test_predicted),4)

            r2_train = round(r2_score(y_train, y_train_predicted),4)
            r2_test = round(r2_score(y_test, y_test_predicted),4)

            r_scorek_train.append(r2_train)
            maek_train.append(mae_train)
            rmsek_train.append(rmse_train)

            r_scorek_test.append(r2_test)
            maek_test.append(mae_test)
            rmsek_test.append(rmse_test)

        self.model = model
        print(f"\n Train R2_Score: {np.mean(r_scorek_train)} Test R2_Score: {np.mean(r_scorek_test)}")
        print(f"Train RMSE: {np.mean(rmsek_train)} Test MSE: {np.mean(rmsek_test)}")
        print(f"Train MAE: {np.mean(maek_train)} Test MAE: {np.mean(maek_test)}")


    def get_fea(self):
        features = self.x.columns
        feature_importances = self.model.feature_importances_
        features_df = pd.DataFrame({'Features':features,'Importance':feature_importances})
        features_df.sort_values('Importance',inplace=True,ascending=False)
        print('feature sort:',features_df)

        topk_idx = math.ceil(self.top_k/self.fea_nums * 100)
        features_filter  = features_df.iloc[:topk_idx,:].reset_index(drop=True)
        importance_sum = sum(features_filter.iloc[:,-1])
        print(f'the top of {self.top_k} features is {features_filter}')
        print('the sum of importance is:',importance_sum)

        sns.barplot(x='Features',y='Importance',data=features_filter)
        sns.despine(bottom=True)
        plt.title('Feature Rank')
        plt.savefig('Result/fea_rank.jpg')
        plt.show()

        features_filter['Features'] = features_df['Features'].apply(lambda x: int(x))
        features_filter['Importance'] = features_df['Importance'].apply(lambda x: round(x, 4))
        res = features_filter.set_index("Features")["Importance"].to_dict()
        return res

        