import os
import logging
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
import xgboost

## Dec 2024
import torch
import torch.nn as nn
import torch.optim as optim
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# PyTorch 기반 MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[128,128]):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim,h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim,1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Regressor():
    def __init__(self,
                 data: pd.DataFrame,  
                 target: pd.DataFrame,
                 test: bool = False,
                 model_path: str = None, 
                 test_ratio: float = 0.2,
                 algo = 'total',
                 loaded_model = None,
                 loaded_scaler = None
        ):
        """
        Train_data, test_data and target dataframe should have 'name' column
        Train_data and target dataframe should have same elements in the 'name' column 
        """
        self.data = data
        self.target = target
        self.test = test
        self.regressor = algo
        self.loaded_model = loaded_model
        self.loaded_scaler = loaded_scaler
        
        ## Dec 2024
        self.load_scaler = False
        self.load_model = False
        
        if loaded_scaler != None:
          self.load_scaler = True
        if loaded_model != None:
          self.load_model = True
        #if self.test:
        #    #print('Initiate Data Prediction')
        #    assert model_path != None, 'There is no Model path to test!!'
        #    
        #    self.load_model = True
        #    self.model_name = model_path.split('/')[-1] 
        #    
        #    #assert 'model.pkl' in os.listdir(model_path), 'Model does not exist in model path!!'
        #    if 'model.pkl' in os.listdir(model_path):
        #        self.loaded_model = joblib.load(model_path+'/model.pkl')
        #    elif 'model.json' in os.listdir(model_path):
        #        self.loaded_model = xgboost.XGBRegressor()
        #        self.loaded_model.load_model(model_path+'/model.json')
        #    #logger.info(f'Load Model {self.model_name}')

        #    if 'scaler.pkl' in os.listdir(model_path):
        #        #logger.info('Load Scaler')
        #        self.load_scaler = True
        #        self.loaded_scaler = joblib.load(model_path+'/scaler.pkl')
        #    else:
        #        self.load_scaler = False
        
        self.load_model 
        self.test_ratio = test_ratio
        self.save_dir = './model/' + str(datetime.today().strftime("%Y-%m-%d")) +'/'
    
    def data_processing(self, data, target):
        
        data['name'] = [str(i) for i in data['name']]
        target['name'] = [str(i) for i in target['name']]
        
        data = data.sort_values('name')
        target = target.sort_values('name')
        data.reset_index(drop=True, inplace=True)
        target.reset_index(drop=True, inplace=True)
        
        name_x = list(data.iloc[:,0])
        name_y = list(target.loc[:,'name'])
        assert name_x == name_y, 'Name column of Target dataframe and name of atoms do not match'
        
        X = data.iloc[:,1:]
        y = target.loc[:,'target']
        
        if self.test:
            if self.load_scaler:
                scaler = self.loaded_scaler
                X_test = scaler.transform(X)
            else:
                X_test = X
            
            X_train = X
            y_train = y
            y_test = y
            name_train = name_x
            name_test = name_x
            
        else:
            X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(X, y, name_x, test_size=self.test_ratio, random_state=42)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
        return X_train, X_test, y_train, y_test, name_train, name_test, scaler
        
    def regression(self, regressor = None):
        if regressor == None:
            regressor = self.regressor

        # device 설정 (PyTorch용)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not self.test:
            #assert self.target != None, 'Need target.csv in atoms folder!!' 
            
            X_train, X_test, y_train, y_test, name_train, name_test, scaler = self.data_processing(self.data, self.target)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            
            if regressor == 'GBR':
                logger.info('Start Gradient Boosting Regression')
                reg = GradientBoostingRegressor(n_estimators=3938, learning_rate=0.14777,max_depth=17,
                                             max_features='sqrt',min_samples_leaf=28, min_samples_split=24,
                                             loss='absolute_error',random_state=42)
                reg.fit(X_train, y_train)
                y_train_pred = reg.predict(X_train)
                y_pred = reg.predict(X_test)
            elif regressor == 'KRR':
                logger.info('Start Kernel Ridge Regression')
                reg = KernelRidge(kernel = 'rbf')
                reg.fit(X_train, y_train)
                y_train_pred = reg.predict(X_train)
                y_pred = reg.predict(X_test)
            elif regressor == 'ELN':
                logger.info('Start ElasticNet Regression')
                reg = ElasticNet(alpha = 0.01)
                reg.fit(X_train, y_train)
                y_train_pred = reg.predict(X_train)
                y_pred = reg.predict(X_test)
            elif regressor == 'SVR':
                logger.info('Start Support Vector Regression')
                reg = SVR(kernel = 'rbf')
                reg.fit(X_train, y_train)
                y_train_pred = reg.predict(X_train)
                y_pred = reg.predict(X_test)
            elif regressor == 'GPR':
                logger.info('Start Gaussian Process Regression')
                kernel = 1.0 * Matern(length_scale=1.0,nu=2.5)
                reg = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10)
                reg.fit(X_train, y_train)
                y_train_pred = reg.predict(X_train)
                y_pred = reg.predict(X_test)
            elif regressor == 'ETR':
                logger.info('Start Gaussian Process Regression')
                reg =ExtraTreesRegressor( bootstrap=False, max_features=0.7500000000000001, 
                                   min_samples_leaf=2, min_samples_split=2, n_estimators=100)
                reg.fit(X_train, y_train)
                y_train_pred = reg.predict(X_train)
                y_pred = reg.predict(X_test)
            elif regressor == 'XGB':
                logger.info('Start XGbooster Regression')
                reg = xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.1, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=11, tree_method='gpu_hist')
                reg.fit(X_train, y_train)
                y_train_pred = reg.predict(X_train)
                y_pred = reg.predict(X_test)
            elif regressor == 'LGBM':
                reg = LGBMRegressor(n_estimators=1000, random_state=42)
                reg.fit(X_train,y_train)
                y_train_pred = reg.predict(X_train)
                y_pred = reg.predict(X_test)
            elif regressor == 'CATB':
                reg = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=8, verbose=0, random_seed=42)
                reg.fit(X_train,y_train)
                y_train_pred = reg.predict(X_train)
                y_pred = reg.predict(X_test)
                
            elif regressor == 'DNN':
                # PyTorch MLP 학습 루프
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = MLP(input_dim=X_train.shape[1], hidden_layers=[128,128]).to(device)

                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
                y_train_tensor = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32).to(device)
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

                # 간단한 학습 루프
                epochs = 100
                batch_size = 32
                n_train = X_train_tensor.size(0)
                for epoch in range(epochs):
                    model.train()
                    permutation = torch.randperm(n_train)
                    for i in range(0, n_train, batch_size):
                        optimizer.zero_grad()
                        indices = permutation[i:i+batch_size]
                        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    y_train_pred = model(X_train_tensor).cpu().numpy().flatten()
                    y_pred = model(X_test_tensor).cpu().numpy().flatten()

                reg = model  # 여기서 reg는 PyTorch 모델 자체를 반환
                
            else:
                raise RuntimeError(f'{regressor} was not defined')
                
            #X_train, X_test, y_train, y_test, name_train, name_test, scaler = self.data_processing(self.data, self.target)
            #y_train = np.array(y_train)
            #y_test = np.array(y_test)

            #reg.fit(X_train,y_train)
            
            #y_train_pred = reg.predict(X_train)
            #y_pred = reg.predict(X_test)
            
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
            test_mae = mean_absolute_error(y_test, y_pred)
            test_rmse = mean_squared_error(y_test, y_pred, squared=False)
            
            trained_df = pd.DataFrame(columns = ['name','target','pred'])
            trained_df['name'] = name_train
            trained_df['target'] = y_train
            trained_df['pred'] = y_train_pred
            
            tested_df = pd.DataFrame(columns = ['name','target','pred'])
            tested_df['name'] = name_test
            tested_df['target'] = y_test
            tested_df['pred'] = y_pred
            
            
            return [train_mae, train_rmse, test_mae, test_rmse], trained_df, tested_df, reg, scaler
            
        if self.test:
            _, X, _, y, _, name, _ = self.data_processing(self.data, self.target)
            
            reg = self.loaded_model
            y_pred = reg.predict(X)
            
            tested_df = pd.DataFrame(columns = ['name','target','pred'])
            tested_df['name'] = name
            tested_df['target'] = y
            tested_df['pred'] = y_pred
            
            return tested_df
    
    def performance_comparison(self):
        models = ['GBR','KRR','ELN','SVR','XGB','DNN','LGBM','CATB']
        result_dict = {}

        for m in models:
            error, train_df, test_df, model, scaler = self.regression(m)
            result_dict[m] = {
                'error': error,
                'train': train_df,
                'test': test_df,
                'model': model,
                'scaler': scaler
            }

        # 개별 모델 성능 출력
        print("=== Individual Model Performance ===")
        mae_dict = {}
        for m in models:
            err = result_dict[m]['error']
            print(f"{m} -> Train MAE: {err[0]:.4f}, Train RMSE: {err[1]:.4f}, Test MAE: {err[2]:.4f}, Test RMSE: {err[3]:.4f}")
            mae_dict[m] = err[2]  # Test MAE 기준으로 저장

        # 가장 좋은 모델 찾기 (Test MAE가 최소인 모델)
        best_model_name = min(mae_dict, key=mae_dict.get)
        best_model_error = result_dict[best_model_name]['error']

        print("\n=====================================================")
        print(f"Best performance model : {best_model_name}")
        print(f"MAE : {best_model_error[2]:.4f}")
        print(f"RMSE : {best_model_error[3]:.4f}")
        print("=====================================================")

        # GPR 제외 모든 모델 앙상블
        from itertools import combinations
        all_combos = []
        for r in range(2, len(models)+1):
            all_combos.extend(list(combinations(models, r)))

        print("\n=== Ensemble Performance (Average of Predictions) ===")
        for combo in all_combos:
            # test set 기준 앙상블
            test_refs = result_dict[combo[0]]['test']
            y_test = test_refs['target'].values
            y_preds = []

            for c in combo:
                y_preds.append(result_dict[c]['test']['pred'].values)

            y_ens = np.mean(y_preds, axis=0)
            ens_mae = mean_absolute_error(y_test, y_ens)
            ens_rmse = mean_squared_error(y_test, y_ens, squared=False)

            combo_name = "+".join(combo)
            print(f"{combo_name} Ensemble -> Test MAE: {ens_mae:.4f}, Test RMSE: {ens_rmse:.4f}")

        return best_model[1], best_model[2], best_model[3]
            
            
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)      
