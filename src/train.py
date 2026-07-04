import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



df = pd.read_csv('data/taxi_clean.csv')

X = df.drop(columns = 'total_amount')
y = df['total_amount']


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

model = DecisionTreeRegressor()
model.fit(x_train,y_train)


y_pred = model.predict(x_test)


r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)


mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('taxi_experiment')


with mlflow.start_run():
    mlflow.log_param('max_depth', 5)
    mlflow.log_metric('r2_score',r2)
    mlflow.log_metric('mse', mse)
    mlflow.log_metric('mae', mae)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="taxi_model",
        registered_model_name="taxi_model"
)
    print("logged in")


joblib.dump(model,'models/taxi_model.pkl')
    