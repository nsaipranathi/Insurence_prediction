import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from data_preprocessing import load_and_split_data
x_train,X_test,y_train,y_test=load_and_split_data()
scaler = StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(X_test)
pd.DataFrame(x_train_scaled).to_csv("../data/processed/X_train.csv",index=False)

pd.DataFrame(x_test_scaled).to_csv("../data/processed/X_test.csv",index=False)

pd.DataFrame(y_train).to_csv("../data/processed/y_train.csv",index=False)

pd.DataFrame(y_test).to_csv("../data/processed/y_test.csv",index=False)
with open("../artifacts/scaler.pkl","wb") as f:
    pickle.dump(scaler,f)
print("Successfully saved the scaler pkl")