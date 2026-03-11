import pickle
import os

class Insurance_Prediction:

    def __init__(self):

        # find project root folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        scaler_path = os.path.join(base_dir, "artifacts", "scaler.pkl")
        model_path = os.path.join(base_dir, "artifacts", "model.pkl")

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def prediction(self, Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):

        data = [[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]]

        scaled_data = self.scaler.transform(data)

        result = self.model.predict(scaled_data)

        return result[0]
