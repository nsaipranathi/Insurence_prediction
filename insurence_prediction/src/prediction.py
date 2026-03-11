import pickle

class Insurance_Prediction:

    def __init__(self):
        with open("../artifacts/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open("../artifacts/model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def prediction(self, Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):

        # create input list
        data = [[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]]

        # scale input
        scaled_data = self.scaler.transform(data)

        # model prediction
        result = self.model.predict(scaled_data)

        return result[0]