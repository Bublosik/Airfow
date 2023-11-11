import pandas as pd
import dill
import json
import glob
import datetime


def predict():
    with open('C:/Users/User/airflow_hw/data/models/cars_pipe_202311111312.pkl', 'rb') as mod:
        model = dill.load(mod)
    df_preds = pd.DataFrame(columns=['car_id', 'predict'])
    for file in glob.iglob('C:/Users/User/airflow_hw/data/test/*.json'):
        with open(file, 'rb') as fin:
            data = json.load(fin)
            df = pd.DataFrame.from_dict([data])
            y = model.predict(df)
            X = {'car_id': df.id, 'predict': y}
            df_pred = pd.DataFrame(X)
            df_preds = pd.concat([df_preds, df_pred])

    df_preds.to_csv(f'C:/Users/User/airflow_hw/data/predictions/preds_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
