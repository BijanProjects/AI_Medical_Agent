import pandas as pd

medical_data = pd.read_csv('train.csv')

# print(len(medical_data))
medical_data.dropna()
# print(len(medical_data))
# print(medical_data.columns)

medical_data['text'] = 'The Question: ' + medical_data['Question'] + '\nExpert response: ' + medical_data['Answer']


# print(medical_data['text'][10])

