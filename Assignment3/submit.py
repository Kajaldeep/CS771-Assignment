import pickle as pkl
import pandas as pd

def my_predict( df ):
	with open( "modelRFNO2.pkl", "rb" ) as file:
		modelDTNO2 = pkl.load( file )
	with open( "modelRFO3.pkl", "rb" ) as file:
		modelDTO3 = pkl.load( file )
	df['Time'] = pd.to_datetime(df['Time'])
	df['Minute'] = df['Time'].dt.hour*60 + df['Time'].dt.minute
	X = df[['Minute', 'o3op1', 'o3op2', 'no2op1', 'no2op2', 'temp', 'humidity']]
	o3 = modelDTO3.predict(X)
	no2 = modelDTNO2.predict(X)
	return (o3, no2)