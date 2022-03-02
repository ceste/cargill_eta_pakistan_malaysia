import os,sys,inspect,getopt,io 
from pathlib import Path
import argparse

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from log import Log
import config, utils

import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import json
import string
from math import sqrt, cos, pi, sin
# import importlib
# import importlib.util
# import requests
from scipy.optimize import minimize
from collections import deque

import pygeohash as gh	
import pickle
import networkx as nx
from math import sin, cos, sqrt, atan2, radians
import json
import xgboost as xgb

class Model:

	def __init__(self): 

		self.uuid = utils.get_uuid()

		self.R = 6373.0
		self.distance_to_geo_8_ratio = config.DIST_TO_DIST_GEO_RATIO
		self.original_features = config.ORIGINAL_FEATURES
		self.features = config.FEATURES 
		self.scaling_features = config.SCALING_FEATURES 

		self.log = Log()		
		# self.database = pd.read_csv('dataset/database.csv')
		self.target = 'epochs_gap'
		# self.database = None
		
		self.path_to_folder = config.PATH_TO_FOLDER

		self.graph_model_file = config.GRAPH_MODEL		
		self.graph_model = None		
		file = Path(self.path_to_folder+self.graph_model_file)
		if file.is_file():            
			with open(file, 'rb') as inp:
				self.graph_model = pickle.load(inp)

		self.label_encoders_file = config.LABEL_ENCODERS		
		self.label_encoders = None
		file = Path(self.path_to_folder+self.label_encoders_file)
		if file.is_file():            
			with open(file, 'rb') as inp:
				self.label_encoders = pickle.load(inp)

		self.one_hot_encoders_file = config.ONE_HOT_ENCODERS		
		self.one_hot_encoders = None
		file = Path(self.path_to_folder+self.one_hot_encoders_file)
		if file.is_file():            
			with open(file, 'rb') as inp:
				self.one_hot_encoders = pickle.load(inp)

		self.scaler_file = config.SCALER		
		self.scaler = None
		file = Path(self.path_to_folder+self.scaler_file)
		if file.is_file():            
			with open(file, 'rb') as inp:
				self.scaler = pickle.load(inp)

		self.geohash5_df = None
		file = Path(self.path_to_folder+'dataset/'+'geohash_5_df.csv')				
		if file.is_file():            
			self.geohash5_df = pd.read_csv(file)

		self.eta_model_file = config.ETA_MODEL
		self.eta_model = None		
		file = Path(self.path_to_folder+self.eta_model_file)
		if file.is_file():            
			with open(file, 'rb') as inp:
				self.eta_model = pickle.load(inp)

		self.vessels = None		
		file = Path(self.path_to_folder+'dataset/'+'vessels.csv')		
		if file.is_file():            
			self.vessels = pd.read_csv(file)

		self.draught = None		
		file = Path(self.path_to_folder+'dataset/'+'avg_draught.csv')		
		if file.is_file():            
			self.draught = pd.read_csv(file)

		self.speed = None		
		file = Path(self.path_to_folder+'dataset/'+'avg_speed.csv')		
		if file.is_file():            
			self.speed = pd.read_csv(file)

		self.ports = None		
		file = Path(self.path_to_folder+'dataset/'+'ports.csv')		
		if file.is_file():            
			self.ports = pd.read_csv(file)

		self.nav_status = None
		file = Path(self.path_to_folder+'dataset/'+'nav_status.csv')		
		if file.is_file():            
			self.nav_status = pd.read_csv(file)

		self.ais_type = None
		file = Path(self.path_to_folder+'dataset/'+'ais_type.csv')		
		if file.is_file():            
			self.ais_type = pd.read_csv(file)
		
		self.estimated_model_geohash_5 = None
		file = Path(self.path_to_folder+'dataset/'+'estimated_model_geohash_5.csv')		
		if file.is_file():            
			self.estimated_model_geohash_5 = pd.read_csv(file)
		
		self.estimated_course_geohash_5 = None
		file = Path(self.path_to_folder+'dataset/'+'estimated_COURSE_geohash_5.csv')		
		if file.is_file():            
			self.estimated_course_geohash_5 = pd.read_csv(file)

		self.estimated_speed_geohash_5 = None
		file = Path(self.path_to_folder+'dataset/'+'estimated_SPEED_geohash_5.csv')		
		if file.is_file():            
			self.estimated_speed_geohash_5 = pd.read_csv(file)

		self.estimated_heading_geohash_5 = None
		file = Path(self.path_to_folder+'dataset/'+'estimated_HEADING_geohash_5.csv')		
		if file.is_file():            
			self.estimated_heading_geohash_5 = pd.read_csv(file)

		self.estimated_navstat_geohash_5 = None
		file = Path(self.path_to_folder+'dataset/'+'estimated_NAVSTAT_geohash_5.csv')		
		if file.is_file():            
			self.estimated_navstat_geohash_5 = pd.read_csv(file)
		
		self.estimated_DRAUGHT_geohash_5 = None
		file = Path(self.path_to_folder+'dataset/'+'estimated_DRAUGHT_geohash_5.csv')		
		if file.is_file():            
			self.estimated_DRAUGHT_geohash_5 = pd.read_csv(file)

		# self.nav_status = None
		# file = Path(self.path_to_folder+'dataset/'+'nav_status.csv')		
		# if file.is_file():            
		# 	self.nav_status = pd.read_csv(file)

		# self.ais_type = None
		# file = Path(self.path_to_folder+'dataset/'+'ais_type.csv')		
		# if file.is_file():            
		# 	self.ais_type = pd.read_csv(file)

		# self.historical_percentiles_pivot = None
		# file = Path(self.path_to_folder+'dataset/'+'historical_percentiles_pivot_pakistan_malaysia_1_4.csv')		
		# if file.is_file():            
		# 	self.historical_percentiles_pivot = pd.read_csv(file)
		

	def get_nearest_geohash(self,geohash_5_df,lat,lon):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		params = locals()
		msg = 'params:'+str(params)
		self.log.print_(msg)


		distances = []
		geohashes = []
		for row in geohash_5_df.itertuples(index=False):
			
			geohash = row[0] 
			lat_ = row[2]
			lon_ = row[3]
			
			distance = self.calculate_distance(lat,lon,lat_,lon_)
			distances.append(distance)
			geohashes.append(geohash)
		
		idx = np.argmin(distances)
		geohash5 = geohashes[idx]
		return geohash5


	def get_path(self,start_lat,start_lon,end_lat,end_lon):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		params = locals()
		msg = 'params:'+str(params)
		self.log.print_(msg)

		starting_geohash5 = self.get_nearest_geohash(self.geohash5_df,start_lat,start_lon)
		ending_geohash5 = self.get_nearest_geohash(self.geohash5_df,end_lat,end_lon)

		# print(starting_geohash5,ending_geohash5)

		paths = nx.shortest_path(self.graph_model,source=starting_geohash5,target=ending_geohash5)
		return paths

	def predict_eta(self,vessel_IMO,ais_type,A,B,C,D,start_lat,start_lon,end_lat,end_lon,month_UTC,port_event):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		params = locals()
		msg = 'params:'+str(params)
		self.log.print_(msg)

		status = None
		error = None


		if (isinstance(vessel_IMO,int) and isinstance(ais_type,int) and isinstance(A,int) and isinstance(B,int) and isinstance(C,int) and isinstance(D,int) and isinstance(start_lat,float) and isinstance(start_lon,float) and isinstance(end_lat,float) and isinstance(end_lon,float) and isinstance(end_lat,float) and isinstance(month_UTC,int) and isinstance(port_event,str)) :

			if month_UTC<1 and month_UTC>12:

				status = 0
				error = 'Wrong value for month_UTC'

			else:		

				try:
					pass
				
					# get the path
					path = self.get_path(start_lat,start_lon,end_lat,end_lon)		

					# predict the eta per path using estimated model
					
					# pair the geohash
					# start = path[0:-1]
					# end = path[1:]

					path_df = pd.DataFrame([path]).T
					path_df.columns = ['geohash5']

					# # add lat lon
					path_df['lat_lon'] = path_df['geohash5'].apply(self.gh_decode) 
					path_df['lat'] = path_df['lat_lon'].apply(lambda x:tuple(x)[0])
					path_df['lon'] = path_df['lat_lon'].apply(lambda x:tuple(x)[1])
					
					path_df = pd.merge(path_df,self.estimated_course_geohash_5,left_on=['geohash5'],right_on=['geohash_5'],how='left')
					if 'geohash_5' in path_df.columns:
						del path_df['geohash_5']

					path_df = pd.merge(path_df,self.estimated_speed_geohash_5,left_on=['geohash5'],right_on=['geohash_5'],how='left')
					if 'geohash_5' in path_df.columns:
						del path_df['geohash_5']

					path_df = pd.merge(path_df,self.estimated_heading_geohash_5,left_on=['geohash5'],right_on=['geohash_5'],how='left')
					if 'geohash_5' in path_df.columns:
						del path_df['geohash_5']

					path_df = pd.merge(path_df,self.estimated_navstat_geohash_5,left_on=['geohash5'],right_on=['geohash_5'],how='left')
					if 'geohash_5' in path_df.columns:
						del path_df['geohash_5']

					path_df = pd.merge(path_df,self.estimated_DRAUGHT_geohash_5,left_on=['geohash5'],right_on=['geohash_5'],how='left')
					if 'geohash_5' in path_df.columns:
						del path_df['geohash_5']					

					if 'NAVSTAT_size' in path_df.columns:
						del path_df['NAVSTAT_size']
					
					path_df['AISTYPE'] = ais_type
					path_df['A'] = A
					path_df['B'] = B
					path_df['C'] = C
					path_df['D'] = D
					path_df['IMO'] = vessel_IMO

					df = pd.merge(path_df,self.vessels,left_on='IMO',right_on='vessel_IMO',how='left')

					df['AISTYPE'] = df['AISTYPE'].astype(str)
					df['ais_type_description'] = 'ais_type_description_'+df['AISTYPE']
					df['month_UTC'] = month_UTC 
					df['month_UTC'] = df['month_UTC'].astype(float)
					df['drift'] = abs(df['COURSE']-df['HEADING'])

					path_df['NAVSTAT'] = path_df['NAVSTAT'].astype(int)
					self.nav_status['Navigation Status'] = self.nav_status['Navigation Status'].astype(int)
					
					df = pd.merge(df,self.nav_status,left_on='NAVSTAT',right_on='Navigation Status',how='left')

					for i in range(1,9):
						df['geohash_'+str(i)]= df.apply(lambda x: gh.encode(x.lat, x.lon, precision=i), axis=1)
						df['geohash_'+str(i)+'_lat_lon'] = df['geohash_'+str(i)].apply(self.gh_decode)
						df['geohash_'+str(i)+'_lat'] = df['geohash_'+str(i)+'_lat_lon'].apply(lambda x:tuple(x)[0])
						df['geohash_'+str(i)+'_lon'] = df['geohash_'+str(i)+'_lat_lon'].apply(lambda x:tuple(x)[1])


					
					df['trip_id'] = df.groupby(['IMO']).cumcount()+1
					df['trip_id_next'] = df['trip_id'] + 1

					df['IMO'] = df['IMO'].astype(int)
					df['trip_id'] = df['trip_id'].astype(int)
					df['trip_id_next'] = df['trip_id_next'].astype(int)
					
					# # df['port_event'] = port_event
					# # df['is_port_call'] = np.where(df['port_event']=='',0,1)
					# # df['port_event'] = np.where(df['port_event']=='',np.nan,df['port_event'])

					tmp = df[['IMO','trip_id','lat','lon','geohash_1_lat','geohash_1_lon','geohash_2_lat','geohash_2_lon','geohash_3_lat','geohash_3_lon','geohash_4_lat','geohash_4_lon','geohash_5_lat','geohash_5_lon','geohash_6_lat','geohash_6_lon','geohash_7_lat','geohash_7_lon','geohash_8_lat','geohash_8_lon','geohash_1','geohash_2','geohash_3','geohash_4','geohash_5','geohash_6','geohash_7','geohash_8']]		


					df = pd.merge(df,tmp,left_on=['IMO','trip_id_next'],right_on=['IMO','trip_id'],how='left')

					if 'trip_id_y' in df.columns:
						del df['trip_id_y']
						
					df.rename(columns={'lat_x':'lat'},inplace=True)
					df.rename(columns={'lon_x':'lon'},inplace=True)

					df.rename(columns={'lat_y':'lat_dest'},inplace=True)
					df.rename(columns={'lon_y':'lon_dest'},inplace=True)

					df.rename(columns={'geohash_1_lat_x':'geohash_1_lat'},inplace=True)
					df.rename(columns={'geohash_1_lon_x':'geohash_1_lon'},inplace=True)
					df.rename(columns={'geohash_2_lat_x':'geohash_2_lat'},inplace=True)
					df.rename(columns={'geohash_2_lon_x':'geohash_2_lon'},inplace=True)
					df.rename(columns={'geohash_3_lat_x':'geohash_3_lat'},inplace=True)
					df.rename(columns={'geohash_3_lon_x':'geohash_3_lon'},inplace=True)
					df.rename(columns={'geohash_4_lat_x':'geohash_4_lat'},inplace=True)
					df.rename(columns={'geohash_4_lon_x':'geohash_4_lon'},inplace=True)
					df.rename(columns={'geohash_5_lat_x':'geohash_5_lat'},inplace=True)
					df.rename(columns={'geohash_5_lon_x':'geohash_5_lon'},inplace=True)
					df.rename(columns={'geohash_6_lat_x':'geohash_6_lat'},inplace=True)
					df.rename(columns={'geohash_6_lon_x':'geohash_6_lon'},inplace=True)
					df.rename(columns={'geohash_7_lat_x':'geohash_7_lat'},inplace=True)
					df.rename(columns={'geohash_7_lon_x':'geohash_7_lon'},inplace=True)
					df.rename(columns={'geohash_8_lat_x':'geohash_8_lat'},inplace=True)
					df.rename(columns={'geohash_8_lon_x':'geohash_8_lon'},inplace=True)

					df.rename(columns={'geohash_1_lat_y':'geohash_1_lat_dest'},inplace=True)
					df.rename(columns={'geohash_1_lon_y':'geohash_1_lon_dest'},inplace=True)
					df.rename(columns={'geohash_2_lat_y':'geohash_2_lat_dest'},inplace=True)
					df.rename(columns={'geohash_2_lon_y':'geohash_2_lon_dest'},inplace=True)
					df.rename(columns={'geohash_3_lat_y':'geohash_3_lat_dest'},inplace=True)
					df.rename(columns={'geohash_3_lon_y':'geohash_3_lon_dest'},inplace=True)
					df.rename(columns={'geohash_4_lat_y':'geohash_4_lat_dest'},inplace=True)
					df.rename(columns={'geohash_4_lon_y':'geohash_4_lon_dest'},inplace=True)
					df.rename(columns={'geohash_5_lat_y':'geohash_5_lat_dest'},inplace=True)
					df.rename(columns={'geohash_5_lon_y':'geohash_5_lon_dest'},inplace=True)
					df.rename(columns={'geohash_6_lat_y':'geohash_6_lat_dest'},inplace=True)
					df.rename(columns={'geohash_6_lon_y':'geohash_6_lon_dest'},inplace=True)
					df.rename(columns={'geohash_7_lat_y':'geohash_7_lat_dest'},inplace=True)
					df.rename(columns={'geohash_7_lon_y':'geohash_7_lon_dest'},inplace=True)
					df.rename(columns={'geohash_8_lat_y':'geohash_8_lat_dest'},inplace=True)
					df.rename(columns={'geohash_8_lon_y':'geohash_8_lon_dest'},inplace=True)

					df.rename(columns={'geohash_1_x':'geohash_1'},inplace=True)
					df.rename(columns={'geohash_2_x':'geohash_2'},inplace=True)
					df.rename(columns={'geohash_3_x':'geohash_3'},inplace=True)
					df.rename(columns={'geohash_4_x':'geohash_4'},inplace=True)
					df.rename(columns={'geohash_5_x':'geohash_5'},inplace=True)
					df.rename(columns={'geohash_6_x':'geohash_6'},inplace=True)
					df.rename(columns={'geohash_7_x':'geohash_7'},inplace=True)
					df.rename(columns={'geohash_8_x':'geohash_8'},inplace=True)

					df.rename(columns={'geohash_1_y':'geohash_1_dest'},inplace=True)
					df.rename(columns={'geohash_2_y':'geohash_2_dest'},inplace=True)
					df.rename(columns={'geohash_3_y':'geohash_3_dest'},inplace=True)
					df.rename(columns={'geohash_4_y':'geohash_4_dest'},inplace=True)
					df.rename(columns={'geohash_5_y':'geohash_5_dest'},inplace=True)
					df.rename(columns={'geohash_6_y':'geohash_6_dest'},inplace=True)
					df.rename(columns={'geohash_7_y':'geohash_7_dest'},inplace=True)
					df.rename(columns={'geohash_8_y':'geohash_8_dest'},inplace=True)

					df.rename(columns={'trip_id_x':'trip_id'},inplace=True)

					df['distance'] = df.apply(lambda x: self.calculate_distance(x['lat'],x['lon'],x['lat_dest'],x['lon_dest']), axis=1 )
					df['distance_geohash_1'] = df.apply(lambda x: self.calculate_distance(x['geohash_1_lat'],x['geohash_1_lon'],x['geohash_1_lat_dest'],x['geohash_1_lon_dest']), axis=1 )
					df['distance_geohash_2'] = df.apply(lambda x: self.calculate_distance(x['geohash_2_lat'],x['geohash_2_lon'],x['geohash_2_lat_dest'],x['geohash_2_lon_dest']), axis=1 )
					df['distance_geohash_3'] = df.apply(lambda x: self.calculate_distance(x['geohash_3_lat'],x['geohash_3_lon'],x['geohash_3_lat_dest'],x['geohash_3_lon_dest']), axis=1 )
					df['distance_geohash_4'] = df.apply(lambda x: self.calculate_distance(x['geohash_4_lat'],x['geohash_4_lon'],x['geohash_4_lat_dest'],x['geohash_4_lon_dest']), axis=1 )
					df['distance_geohash_5'] = df.apply(lambda x: self.calculate_distance(x['geohash_5_lat'],x['geohash_5_lon'],x['geohash_5_lat_dest'],x['geohash_5_lon_dest']), axis=1 )
					df['distance_geohash_6'] = df.apply(lambda x: self.calculate_distance(x['geohash_6_lat'],x['geohash_6_lon'],x['geohash_6_lat_dest'],x['geohash_6_lon_dest']), axis=1 )
					df['distance_geohash_7'] = df.apply(lambda x: self.calculate_distance(x['geohash_7_lat'],x['geohash_7_lon'],x['geohash_7_lat_dest'],x['geohash_7_lon_dest']), axis=1 )
					df['distance_geohash_8'] = df.apply(lambda x: self.calculate_distance(x['geohash_8_lat'],x['geohash_8_lon'],x['geohash_8_lat_dest'],x['geohash_8_lon_dest']), axis=1 )

					# print(df.shape)

					df = df[~(df['lat_dest'].isnull()) & (~df['lon_dest'].isnull())]

					# # # print(self.historical_percentiles_pivot)	

					# # # df[['geohash_1','geohash_2','geohash_3','geohash_4','geohash_5','geohash_6','geohash_7','geohash_8','geohash_1_dest','geohash_2_dest','geohash_3_dest','geohash_4_dest','geohash_5_dest','geohash_6_dest','geohash_7_dest','geohash_8_dest']].to_csv('dataset/tmp.csv')

					# # # print(df[['geohash_1','geohash_2','geohash_3','geohash_4','geohash_5','geohash_6','geohash_7','geohash_8','geohash_1_dest','geohash_2_dest','geohash_3_dest','geohash_4_dest','geohash_5_dest','geohash_6_dest','geohash_7_dest','geohash_8_dest']])
					# # # print(self.historical_percentiles_pivot[['geohash_1','geohash_2','geohash_3','geohash_4','geohash_5','geohash_6','geohash_7','geohash_8','geohash_1_dest','geohash_2_dest','geohash_3_dest','geohash_4_dest','geohash_5_dest','geohash_6_dest','geohash_7_dest','geohash_8_dest']])

					# df = pd.merge(df,self.historical_percentiles_pivot,on=['geohash_1','geohash_2','geohash_3','geohash_4','geohash_1_dest','geohash_2_dest','geohash_3_dest','geohash_4_dest'],how='left')
					
					# print(df[df['IQR'].isnull()][['geohash_1','geohash_2','geohash_3','geohash_4','geohash_1_dest','geohash_2_dest','geohash_3_dest','geohash_4_dest','IQR']])

					df = df[self.original_features]

					categorical_features = []
					numerical_features = []

					for col,type_ in zip(df.columns,df.dtypes):
						
						if str(type_)=='object':        
							categorical_features.append(col)        
						else:
							numerical_features.append(col)

					dic = {}
					for col in categorical_features:

						if col in self.label_encoders.keys():

							df[col] = self.label_encoders[col].transform(df[col])						
							dic[col] = dict(zip(self.label_encoders[col].classes_, self.label_encoders[col].transform(self.label_encoders[col].classes_)))						
							
							tmp = self.one_hot_encoders[col].transform(df[col].values.reshape(-1,1)).toarray()[:,1:]
							tmp_df = pd.DataFrame(tmp)
							tmp_df = pd.DataFrame(tmp, columns=utils.get_ohe_column_names(dic,col))
							
							df = pd.DataFrame(np.hstack([df,tmp_df]), columns=list(df.columns)+list(tmp_df.columns))
							
							del df[col]

					feature_names = [col for col in df.columns]
					for col in feature_names:
					    df[col] = df[col].astype(float) 
					    
					if 'index' in feature_names:
					    feature_names.remove('index')

					# for col in self.scaling_features:
					# 	if col not in df.columns:
					# 		print(col)
					df.rename(columns={'lat':'LATITUDE','lon':'LONGITUDE'},inplace=True)		
					
					df_scaled = self.scaler.transform(df[self.scaling_features])

					scaled_features = []
					for col in df[self.scaling_features].columns:
					    tmp = col+'_scaled'
					    scaled_features.append(tmp)		  

					df_scaled = pd.DataFrame(df_scaled,columns=scaled_features)

					final_df = pd.concat([df, df_scaled], axis=1)
					
					eta = self.eta_model.predict(final_df[self.features])

					final_df['ETA'] = np.exp(eta)

					output_df = final_df[['LATITUDE','LONGITUDE','COURSE','SPEED','HEADING','IMO','A','B','C','D','DRAUGHT','month_UTC','ETA']]

					details = output_df.to_dict('index')

					status = 1


				except Exception as e:
					
					status = 0
					error = str(e)


				else:
					pass
				finally:
					pass


		else:

			status = 0
			error = 'Wrong Data Type(s)'

		return_ = {}
		return_["status"] = status
		return_["error"] = error
		if status==1:
			return_["data"] = details
		else:
			return_["data"] = None
		return_json = json.dumps(return_)

		return return_json

	# # def simulation(self,vessel_IMO,start_lat,start_lon,end_lat,end_lon,COURSE,SPEED,HEADING,C,port_code,port_event,month_UTC,ais_type_description,navigation_status):
	# def estimate(self,vessel_IMO,start_lat,start_lon,end_lat,end_lon):

	# 	msg = self.__class__.__name__+'.'+utils.get_function_caller()
	# 	self.log.print_(msg)
	# 	print(msg)

	# 	params = locals()
	# 	msg = 'params:'+str(params)
	# 	self.log.print_(msg)

		
	# 	# get the path
	# 	path = self.get_path(start_lat,start_lon,end_lat,end_lon)		

	# 	# predict the eta per path using estimated model
		
	# 	# pair the geohash
	# 	start = path[0:-1]
	# 	end = path[1:]

	# 	path_df = pd.DataFrame([start,end]).T
	# 	path_df.columns = ['geohash','geohash_dest']
		
	# 	# path_df.to_csv('dataset/path_df_simulation.csv',index=False)

	# 	tmp = pd.merge(path_df,self.estimated_model_geohash_5,left_on=['geohash','geohash_dest'],right_on=['geohash_5','geohash_5_dest'],how='left')
		
	# 	empty = tmp['epochs_gap'].isnull().sum()

	# 	eta = tmp['epochs_gap'].sum()		

	# 	# print('ETA:',eta)

	# 	# print(tmp)		

	# 	details = tmp[['geohash','geohash_dest','epochs_gap']].copy()		
	# 	details.fillna(-1,inplace=True)
	# 	details = details.to_dict('index')

	# 	return_ = {}
	# 	return_["status"] = 1
	# 	return_["error"] = None
	# 	return_["ETA"] = eta
	# 	return_["path"] = path
	# 	return_["details"] = details
	# 	return_json = json.dumps(return_)

	# 	# print(return_json)

	# 	return return_json


		


	def gh_decode(self,hash):
		lat, lon = gh.decode(hash)
		return lat, lon

	def calculate_distance(self,lat1,lon1,lat2,lon2):

		# msg = self.__class__.__name__+'.'+utils.get_function_caller()
		# self.log.print_(msg)
		# print(msg)

		# params = locals()
		# msg = 'params:'+str(params)
		# self.log.print_(msg)

		
		lat1 = radians(lat1)
		lon1 = radians(lon1)
		lat2 = radians(lat2)
		lon2 = radians(lon2)

		dlon = lon2 - lon1
		dlat = lat2 - lat1

		a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
		c = 2 * atan2(sqrt(a), sqrt(1 - a))

		distance = self.R * c

		return distance

