import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
import json

from Model import Model 
from log import Log
import config, utils, inspect, sys

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('list', type=list)

model = Model()

@app.route('/')
def hello():
	
	return jsonify('Welcome to Vessel ETA')

@app.route('/vessel_eta', methods=['POST'])
def vessel_eta():

	ABC = parser.parse_args()
	data_decoded = request.data.decode("utf-8") 

	#convert to json
	data_json = json.loads(data_decoded)

	uuid = utils.get_uuid()
	log = Log(uuid)		

	params = locals()
	msg = 'params:'+str(params)
	log.print_(msg)

	print(data_json)

	if 'vessel_IMO' in data_json:
		vessel_IMO = data_json['vessel_IMO']
	else:
		vessel_IMO = ''

	if 'ais_type' in  data_json:
		ais_type = data_json['ais_type']
	else:
		ais_type = ''
	
	if 'A' in  data_json:
		A = data_json['A']
	else:
		A = ''
	
	if 'B' in  data_json:
		B = data_json['B']
	else:
		B = ''

	if 'C' in  data_json:
		C = data_json['C']
	else:
		C = ''

	if 'D' in  data_json:
		D = data_json['D']
	else:
		D = ''
	
	if 'start_lat' in  data_json:
		start_lat = data_json['start_lat']
	else:
		start_lat = ''

	if 'start_lon' in  data_json:
		start_lon = data_json['start_lon']
	else:
		start_lon = ''

	if 'end_lat' in  data_json:
		end_lat = data_json['end_lat']
	else:
		end_lat = ''

	if 'end_lon' in  data_json:
		end_lon = data_json['end_lon']
	else:
		end_lon = ''

	if 'month_UTC' in  data_json:
		month_UTC = data_json['month_UTC']
	else:
		month_UTC = ''

	if 'port_event' in  data_json:
		port_event = data_json['port_event']
	else:
		port_event = ''

	if vessel_IMO!='' and ais_type!='' and A!='' and B!='' and C!='' and D!='' and start_lat!='' and start_lon!='' and end_lat!='' and end_lon!='' and month_UTC!='':

		output = model.predict_eta(int(vessel_IMO),int(ais_type),int(A),int(B),int(C),int(D),float(start_lat),float(start_lon),float(end_lat),float(end_lon),int(month_UTC),str(port_event))	
		output = json.dumps(output)

	else:

		status = 0 
		error = 'There is a problem on the parameters'
		data = None

		output = dict()
		output["status"] = status
		output["error"] = error
		output["data"] = None

		output = json.dumps(output)


	return jsonify(output)



if __name__ == '__main__':
	port = int(os.environ.get("PORT", 5050))
	# app.run(host='0.0.0.0', port = port, debug=True)

	# local
	app.run(host='127.0.0.1', port = port, debug=True)
	