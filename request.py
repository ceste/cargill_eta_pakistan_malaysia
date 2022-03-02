import requests
import json
import config, utils
import argparse
from pathlib import Path
import curlify
from Model import Model 

	
def curl_request(url,method,headers,payloads):
	# construct the curl command from request
	command = "curl -v -H {headers} {data} -X {method} {uri}"
	data = "" 
	if payloads:
		payload_list = ['"{0}":"{1}"'.format(k,v) for k,v in payloads.items()]
		data = " -d '{" + ", ".join(payload_list) + "}'"
	header_list = ['"{0}: {1}"'.format(k, v) for k, v in headers.items()]
	header = " -H ".join(header_list)
	print(command.format(method=method, headers=header, data=data, uri=url))



if __name__ == '__main__':

	# local url
	url = config.LOCAL_URL
	# url = config.HEROKU_URL
	# url = config.DEV_URL
	# url  = config.DOCKER_URL


	method = 'POST'
	headers = {'Content-type': 'application/json', 'Accept': 'application/json','User-Agent' : 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}

	from_port = 'PKBQM'
	from_port_lat = 24.7735
	from_port_lon = 67.337

	to_port = 'MYPGU'
	to_port_lat = 1.432973
	to_port_lon = 103.9119

	vessel_IMO = 9340752
	COURSE = 200 
	SPEED = 13.1
	HEADING = 199 
	C = 24 
	port = 'No port'
	port_event = ''
	month_UTC = 8
	ais_type = 70
	A=192
	B=68
	C=24
	D=8
	month_UTC = 8
	navigation_status = 5
	port_event = ''
	# vessel_IMO,ais_type,A,B,C,D,start_lat,start_lon,end_lat,end_lon,month_UTC,port_event

	function = 'vessel_eta' 
	url_ = url+function 
	data = '{"vessel_IMO":'+str(vessel_IMO)+', "ais_type":'+str(ais_type)+', "A":'+str(A)+', "B":'+str(B)+', "C":'+str(C)+', "D":'+str(D)+', "start_lat":'+str(from_port_lat)+', "start_lon":'+str(from_port_lon)+', "end_lat":'+str(to_port_lat)+', "end_lon":'+str(to_port_lon)+', "month_UTC":'+str(month_UTC)+', "port_event":"'+str(port_event)+'"}'
	data = data.replace("'",'"')	
	data_json = json.loads(data)

	print(url_,	data_json)

	send_request = requests.post(url_, data, headers=headers, verify=False)

	print(curlify.to_curl(send_request.request))

	if send_request.status_code == 200:

		print(send_request.json())
	else:
		print('There is an error occurs')
