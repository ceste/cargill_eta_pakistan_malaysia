import os
import pytest
from Model import Model 
import json
import requests
import pandas as pd



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



# happy path
def test_good_case():

    model = Model()
    output = model.predict_eta(vessel_IMO,ais_type,A,B,C,D,from_port_lat,from_port_lon,to_port_lat,to_port_lon,month_UTC,port_event)  
    
    output_json = json.loads(output)

    assert isinstance(output_json, dict)

    assert output_json['status']==1
    assert output_json['error'] is None or output_json['error'] == ''    
    

def test_bad_case():

    vessel_IMO = 'abc'

    model = Model()
    output = model.predict_eta(vessel_IMO,ais_type,A,B,C,D,from_port_lat,from_port_lon,to_port_lat,to_port_lon,month_UTC,port_event)  
    
    output_json = json.loads(output)

    assert isinstance(output_json, dict)

    assert output_json['status']==0
    assert output_json['error'] is not None or output_json['error'] != ''    

