import os
import pytest
import json
import requests
import config
from Model import Model 
from app import app

# local url
url = config.LOCAL_URL
# url = config.HEROKU_URL

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



def test_api(app, client):

    function = 'vessel_eta' 
    url_ = url+function 
    data = '{"vessel_IMO":'+str(vessel_IMO)+', "ais_type":'+str(ais_type)+', "A":'+str(A)+', "B":'+str(B)+', "C":'+str(C)+', "D":'+str(D)+', "start_lat":'+str(from_port_lat)+', "start_lon":'+str(from_port_lon)+', "end_lat":'+str(to_port_lat)+', "end_lon":'+str(to_port_lon)+', "month_UTC":'+str(month_UTC)+', "port_event":"'+str(port_event)+'"}'
    data = data.replace("'",'"')    

    send_request = client.post(url_, data=data, follow_redirects=True)    

    assert send_request.status_code == 200

