# from app import result
from upbitpy import Upbitpy
import datetime
import urllib.request
from urllib.request import Request, urlopen
import json
import pyupbit

class Bitcoin:
    def __init__(self):
        pass

    def get_bitcoin(self):
    
        urlTicker = urllib.request.urlopen('https://api.bithumb.com/public/ticker/all')
        readTicker = urlTicker.read()
        jsonTicker = json.loads(readTicker)
        FindBTC = jsonTicker['data']['BTC']['closing_price']
        BTC = int(FindBTC)

        print(BTC)

        # now = datetime.datetime.now()
        # day = now.day-1
        # second = round(now.second)
        # start = datetime.datetime(now.year, now.month, day, now.hour, now.minute, second)
        # print(second)
        # print(start)

        # URL = 'https://api.cryptowat.ch/markets/bitfinex/btcusd/ohlc'

        # params = {'atfer':start, 'periods':'3600'}

        # response = requests.get(URL, params=params)
        # response = response.json()
        # # print(response['result']['3600'])

        # datas = response['result']['3600']
        # dataset = []
        # timeset = []

        # for d in datas:
        #     dataset.append(d[4])

        # for t in datas:
        #     time = datetime.datetime.fromtimestamp(int(t[0])).strftime('%Y-%m-%d %H:%M:')
        #     timeset.append(time)
        # dataset = dataset[-25:-1]
        # timeset = timeset[-25:-1]
        # print(timeset)
        # print(dataset)

