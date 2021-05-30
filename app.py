import time
import datetime
import pickle
import joblib
import os
from model import Model
from news import News
from bitcoin import Bitcoin
# from upbit import Upbit
from flask import Flask, request, render_template

app = Flask(__name__)
# upbit = Upbit()
newss = News()
model = Model()
bitcoin = Bitcoin()

chart_path = 'static/img/expr_14'

@app.route('/')
def main():
    bitcoin.get_bitcoin()
    return render_template('main.html')

@app.route('/result')
def result():
    now = datetime.datetime.now()
    nowDate = now.strftime('%Y-%m-%d')
    now_path = chart_path + nowDate + '.png'
    data = 'img/expr_14'+nowDate+'.png'

    #new_model = model.create_model()                       새로운 데이터로 모델 학습시

    if os.path.isfile(now_path):
        print("차트 이미지 확인")
        return render_template('result.html', img=data)
    else:
        print("차트 이미지 없음, 이미지 생성")
        model_data = model.get_model()
        return render_template('result.html', img=data )


@app.route('/reddit')
def reddit():

    return render_template('reddit.html')

@app.route('/introduce')
def introduce():

    return render_template('introduce.html')

@app.route('/news')
def news():
    
    today_news = newss.get_news()
    news_img = newss.get_img()

    return render_template('news.html', **locals())


# @app.route('/main')
# def get_main():
#     market = request.args.get('market')
#     market2 = 'KRW-ETH'
#     if market is None or market == '':
#         return 'No market parameter'

#     candles = upbit.get_hour_candles(market)
#     if candles is None:
#         return 'invalid market: {}'.format(market)
    
#     candles2 = upbit.get_hour_candles(market2)
#     if candles2 is None:
#         return 'invalid market: {}'.format(market2)
    
#     label = market
#     xlabels = []
#     dataset = []
#     label2 = market2
#     dataset2 = []
#     i = 1
#     j = 1

#     for candle in candles:
#         xlabels.append(i)
#         dataset.append(candle['trade_price'])
#         i += 1
    
#     for candle2 in candles2:
#         dataset2.append(candle2['trade_price'])
        

#     return render_template('chart.html', **locals())

# @app.route('/test')
# def test():
#     label = '테스트됨?'
#     dataset = [50, 30, 50, 30, 50, 10, 10]
#     return render_template('test.html', **locals())

if __name__ == '__main__':
    app.debug = True