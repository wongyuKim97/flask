import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime


class News:
    def __init__(self):
        pass

    def get_news(self):
     
        news_url = 'https://search.naver.com/search.naver?where=news&sm=tab_jum&query=비트코인'

        req = requests.get(news_url)
        soup = BeautifulSoup(req.text, 'html.parser')

        news_dict = {}
        idx = 0
        cur_page = 1

        print()
        print('크롤링 중...')

        while idx < 10:
        ### 네이버 뉴스 웹페이지 구성이 바뀌어 태그명, class 속성 값 등을 수정함(20210126) ###
            
            table = soup.find('ul',{'class' : 'list_news'})
            li_list = table.find_all('li', {'id': re.compile('sp_nws.*')})
            area_list = [li.find('div', {'class' : 'news_area'}) for li in li_list]
            a_list = [area.find('a', {'class' : 'news_tit'}) for area in area_list]

            for n in a_list[:min(len(a_list), 10-idx)]:
                news_dict[idx] = {'title' : n.get('title'),
                                'url' : n.get('href') }
                idx += 1

            cur_page += 1

            pages = soup.find('div', {'class' : 'sc_page_inner'})
            next_page_url = [p for p in pages.find_all('a') if p.text == str(cur_page)][0].get('href')
            
            req = requests.get('https://search.naver.com/search.naver' + next_page_url)
            soup = BeautifulSoup(req.text, 'html.parser')

        print('완료')

        return news_dict


    def get_img(self):
        news_url = 'https://search.naver.com/search.naver?where=news&sm=tab_jum&query=비트코인'

        req = requests.get(news_url)
        soup = BeautifulSoup(req.text, 'html.parser')

        img_dict = {}
        idx = 0
        cur_page = 1

        print()
        print('크롤링 중...')

        while idx < 10:
        ### 네이버 뉴스 웹페이지 구성이 바뀌어 태그명, class 속성 값 등을 수정함(20210126) ###
            
            table = soup.find('ul',{'class' : 'list_news'})
            li_list = table.find_all('li', {'id': re.compile('sp_nws.*')})
            img_list = [ls.find('a', {'class' : 'dsc_thumb'}) for ls in li_list]        #새로 함          
            i_list = [img.find('img', {'class' : 'thumb api_get'}) for img in img_list]     #새로 함
           

            for n in i_list[:min(len(i_list), 10-idx)]:
                img_dict[idx] = { 'img' : n.get('src') }
                idx += 1

            cur_page += 1

            pages = soup.find('div', {'class' : 'sc_page_inner'})
            next_page_url = [p for p in pages.find_all('a') if p.text == str(cur_page)][0].get('href')
            
            req = requests.get('https://search.naver.com/search.naver' + next_page_url)
            soup = BeautifulSoup(req.text, 'html.parser')

        print('완료')

        return img_dict

