import requests
import re
import os
import json
import http.cookiejar
import cv2
from bs4 import BeautifulSoup
import numpy as np

post_url = "https://accounts.pixiv.net/login?lang=zh&source=pc&view_type=page&ref=wwwtop_accounts_index"

session = requests.session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
    'Referer': 'https://www.pixiv.net/'}
session.headers = headers
session.cookies = http.cookiejar.LWPCookieJar(filename="pixiv_cookies")

params = {
    "lang": "zh",
    "source": "pc",
    "view_type": "page",
    "ref": "wwwtop_accounts_index"
}

datas = {
    'pixiv_id': '',
    'password': '',
    'captcha': '',
    'g_recaptcha_response': '',
    'post_key': "",
    'source': 'pc',
    'ref': 'wwwtop_accounts_index',
    'return_to': 'http://www.pixiv.net/',
}


def get_postkey():
    r = session.get(post_url, params=params)
    soup = BeautifulSoup(r.content, 'lxml')
    post_key = soup.find_all('input')[0]['value']
    datas['post_key'] = post_key


def login_in(username, password):
    get_postkey()
    datas['pixiv_id'] = username
    datas['password'] = password
    post_data = session.post(post_url, data=datas)
    print(post_data)
    session.cookies.save(ignore_discard=True, ignore_expires=True)

    try:
        session.cookies.load(filename="pixiv_cookies", ignore_discard=True)
        print("Load cookies successfully")
    except Exception as e:
        print("Can't load cookies")


def check_login():
    check_url = "https://www.pixiv.net/setting_user.php"
    login_code = session.get(check_url).status_code
    print(login_code)
    if login_code == 200:
        return True
    else:
        return False


def get_image(url: str):
    check_url = url
    rq = session.get(check_url, allow_redirects=False, timeout=2, stream=True)
    print(rq)
    return rq


login_in('andreslamosk124@hotmail.com', 'yQgUx9TeSFSfuj7')



