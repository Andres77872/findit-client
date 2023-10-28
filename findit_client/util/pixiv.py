import requests
import http.cookiejar
from bs4 import BeautifulSoup
import json
import concurrent.futures

post_url = "https://accounts.pixiv.net/login?lang=zh&source=pc&view_type=page&ref=wwwtop_accounts_index"

session = requests.session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
    'Referer': 'https://www.pixiv.net/'}
session.headers = headers
cookies = http.cookiejar.LWPCookieJar(filename="pixiv_cookies")
session.cookies = cookies

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


session_raw = requests.session()


def download(u, t):
    for i in ['.png', '.jpg', '.jpeg']:
        _u = u[1].replace('.png', i)
        res = session_raw.post(f'https://crawler.arz.ai/',
                               headers={'token-access': t,
                                        'accept': 'application/json',
                                        'Content-Type': 'application/x-www-form-urlencoded'},
                               data={'url': _u,
                                     'headers': json.dumps(headers, ensure_ascii=True),
                                     # 'cookies': cookies
                                     },
                               stream=True)
        if res.status_code == 200:
            return u[0].replace('.png', i), res.content
    return None


def get_crawler_image(url: list, token: str):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        resultados = list(executor.map(download, url, [token] * len(url)))
    return [x for x in resultados if x is not None]


def get_pixiv_image_url(idx: int):
    ptr = 'https://i.pximg.net/img-original/img/'
    r = session_raw.get(f'https://www.pixiv.net/en/artworks/{idx}')
    soup = BeautifulSoup(r.content, 'lxml')
    f = soup.find('meta', {'id': 'meta-preload-data'})
    d = json.loads(f['content'])
    url_base = d['illust'][str(idx)]['userIllusts'][str(idx)]['url']
    url_count = d['illust'][str(idx)]['userIllusts'][str(idx)]['pageCount']
    url_org = []
    for i in range(url_count):
        url_org.append(
            (f'{idx}_p{i}.png', ptr + ('/'.join(url_base.split('/')[7:]).split('p0')[0][:-1] + f'_p{i}.png'))
        )
    return url_org
