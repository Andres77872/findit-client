import time
import unittest
from glob import glob

from findit_client import FindItClient
from findit_client.exceptions import (ImageNotFetchedException,
                                      ImageSizeTooBigException,
                                      QueryCantBeDecodedException,
                                      SearchBooruNotFound,
                                      TooFewSearchResultsException)

# Instead of creating the client at module level, create it in the setUp method
local_file = '/home/andres/Pictures/test/0000A3CEA1F8818C260C7E4FA37A80C8.webp'
local_files = glob('/home/andres/Pictures/test/*.webp')
all_pool = [
    'danbooru',
    'gelbooru',
    'zerochan',
    'anime-pictures',
    'yande.re',
    'e-shuushuu',
    'safebooru',
    'konachan',
    'tbib'
]


class MyTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create a new client for each test
        self.client = FindItClient(
            url_api_embedding='https://nn.arz.ai/',
            # url_api_embedding='http://127.0.0.1:7999/',
            # url_api_back_search='https://search.arz.ai/',
            url_api_back_search='http://127.0.0.1:8002/',
            url_image_backend='http://192.168.1.90:5001',
        )

    async def asyncTearDown(self):
        # Close all open resources
        if hasattr(self.client, 'search') and hasattr(self.client.search, 'ApiRequests'):
            if hasattr(self.client.search.ApiRequests, 'sess') and self.client.search.ApiRequests.sess:
                await self.client.search.ApiRequests.sess.close()

        if hasattr(self.client, 'tagger') and hasattr(self.client.tagger, 'ApiRequests'):
            if hasattr(self.client.tagger.ApiRequests, 'sess') and self.client.tagger.ApiRequests.sess:
                await self.client.tagger.ApiRequests.sess.close()

        if hasattr(self.client, 'util') and hasattr(self.client.util, 'ApiRequests'):
            if hasattr(self.client.util.ApiRequests, 'sess') and self.client.util.ApiRequests.sess:
                await self.client.util.ApiRequests.sess.close()

    # Fix the ResourceWarning for the unclosed file
    async def safe_open_file(self, filepath, mode='rb'):
        with open(filepath, mode) as f:
            return f.read()

    async def test_search_by_file_image_001(self):
        pool = ['danbooru', 'gelbooru']
        limit = 32
        r = await self.client.search.by_file(img=local_file,
                                             limit=limit,
                                             pool=pool)
        print(r)
        self.assertEqual(limit, r.search_meta.qdrant_meta.config.limit)
        self.assertEqual(pool, r.search_meta.qdrant_meta.config.pools)

    async def test_search_by_file_image_002(self):
        pool = ['danbooru', 'gelbooru']
        limit = 128
        r = await self.client.search.by_file(img=local_file,
                                             limit=limit,
                                             pool=pool)
        self.assertEqual(limit, r.search_meta.qdrant_meta.config.limit)
        self.assertEqual(pool, r.search_meta.qdrant_meta.config.pools)

    async def test_search_by_file_image_003(self):
        pool = ['danbooru', 'gelbooru']
        limit = 129
        r = await self.client.search.by_file(img=local_file,
                                             limit=limit,
                                             pool=pool)
        self.assertEqual(32, r.search_meta.qdrant_meta.config.limit)
        self.assertEqual(pool, r.search_meta.qdrant_meta.config.pools)

    async def test_search_by_file_image_004(self):
        pool = ['danbooru', 'gelbooru']
        limit = 0
        r = await self.client.search.by_file(img=local_file,
                                             limit=limit,
                                             pool=pool)
        self.assertEqual(32, r.search_meta.qdrant_meta.config.limit)
        self.assertEqual(pool, r.search_meta.qdrant_meta.config.pools)

    async def test_search_by_file_image_005(self):
        pool = ['danbooru', 'gelbooru', 'zerochan', 'anime-pictures', 'e-shuushuu', 'yande.re', 'safebooru']
        limit = 32
        r = await self.client.search.by_file(img=local_file,
                                             limit=limit,
                                             pool=pool)
        self.assertEqual(32, r.search_meta.qdrant_meta.config.limit)
        self.assertEqual(pool, r.search_meta.qdrant_meta.config.pools)

    async def test_search_by_file_image_006(self):
        pool = all_pool
        limit = 32
        r = await self.client.search.by_file(img=local_file,
                                             limit=limit)
        self.assertEqual(32, r.search_meta.qdrant_meta.config.limit)
        self.assertEqual(pool, r.search_meta.qdrant_meta.config.pools)

    async def test_search_by_file_image_007(self):
        pool = all_pool
        r = await self.client.search.by_file(img=local_file)
        self.assertEqual(32, r.search_meta.qdrant_meta.config.limit)
        self.assertEqual(pool, r.search_meta.qdrant_meta.config.pools)

    async def test_search_by_file_image_008(self):
        pool = all_pool

        with self.assertRaises(SearchBooruNotFound):
            await self.client.search.by_file(img=local_file,
                                             pool=pool + ['other'])

    async def test_search_by_url_image_000(self):
        pool = all_pool
        r = await self.client.search.by_url(url='https://img.arz.ai')
        self.assertEqual(32, r.search_meta.qdrant_meta.config.limit)
        self.assertEqual(pool, r.search_meta.qdrant_meta.config.pools)

    async def test_search_by_url_image_001(self):
        with self.assertRaises(ImageNotFetchedException):
            await self.client.search.by_url(url='https://img.arz.ai/a')

    async def test_search_by_url_image_002(self):
        with self.assertRaises(ImageSizeTooBigException):
            await self.client.search.by_url(
                url='https://cdn.donmai.us/original/07/8c/__nanashi_mumei_hololive_and_1_more_drawn_by_panpanmc4__078c86443fe8f8740c0ae617adb598bb.png')

    async def test_search_by_url_image_003(self):
        await self.client.search.by_url(
            url='https://cdn.donmai.us/original/dd/28/__original_drawn_by_waneella__dd28748e921b423a28029ed1b0dd3332.gif')

    async def test_search_by_query_000(self):
        pool = all_pool
        r = await self.client.search.by_url(url='https://img.arz.ai/qOaAWsm5', limit=1)
        query = r.results.data[0].content[0].query
        r = await self.client.search.by_query(query=query)
        self.assertEqual(32, r.search_meta.qdrant_meta.config.limit)
        self.assertEqual(pool, r.search_meta.qdrant_meta.config.pools)

    async def test_search_by_query_001(self):
        with self.assertRaises(QueryCantBeDecodedException):
            await self.client.search.by_query(query='NOQUERY')

    async def test_search_by_query_002(self):
        with self.assertRaises(QueryCantBeDecodedException):
            await self.client.search.by_query(query='uttDp8FZCJUBGEHa')

    async def test_search_by_query_003(self):
        pool = all_pool
        query = '88Z8tyzTxJYBDfis'
        r = await self.client.search.by_query(query=query)
        self.assertEqual(32, r.search_meta.qdrant_meta.config.limit)
        self.assertEqual(pool, r.search_meta.qdrant_meta.config.pools)

    async def test_search_scroll_000(self):
        r = await self.client.search.by_url(url='https://img.arz.ai', limit=32)
        r = await self.client.search.scroll(
            scroll_token=r.scroll_token,
            limit=32)
        self.assertEqual(32, r.search_meta.qdrant_meta.config.limit)

    async def test_search_scroll_001(self):
        r = await self.client.search.by_url(url='https://img.arz.ai', limit=32)
        for i in range(10):
            r = await self.client.search.scroll(
                scroll_token=r.scroll_token,
                limit=32)
        # self.assertEqual(32, r.search_meta.qdrant_meta.config.limit)

    async def test_search_scroll_002(self):
        with self.assertRaises(QueryCantBeDecodedException):
            await self.client.search.scroll(
                scroll_token='-9UhbKolbEFF2tnLD92XFi6w8trohvWcJJbHhHzOKgWCemuTmGP8fS90FMWhuXf6hq0fZIr-GImSlqNTB-WW')

    async def test_search_by_booru_image_id_000(self):
        r = await self.client.search.by_booru_image_id(image_id=1,
                                                       booru_name='gelbooru',
                                                       limit=32)
        self.assertEqual(32, r.search_meta.qdrant_meta.config.limit)

    async def test_search_by_booru_image_id_001(self):
        r = await self.client.search.by_booru_image_id(image_id=3950483,
                                                       booru_name='zerochan',
                                                       limit=32)
        self.assertEqual(32, len(r.results.data))

    async def test_search_by_vector_000(self):
        r = await self.client.search.by_vector(vector=[
            -0.12229436,
            0.0045484416,
            -0.09138441,
            -0.25359866,
            -0.43645084,
            -0.13887814,
            -0.6080942,
            -0.009801474,
            0.32522437,
            -0.3590045,
            -0.2137959,
            -0.4305024,
            0.22130184,
            0.39412385,
            -0.08539981,
            -0.17428012,
            0.48984602,
            -0.5243032,
            -0.25160098,
            -1.3007671,
            0.5203662,
            -0.6155676,
            1.0518993,
            0.2774297,
            0.36143935,
            -0.22066054,
            0.6201038,
            -0.060720712,
            -0.5758535,
            -1.6034433,
            -0.14686367,
            0.35615423,
            -0.14859128,
            -0.2954404,
            -0.3842869,
            0.42126122,
            0.025667686,
            -0.19650052,
            0.1883912,
            0.21510684,
            -0.20989983,
            -0.13947505,
            -0.22303061,
            -0.3523093,
            0.09148791,
            0.4189985,
            0.50538474,
            0.23035495,
            -0.0120931715,
            -0.050060753,
            -0.75651973,
            0.014597936,
            -0.61656666,
            0.51897746,
            0.9053695,
            4.79773,
            0.51792735,
            0.48207757,
            0.54228556,
            0.10002982,
            -0.11659889,
            0.3673966,
            0.32360145,
            -1.4328777,
            -0.020029623,
            -0.3387726,
            -0.2528066,
            0.4725525,
            -0.9169043,
            -0.3432888,
            -0.19335893,
            -0.10932932,
            0.0553561,
            0.08341453,
            0.37579533,
            0.009551653,
            -0.14023666,
            1.4123012,
            -0.28541914,
            0.33990353,
            0.74907607,
            -0.8955964,
            0.09410069,
            0.7589093,
            0.72947264,
            0.10743425,
            -0.30906245,
            -0.81799614,
            -0.0406957,
            -0.0042297044,
            -0.49801537,
            0.08948621,
            -0.22376788,
            0.3836967,
            0.04882407,
            0.3792857,
            -0.23332895,
            -0.54109204,
            0.40745082,
            -0.4787006,
            0.08227306,
            -0.008423853,
            -0.33400938,
            0.0668553,
            -0.87472737,
            0.082465105,
            0.06320382,
            0.02176764,
            0.552988,
            0.49601692,
            0.0028399862,
            -0.91109043,
            0.29336053,
            0.26184097,
            -0.8589163,
            -0.5068419,
            -0.3701001,
            -0.25115746,
            1.3346592,
            0.36879078,
            -0.12173424,
            0.14820272,
            -1.4105709,
            0.26731035,
            0.05965319,
            0.17172731,
            -0.6661799,
            0.35212374,
            0.13875984,
            0.03616742,
            0.47872213,
            0.390375,
            0.5496537,
            -0.4586076,
            -0.54954016,
            -0.24737203,
            0.41592854,
            0.19803442,
            -0.700582,
            -0.3404162,
            0.6750682,
            0.089604676,
            0.10448629,
            -0.37917504,
            -0.17382191,
            0.13591959,
            -0.49634555,
            0.46570516,
            0.003440821,
            -0.10945232,
            -0.105095156,
            -0.40667593,
            -0.18871064,
            -0.75652826,
            -0.28935638,
            -0.29880896,
            -0.28670698,
            -0.67793983,
            0.55389947,
            0.08060681,
            0.052920166,
            0.15481545,
            -0.06507208,
            0.3459592,
            -0.5587581,
            0.45046595,
            0.28946236,
            -0.12703256,
            -0.24609339,
            -0.079163104,
            0.5242693,
            -0.26883212,
            -0.22138391,
            0.32906786,
            -0.23990782,
            -0.26530275,
            0.31002656,
            -0.13368951,
            0.10048307,
            0.15891998,
            0.464912,
            -0.48067397,
            0.13883625,
            0.10018261,
            0.1475999,
            -0.032992948,
            0.3226877,
            1.1572021,
            0.17575358,
            -0.2732988,
            0.17473087,
            0.16185734,
            -0.67621887,
            0.25500408,
            -0.24547386,
            -0.35315394,
            -0.052405246,
            -0.37504598,
            -0.36546886,
            -0.16241987,
            -0.31283292,
            0.5473533,
            -0.088636756,
            0.4854019,
            0.17380008,
            0.12511809,
            0.500519,
            -0.0830479,
            0.8241987,
            -0.23200805,
            0.23423909,
            0.29228112,
            -0.8096474,
            0.0816781,
            -0.053590998,
            -0.4105354,
            -0.10170358,
            -0.24579848,
            -0.028787443,
            0.48378825,
            -0.9203491,
            0.28365678,
            -0.059307177,
            0.34017575,
            -0.30554694,
            -0.09782576,
            0.4538633,
            0.33624965,
            -0.27883765,
            0.31938764,
            -0.018213827,
            -0.502866,
            -0.34893686,
            -0.5434765,
            1.0938166,
            0.06727407,
            -0.1979,
            -0.5060997,
            0.61230224,
            0.24839568,
            0.75077766,
            -0.026168635,
            0.18603413,
            -0.14042759,
            -0.20101155,
            0.5624077,
            -0.37118277,
            0.32543927,
            -0.28629592,
            0.07946991,
            -0.1885063,
            0.9286908,
            0.23563913,
            0.004608473,
            -0.10857315,
            -0.26885307,
            0.0026817261,
            0.67861,
            0.2693732,
            1.3608094,
            0.63772136,
            -0.25023928,
            -0.02515699,
            0.6726071,
            -0.65822047,
            -0.50942934,
            0.34261703,
            -1.0440073,
            0.4425896,
            0.33870867,
            0.5065471,
            0.43869296,
            0.6889692,
            0.1472846,
            -0.55435324,
            -0.4654828,
            0.021530928,
            -0.12077976,
            -0.48727357,
            -0.42575353,
            -0.3011228,
            -0.19829312,
            -0.030672174,
            -0.54858226,
            -0.25116605,
            0.41859487,
            0.05947181,
            0.008110643,
            -0.24345182,
            -0.5013624,
            -0.40130162,
            0.004655048,
            -0.27032596,
            0.25649562,
            -1.0191523,
            0.245577,
            -0.46771368,
            0.60216653,
            0.32355562,
            -0.5570531,
            0.055310674,
            -0.16288054,
            0.23502901,
            0.12649973,
            0.7667937,
            0.5858721,
            0.105517894,
            0.1579149,
            -0.5441897,
            -0.20994131,
            0.077304564,
            0.42023838,
            0.6387451,
            0.49067637,
            -0.80568284,
            -0.08517247,
            -0.2923071,
            -0.28995392,
            -0.36985222,
            -0.040049646,
            0.19544171,
            0.40427446,
            0.24289206,
            0.33978987,
            0.7150007,
            -0.6556802,
            0.12918676,
            0.051792815,
            0.11952282,
            0.46925655,
            0.58016306,
            0.35397196,
            -0.59597164,
            -0.17885552,
            0.16161183,
            -0.2534559,
            -0.43605033,
            0.9669109,
            0.68774927,
            0.111664176,
            2.9960215,
            0.76128817,
            -0.22046183,
            -0.26124415,
            -0.52394265,
            -0.2479945,
            0.7120588,
            -0.15690413,
            -0.0031644083,
            0.42002586,
            0.38818312,
            -0.31817842,
            -0.15945344,
            0.27456558,
            -0.9537954,
            -0.020377189,
            0.85266244,
            0.08020058,
            0.47136244,
            -0.07273155,
            -0.34998515,
            1.1190062,
            -0.074634165,
            -0.20002022,
            0.35998487,
            -0.009604002,
            0.34554556,
            -0.48146525,
            -0.7995054,
            -0.59002525,
            1.3452855,
            0.25848898,
            -0.63399255,
            0.16190505,
            1.2562747,
            0.017860789,
            0.77279836,
            -0.83713573,
            -0.10582634,
            -0.12309453,
            -0.015208252,
            -0.02723013,
            -0.2547058,
            -0.13919295,
            -0.5327815,
            -0.24092802,
            -0.5951274,
            1.6011013,
            -0.027823774,
            0.12078396,
            -0.20674816,
            0.031370036,
            -0.06458666,
            0.37888473,
            -0.1512048,
            0.33063728,
            -0.2702095,
            0.14252026,
            -0.11372108,
            -0.27501252,
            0.16143925,
            0.084121644,
            0.08495891,
            0.10752606,
            0.18694037,
            -0.18408348,
            0.45702872,
            0.30785456,
            0.103590876,
            -0.12218215,
            -0.35215414,
            -0.30311137,
            0.14031455,
            -0.007924944,
            0.086943775,
            -0.4519642,
            0.7768197,
            -0.36781767,
            0.26052403,
            0.063713074,
            -0.5260669,
            0.17191122,
            -0.4950406,
            0.15724346,
            0.041718405,
            0.32387647,
            0.61338925,
            -0.14617655,
            0.25260705,
            0.31663463,
            0.08047782,
            0.5906157,
            -0.24174526,
            0.17381215,
            0.25290942,
            -0.4962409,
            0.24177851,
            0.33852282,
            -0.31495982,
            -0.6759607,
            0.089084394,
            -0.38293448,
            0.32871425,
            0.44041878,
            -0.14780141,
            -0.03860934,
            -0.40284637,
            0.44067204,
            -1.2856594,
            0.079857595,
            0.071904264,
            0.03633744,
            -0.03680572,
            -0.07273803,
            1.0750965,
            0.12379208,
            -0.10388216,
            -0.17452767,
            -0.46419597,
            0.2747822,
            -0.58780974,
            0.102012955,
            -0.183326,
            -0.4068696,
            0.17878856,
            -0.09830891,
            -1.2524924,
            0.5551258,
            -0.29145232,
            0.41777384,
            -0.356364,
            0.24634045,
            0.3665665,
            0.46279535,
            -0.043565236,
            0.078283645,
            -0.029523859,
            0.14795448,
            0.12520581,
            0.32691604,
            -0.13283844,
            0.8217792,
            -0.22297345,
            0.32591757,
            -0.67385525,
            -0.24244078,
            0.20203272,
            0.11844421,
            0.25981906,
            0.3233137,
            0.110759236,
            0.29366505,
            0.41238618,
            0.47683474,
            -0.0005937879,
            -0.8486183,
            -0.09455296,
            0.2494944,
            0.5166641,
            0.006710029,
            -0.039228503,
            0.5150596,
            0.112204015,
            0.65213716,
            0.45911855,
            -0.17075649,
            -0.35166714,
            0.1569346,
            0.30089584,
            -0.9869364,
            -0.17195305,
            0.02725066,
            -0.030817032,
            0.23509039,
            1.1466457,
            0.132744,
            0.36248145,
            -1.0552995,
            -0.4168359,
            0.60457563,
            0.7312489,
            -0.4687002,
            -0.055125758,
            -0.26891312,
            0.707532,
            0.010822425,
            -0.10839564,
            -0.3035406,
            0.1926222,
            -0.11131615,
            0.3789375,
            -0.32952854,
            -0.18740855,
            1.1225998,
            -0.013927296,
            0.041285977,
            0.20512843,
            0.22838,
            -0.41846272,
            0.09259913,
            0.6439147,
            0.10060209,
            0.1062548,
            0.016917538,
            -0.29084718,
            -0.59197474,
            0.4768941,
            0.19228707,
            0.6117219,
            0.38197213,
            -0.042147193,
            -0.6115241,
            0.43325844,
            -0.24730632,
            -0.0580617,
            -0.22842072,
            0.11755956,
            -0.2648012,
            0.6146398,
            0.101848625,
            -0.41550687,
            -0.14818949,
            -1.1142877,
            0.1540348,
            -0.04594839,
            0.19680384,
            -0.045810953,
            -0.27050474,
            -1.125882,
            0.4204669,
            0.04866486,
            -0.017056942,
            0.000839977,
            0.1138849,
            0.16634813,
            0.14548771,
            -0.07057003,
            -0.16901131,
            0.6259554,
            0.26622552,
            0.01969629,
            -0.5742189,
            0.37717044,
            0.19665767,
            0.3011644,
            0.084270604,
            -0.42400917,
            -0.4666321,
            -0.05849158,
            0.3981304,
            -0.70970386,
            -0.26276755,
            0.31241128,
            -0.6916207,
            0.33319652,
            0.49691528,
            0.01730378,
            -0.2332117,
            0.028543878,
            -0.40287516,
            -0.36401215,
            -0.027115623,
            -0.23604058,
            -0.39536008,
            0.14754827,
            0.092266984,
            -0.61072636,
            0.5980669,
            0.3479228,
            -0.1808466,
            -0.08774587,
            -0.1605058,
            0.6576754,
            0.13619599,
            -0.8542455,
            -0.14284383,
            -0.33019355,
            0.6473855,
            0.22744393,
            0.17739111,
            -0.014634259,
            -0.46276453,
            0.30688423,
            -0.20494671,
            0.081850916,
            -0.46421105,
            0.0684105,
            0.16788451,
            1.1643449,
            -0.2738288,
            0.6058956,
            0.22842678,
            0.13235916,
            0.41432118,
            1.1859313,
            0.2678513,
            -1.2202433,
            -0.57067686,
            0.005838309,
            -0.002894491,
            0.151707,
            -0.19098216,
            0.1667883,
            0.2923788,
            -0.5804702,
            0.4558766,
            0.12457802,
            -1.0597951,
            -0.5460154,
            0.080005765,
            -0.6369619,
            -0.17558207,
            -0.1833269,
            0.18361506,
            0.85482943,
            0.6939103,
            0.2988203,
            -0.70626086,
            0.18261229,
            -0.33194572,
            0.18265189,
            -0.4571111,
            -0.34841132,
            0.50119686,
            -0.11294807,
            0.2527298,
            -0.20309678,
            -0.12867536,
            -0.34728354,
            0.011148104,
            0.5245332,
            -0.081761576,
            0.036993716,
            0.73259306,
            0.05964802,
            -0.2526721,
            0.43459895,
            -0.2813849,
            0.1685472,
            0.32836595,
            -0.37466288,
            -0.07234447,
            -0.18130428,
            -0.35025656,
            -0.54164106,
            -0.1195384,
            0.4169857,
            0.24132216,
            -0.0053030546,
            0.19260402,
            -0.11580988,
            -0.040511124,
            -0.08299447,
            0.34444213,
            -1.6659261,
            0.11945233,
            -0.22393648,
            0.24034368,
            -0.39813948,
            -0.6564529,
            0.68891376,
            -0.15437233,
            -0.46516147,
            0.044685587,
            0.3539907,
            -0.37478068,
            -0.030019421,
            -0.61800265,
            -0.214137,
            0.15409398,
            0.19984917,
            -0.13560233,
            -0.077383704,
            0.61124694,
            -0.042955443,
            -0.13199605,
            -0.022386875,
            0.22761446,
            0.3632202,
            1.1289591,
            0.60503036,
            0.009016054,
            1.7862955,
            -0.26564497,
            -1.0846455,
            -0.087227695,
            6.6224914,
            -0.19248505,
            -0.42533356,
            -0.40631357,
            -0.21047172,
            0.29168123,
            0.10263452,
            0.14350235,
            0.1948046,
            0.26973492,
            0.36243644,
            -0.20857453,
            0.37654474,
            -0.4104849,
            -0.21753678,
            -0.11963448,
            0.8633883,
            -0.23196469,
            0.18356188,
            -0.17272538,
            -0.046718437,
            -0.16897012,
            -0.43140543,
            -0.22857699,
            -0.5426637,
            0.12656656,
            -0.07921625,
            -0.6904338,
            0.15601051,
            -0.21032348,
            -0.030826721,
            1.1453091,
            -0.7199559,
            0.16115254,
            0.023847403,
            0.40303132,
            -0.7170055,
            0.21950118,
            -0.21797168,
            -0.12905616,
            -0.5825663,
            -2.4367154,
            -0.07608505,
            -0.2182548,
            0.5368814,
            -0.3972936,
            -0.10735964,
            0.9413781,
            0.23950654,
            0.7312541,
            0.08287052,
            -0.24218068,
            0.47532815,
            0.11011035,
            0.1820867,
            0.2999337,
            -0.3696241,
            -1.0544405,
            -0.12761244,
            -0.30173406,
            -0.48676068,
            -0.38844526,
            -0.0030386117,
            0.43164945,
            -0.27947816,
            0.5892209,
            0.4027028,
            0.025011824,
            0.38335443,
            0.041272,
            -0.2615272,
            -0.17264879,
            0.0038024865,
            -0.32081202,
            0.28441042,
            0.65034974,
            -0.10797419,
            -0.34808448,
            0.47229692,
            -0.58501565,
            0.75408703,
            -0.0018058311,
            -0.35123956,
            -0.41464078,
            -0.16114633,
            -0.39601126,
            0.16555718,
            0.48483393,
            0.24934483,
            -0.5750284,
            0.18096326,
            0.3107676,
            0.46187946,
            0.31454057,
            0.20818037,
            0.2002598,
            -0.79689664,
            0.51283884,
            -0.20748736,
            0.09875492,
            0.005211854,
            -0.20225933,
            0.0021912886,
            -1.8610203,
            0.0683854,
            0.31308836,
            0.3019902,
            -0.2035153,
            -0.39590916,
            -0.10663158,
            -0.22412245,
            0.17503853,
            0.35516268,
            0.25302747,
            -0.33188367,
            0.22000681,
            0.51848954,
            0.49754143,
            0.6872965,
            -1.9138097,
            -0.17504862,
            -0.14583738,
            0.0731699,
            -0.61947143,
            0.10265666,
            0.041294776,
            0.4522911,
            -0.1281965,
            0.111735925,
            0.42979047,
            -0.3844,
            1.0067405,
            1.1610084,
            0.039366234,
            -0.08054277,
            0.13245597,
            -0.35949734,
            0.9505771,
            0.2428515,
            0.45985973,
            0.5464468,
            0.15478031,
            -0.3251956,
            0.68139696,
            0.3847134,
            -0.39418396,
            -1.2197917,
            -0.36136943,
            -0.06536044,
            -0.029996365,
            -0.10248727,
            0.13825405,
            -0.32119924,
            0.7102081,
            -0.846253,
            0.2546342,
            0.3914953,
            0.08208977,
            -0.12213089,
            -0.29841605,
            0.1790774,
            0.33058444,
            -0.30795678,
            -0.07939489,
            0.012156962,
            0.5462852,
            0.96067584,
            0.5423222,
            0.2425941,
            -0.24320899,
            -0.8790747,
            -0.7391385,
            0.71500254,
            0.6072405,
            0.2727042,
            0.14944182,
            -0.75047284,
            0.16640769,
            -0.21107647,
            -0.42485416,
            -0.084544756,
            0.6835633,
            0.0959745,
            -0.034180433,
            0.042533338,
            -0.5848137,
            -0.29342806,
            -0.20122801,
            0.8836547,
            0.91007113,
            0.6166408,
            -0.11545531,
            -0.24243076,
            -0.1866237,
            -0.633504,
            0.18876465,
            0.4495398,
            0.2622061,
            0.088852,
            0.16379574,
            -0.78725994,
            0.073749036,
            -0.43920514,
            -0.56825817,
            -0.8749619,
            -0.31339088,
            0.45215535,
            0.1501325,
            0.13723795,
            -0.2958735,
            -0.37963992,
            -0.06963328,
            0.046235528,
            0.11371686,
            -0.13132145,
            0.1488795,
            0.25460592,
            0.69921917,
            0.5793131,
            -0.03487,
            0.7451226,
            -0.39477408,
            -0.044659216,
            -0.81414294,
            -0.16612695,
            0.37633103,
            0.40811014,
            0.30188525,
            -0.008749612,
            0.26654354,
            0.48221517,
            0.3146929,
            0.14500408,
            0.66788816,
            -0.37227872,
            0.03661711,
            -0.3597401,
            0.27580276,
            0.11689848,
            -0.7609937,
            0.52892935,
            -0.5967872,
            0.2978977,
            0.36734128,
            0.20612308,
            0.033001654,
            0.3569358,
            -0.05737645,
            0.5220316,
            0.35359052,
            -0.27530605,
            -0.21541063,
            -0.3163059,
            0.79836315,
            -0.9172589,
            0.29434338,
            -0.1993737,
            0.62280536,
            -0.0979093,
            0.20699082,
            0.7497035,
            0.32244843,
            0.42509362,
            -0.41004285,
            0.81904066,
            -0.89449877,
            -0.20111607,
            -0.46963817,
            0.080460906,
            0.12952714,
            0.06948678,
            -0.760566,
            -0.50547796,
            -0.38876545,
            0.18051288,
            0.4706596,
            0.75109,
            0.7399496,
            -0.094831906,
            0.23756519,
            0.28115788,
            0.31090656,
            0.18870728,
            0.0039777285,
            0.10028537,
            -0.27731985,
            1.0530372,
            0.032488164,
            0.42480633,
            0.028932141,
            -0.1737386,
            0.2794588,
            -0.3479726,
            -0.21962893,
            0.05578904,
            -0.20316525,
            0.08730026,
            -0.15524976
        ],
            limit=32)
        self.assertEqual(32, r.search_meta.qdrant_meta.config.limit)

    async def test_tagger_by_file_000(self):
        r = await self.client.tagger.by_file(img=local_file)
        print(r)
        self.assertEqual(13, r.results.count)

    async def test_tagger_by_file_001(self):
        r = await self.client.tagger.by_file(img=local_file,
                                             th_general=0,
                                             th_character=0,
                                             th_rating=0)
        self.assertEqual(9083, r.results.count)

    async def test_tagger_by_file_002(self):
        r = await self.client.tagger.by_file(img=local_file,
                                             th_general=1,
                                             th_character=1,
                                             th_rating=1)
        self.assertEqual(1, r.results.count)

    async def test_tagger_by_file_003(self):
        r = await self.client.tagger.by_file(img=local_file,
                                             th_general=1,
                                             th_character=1,
                                             th_rating=1)
        self.assertGreater(0.91, r.results.data.rating[0].score)
        self.assertLess(0.51, r.results.data.rating[0].score)
        self.assertEqual('sensitive', r.results.data.rating[0].tag)

    async def test_tagger_by_url_000(self):
        r = await self.client.tagger.by_url(url='https://img.arz.ai/qOaAWsm5',
                                            th_general=0.35,
                                            th_character=0.8,
                                            th_rating=0)
        self.assertGreater(0.81, r.results.data.rating[0].score)
        self.assertLess(0.51, r.results.data.rating[0].score)
        self.assertEqual('sensitive', r.results.data.rating[0].tag)
        self.assertEqual(31, r.results.count)

    async def test_tagger_by_query_000(self):
        r = await self.client.tagger.by_query(query='hAfAMxp9')
        self.assertGreater(0.77, r.results.data.rating[0].score)
        self.assertLess(0.49, r.results.data.rating[0].score)
        self.assertEqual('general', r.results.data.rating[0].tag)
        self.assertEqual(19, r.results.count)

    async def test_tagger_by_booru_image_id_000(self):
        r = await self.client.tagger.by_booru_image_id(booru_name='gelbooru', image_id=1)
        self.assertGreater(0.95, r.results.data.rating[0].score)
        self.assertLess(0.56, r.results.data.rating[0].score)
        self.assertEqual('sensitive', r.results.data.rating[0].tag)
        self.assertEqual(10, r.results.count)

    async def test_random_search_generator_000(self):
        r = await self.client.util.random_search_generator()
        self.assertEqual(32, r.results.count)

    async def test_random_search_generator_001(self):
        r = await self.client.util.random_search_generator(limit=128)
        self.assertEqual(128, r.results.count)

    async def test_random_search_generator_002(self):
        r = await self.client.util.random_search_generator(limit=0)
        self.assertEqual(32, r.results.count)

    async def test_random_search_generator_003(self):
        r = await self.client.util.random_search_generator(limit=-10)
        self.assertEqual(32, r.results.count)

    async def test_random_search_generator_004(self):
        r = await self.client.util.random_search_generator(limit=128,
                                                           pool=['danbooru'])

        for i in r.results.data:
            for j in i.content:
                self.assertEqual('danbooru', j.pool)

    async def test_masonry_000(self):
        r = await self.client.search.by_url(url='https://img.arz.ai', limit=16)
        r = await self.client.util.generate_masonry_collage(r)

        self.assertEqual(3, len(r))
        self.assertEqual(1024, max(r[0].shape))
        self.assertEqual(512, max(r[1].shape))
        self.assertEqual(256, max(r[2].shape))

    async def test_masonry_001(self):
        r = await self.client.search.by_url(url='https://img.arz.ai', limit=3)
        with self.assertRaises(TooFewSearchResultsException):
            await self.client.util.generate_masonry_collage(r)

    async def test_embedding_000(self):
        r = await self.client.util.image_encoder_by_url(url='https://img.arz.ai')
        self.assertEqual(1024, len(r))

    async def test_embedding_001(self):
        r = await self.client.util.image_encoder_by_file(img=local_file)
        self.assertEqual(1024, len(r))

    async def test_search_by_string_000(self):
        pool = ['danbooru', 'zerochan', 'gelbooru']
        limit = 32
        r = await self.client.search.by_text(text='a fox girl',
                                             limit=limit,
                                             pool=pool)
        print(r)
        self.assertEqual(limit, r.search_meta.qdrant_meta.config.limit)
        self.assertEqual(['danbooru', 'zerochan', 'gelbooru'], r.search_meta.qdrant_meta.config.pools)

    async def test_search_by_string_scroll_000(self):
        pool = ['zerochan']
        limit = 32
        r = await self.client.search.by_text(text='a fox girl',
                                             limit=limit,
                                             pool=pool)
        print(r)
        r = await self.client.search.scroll(r.scroll_token, limit=limit)
        r = await self.client.search.scroll(r.scroll_token, limit=limit)
        r = await self.client.search.scroll(r.scroll_token, limit=limit)
        r = await self.client.search.scroll(r.scroll_token, limit=limit)
        self.assertEqual(limit, r.search_meta.qdrant_meta.config.limit)
        # self.assertEqual(['zerochan'], r.search_meta.qdrant_meta.config.pools)

    async def test_search_by_url_image_pixiv_000(self):
        pool = all_pool
        r = await self.client.search.by_url(
            url='https://i.pximg.net/img-original/img/2023/08/03/05/29/49/110480732_p0.jpg')
        self.assertEqual(32, r.search_meta.qdrant_meta.config.limit)
        self.assertEqual(pool, r.search_meta.qdrant_meta.config.pools)

    async def test_search_by_url_image_pixiv_001(self):
        r2 = await self.client.util.generate_md5_by_url(
            url='https://i.pximg.net/img-original/img/2023/08/03/05/29/49/110480732_p0.jpg')
        self.assertEqual('8de186e88781d7550827011a67f19fdb', r2)

    async def test_pixiv_download_original_by_id_002(self):
        # r = await self.client.util.download_pixiv_image(idx=110480732, token='TOKEN')
        r = await self.client.util.download_pixiv_image(idx=110480732, token=None)
        with open('/mnt/RAID5/findit.moe/temp/a.zip', 'wb') as f:
            f.write(r)

    async def test_search_by_batch_file_image_001(self):
        limit = 32
        r = await self.client.search.by_file(img=local_files,
                                             limit=limit)
        print(r)

    async def test_X(self):
        r = await self.client.tagger.by_booru_image_id(booru_name='gelbooru',
                                                       image_id=1)
        print(', '.join([x.tag + ' : ' + str(round(x.score, 4)) for x in r.results.data.general]))

    async def test_random_generator_image_speed_001(self):
        st = time.time()
        res = await self.client.util.random_search_generator(limit=32,
                                                             pool=None,
                                                             content=None)
        print(len(res.results.data), time.time() - st)


if __name__ == '__main__':
    unittest.main()