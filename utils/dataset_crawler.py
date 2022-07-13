import wget
import logging
url_0 = 'http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-'
url_1 = '-cam0-rgb.zip'  # 目标路由，下载的资源是图片
path = 'dataset/'  # 保存的路径
url = None
for i in range(1,31):
    try:
        url = url_0+str(i)+url_1 if i>=10 else url_0+'0'+str(i)+url_1
        logging.info('Download from :',url)
        wget.download(url, path+str(i)+'.zip')  # 下载
        logging.info('finish')
    except:
        logging.error('Wrong at ' + url)