import shutil
import requests
import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re
import time
from .constants import SOURCE_DOWNLOAD_PATH
from .exceptions import ImagesNotFoundInRequest, NotValidUrl
from utils.logger import logger as LOGGER
import os


class SourceBase:
    def __init__(self):
        self.url: str = ''
        self.path: str = ''
        self.title: str = ''
        self.template: list[str] = ['cover']
        self.image_urls: list[str] = []
        self.last_page_num: int = 0

    def SetUrl(self, url):
        self.url = url

    def SetTitle(self, title):
        self.title = title

    def SaveNumberOfPages(self, path):
        #  clear file before saving last page number
        open(path, 'w').close()

        with open(path, 'w') as txt:
            txt.write(str(self.last_page_num))

    def ReturnNumberOfPages(self) -> int:
        return self.last_page_num

    def ReturnFullPathToProject(self) -> str:
        return self.path

    def CheckLink(self):
        if 'https://' not in self.url:
            raise NotValidUrl(self.url)

    def CheckFiles(self, path):

        #  read known page number
        try:
            with open(path, 'r') as txt:
                try:
                    self.last_page_num = txt.readlines()[0]
                except IndexError:
                    return False
        except FileNotFoundError:
            return False

        #  count images in directory
        files = os.listdir(self.path)
        number_of_images = 0
        for i in files:
            if '.jpg' in i:
                number_of_images += 1

        if number_of_images == int(self.last_page_num):
            return True
        else:
            return False

    def FetchImageUrls(self, force_redownload: bool = False):
        if self.url:
            LOGGER.info('Scraping website for images')

            if not self.title:
                _url = self.url.translate({ord(c): None for c in '\./:*?"<>|'})
                self.path = rf'{SOURCE_DOWNLOAD_PATH}\{_url}'

            else:
                self.path = rf'{SOURCE_DOWNLOAD_PATH}\{self.title}'

            path_to_page_num = rf'{self.path}\pages.txt'

            are_downloaded = False
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            elif os.path.exists(self.path) and force_redownload is False:
                are_downloaded = self.CheckFiles(path_to_page_num)

            if are_downloaded is False:

                options = Options()
                # options.add_argument("--headless")
                driver = uc.Chrome(options=options)

                #  wait for cloudflare to pass
                driver.get(self.url)
                time.sleep(10)
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                driver.close()

                _elements = soup.find_all('img')
                urls = [img['src'] for img in _elements]
                images = [k for k in urls if 'https' in k]
                temp_list = []

                #  filter images for only those with numbers at the end and makes sure there are no duplicates
                for i in images:

                    try:
                        temp = (re.search('https://(.*)/(.*?)jpg', i)).group(2)

                        if any(k.isdigit() for k in temp) and temp not in temp_list and len(temp) < 10 and temp not in self.template:
                            i = i.replace(' ', '%20')
                            self.image_urls.append(i)

                        temp_list.append(temp)

                    except AttributeError:
                        pass

                if not self.image_urls:
                    raise ImagesNotFoundInRequest(self.image_urls)

                self.WebsiteExceptions()
                LOGGER.info(self.image_urls)
                self.DownloadImages()

    def WebsiteExceptions(self):
        urls = self.image_urls
        if any('nhentai' in k for k in urls):

            for i, s in enumerate(urls):
                urls[i] = s.replace('https://t', 'https://i').replace('t.jpg', '.jpg')

            self.image_urls = urls

    def DownloadImages(self):
        n = 1
        LOGGER.info('Downloading images')

        for i in self.image_urls:
            img_data = requests.get(i, stream=True)

            with open(rf'{self.path}\{n:03}.jpg', 'wb') as image:
                shutil.copyfileobj(img_data.raw, image)
            n += 1
            #  Avoid IP ban
            time.sleep(1)

        self.last_page_num = len(self.image_urls)

        self.SaveNumberOfPages(rf'{self.path}\pages.txt')

        LOGGER.info('Download complete')

    def run(self, url: str, force_redownload: bool, title: str = ''):
        self.SetUrl(url)
        if title:
            self.SetTitle(title)
        self.FetchImageUrls(force_redownload)

