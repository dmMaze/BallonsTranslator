import time
import requests
import re
import urllib.request
from .constants import SOURCE_DOWNLOAD_PATH
from .exceptions import SourceNotImplemented
import os

PROXY = urllib.request.getproxies()


class SourceBase:
    def __init__(self):
        self.link = ''
        self.SOURCEMAP = {
            0: 'manual',
            1: 'nhentai'
        }
        self.source: str = ''
        self.number_of_pages: int = 0

    def SetLink(self, link):
        self.link = link

    def CheckLink(self):
        if 'https://' not in self.link:
            self.link = 'https://' + self.link
        for v in self.SOURCEMAP.items():
            if v[1] in self.link.lower():
                self.source = v[1]
        if not self.source:
            self.link = ''
            raise SourceNotImplemented


class nhentai(SourceBase):
    def __init__(self):
        super().__init__()
        self.gallery_number: int = 0
    def Help(self):
        print("""Please use gallery links. To get link follow these steps:
        + Go to desired doujin
        + click on first image
        + right click it and select open image in new tab
        + in the url bar you can see gallery link
        These steps are necessary as I don't know of any methods to bypass cloudflare""")

    def FetchImages(self):
        size = len(self.link)
        url = self.link[:size - 5]
        number = (re.search('galleries/(.*)/', url)).group(1)
        self.gallery_number = number
        i = 1
        path = fr'{SOURCE_DOWNLOAD_PATH}\{number}'
        if not os.path.exists(path):
            os.makedirs(path)
        while True:
            try:
                img_data = requests.get(f'{url}{i}.jpg').content
                if '404 Not Found' in str(img_data):
                    break
                with open(rf'{path}\{i}.jpg', 'wb') as image:
                    image.write(img_data)
                i += 1
                time.sleep(1)  # Avoiding anti ddos ban
            except Exception as e:
                print(e)
                break
        self.number_of_pages = i

    def ReturnGalleryNumber(self):
        return self.gallery_number

    def run(self, url):
        self.SetLink(url)
        self.CheckLink()
        self.FetchImages()






