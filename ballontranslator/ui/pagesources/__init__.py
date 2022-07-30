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
        self.path: str = ''

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

    def ReturnNumberOfPages(self) -> int:
        return self.number_of_pages

    def CheckFiles(self, path):
        try:
            with open(path, 'r') as txt:
                try:
                    self.number_of_pages = txt.readlines()[0]
                except IndexError:
                    return False
        except FileNotFoundError:
            return False
        files = os.listdir(self.path)
        number_of_images = 0
        for i in files:
            if '.jpg' in i:
                number_of_images += 1
        if number_of_images == self.number_of_pages:
            return True

    def SaveNumberOfPages(self, path):
        open(path, 'w').close()  # clears file before saving number of pages
        with open(path, 'w') as txt:
            txt.write(str(self.number_of_pages))


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

    def FetchImages(self, skip_check: bool = False):
        url = self.link
        number = (re.search('galleries/(.*)/', url)).group(1)
        self.gallery_number = number
        i = 1
        self.path = fr'{SOURCE_DOWNLOAD_PATH}\{number}'
        path_to_txt = rf'{self.path}\pages.txt'
        skip = False
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        elif os.path.exists(self.path) and skip_check is False:
            skip = self.CheckFiles(path_to_txt)
        if skip is False:
            while True:
                try:
                    img_data = requests.get(f'https://i3.nhentai.net/galleries/{number}/{i}.jpg', proxies=PROXY).content
                    if '404 Not Found' in str(img_data):
                        break
                    with open(rf'{self.path}\{i}.jpg', 'wb') as image:
                        image.write(img_data)
                    i += 1
                    time.sleep(1)  # Avoiding anti ddos ban
                except Exception as e:
                    print(e)
                    break
            self.number_of_pages = i - 1
            self.SaveNumberOfPages(path_to_txt)

    def ReturnGalleryNumber(self):
        return self.gallery_number

    def run(self, url, skip_check: bool):
        self.SetLink(url)
        self.CheckLink()
        self.FetchImages(skip_check)






