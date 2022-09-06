import sys, os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from dl.translators import *
from ui.constants import PROGRAM_PATH
os.chdir(PROGRAM_PATH)

def test_translator(translator: TranslatorBase, test_list: List):
    for test_dict in test_list:
        translator.set_source(test_dict['source'])
        translator.set_target(test_dict['target'])
        for text in test_dict['text_list']:
            translation = translator.translate(text)
            print(f'src: {text}, translation: {translation}')
            assert type(translation) == type(text)
            if isinstance(translation, List):
                assert len(translation) == len(text)

    text = ['', '', '', '', '', '', '']
    translation = translator.translate(text)
    assert len(translation) == len(text)
    print(f'src: {text}, translation: {translation}')
    text = ''
    translation = translator.translate(text)
    print(f'src: {text}, translation: {translation}')

engchscht_test_list = [
    {
        'source': '简体中文',
        'target': 'English',
        'text_list': [
            '中文测试',     # can be text or text list
            ['', '', '测试', '', '', '简中', ''],
        ]
    },
    {
        'source': 'English',
        'target': '简体中文',
        'text_list': [
            ['', '', 'test ', '', '', ' English', '']
        ]
    }
]

jaeng_test_list = [
    {
        'source': '日本語',
        'target': 'English',
        'text_list': [
            '日本語のテスト',
            ['日本語の...テスト', 'ククク…何かしらねぇ 当ててごらんなさい']
        ]
    },
]

if __name__ == '__main__':

    device = 'cuda'

    caiyun_setup_params = {
        'token': 'invalidtoken',
    }
    # ctranslator = CaiyunTranslator('简体中文', 'English', **caiyun_setup_params)
    # ptranslator = PapagoTranslator('简体中文', 'English')
    # gtranslator = GoogleTranslator('简体中文', 'English')
    # dtranslator = DeeplTranslator('简体中文', 'English')
    # sugoi_translator = SugoiTranslator('日本語', 'English', device= {'select': device})

    yandex_setup_params = {
        'api_key': 'invalidtoken'
    }
    yandex_translator = YandexTranslator('日本語', 'English', **yandex_setup_params)
    test_translator(yandex_translator, engchscht_test_list)

