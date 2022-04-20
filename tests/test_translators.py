import sys, os
sys.path.append(os.getcwd())
from dl.translators import TranslatorBase, GoogleTranslator, PapagoTranslator, TRANSLATORS, CaiyunTranslator, DeeplTranslator

def test_translator(translator: TranslatorBase, test_list):
    for test_dict in test_list:
        translator.set_source(test_dict['source'])
        translator.set_target(test_dict['target'])
        for text in test_dict['text_list']:
            print(f'src: {text}, translation: {translator.translate(text)}')

    text = ['', '', '', '', '', '', '']
    print(f'src: {text}, translation: {translator.translate(text)}')
    text = ''
    print(f'src: {text}, translation: {translator.translate(text)}')

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
    },
    {
        'source': 'English',
        'target': '繁體中文',
        'text_list': [
            '中文测试',
            ['', '', 'test ', '', '', ' English', '']
        ]
    }
]

if __name__ == '__main__':

    caiyun_setup_params = {
        'token': 'invalidtoken',
    }
    # ctranslator = CaiyunTranslator('简体中文', 'English', **caiyun_setup_params)
    ptranslator = PapagoTranslator('简体中文', 'English')
    ptranslator = PapagoTranslator('简体中文', 'English')
    gtranslator = GoogleTranslator('简体中文', 'English')
    dtranslator = DeeplTranslator('简体中文', 'English')
    test_translator(ptranslator, engchscht_test_list)

