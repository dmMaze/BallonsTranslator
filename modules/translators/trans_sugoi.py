from .base import *
import ctranslate2, sentencepiece as spm

SUGOIMODEL_TRANSLATOR_DIRPATH = 'data/models/sugoi_translator/'
SUGOIMODEL_TOKENIZATOR_PATH = SUGOIMODEL_TRANSLATOR_DIRPATH + "spm.ja.nopretok.model"
@register_translator('Sugoi')
class SugoiTranslator(BaseTranslator):

    concate_text = False
    params: Dict = {
        'device': {
            'type': 'selector',
            'options': ['cpu', 'cuda'],
            'select': 'cpu'
        }
    }

    def _setup_translator(self):
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'
        
        self.translator = ctranslate2.Translator(SUGOIMODEL_TRANSLATOR_DIRPATH, device=self.params['device']['select'])
        self.tokenizator = spm.SentencePieceProcessor(model_file=SUGOIMODEL_TOKENIZATOR_PATH)

    def _translate(self, text: Union[str, List]) -> Union[str, List]:
        input_is_lst = True
        if isinstance(text, str):
            text = [text]
            input_is_lst = False
        
        text = [i.replace(".", "@").replace("．", "@") for i in text]
        tokenized_text = self.tokenizator.encode(text, out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)
        tokenized_translated = self.translator.translate_batch(tokenized_text)
        text_translated = [''.join(text[0]["tokens"]).replace('▁', ' ').replace("@", ".") for text in tokenized_translated]
        
        if not input_is_lst:
            return text_translated[0]
        return text_translated

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'device':
            if hasattr(self, 'translator'):
                delattr(self, 'translator')
            self.translator = ctranslate2.Translator(SUGOIMODEL_TRANSLATOR_DIRPATH, device=self.params['device']['select'])

    @property
    def supported_tgt_list(self) -> List[str]:
        return ['English']

    @property
    def supported_src_list(self) -> List[str]:
        return ['日本語']