from .base import *
import ctranslate2, sentencepiece as spm

SUGOIMODEL_TRANSLATOR_DIRPATH = 'data/models/sugoi_translator/'
SUGOIMODEL_TOKENIZATOR_PATH = SUGOIMODEL_TRANSLATOR_DIRPATH + "spm.ja.nopretok.model"
@register_translator('Sugoi')
class SugoiTranslator(BaseTranslator):

    concate_text = False
    params: Dict = {
        'device': DEVICE_SELECTOR()
    }

    def _setup_translator(self):
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'
        
        self.translator = ctranslate2.Translator(SUGOIMODEL_TRANSLATOR_DIRPATH, device=self.params['device']['value'])
        self.tokenizator = spm.SentencePieceProcessor(model_file=SUGOIMODEL_TOKENIZATOR_PATH)

    def _translate(self, src_list: List[str]) -> List[str]:
        
        text = [i.replace(".", "@").replace("．", "@") for i in src_list]
        tokenized_text = self.tokenizator.encode(text, out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)
        tokenized_translated = self.translator.translate_batch(tokenized_text)
        text_translated = [''.join(text[0]["tokens"]).replace('▁', ' ').replace("@", ".") for text in tokenized_translated]
        
        return text_translated

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'device':
            if hasattr(self, 'translator'):
                delattr(self, 'translator')
            self.translator = ctranslate2.Translator(SUGOIMODEL_TRANSLATOR_DIRPATH, device=self.params['device']['value'])

    @property
    def supported_tgt_list(self) -> List[str]:
        return ['English']

    @property
    def supported_src_list(self) -> List[str]:
        return ['日本語']