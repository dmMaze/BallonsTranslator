import copy
import os

from .base import *



if os.environ.get('BALLOONTRANS_DEBUG', ''):

    @register_translator('Debug Original')
    class TransOriginal(BaseTranslator):

        concate_text = False
        cht_require_convert = True
        params: Dict = {
            'api_key': '',
            'delay': 0.0,
        }

        def _setup_translator(self):
            for k in self.lang_map.keys():
                self.lang_map[k] = 'dummy language'
            
        def _translate(self, src_list: List[str]) -> List[str]:
            return copy.copy(src_list)
        
    def transhook_copy_original(translations: List[str] = None, textblocks: List[TextBlock] = None, translator: BaseTranslator = None, **kwargs):
        if textblocks is not None and isinstance(translator, TransOriginal):
            for ii, _ in enumerate(translations):
                translations[ii] = textblocks[ii].translation

    TransOriginal.register_postprocess_hooks({'copy_original': transhook_copy_original})

    @register_translator('Debug Source')
    class TransSource(BaseTranslator):

        concate_text = False
        cht_require_convert = True
        params: Dict = {
            'api_key': '',
            'delay': 0.0,
        }

        def _setup_translator(self):
            for k in self.lang_map.keys():
                self.lang_map[k] = 'dummy language'
            self.register_preprocess_hooks
            
        def _translate(self, src_list: List[str]) -> List[str]:
            return copy.copy(src_list)