import ballontranslator.utils.shared as shared

if shared.ON_WINDOWS:
    from .base import *
    import os
    from typing import Literal
    from msl.loadlib import Client64


    class MyClient(Client64):
        def __init__(self, engine_path, engine_type: Literal['J2K', 'K2J'], dat_path):
            super(MyClient, self).__init__(module32=str(os.path.dirname(os.path.realpath(__file__))) + '/module_eztrans32.py',
                                        engine_path=engine_path,
                                        engine_type=engine_type,
                                        dat_path=dat_path)

        def translate(self, src_text: Union[str, list]):
            return self.request32('translate', src_text)


    def fullwidth_to_halfwidth(text):
        mapping = {i: i - 0xFEE0 for i in range(0xFF01, 0xFF5F)}
        mapping[0x3000] = 0x0020  # 전각 공백 → 반각 공백
        return text.translate(mapping)

    @register_translator('ezTrans')
    class ezTransTranslator(BaseTranslator):
        dependencies = ['msl-loadlib']

        concate_text = True

        params: Dict = {
            'path_dat': r"C:\Program Files (x86)\ChangShinSoft\ezTrans XP\Dat",
            'path_j2k(J2KEngine.dll)': r"C:\Program Files (x86)\ChangShinSoft\ezTrans XP\J2KEngine.dll",
            'path_k2j(ehnd-kor.dll, Optional)': r"C:\Program Files (x86)\ChangShinSoft\ezTrans XP\ehnd-kor.dll"
        }

        def _setup_translator(self):
            self.textblk_break = '\n'
            self.lang_map['日本語'] = 'j'
            self.lang_map['한국어'] = 'k'

            self.j2k_engine, self.k2j_engine = (None, None)

            if os.path.exists(self.params['path_j2k(J2KEngine.dll)']):
                self.j2k_engine = MyClient(self.params['path_j2k(J2KEngine.dll)'], "J2K", self.params['path_dat'])
            if os.path.exists(self.params['path_k2j(ehnd-kor.dll, Optional)']):
                self.k2j_engine = MyClient(self.params['path_k2j(ehnd-kor.dll, Optional)'], "K2J", self.params['path_dat'])

        def _translate(self, src_list: List[str]) -> List[str]:
            source = self.lang_map[self.lang_source]
            target = self.lang_map[self.lang_target]

            if source != target:
                engine: MyClient = getattr(self, f"{source}2{target}_engine")
                return engine.translate(src_list) if source != "k" else fullwidth_to_halfwidth(engine.translate(src_list))
            else:
                return src_list

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)

            if not self.j2k_engine and os.path.exists(self.params['path_j2k(J2KEngine.dll)']):
                self.j2k_engine = MyClient(self.params['path_j2k(J2KEngine.dll)'], "J2K", self.params['path_dat'])
            if not self.k2j_engine and os.path.exists(self.params['path_k2j(ehnd-kor.dll, Optional)']):
                self.k2j_engine = MyClient(self.params['path_k2j(ehnd-kor.dll, Optional)'], "K2J", self.params['path_dat'])

        @property
        def supported_tgt_list(self) -> List[str]:
            return ['한국어', '日本語'] if self.j2k_engine else ['한국어']

        @property
        def supported_src_list(self) -> List[str]:
            return ['한국어', '日本語'] if self.k2j_engine else ['日本語']
