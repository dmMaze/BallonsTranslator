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
            'path_dat': {
                'value': r"C:\Program Files (x86)\ChangShinSoft\ezTrans XP\Dat",
                'display_name': 'DAT Path'
            },
            'path_j2k(J2KEngine.dll)': {
                'value': r"C:\Program Files (x86)\ChangShinSoft\ezTrans XP\J2KEngine.dll",
                'display_name': 'J2KEngine.dll Path'
            },
            'path_k2j(ehnd-kor.dll, Optional)': {
                'value': r"C:\Program Files (x86)\ChangShinSoft\ezTrans XP\ehnd-kor.dll",
                'display_name': 'ehnd-kor.dll Path (optional)'
            }
        }

        def _setup_translator(self):
            self.textblk_break = '\n'
            self.lang_map['日本語'] = 'j'
            self.lang_map['한국어'] = 'k'

            self.j2k_engine, self.k2j_engine = (None, None)
            dat_path = self.get_param_value('path_dat')
            j2k_path = self.get_param_value('path_j2k(J2KEngine.dll)')
            k2j_path = self.get_param_value('path_k2j(ehnd-kor.dll, Optional)')

            if os.path.exists(j2k_path):
                self.j2k_engine = MyClient(j2k_path, "J2K", dat_path)
            if os.path.exists(k2j_path):
                self.k2j_engine = MyClient(k2j_path, "K2J", dat_path)

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
            dat_path = self.get_param_value('path_dat')
            j2k_path = self.get_param_value('path_j2k(J2KEngine.dll)')
            k2j_path = self.get_param_value('path_k2j(ehnd-kor.dll, Optional)')

            if not self.j2k_engine and os.path.exists(j2k_path):
                self.j2k_engine = MyClient(j2k_path, "J2K", dat_path)
            if not self.k2j_engine and os.path.exists(k2j_path):
                self.k2j_engine = MyClient(k2j_path, "K2J", dat_path)

        @property
        def supported_tgt_list(self) -> List[str]:
            return ['한국어', '日本語'] if self.j2k_engine else ['한국어']

        @property
        def supported_src_list(self) -> List[str]:
            return ['한국어', '日本語'] if self.k2j_engine else ['日本語']
