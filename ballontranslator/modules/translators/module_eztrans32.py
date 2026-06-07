# -*- coding: utf-8 -*-
from ctypes import c_char_p, c_int, c_wchar_p
from ctypes.wintypes import BOOL
from typing import Literal
from msl.loadlib import Server32

import re

ENGINE_TYPES = Literal['J2K', 'K2J']


class MyServer(Server32):
    def __init__(self, host, port, engine_path, engine_type: ENGINE_TYPES, dat_path):
        super(MyServer, self).__init__(engine_path, 'windll', host, port)
        self.engine = TransEngine(self.lib, engine_type, dat_path)

    def translate(self, src_text: str | list) -> str | list:
        def work(t):
            return encode_text(self.engine.translate(decode_text(t)))
        if type(src_text) == list:
            if len(src_text) == 1:
                return [work(src_text[0])]
            return self.translate(src_text[:-1]) + [work(src_text[-1])]
        elif type(src_text) == str:
            return work(src_text)


class TransEngine:
    def __init__(self, engine, engine_type: ENGINE_TYPES, dat_path):
        self.start = getattr(engine, f"{engine_type}_InitializeEx")
        self.start.argtypes = [c_char_p, c_char_p]
        self.start.restype = BOOL
        self.trans = getattr(engine, f"{engine_type}_TranslateMMNTW")
        self.trans.argtypes = [c_int, c_wchar_p]
        self.trans.restype = c_wchar_p
        self.start_obj = self.start(b"CSUSER123455", dat_path.encode('utf-8'))

    def translate(self, src_text):
        trans_obj = self.trans(0, src_text)
        return trans_obj


def decode_text(txt):
    chars = "↔◁◀▷▶♤♠♡♥♧♣⊙◈▣◐◑▒▤▥▨▧▦▩♨☏☎☜☞↕↗↙↖↘♩♬㉿㈜㏇™㏂㏘＂＇∼ˇ˘˝¡˚˙˛¿ː∏￦℉€㎕㎖㎗ℓ㎘㎣㎤㎥㎦㎙㎚㎛㎟㎠㎢㏊㎍㏏㎈㎉㏈㎧㎨㎰㎱㎲㎳㎴㎵㎶㎷㎸㎀㎁㎂㎃㎄㎺㎻㎼㎽㎾㎿㎐㎑㎒㎓㎔Ω㏀㏁㎊㎋㎌㏖㏅㎭㎮㎯㏛㎩㎪㎫㎬㏝㏐㏓㏃㏉㏜㏆┒┑┚┙┖┕┎┍┞┟┡┢┦┧┪┭┮┵┶┹┺┽┾╀╁╃╄╅╆╇╈╉╊┱┲ⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ½⅓⅔¼¾⅛⅜⅝⅞ⁿ₁₂₃₄ŊđĦĲĿŁŒŦħıĳĸŀłœŧŋŉ㉠㉡㉢㉣㉤㉥㉦㉧㉨㉩㉪㉫㉬㉭㉮㉯㉰㉱㉲㉳㉴㉵㉶㉷㉸㉹㉺㉻㈀㈁㈂㈃㈄㈅㈆㈇㈈㈉㈊㈋㈌㈍㈎㈏㈐㈑㈒㈓㈔㈕㈖㈗㈘㈙㈚㈛ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂"
    for c in chars:
        if c in txt:
            txt = txt.replace(c, "\\u" + str(hex(ord(c)))[2:])
    return txt


def encode_text(txt):
    return re.sub(r'(?i)(?<!\\)(?:\\\\)*\\u([0-9a-f]{4})', lambda m: chr(int(m.group(1), 16)), txt)
