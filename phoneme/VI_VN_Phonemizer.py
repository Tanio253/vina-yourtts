from typing import Dict

from phoneme.base import BasePhonemizer
from phoneme.vPhon import trans, convert
import sys, re, io, string, argparse
from rules import *

class VI_VN_Phonemizer(BasePhonemizer):

    language = "vi-vn"

    def __init__(self, punctuations=_DEF_JA_PUNCS, keep_puncs=True, **kwargs):  # pylint: disable=unused-argument
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)
        self.chao = False
        self.delimit = ''
        self.dialect = 's'
        self.nosuper = False
        self.glottal = False
        self.phonemic = False
        self.output_ortho = '' 
        self.eight = False
        self.tokenize = False
    @staticmethod
    def name():
        return "vi_vn_phonemizer"

    def _phonemize(self, text: str, separator: str = "|") -> str:
        compound = ''
        ortho = ''
        words = text.split()
        # toss len==0 junk
        words = [word for word in words if len(word)>0]
        # hack to get rid of single hyphens
        words = [word for word in words if word!='-']
        # hack to get rid of single underscores
        words = [word for word in words if word!='_']
        for i in range(0,len(words)):
            word = words[i].strip()
            ortho += word
            word = word.strip(string.punctuation).lower()
            # if tokenize==true: 
            # call this routine for each substring and re-concatenate
            if (self.tokenize and '-' in word) or (self.tokenize and '_' in word):
                substrings = re.split(r'(_|-)', word)
                values = substrings[::2]
                delimiters = substrings[1::2] + ['']
                ipa = [convert(x, self.dialect, self.chao, self.eight, self.nosuper, self.glottal, self.phonemic, self.delimit).strip() for x in values]
                seq = ''.join(v+d for v,d in zip(ipa, delimiters))
            else:
                seq = convert(word, self.dialect, self.chao, self.eight, self.nosuper, self.glottal, self.phonemic, self.delimit).strip()
            # concatenate
            if len(words) >= 2:
                ortho += ' '
            if i < len(words)-1:
                seq = seq + ' '
            compound = compound + seq

        # entire text has been parsed
        if ortho == '':
            pass
        else:
            ortho = ortho.strip()
            # print orthography?
            if self.output_ortho: print(ortho, self.output_ortho, sep= '', end='')
            
    
        return compound

    def phonemize(self, text: str, separator="|", language=None) -> str:
        """Custom phonemize for JP_JA

        Skip pre-post processing steps used by the other phonemizers.
        """
        return self._phonemize(text, separator)

    @staticmethod
    def supported_languages() -> Dict:
        return {"vi-vn": "Vietnamese (Vietnam)"}

    def version(self) -> str:
        return "0.0.1"

    def is_available(self) -> bool:
        return True


# if __name__ == "__main__":
#     text = "これは、電話をかけるための私の日本語の例のテキストです。"
#     e = JA_JP_Phonemizer()
#     print(e.supported_languages())
#     print(e.version())
#     print(e.language)
#     print(e.name())
#     print(e.is_available())
#     print("`" + e.phonemize(text) + "`")