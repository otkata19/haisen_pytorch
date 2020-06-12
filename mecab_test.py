import re
import io,sys
from janome.tokenizer import Tokenizer

def user_input(text):
    # 記号の削除
    clean_i = re.sub(re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]'), '', text)
    sentence = clean_i.replace("　", "").replace(" ", "") # スペースの消去
    # janomeで形態素解析
    t = Tokenizer()
    word_list = [token.surface for token in t.tokenize(sentence)]
    return word_list
