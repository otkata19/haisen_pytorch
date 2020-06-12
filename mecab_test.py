import re
import io,sys
import MeCab
import subprocess

cmd='echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
path = (subprocess.Popen(cmd, stdout=subprocess.PIPE,shell=True).communicate()[0]).decode('utf-8')
m=MeCab.Tagger("-d {0}".format(path))

def user_input(text):
    # 記号の削除
    clean_i = re.sub(re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]'), '', text)
    sentence = clean_i.replace("　", "").replace(" ", "") # スペースの消去
    node = m.parseToNode(sentence).next #解析してノードに変換
    word_list = []
    while node.next: #nodeがfalseになるまで
        word_list.append(node.surface)
        node = node.next #次の要素に移動（末尾の場合はfalseになる）

    return word_list
