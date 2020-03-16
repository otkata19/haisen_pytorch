from chainer import Chain, Variable, links, optimizers, optimizer, functions, serializers
import chainer
import numpy as np
import pickle
import csv
import codecs
import pandas as pd

from mecab_test import user_input

#[[名詞],[俳句]]を数値で組合せたリストをダウンロード
def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

id_to_word = pickle_load('static/id_to_word_marusen575.pickle')
word_to_id = pickle_load('static/word_to_id_marusen575.pickle')

#語彙数,embed_size,hidden_size,batch_size,epoch_numを指定
FLAG_GPU = False
vocab_size = len(id_to_word)
EMBED_SIZE = 300
HIDDEN_SIZE = 150
BATCH_SIZE = 128
EPOCH_NUM = 30

#### Attentionモデルの定義 ####
class LSTM_Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :param vocab_size: 使われる単語の種類数（語彙数）
        :param embed_size: 単語をベクトル表現した際のサイズ
        :param hidden_size: 中間層のサイズ
        """
        super(LSTM_Encoder, self).__init__(
            # 単語を単語ベクトルに変換する層
            xe = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            eh = links.Linear(embed_size, 4 * hidden_size),
            # 出力された中間層を4倍のサイズに変換するための層
            hh = links.Linear(hidden_size, 4 * hidden_size)
        )

    def __call__(self, x, c, h):
        """
        Encoderの計算
        :param x: one-hotなベクトル
        :param c: 内部メモリ
        :param h: 隠れ層
        :return: 次の内部メモリ、次の隠れ層
        """
        # xeで単語ベクトルに変換して、そのベクトルをtanhで活性化
        e = functions.tanh(self.xe(x))
        # 前の内部メモリの値cと単語ベクトルの4倍サイズ+中間層の4倍サイズを入力
        return functions.lstm(c, self.eh(e) + self.hh(h))

class Att_LSTM_Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        Attention ModelのためのDecoderのインスタンス化
        :param vocab_size: 語彙数
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(Att_LSTM_Decoder, self).__init__(
            # 単語を単語ベクトルに変換する層
            ye = links.EmbedID(vocab_size, embed_size, ignore_label = -1),
            # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            eh = links.Linear(embed_size, 4 * hidden_size),
            # Decoderの中間ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            hh = links.Linear(hidden_size, 4 * hidden_size),
            # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
            fh = links.Linear(hidden_size, 4 * hidden_size),
            # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
            bh = links.Linear(hidden_size, 4 * hidden_size),
            # 隠れ層サイズのベクトルを単語ベクトルのサイズに変換する層
            he = links.Linear(hidden_size, embed_size),
            # 単語ベクトルを語彙数サイズのベクトルに変換する層
            ey = links.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h, f, b):
        """
        Decoderの計算
        :param y: Decoderに入力する単語
        :param c: 内部メモリ
        :param h: Decoderの中間ベクトル
        :param f: Attention Modelで計算された順向きEncoderの加重平均
        :param b: Attention Modelで計算された逆向きEncoderの加重平均
        :return: 語彙数サイズのベクトル、更新された内部メモリ、更新された中間ベクトル
        """
        # 単語を単語ベクトルに変換
        e = functions.tanh(self.ye(y))
        # 単語ベクトル、Decoderの中間ベクトル、順向きEncoderのAttention、逆向きEncoderのAttentionを使ってLSTM
        c, h = functions.lstm(c, self.eh(e) + self.hh(h) + self.fh(f) + self.bh(b))
        # LSTMから出力された中間ベクトルを語彙数サイズのベクトルに変換する
        t = self.ey(functions.tanh(self.he(h)))
        return t, c, h

class Attention(Chain):
    def __init__(self, hidden_size, flag_gpu):
          """
          Attentionのインスタンス化
          :param hidden_size: 隠れ層のサイズ
          :param flag_gpu: GPUを使うかどうか
          """
          super(Attention, self).__init__(
              # 順向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
              fh = links.Linear(hidden_size, hidden_size),
              # 逆向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
              bh = links.Linear(hidden_size, hidden_size),
              # Decoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
              hh = links.Linear(hidden_size, hidden_size),
              # 隠れ層サイズのベクトルをスカラーに変換するための線形結合層
              hw = links.Linear(hidden_size, 1)
              )
          # 隠れ層のサイズを記憶
          self.hidden_size = hidden_size
          # GPUを使う場合はcupyを使わないときはnumpyを使う
          if flag_gpu:
              self.ARR = cuda.cupy
          else:
              self.ARR = np

    def __call__(self, fs, bs, h):
          """
          Attentionの計算
          :param fs: 順向きのEncoderの中間ベクトルが記録されたリスト
          :param bs: 逆向きのEncoderの中間ベクトルが記録されたリスト
          :param h: Decoderで出力された中間ベクトル
          :return: 順向きのEncoderの中間ベクトルの加重平均と逆向きのEncoderの中間ベクトルの加重平均
          """
          #ミニバッチのサイズを記憶
          batch_size = h.data.shape[0]
          # ウェイトを記録するためのリストの初期化
          ws = []
          # ウェイトの合計値を計算するための値を初期化
          sum_w = Variable(self.ARR.zeros((batch_size, 1), dtype='float32'))
          # Encoderの中間ベクトルとDecoderの中間ベクトルを使ってウェイトの計算
          for f, b in zip(fs, bs):
              # 順向きEncoderの中間ベクトル、逆向きEncoderの中間ベクトル、Decoderの中間ベクトルを使ってウェイトの計算
              w = functions.tanh(self.fh(f) + self.bh(b) + self.hh(h))
              # softmax関数を使って正規化する
              w = functions.exp(self.hw(w))
              # 計算したウェイトを記録
              ws.append(w)
              sum_w += w
          # 出力する加重平均ベクトルの初期化
          att_f = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype = 'float32'))
          att_b = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype = 'float32'))
          for f, b, w in zip (fs, bs, ws):
              # ウェイトの和が1になるように正規化
              w /= sum_w
              # ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
              att_f += functions.reshape(functions.batch_matmul(f, w), (batch_size, self.hidden_size))
              att_b += functions.reshape(functions.batch_matmul(b, w), (batch_size, self.hidden_size))
          return att_f, att_b

class Att_Seq2Seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, flag_gpu = True):
        """
        Seq2Seq + Attentionのインスタンス化
        :param vocab_size: 語彙数のサイズ
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 隠れ層のサイズ
        :param batch_size: ミニバッチのサイズ
        :param flag_gpu: GPUを使うかどうか
        """
        super(Att_Seq2Seq, self).__init__(
            # 順向きのEncoder
            f_encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
            # 逆向きのEncoder
            b_encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
            # Attention Model
            attention = Attention(hidden_size, flag_gpu),
            # Decoder
            decoder = Att_LSTM_Decoder(vocab_size, embed_size, hidden_size)
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # GPUを使うときはcupy、使わないときはnumpy
        if flag_gpu:
            self.ARR = cuda.cupy
        else:
            self.ARR = np

        # 順向きのEncoderの中間ベクトル、逆向きのEncoderの中間ベクトルを保存するためのリストを初期化
        self.fs = []
        self.bs = []

    def encode(self, words):
        """
        Encoderの計算
        :param words: 入力で使用する単語記録されたリスト
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype = 'float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype = 'float32'))
        # まずは順向きのEncoderの計算
        for w in words:
            c, h = self.f_encoder(w, c, h)
            # 計算された中間ベクトルを記録
            self.fs.append(h)

        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype = 'float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype = 'float32'))
        # 逆向きのEncoderの計算
        for w in reversed(words):
            c, h = self.b_encoder(w, c, h)
            # 計算された中間ベクトルを記録
            self.bs.insert(0, h)

        # 内部メモリ、中間ベクトルの初期化
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype = 'float32'))
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype = 'float32'))

    def decode(self, w):
        """
        Decoderの計算
        :param w: Decoderで入力する単語
        :return: 予測単語
        """
        # Attention Modelを使ってEncoderの中間層の加重平均を計算
        att_f, att_b = self.attention(self.fs, self.bs, self.h)
        # Decoderの中間ベクトル、順向きのAttention、逆向きのAttentionを使って次の中間ベクトル、内部メモリ、予測単語の計算
        t, self.c, self.h = self.decoder(w, self.c, self.h, att_f, att_b)
        return t

    def reset(self):
        """
        インスタンス変数を初期化する
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype = 'float32'))
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype = 'float32'))
        # Encoderの中間ベクトルを記録するリストの初期化
        self.fs = []
        self.bs = []
        # 勾配の初期化
        self.cleargrads()
#### モデルの定義終わり ####

def forward_test(enc_words, model, ARR):
    ret = []
    model.reset()
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    model.encode(enc_words)
    t = Variable(ARR.array([0], dtype='int32'))
    counter = 0
    while counter < 50:
        y = model.decode(t)
        # 次の単語のindexを返す
        label = y.data.argmax()
        ret.append(label)
        t = Variable(ARR.array([label], dtype='int32'))
        counter += 1
        if label == 1:
            counter = 50
    return ret

def id_list_to_senryu(id_list):
    ai_senryu_wordlist = []
    for each_id in id_list:
        ai_senryu_wordlist.append(id_to_word[each_id])
    ai_senryu = ''.join(ai_senryu_wordlist)
    return ai_senryu

def word_to_encode_id(word_list):
    word_list_to_id = []
    for each_word in word_list:
        if each_word not in word_to_id:
            each_word = '<unknown>'
            word_list_to_id.append(word_to_id[each_word])
        else:
            word_list_to_id.append(word_to_id[each_word])
        a = [[x] for x in word_list_to_id]
    b = np.array(list(reversed(a)))
    return b

#新しいニューラルネットワークモデルの作成
loaded_model = Att_Seq2Seq(vocab_size,
                       embed_size=EMBED_SIZE,
                       hidden_size=HIDDEN_SIZE,
                       batch_size=BATCH_SIZE,
                       flag_gpu=FLAG_GPU)
#重みとバイアスをロードし、loaded_modelに当てはめる
serializers.load_npz('static/marusen575_EMBED300_HIDDEN150_BATCH128_EPOCH60.weights', loaded_model)

def ai_return(text):
    word_list = user_input(text)
    index = [i for i in word_list]
    new_encode_id = word_to_encode_id(index)
    #推論
    ai_senryu_id = forward_test(new_encode_id, loaded_model, np)
    del ai_senryu_id[-1]
    ai_naka7 = id_list_to_senryu(ai_senryu_id)

    return ai_naka7
