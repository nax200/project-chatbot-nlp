
from flask import Flask,request,abort
import requests
from app.Config import *
import json

from pythainlp.tokenize import sent_tokenize, word_tokenize
from pythainlp import Tokenizer
from pythainlp.util import dict_trie
from pythainlp.corpus.common import thai_words
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf


def load_model() :
    loaded_model = tf.keras.models.load_model('Model')
    loaded_model.summary()
    return loaded_model
    
Model = load_model()

def load_w2vec_model():
    word2vec_model = KeyedVectors.load_word2vec_format('Src/LTW2V_v0.1.bin', binary=True, unicode_errors='ignore')
    return word2vec_model

Word2vec_model = load_w2vec_model()

def map_word_index(word_seq):
    indices = [] 
    for word in word_seq:
        if word in Word2vec_model.vocab:
            indices.append(Word2vec_model.vocab[word].index + 1)
        else:
            indices.append(1)
    return indices

app=Flask(__name__)

@app.route('/webhook',methods=['POST','GET'])

def webhook():
    if request.method=='POST':
        payload =request.json
        Reply_token=payload['events'][0]['replyToken']
        message=payload['events'][0]['message']['text']
        Reply_text= prediction(message)
        print(Reply_text,flush=True)
        ReplyMessage(Reply_token,Reply_text,Channel_access_token)
        return request.json,200
    elif request.method=='GET':
        return "this is method GET!!!",200
    else:
        abort(400)



def prediction(message):
    words = ["ภาคพิเศษ","ภาคปกติ","ภาคเปย์","ภาคปก","ภาคเป","สหกิจ","เฟซบุ๊ก","เฟสบุ๊ค","วิทย์คอม"]
    custom_words_list = set(thai_words())
    custom_words_list.update(words)
    trie = dict_trie(dict_source=custom_words_list)
    word_seq = word_tokenize(message, engine="newmm", custom_dict=trie)
    word_indices = map_word_index(word_seq)
    max_leng = Model.layers[0].output_shape[0][1]
    padded_wordindices = pad_sequences([word_indices], maxlen=max_leng, value=0)
    logit = Model.predict(padded_wordindices, batch_size=32)
    index = [ logit[0][pred] for pred in np.argmax(logit, axis=1) ][0]
    index_to_label = sorted(['การเรียนแตกต่างกันไหม','กิจกรรมในภาควิชา','ค่าใช้จ่าย','ช่องทางติดต่อภาควิชา','ทักทาย','ทุนในภาควิชา','อุปกรณ์การเรียน','เรียนเวลาใด','เรียนแล้วคุ้มไหม'])
    predict = [index_to_label[pred] for pred in np.argmax(logit, axis=1) ][0]
    if index > 0.675:
        if predict == "ทักทาย":
            Reply_text="สวัสดีค่ะ คุณต้องการสอบถามเรื่องอะไรคะ"
        elif predict == "เรียนเวลาใด":
            Reply_text="เวลาเรียนไม่ต่างจากภาคปกติมากนัก ด้วยในการจัดตารางเรียน จะจัดให้ภาคปกติก่อน เมื่อเสร็จจึงจัดให้กับภาคพิเศษทำให้เวลาเรียนของภาคพิเศษ มีทั้งเช้ามาก (8:00) กลางวัน เย็น และ เสาร์-อาทิตย์ ซึ่งภาคปกติก็มีตารางเรียนแบบนี้เช่นกัน"
        elif predict == "ค่าใช้จ่าย":
            Reply_text="ภาคพิเศษ เป็นการเรียนที่ผู้ปกครองออกค่าใช้จ่ายในการเรียนของนิสิต โดยรัฐ ฯ มิได้ให้การสนับสนุน มีค่าใช้จ่ายดังนี้\n1) เทอมตามประกาศของมหาวิทยาลัย 45,700 บาทโดยประมาณ (อาจมีการเปลี่ยนแปลงตามประกาศ) และไม่รวมภาคฤดูร้อน (ส่วนภาคฤดูร้อนแบบเหมาจ่ายเทอมละ 11,000 บาท)\n2) เทอมที่มีการลงทะเบียนสหกิจศึกษา (ปี 4/1 หรือ 4/2) เป็นค่าเทอมเหมาจ่ายสหกิจศึกษา ประมาณ 26,100 บาท (ดูประกาศของมหาวิทยาลัย)\n3) ค่าอุปกรณ์ประกอบการเรียน อาทิ Computer, Notebook Computer, Tablet, ตำราเรียน, คู่มือ, ค่าใช้จ่ายอินเตอร์เน็ต, ค่าทำกิจกรรม-รายงาน"
        elif predict == 'เรียนแล้วคุ้มไหม':
            Reply_text="หากนิสิตมีความตั้งใจ อุตสาหะ หมั่นฝึกฝนตนเองอย่างสม่ำเสมอ ทำงานหนัก พัฒนาตนเอง พัฒนาทักษะการเขียนโปรแกรม คณิตศาสตร์ และ ภาษาอังกฤษ และเข้าร่วมกิจกรรมพิเศษของทางภาควิชา อย่างสม่ำเสมอ และสามารถจบได้ด้วยผลการเรียนที่ดี สามารถหางานที่มีเงินเดือนเฉลี่ยที่ 23,000 – 50,000 บาท (ไม่นับเงินเดือนราชการ และ รัฐวิสาหกิจ) มีโอกาสได้งานสูง ตลาดงานยังมีความต้องการในระดับสูงมากแต่ก็ต้องการคนที่มีความตั้งใจและมีฝีมือดีจริงๆ"
        elif predict == 'การเรียนแตกต่างกันไหม':
            Reply_text="ในวิชาหลักๆของภาควิชาที่นิสิตต้องลงเรียนไม่น้อยกว่า 30 รายวิชา นั้น จะตัดเกรดนิสิตภาคปกติกับภาคพิเศษด้วยกัน และในวิชานอกภาคฯ จำนวนหนึ่งก็จะตัดเกรดด้วยกัน"
        elif predict == 'กิจกรรมในภาควิชา':
            Reply_text="- วันปฐมนิเทศ\n- กิจกรรมนิสิตพบที่ปรึกษา\n- วันรับปริญญา\n- กิจกรรมรับน้อง\n- การหาที่ฝึกสหกิจศึกษาให้\n- โครงการความร่วมมือกับ SCB, SaleForce, CP All, CPF, ….. เพื่อเพิ่มโอกาสในการหางาน\n- สามารถสมัครเข้าโครงการแลกเปลี่ยน ในต่างประเทศ (ไต้หวัน) 1 ปี\n- KU DEPA Data Analytics\n- Dev Camp"
        elif predict == 'ช่องทางติดต่อภาควิชา':
            Reply_text="ภาควิชาวิทยาการคอมพิวเตอร์ คณะวิทยาศาสตร์ มหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตบางเขน\n\nอาคารวิทยาศาสตร์กายภาพ 45 ปี ชั้น 7 (สำนักงานธรุการภาควิชา) และชั้น 8\nอาคารทวี ญาณสุคนธ์ ชั้น 3\n\nเลขที่ 50 ถนนงามวงศ์วาน เขตจตุจักร กรุงเทพฯ 10900\nhttps://goo.gl/maps/CrTZbfXGXAQDo5D69\n\nโทรศัพท์ 02-562-5555 ต่อ 647209, 647210\n\nFacebook: ภาควิชาวิทยาการคอมพิวเตอร์ เกษตรศาสตร์\nhttps://www.facebook.com/comsci.ku"
        elif predict == 'ทุนในภาควิชา':
            Reply_text="โครงการปริญญาตรีสาขาวิทยาการคอมพิวเตอร์ภาคพิเศษ ได้ให้การสนับสนุนนิสิตภาคพิเศษ ในหลากหลายรูปแบบ ดังตัวอย่างต่อไปนี้\n- ทุนการศึกษาสำหรับนิสิตชั้นปีที่ 1 ใน 2 ภาคการศึกษา หากนิสิตเรียนไม่ช้าไปกว่าแผนการเรียนที่กำหนดไว้\n- ทุนผลการเรียนดีเด่น ได้ A 3 รายวิชา และ 5 รายวิชา\n- ทุนเกรดเฉลี่ย 3.50 ขึ้นไป\n- ทุนโครงงานวิทยาการคอมพิวเตอร์\n- ทุนสหกิจศึกษา ณ ต่างประเทศ เช่น South Korea : Dongseo University, Taiwan : National Central University, Tamkang University, Yuan Ze University, Asia University, India : Christ University\n- ทุนโครงการ SUMMER CAMP ประเทศ Taiwan\n- ทุนผู้ช่วยสอน ทุนผู้ช่วยวิจัย\n- ทุนนำเสนอผลงานทางวิชาการ ทั้งในและต่างประเทศ"
        elif predict == 'อุปกรณ์การเรียน':
            Reply_text="ทางคณะมีอุปกรณ์ให้ในคาบปฎิบัติ แต่ถ้ามีความประสงค์ต้องการยืมกลับบ้าน นิสิตจะต้องทำเรื่องขอยืม notebook โดยผ่านห้องสมุด"
    else: Reply_text="ไม่เข้าใจคำถามค่ะ"
    return Reply_text

def ReplyMessage(Reply_token,TextMessage,Line_Acees_Token):
    LINE_API='https://api.line.me/v2/bot/message/reply/'
    
    Authorization='Bearer {}'.format(Line_Acees_Token)
    print(Authorization)
    headers={
        'Content-Type':'application/json; char=UTF-8',
        'Authorization':Authorization
    }

    data={
        "replyToken":Reply_token,
        "messages":[{
            "type":"text",
            "text":TextMessage
        }
        ]
    }
    data=json.dumps(data) # ทำเป็น json
    r=requests.post(LINE_API,headers=headers,data=data)
    return 200