# %%
from pyngrok import ngrok # pyngok 套件
from pyngrok.conf import PyngrokConfig
webhook_url = ngrok.connect(addr='127.0.0.1:5000', pyngrok_config=PyngrokConfig(start_new_session=True))
print( '外部網址 => {}'.format(webhook_url) )

from flask import Flask, request, abort # flask套件
from linebot import LineBotApi, WebhookHandler # line-bot-sdk套件
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import google.generativeai as genai # google-generativeai套件
import json, requests # requests及其餘基本套件

app = Flask(__name__)

# LINE Channel access token
line_bot_api = LineBotApi('')
# LINE Channel secret
handler = WebhookHandler('')
# GEMINI API KEY
genai.configure(api_key='')

generation_config = {
  "temperature": 1,
  "top_p": 0.95, "top_k": 0,
  "max_output_tokens": 200
}

safety_settings = [ # google-generativeai預設設定
  {
    "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  }
]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", # google-generativeai模型設定
                              generation_config=generation_config, safety_settings=safety_settings)

@app.route('/', methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    json.loads(body) # 使用json格式載入資料
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

dict_user_context = {} # 設定儲存上下文全域變數

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id # 使用者編號
    user_text = event.message.text # 使用者訊息

    context = dict_user_context.get(user_id, {}) # 取得上下文

    country = ['蘇澳', '頭城', '宜蘭', '南澳', '羅東', '三星', '大同', '五結', '員山', '冬山', '礁溪', '壯圍']
    weather = ['降雨', '溫度', '濕度', '', '風速', '體感', '現象', '', '', '指數', '描述']

    def API(country_index, weather_index):
        base_url = 'https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-D0047-003?Authorization=CWA-0C63261F-E152-485F-B43E-F17EE1CADF6B'
        params = {
            'Authorization': '' # 中央氣象局 API KEY
        }
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json() # 回傳的json格式

            word = data['records']['locations'][0]['location'][country_index]['weatherElement'][weather_index]['time'][2]['elementValue'][0]['value']
            context['message'] = word

            convo = model.start_chat(history=[]) # 設定聊天紀錄為空list
            convo.send_message('請將{}和{}{}組成一般人會告知氣象的方式'.format(country[country_index], weather[weather_index], word))
            msg = convo.last.text
            return msg
        else:
            msg = response.statues_code
            return msg

    def APIS(country_index):
        base_url = 'https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-D0047-003?Authorization=CWA-0C63261F-E152-485F-B43E-F17EE1CADF6B'
        params = {
            'Authorization': '' # 中央氣象局 API KEY
        }
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json() # 回傳的json格式

            word0 = data['records']['locations'][0]['location'][country_index]['weatherElement'][0]['time'][2]['elementValue'][0]['value']
            word1 = data['records']['locations'][0]['location'][country_index]['weatherElement'][5]['time'][2]['elementValue'][0]['value']
            context['message0'] = word0
            context['message1'] = word1

            convo = model.start_chat(history=[]) # 設定聊天紀錄為空list
            convo.send_message('請用降雨機率={}和體感溫度={}去判斷是否攜帶雨具'.format(word0, word1))
            msg = convo.last.text
            return msg
        else:
            msg = response.statues_code
            return msg

    result = [] # 切割訊息
    for i in range(0, len(user_text)):
        result.append(user_text[i:i+2])

    for item in result:
        if item == '選項':
            break

        if item == '結束':
            context.clear() # 清空上下文
            convo = model.start_chat(history=[])
            convo.send_message('答謝使用者使用*宜蘭地區天氣之ChatBot*，不要回答其它東西。')
            msg = convo.last.text
            break

        if item in country:
            if context.get('message'):
                country_index = country.index(item) # country list 索引位置
                context['country'] = country[country_index]
                context['country_index'] = country_index
                weather_index = context['weather_index']
                weather[weather_index] = context['weather']
                msg = API(country_index, weather_index)
                break
            else:
                country_index = country.index(item) # country list 索引位置
                context['country'] = country[country_index]
                context['country_index'] = country_index
                convo = model.start_chat(history=[])
                convo.send_message('詢問使用者要{}的什麼資訊就好了，不要回答其它東西。'.format(country[country_index]))
                msg = convo.last.text
                break

        if context.get('country') is not None:
            if item in weather:
                weather_index = weather.index(item) # weather list 索引位置
                context['weather'] = weather[weather_index]
                context['weather_index'] = weather_index
                country_index = context.get('country_index')
                msg = API(country_index, weather_index)
                break

        if context.get('message') is not None:
            if item == '雨具':
                country_index = context['country_index']
                msg = APIS(country_index)
                break

    else:
        convo = model.start_chat(history=[])
        mistake = ['您輸入的地區不在本系統服務範圍內，請重新輸入，謝謝。',
                   '不好意思，我不懂您的意思，請耐心等候一下後臺人員上線協助您。']
        convo.send_message('請用以上{}模板隨機回答我，只回答我內容就好了'.format(mistake))
        msg = convo.last.text

    dict_user_context[user_id] = context # 更新上下文

    try:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))
    except Exception as e:
        print(f'Line Bot API 回覆訊息時發生錯誤: {e}')

if __name__ == '__main__':
    app.run() # 程式啟動


