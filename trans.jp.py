#!python3.9
from httpx import Client, AsyncClient
import re
from openai import OpenAI
from collections import deque
from google.genai import types
from google import genai
from pydantic import BaseModel
import tqdm
import sys
import time
import os
sys.path.append('e:/')
###############
from api_key import google as API_KEY
from api_key import deepseek as DEEPSEEK_API_KEY
################
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
ai_service="deepseek"
sys_message = '''你是中国本土的专业日语翻译家，你能够精确、忠实、流畅地翻译日语为生活化、通俗易懂的简体中文。现在有一些待翻译的漫画文本（存在OCR识别错误的可能），请结合语境给出一句完整的翻译。直接回答翻译语句，不做任何解释，不添加引号。对于无法翻译的句子，直接返回原文。
'''
'''
1. ナナチ：娜娜奇
2. リコ：莉可
3. レグ　：雷古
4. ヤタラマル：亚塔拉马努
5. テバステ：迪帕斯蒂'''


first = ''  # '''历史记录：'MAS PRAM!!'='普拉姆先生！！''''
id = 0
cache = []


def waitfor():
    time.sleep(10)


def read_cache():
    global cache
    if os.path.exists("./cache.txt"):
        with open("./cache.txt", encoding='utf-8') as f:
            cache = [line.strip() for line in f]


read_cache()


def writecache():
    with open("./cache.txt", "w", encoding='utf-8') as f:
        f.write('\n'.join(cache))


def sanitize(text):
    text = text.replace('?', '？')
    text = text.replace('!', '！')
    text = text.replace('？！', '?!')
    text = text.replace('...', '…')
    return text


def pre_clean(text):
    text = text.strip().replace("\n", "").lower()
    return text


oldClient = Client.__init__
oldAsyncClient = AsyncClient.__init__

proxy = "http://localhost:7890"


def newClient(self, *args, **kwargs):
    kwargs["proxy"] = proxy
    oldClient(self, *args, **kwargs)


def newAsyncClient(self, *args, **kwargs):
    kwargs["proxy"] = proxy
    oldAsyncClient(self, *args, **kwargs)


Client.__init__ = newClient
AsyncClient.__init__ = newAsyncClient

google_client = genai.Client(api_key=API_KEY)


class StructuredJSON(BaseModel):
    recipe_name: str
    ingredients: list[str]

def translate_text(text, context: deque):
    global id
    if len(cache) > id:
        id += 1
        c = cache[id - 1]
        if c:
            return c
    text = pre_clean(text)
    if text:
        ret = ai_google(text, context) if ai_service=="google" else ai_openai(text,context)
    else:
        ret=""
    if ret:
        cache.append(ret)
    return ret

def ai_google(text, context):
    global id
    messages = []
    for i in context:
        messages.append(f"上文 第{id}行：" + i)
    messages.append(f"当前文本：'{text}'")
    error = False
    try:
        response = google_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=sys_message,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0)  # Disables thinking
                # "response_mime_type": "application/json",
                # "response_schema": list[Recipe],
            ),
        )
    except:
        error = True
    if error:
        return None
    id += 1
    ret = response.text
    ret = sanitize(ret)
    return ret




def ai_openai(text, context):
    global id
    messages = [
        {"role": "system", "content": f"{sys_message}"},
    ]
    for i in context:
        messages.append({"role": "assistant", "content": i})
    messages.append({"role": "user", "content": f"当前文本：'{text}'"})
    id += 1
    response = client.chat.completions.create(
        temperature=1.3,
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    ret = response.choices[0].message.content
    ret = sanitize(ret)
    return ret


def translate_markdown(markdown_content):
    pattern = re.compile(r'(\d+\.\s+)(.*)')
    translated_content = []
    context = deque(maxlen=10)
    context.append(first)
    error = False
    lines=markdown_content.splitlines()
    prog=tqdm.tqdm(total=len(lines))
    for line in lines:
        prog.update(1)
        match = pattern.match(line)
        if match:
            prefix = match.group(1)
            text = match.group(2)
            if not error:
                for i in range(5):
                    translated_text = translate_text(text, context)
                    if not translated_text:
                        waitfor()
                    else:
                        break
                if not translated_text:
                    translated_text = ""
                    error = True
                    writecache()
            else:
                translated_text = ""
            prog.set_description(translated_text)
            translated_line = f"{prefix}{translated_text}"
            translated_content.append(translated_line)
            context.append(f"'{text}'='{translated_text}'")  # 更新上下文
        else:
            translated_content.append(line)

    return '\n'.join(translated_content)


def main():
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'translation.md'

    with open(input_file, 'r', encoding='utf-8') as file:
        markdown_content = file.read()

    translated_content = translate_markdown(markdown_content)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(translated_content)

    print(f"**Translated content has been saved to {output_file}")


if __name__ == "__main__":
    main()
