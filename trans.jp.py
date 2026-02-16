#!python3.9
from httpx import Client, AsyncClient
import re
from openai import OpenAI
from collections import deque
from google.genai import types
from google import genai
import simplejson as json
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
proxy = "http://localhost:7890"


class SERVICE():
    @staticmethod
    def deepseek(*args, **kwargs):
        return ai_openai(*args, **kwargs)

    @staticmethod
    def google(*args, **kwargs):
        return ai_google(*args, **kwargs)


ai_service = SERVICE.deepseek
sys_message = '''你是中国本土的专业日语翻译家，你能够精确、忠实、流畅地翻译日语为生活化、通俗易懂的简体中文。现在有一些待翻译的漫画文本（存在OCR识别错误的可能），请结合语境给出一句完整的翻译。直接回答翻译语句，不做任何解释，不添加引号。对于无法翻译的句子，直接返回原文。
'''
terms_begin = '''以下是术语介绍：'''
terms_body = ''
terms_end = ''''''
if os.path.exists("terms.txt"):
    with open("terms.txt", 'r', encoding='utf-8') as f:
        terms_body = f.read()
if terms_body:
    sys_message += terms_begin
    sys_message += terms_body
    sys_message += terms_end
first_context = ''
id = 0
cache = []
cache_name = ""


def waitfor():
    time.sleep(10)


def read_cache():
    global cache
    if os.path.exists(f"cache/{cache_name}.json"):
        with open(f"cache/{cache_name}.json", encoding='utf-8') as f:
            cache = json.load(f)


def read_from_previous(data):
    if not data:
        return
    lines = data.split('\n\n')
    validlines = [i.strip() for i in lines]
    v = []
    for i in validlines:
        if i.startswith('#'):
            continue
        dot = i.find('.')
        if dot > 0:
            dot_prev = i[:dot].strip()
            dot_after = i[dot + 1:].strip()
            try:
                l = int(dot_prev)
                if not dot_after:
                    break
                v.append(dot_after)
            except:
                break
    cache.clear()
    cache.extend(v)


def writecache():
    if not os.path.exists('cache'):
        os.makedirs('cache', exist_ok=True)
    with open(f"./cache/{cache_name}.json", "w", encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)


def sanitize(text):
    if text.startswith('```'):
        lines = text.split('\n')
        begin = 1
        end = None
        if lines[-1].startswith('```'):
            end = -1
        text = '\n'.join(lines[begin:end])
    text = text.replace('?', '？')
    text = text.replace('!', '！')
    text = text.replace('？！', '?!⁈')
    text = text.replace('！？', '?!⁈')
    text = text.replace('...', '…')
    text = text.replace('．．．', '…')
    return text


def pre_clean(text):
    text = text.strip().replace("\n", "").lower()
    text = text.replace('...', '…')
    text = text.replace('．．．', '…')
    return text


oldClient = Client.__init__
oldAsyncClient = AsyncClient.__init__


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
        ret = ai_service(text, context)
    else:
        ret = ""
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
    if first_context:
        context.append(first_context)
    error = False
    lines = markdown_content.split('\n\n')
    prog = tqdm.tqdm(total=len(lines))
    for line in lines:
        prog.update(1)
        match = pattern.match(line.replace('\n','$'))
        if match:
            prefix = match.group(1)
            text = match.group(2).replace('$','\n')
            if not error:
                for i in range(5):
                    translated_text = translate_text(
                        text, context)
                    if not translated_text:
                        waitfor()
                    else:
                        break
                if not translated_text:
                    translated_text = ""
                    error = True
            else:
                translated_text = ""
            prog.set_description(f"{text:10s}->{translated_text:10s}")
            translated_line = f"{prefix}{translated_text}"
            translated_content.append(translated_line)
            context.append(f"'{text}'='{translated_text}'")  # 更新上下文
            writecache()
        else:
            translated_content.append(line)

    writecache()
    return '\n'.join(translated_content)


def string_hash(s):
    """
    简单字符串哈希函数（DJB2 算法变种）
    输入：字符串 s
    输出：一个非负整数哈希值（32位范围）
    特点：确定性、简单、对相同输入始终输出相同结果
    """
    if not isinstance(s, str):
        raise TypeError("输入必须是字符串")

    hash_value = 5381  # 初始值
    for char in s:
        # hash * 33 + char
        hash_value = (hash_value * 33 + ord(char)) & 0xFFFFFFFF  # 32位无符号整数
    return hash_value


def main():
    import sys
    input_file = sys.argv[1]
    previous_file = sys.argv[2] if len(sys.argv) > 2 else None
    global cache_name
    cache_name = str(string_hash(input_file))
    read_cache()
    with open(input_file, 'r', encoding='utf-8') as file:
        markdown_content = file.read()
    if previous_file and os.path.exists(previous_file):
        with open(previous_file, 'r', encoding='utf-8') as file:
            c = file.read()
            read_from_previous(c)

    translated_content = translate_markdown(markdown_content)

    with open('translation.md', 'w', encoding='utf-8') as file:
        file.write(translated_content)

    print(
        f"**Translated content has been saved to {previous_file or 'translation.md'}")


if __name__ == "__main__":
    main()
