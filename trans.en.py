from collections import deque
from openai import OpenAI
import re
API_KEY = 'sk-69f7b66489564a90aa2698de67b33b7c'
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
prompt = '''你是中国本土的专业日语翻译家，你能够精确、忠实、流畅地翻译日语为生活化、通俗易懂的简体中文。现在有一些待翻译的漫画文本（存在OCR识别错误的可能），请结合语境给出一句完整的翻译。直接回答翻译语句，不做任何解释，不添加引号。对于无法翻译的句子，直接返回原文。
术语：
1. ナナチ：娜娜奇
2. リコ：莉可
3. レグ　：雷古
4. ヤタラマル：亚塔拉马努
5. テバステ：迪帕斯蒂
'''

id = 0


def sanitize(text):
    text = text.replace('?', '？')
    text = text.replace('!', '！')
    text = text.replace('？！', '?!')
    text = text.replace('...', '…')
    return text

def pre_clean(text):
    text = text.strip().replace("\n", "").lower()
    return text


def translate_text(text, context: deque):
    text=pre_clean(text)
    global id
    messages = [
        {"role": "system", "content": f"{prompt}"},
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
    print(ret)
    return ret


def translate_markdown(markdown_content):
    pattern = re.compile(r'(\d+\.\s+)(.*)')
    translated_content = []
    context = deque(maxlen=10)
    context.append("历史记录：'MAS PRAM!!'='普拉姆先生！！'")

    for line in markdown_content.splitlines():
        match = pattern.match(line)
        if match:
            prefix = match.group(1)
            text = match.group(2)
            translated_text = translate_text(text, context)
            translated_line = f"{prefix}{translated_text}"
            translated_content.append(translated_line)
            context.append(f"历史记录：'{text}'='{translated_text}'")  # 更新上下文
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
