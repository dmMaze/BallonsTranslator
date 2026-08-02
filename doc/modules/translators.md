# Ballon Translator: Translation Modules

*   Available translators: Google, DeepL, ChatGPT, Sugoi, Caiyun, Baidu, Papago, and Yandex.

[**Table of Contents**](#table-of-contents)
- [Ballon Translator: Translation Modules](#ballon-translator-translation-modules)
  - [LLM (Large Language Models)](#llm-large-language-models)
      - [ChatGPT](#chatgpt)
      - [ChatGPT (Experimental)](#chatgpt-experimental)
      - [Text Generation WebUI (TGW)](#text-generation-webui-tgw)
      - [Sakura](#sakura)
    - [LLM (General Module)](#llm-general-module)
  - [Other Translators](#other-translators)
    - [Paid Translators](#paid-translators)
      - [Baidu](#baidu)
      - [Caiyun](#caiyun)
      - [DeepL (Official API)](#deepl-official-api)
      - [Youdao API](#youdao-api)
      - [Yandex (Official API)](#yandex-official-api)
    - [Free Translators](#free-translators)
      - [DeepL Free](#deepl-free)
      - [DeepLX API](#deeplx-api)
      - [EzTrans](#eztrans)
      - [Google](#google)
      - [M2M100 (Facebook)](#m2m100-facebook)
      - [Papago](#papago)
      - [Sugoi](#sugoi)
      - [Translators](#translators)
      - [Yandex Free](#yandex-free)
    - [Additional Resources](#additional-resources)
  - [Acknowledgments](#acknowledgments)
  - [Contributing to the Project](#contributing-to-the-project)

---

## LLM (Large Language Models)

*Includes ChatGPT, Google Gemini, Text Generation WebUI, Sakura, and others.*

#### ChatGPT

For detailed setup instructions and using other OpenAI-compatible APIs, please refer to this [Discussion(We'll write soon.)](link-to-discussion-about-chatgpt-setup-here). *(Please replace 'link-to-discussion-about-chatgpt-setup-here' with the actual link to a relevant discussion about ChatGPT setup and alternative APIs)*

**Settings Fields:**

*   **api key:** API Key for accessing the OpenAI API. You need to obtain an API key from the OpenAI platform ([https://platform.openai.com/playground](https://platform.openai.com/playground)).
*   **model:** Model selection. Choose the desired OpenAI model from the dropdown list. Available options include: `gpt-4o`, `gpt-4-turbo`, `gpt3`, `gpt35-turbo`, `gpt4`. `gpt-4o` is recommended for the best performance.
*   **3rd party api url:** 3rd party API URL (Endpoint).  If you are using a third-party OpenAI-compatible API, enter its URL here. Leave blank to use the official OpenAI API endpoint (`https://api.openai.com/v1`).
*   **override model:** Override Model.  Optionally, specify a model name here to override the selected model. This is useful for testing specific models or using models not listed in the dropdown.
*   **max tokens:** Maximum tokens.  Sets the maximum number of tokens for the response from the API.
*   **temperature:** Temperature. Controls the randomness of the output. Higher values (e.g., 0.7) make the output more random and creative, while lower values (e.g., 0.2) make it more focused and deterministic.
*   **top p:** Top P.  Another way to control the randomness of the output, similar to temperature.

#### ChatGPT (Experimental)

For detailed setup instructions and using other OpenAI-compatible APIs, please refer to this [Discussion(We'll write soon.)](link-to-discussion-about-chatgpt-exp-setup-here). *(Please replace 'link-to-discussion-about-chatgpt-exp-setup-here' with the actual link to a relevant discussion about ChatGPT (Experimental) setup and alternative APIs)*

*   This is another version of the OpenAI-compatible translator. It may require more tokens to produce results, but it could be more accurate and reliable.
*   Two versions of OpenAI API-compatible translators are supported, working with official or third-party LLM providers, requiring configuration in the settings panel:
    *   The non-suffix version (ChatGPT) consumes fewer tokens but has slightly weaker sentence splitting stability, which may cause issues with long text translations.
    *   The 'exp' suffix version (ChatGPT (Experimental)) uses more tokens but has better stability and includes "jailbreaking" in the Prompt, making it suitable for long text translations.

**Settings Fields:**

*   **api key:** API Key for accessing the OpenAI API. You need to obtain an API key from the OpenAI platform ([https://platform.openai.com/playground](https://platform.openai.com/playground)).
*   **model:** Model selection. Choose the desired OpenAI model from the dropdown list. Available options include: `gpt-4o`, `gpt-4-turbo`, `gpt-4o-mini`. `gpt-4o` is recommended for the best performance.
*   **3rd party api url:** 3rd party API URL (Endpoint).  If you are using a third-party OpenAI-compatible API, enter its URL here. Leave blank to use the official OpenAI API endpoint (`https://api.openai.com/v1`).
*   **override model:** Override Model.  Optionally, specify a model name here to override the selected model. This is useful for testing specific models or using models not listed in the dropdown.
*   **max tokens:** Maximum tokens.  Sets the maximum number of tokens for the response from the API.
*   **temperature:** Temperature. Controls the randomness of the output. Higher values (e.g., 0.7) make the output more random and creative, while lower values (e.g., 0.2) make it more focused and deterministic.
*   **top p:** Top P.  Another way to control the randomness of the output, similar to temperature.


#### Text Generation WebUI (TGW)

This module is recommended only for users who can easily set up their own [TGW](https://github.com/oobabooga/text-generation-webui) server. If you cannot, please use other translation methods.

#### Sakura

**Sakura-13B-Galgame: [GitHub Repository](https://github.com/SakuraLLM/Sakura-13B-Galgame)**

*   When running locally on a single device and encountering crashes due to VRAM OOM (Out Of Memory), it is recommended to enable ```low vram mode``` in the settings panel (enabled by default).

**Settings Fields:**

*   **low vram mode:** Low VRAM mode. If checked (true), the Sakura module will use less video memory.
*   **api baseurl:** API base URL to access the Sakura server. Default is `http://127.0.0.1:8080/v1/`. Change this if your Sakura server is running on a different address or port.
*   **dict path:** Dictionary path for Sakura. Improves translation quality.
*   **version:** Model version for Sakura. Choose from the list of available versions.
*   **retry attempts:** Retry attempts in case of an error.
*   **timeout:** Timeout for server response in milliseconds.
*   **max tokens:** Maximum tokens in the response. Limits the length of the translation.
*   **repeat detect threshold:** Repeat detection threshold. Decoding algorithm parameter.
*   **force apply dict:** Force apply dictionary.
*   **do enlarge small kana:** Enlarge small kana. Option for the Japanese language.

### LLM (General Module)

Temporary module. Essentially carries the functionality of ChatGPT, but with a preset for Google. Created as a temporary solution until a clear and working guide is written on how to use it within ChatGPT and ChatGPT (Experimental), and until proxy support is added.

---

## Other Translators

*   **The following translators may require a token or API key to function:** [Caiyun](https://dashboard.caiyunapp.com/), [ChatGPT](https://platform.openai.com/playground), [Yandex](https://yandex.com/dev/translate/), [Baidu](http://developers.baidu.com/), and [DeepL](https://www.deepl.com/docs-api/api-access).

---

### Paid Translators

#### Baidu

**Settings Fields:**

*   **token:** In this case, likely refers to the Secret Key for accessing the Baidu Translate API. **How to obtain (along with appId):**
    1.  Go to [Baidu AI开放平台 (Baidu AI Open Platform)](https://ai.baidu.com/tech/translate/translation_http). (Website in Chinese).
    2.  Register or log in to your Baidu account.
    3.  Navigate to the "产品服务" (Products and Services) section and find "翻译开放平台" (Open Translation Platform).
    4.  Create a new translation application or activate the translation service.
    5.  In your application settings, you will find **`AppID`** and **`API Key` (or `Secret Key`)**. Enter the `Secret Key` value in the `token` field and the `AppID` in the `appId` field. *The Baidu AI website is also in Chinese and may require a page translator.*
*   **appId:** Application ID (AppID), obtained along with the Secret Key when registering for the Baidu Translate API. Required for authenticating requests to the Baidu API.
*   **delay:** Delay in seconds between requests to the translation service. Used to control the frequency of requests.

#### Caiyun

**Settings Fields:**

*   **token:** Access token for the Caiyun API. **How to obtain:**
    1.  Go to the [Caiyunfanyi](https://fanyi.caiyunapp.com/) website. (Website in Chinese)
    2.  Register or log in to your account.
    3.  Find the developer or API section (usually in profile settings or at the bottom of the page).
    4.  Create an application or get API access to obtain a **token**. *Please note that the website is in Chinese, and the token acquisition process may require using an online translator.*
*   **delay:** Delay in seconds between requests to the translation service. Used to control the frequency of requests.

#### DeepL (Official API)

**Settings Fields:**

*   **api_key:** API Key to access the DeepL API.  **How to obtain:**
    1.  Go to the [DeepL for developers](https://www.deepl.com/pro-api) website.
    2.  Register and subscribe to the DeepL API (paid service).
    3.  After registration, you will receive an API key, which you need to enter in this field.
*   **formality:** Allows controlling the formality level of the translation. For example, the value `less` will make the translation less formal and more conversational. Available options depend on the DeepL API.
*   **context:** Field to add context to the text being translated. Providing context can improve translation quality, especially for ambiguous phrases. Enter additional information here to help DeepL understand the meaning of the text.
*   **preserve_formatting:** Option to preserve the formatting of the original text during translation. If `enabled` is selected, DeepL will try to preserve formatting such as bold, italics, etc.
*   **delay:** Delay in seconds between requests to the translation service. Used to control the frequency of requests to avoid blocking or exceeding API limits. A value of `0.0` (no delay) is usually sufficient, or use a small value if you encounter issues.

#### Youdao API

**Settings Fields:**

*   **api_key:** API Key for accessing the Youdao Translate API. **How to obtain:**
    1.  Go to [Youdao智云 (Youdao Zhiyun - Youdao Intelligent Cloud)](https://ai.youdao.com/). (Website in Chinese).
    2.  Register or log in to your Youdao account.
    3.  Find the "自然语言翻译" (Natural Language Translation) or "机器翻译" (Machine Translation) section in the product list.
    4.  Create a new "应用" (application) for translation.
    5.  In the settings of the created application, you will get **`应用ID (App ID)`** and **`应用密钥 (App Secret)`**. Use the `应用ID` value as `api_key` and the `应用密钥` as `app_secret`.  *The Youdao Zhiyun website is in Chinese and may require a page translator.*
*   **app_secret:** App Secret, obtained along with the API Key when registering for the Youdao Translate API. Used for authenticating requests.

#### Yandex (Official API)

**Settings Fields:**

*   **api_key:** API key to access the Yandex Translate API.  **How to obtain:**
    1.  Go to [Yandex Cloud](https://cloud.yandex.ru/en/).
    2.  Register or log in to your Yandex Cloud account.
    3.  Create a "Service Account" and obtain an "API key" for this account. Ensure the service account has permissions to use the Yandex Translate API.
    4.  Copy the obtained **API key** and paste it into this field. *Yandex Cloud is a paid service but may offer a free trial period.*
*   **delay:** Delay in seconds between requests to the translation service. Used to control the frequency of requests.

* Attention! If you previously used the [v1.5](https://translate.yandex.com/developers) version of the API, it has been closed and moved to Yandex Cloud. Therefore, if you have an old key, it will work if there are funds remaining in your balance. Once they run out, your key will become invalid.

---

### Free Translators

#### DeepL Free

**Settings Fields:**

*   **delay:** Delay in seconds between requests. It's useful to set a value greater than `0`, for example, `3` seconds, to avoid overloading the free service and prevent temporary blocking.
*   **proxy:** Field to enter the proxy server address if you want to use a proxy to access DeepL Free. The input format depends on the proxy type (HTTP, SOCKS, etc.). Leave the field blank if no proxy is needed.

#### DeepLX API

**DeepLX: Repositories:** [Vercel ver.](https://github.com/bropines/Deeplx-vercel) or [Self-host ver.](https://github.com/OwO-Network/DeepLX)

**Settings Fields:**

*   **api_url:** API URL for DeepLX. Instructions on how to install and run DeepLX can be found in the project repositories listed at the beginning of this section.

*   **delay:** Delay in seconds between requests to the translation service. Used to control the frequency of requests.

#### EzTrans

**Installation instructions for ezTrans XP can be found, for example, in this blog:** [Naver Blog - WaltherP38 (Korean)](https://blog.naver.com/waltherp38/221062272423)

**Settings Fields:**

*   **path_dat:** Path to the Dat folder of your installed ezTrans XP program. This is usually the folder containing ezTrans data files needed for translation. **Example path:** `C:\Program Files (x86)\ChangShinSoft\ezTrans XP\Dat`. **Important:** Make sure the path leads to the correct `Dat` folder of your ezTrans XP installation.
*   **path_j2k:** Path to the J2KEngine.dll file from your ezTrans XP installation. This file is the main library for Japanese to Korean (J2K) translation. **Example path:** `C:\Program Files (x86)\ChangShinSoft\ezTrans XP\J2KEngine.dll`.
*   **path_k2j (Optional):** Path to the ehnd-kor.dll file from your ezTrans XP installation. This file is used for Korean to Japanese (K2J) translation. **Optional field:** required only if you need Korean to Japanese translation. **Example path:** `C:\Program Files (x86)\ChangShinSoft\ezTrans XP\ehnd-kor.dll`.

#### Google

**Attention:** Google Translate service has ceased operations in China.  If you are in China, you may need to use a VPN or proxy server to access Google Translate.

*   **delay:** Delay in seconds between requests to the translation service. Used to control the frequency of requests. A value of `0.0` is usually sufficient.

#### M2M100 (Facebook)

**Settings Fields:**

*   **device:** Device for running the M2M100 model. Choose `CPU` to use the processor or `CUDA` to use an NVIDIA graphics card (if supported and configured). Using `CUDA` usually provides faster performance.

**To use the M2M100 module, you need to download the model:**

1.  **Download the M2M100 1.2B model in CTranslate2 format.** Pre-converted models can be found [this.](https://huggingface.co/facebook/m2m100_1.2B)
2.  **Place the downloaded model in the `data/models/m2m100-1.2B-ctranslate2` folder** (or the path specified as `CT_MODEL_PATH` in the `m2m100` module code). Make sure that the CTranslate2 model files (e.g., `model.bin`, `config.json`, `vocabulary.txt`) are in this folder.

#### Papago

**Settings Fields:**

*   **delay:** Delay in seconds between requests to the translation service. Used to control the frequency of requests. A value of `0.0` is usually sufficient.

#### Sugoi

**Sugoi Translator: Japanese-English translation completely offline.**

*   **Download the offline model:** [Google Drive](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm) and move the "sugoi_translator" folder to `BallonsTranslator/ballontranslator/data/models`.

**Settings Fields:**

*   **device:** Device for running the Sugoi Translator model. Choose `CPU` to use the processor or `CUDA` to use an NVIDIA graphics card (if supported and configured). Using `CUDA` usually provides faster performance.

#### Translators

**Translators library:** [GitHub Repository](https://github.com/UlionTse/translators).
*   Supports access to some translation services without API keys. You can find out about supported services [here](https://github.com/UlionTse/translators#supported-translation-services).

**Settings Fields:**

*   **translator provider:** Dropdown menu to select the translation provider from the `Translators Pack` library ([https://pypi.org/project/translators/](https://pypi.org/project/translators/)). `Translators Pack` integrates many different translation services (Bing, Yandex, Google, and others). Select the desired service from the list (e.g., `bing`, `google_v2`, `yandex`).
*   **sleep_seconds:** Delay in seconds between requests. Similar to the `delay` field in other modules, used to control the frequency of requests.

#### Yandex Free

**Settings Fields:**

*   **endpoint:** Endpoint URL to access Yandex Free Translate. Currently, only a self-hosted solution is available. You can find more details about the installation [here](https://github.com/FOSWLY/translate-backend)
*   **delay:** Delay in seconds between requests.

---

### Additional Resources

*   Other good offline English translators can be found or suggested in this [discussion thread](https://github.com/dmMaze/BallonsTranslator/discussions/515).

---

## Acknowledgments

*   DeepL and Sugoi translators (and their CT2 Translation conversion) are developed thanks to [Snowad14](https://github.com/Snowad14).

---

## Contributing to the Project

*   To add a new translator, please refer to the [instructions](doc/how_to_add_new_translator.md). It's as simple as subclassing a BaseClass and implementing two interfaces. You are welcome to contribute to the project.
