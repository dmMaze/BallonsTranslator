[简体中文](../doc/加别的翻译器.md) | [English](../doc/how_to_add_new_translator.md) | [pt-BR](../doc/Como_add_um_novo_tradutor.md) | Русский

---

Если у вас есть базовые знания программирования на python, вы будете знать, как использовать python для вызова необходимого api переводчика или модели перевода, Напишите класс в dl/translators.__init__.py следующим образом, чтобы использовать его непосредственно в программе.      
Следующий пример DummyTranslator закомментирован в dl/translator/__init__.py, и его можно не комментировать, чтобы увидеть результат в программе.  

``` python
@register_translator('dummy translator')
class DummyTranslator(BaseTranslator):
    concate_text = True

    # parameters showed in the config panel. 
    # keys are parameter names, if value type is str, it will be a text editor(required key)
    # if value type is dict, you need to spicify the 'type' of the parameter, 
    # following 'device' is a selector, options a cpu and cuda, default is cpu
    params: Dict = {
        'required_key': '', 
        'device': {
            'type': 'selector',
            'options': ['cpu', 'cuda'],
            'value': 'cpu'
        }
    }

    def _setup_translator(self):
        '''
        do the setup here.  
        keys of lang_map are those languages options showed in the app, 
        assign corresponding language keys accepted by API to supported languages.  
        This translator only supports Chinese, Japanese, and English.
        '''
        self.lang_map['简体中文'] = 'zh'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'  
        
    def _translate(self, src_list: List[str]) -> List[str]:
        '''
        do the translation here.  
        This translator do nothing but return the original text.
        '''
        source = self.lang_map[self.lang_source]
        target = self.lang_map[self.lang_target]
        return 'translate ' + text + f'from {source} to target'

    def updateParam(self, param_key: str, param_content):
        '''
        required only if some state need to be updated immediately after user change the translator params,
        for example, if this translator is a pytorch model, you can convert it to cpu/gpu here.
        '''
        super().updateParam(param_key, param_content)
        if param_key == 'device':
            # self.model.to(self.params['device']['value'])
            pass
```

Во-первых, переводчик должен быть декорирован с помощью register_translator и наследоваться от базового класса BaseTranslator, параметр 'dummy translator' внутри декоратора - это имя переводчика, которое будет отображаться в интерфейсе, будьте осторожны, чтобы не дублировать имя существующего переводчика.  
Сохраните concate_text на потом.  
``` python
@register_translator('dummy translator')
class DummyTranslator(BaseTranslator):  
    concate_text = True
```

Если новый переводчик требует настраиваемых пользователем параметров, создайте словарь params, как показано ниже, в противном случае оставьте его в покое или присвойте значение None.  
Ключом в params является соответствующее имя параметра, отображаемое в интерфейсе, значение может быть str, api_key ниже будет текстовый редактор с пустым значением по умолчанию в интерфейсе.  
Значение параметра также может быть словарем, но должно быть указано как тип 'type', который будет показан как селектор в интерфейсе, следующее устройство является селектором, либо cpu, либо cuda, по умолчанию cpu.  

``` python
    params: Dict = {
        'api_key': '', 
        'device': {
            'type': 'selector',
            'options': ['cpu', 'cuda'],
            'value': 'cpu'
        }
    }
```  

<p align = "center">
<img src="./src/new_translator.png">
</p>
<p align = "center">
Результат вышеуказанного словаря параметров в панели настроек интерфейса
</p>  

Переводчик должен реализовать _setup_translator, который выполняет инициализацию здесь. Ключ словаря lang_map - это языковая опция, отображаемая в интерфейсе, и присваивается языковому ключевому слову, принятому API, например, 'zh' для упрощенного китайского языка Google Translate. Здесь указаны только языки, поддерживаемые переводчиком, полный список языков см. в LANGMAP_GLOBAL в translator.__init__. 

``` python
    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'  
```

Переводчику также необходимо реализовать _translate, где lang_source и lang_target - это языки, выбранные в интерфейсе на данном этапе, а соответствующие ключевые слова api можно получить из предыдущей lang_map, чтобы сшить вместе параметры api и отправить запрос.  
Обратите внимание, что если предыдущий параметр concate_text имеет значение False, то переданный сюда текст будет представлять собой таблицу строк, соответствующую оригинальному содержимому каждого текстового блока на текущей странице перевода, и переведенный результат также должен представлять собой таблицу переведенного текста один к одному. Если установлено значение True, входящий текст будет представлять собой обычную строку всех текстовых блоков, а выходной текст должен быть переведенной строкой.  
Слишком медленно посылать запрос для каждого текстового блока, поэтому вся страница сшивается и переводится. concate_text настроен на автоматическое сшивание/разделение, и по умолчанию сшивает весь блок вместе с '\n###\n' в качестве разделителя, а затем разделяет переведенный текст обратно в текстовую таблицу с помощью '####'. Это работает для большинства проверенных мной переводчиков, но некоторые из них избавляются от #, поэтому вы можете отключить перевод concate_text блок за блоком или реализовать свой собственный метод сшивания.  
Некоторые апи, такие как Caiyun, поддерживают прямые текстовые таблицы в сообщениях, поэтому можно установить значение False.  
``` python
    def _translate(self, src_list: List[str]) -> List[str]:
        api_key = self.params['api_key']  # 如此获取用户修改过的api_key
        source = self.lang_map[self.lang_source]
        target = self.lang_map[self.lang_target]
        return text
```
Фиктивный переводчик не делает ничего, кроме возвращения оригинального текста.  
После внедрения переводчика рекомендуется написать собственный тест переводчика для проверки правильности вывода, следуя примеру в tests/test_translators.py. Как только тест пройден, вы можете использовать его в своем приложении.   

Наконец, updateParam выше будет вызываться автоматически, когда пользователь изменит параметр, по умолчанию он будет изменять только значение в params, такое как api_key выше. Обычно это можно игнорировать, но если вам нужно изменить состояние транслятора, например, если это локальная модель трансляции, которая может переключаться между cuda и cpu, вы можете сделать это здесь.  