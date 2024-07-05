[简体中文](../doc/加别的翻译器.md) | [English](../doc/how_to_add_new_translator.md) | pt-BR | [Русский](../doc/add_translator_ru.md)

---

## Como Adicionar um Novo Tradutor ao BallonsTranslator

Se você sabe como utilizar a API do tradutor ou o modelo de tradução desejado em Python, siga os passos abaixo para integrá-lo ao BallonsTranslator.

### Implementação da Classe do Tradutor

Se você sabe como chamar a API do tradutor alvo ou modelo de tradução em Python, implemente uma classe em `ballontranslator/dl/translators.__init__.py` da seguinte forma para usá-la no aplicativo. O exemplo a seguir, DummyTranslator, está comentado em `ballontranslator/dl/translator/__init__.py` e pode ser descomentado para testar no programa.

1. **Crie uma nova classe em `ballontranslator/dl/translators/__init__.py`:**

```python
# "dummy translator" é o nome exibido no aplicativo
@register_translator('dummy translator')
class DummyTranslator(BaseTranslator):

    concate_text = True

    # parâmetros exibidos no painel de configuração.
    # chaves são nomes dos parâmetros, se o tipo do valor for str, será um editor de texto (chave obrigatória)
    # se o tipo do valor for dict, você precisa especificar o 'type' do parâmetro,
    # o seguinte 'device' é um seletor, as opções são cpu e cuda, o padrão é cpu
    params: Dict = {
        'api_key': '', 
        'device': {
            'type': 'selector',
            'options': ['cpu', 'cuda'],
            'value': 'cpu'
        }
    }

    def _setup_translator(self):
        '''
        faça a configuração aqui.
        as chaves de lang_map são aquelas opções de idiomas exibidas no aplicativo,
        atribua as chaves de idioma correspondentes aceitas pela API aos idiomas suportados.
        Apenas os idiomas suportados pelo tradutor são atribuídos aqui, este tradutor suporta apenas japonês e inglês.
        Para uma lista completa de idiomas, veja LANGMAP_GLOBAL em translator.__init__
        '''
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'
        
    def _translate(self, src_list: List[str]) -> List[str]:
        '''
        faça a tradução aqui.
        Este tradutor não faz nada além de retornar o texto original.
        '''
        source = self.lang_map[self.lang_source]
        target = self.lang_map[self.lang_target]
        
        translation = text
        return translation

    def updateParam(self, param_key: str, param_content):
        '''
        necessário apenas se algum estado precisar ser atualizado imediatamente após o usuário alterar os parâmetros do tradutor,
        por exemplo, se este tradutor for um modelo pytorch, você pode convertê-lo para cpu/gpu aqui.
        '''
        super().updateParam(param_key, param_content)
        if (param_key == 'device'):
            # obtenha o estado atual dos parâmetros
            # self.model.to(self.params['device']['value'])
            pass

    @property
    def supported_tgt_list(self) -> List[str]:
        '''
        necessário apenas se o suporte a idiomas do tradutor for assimétrico,
        por exemplo, este tradutor suporta apenas inglês -> japonês, não japonês -> inglês.
        '''
        return ['English']

    @property
    def supported_src_list(self) -> List[str]:
        '''
        necessário apenas se o suporte a idiomas do tradutor for assimétrico.
        '''
        return ['日本語']
```

- Decore a classe com `@register_translator` e forneça o nome do tradutor que será exibido na interface. No exemplo, o nome passado para o decorador é `'dummy translator'`, tome cuidado para não renomeá-lo com um tradutor existente.
- A classe deve herdar de `BaseTranslator`.

2. **Defina o atributo `concate_text`:**

```python
@register_translator('dummy translator')
class DummyTranslator(BaseTranslator):  
    concate_text = True  # Se o tradutor aceitar apenas strings concatenadas
    concate_text = False # Se o tradutor aceitar lista de strings ou modelo offline
```

- Indique se o tradutor aceita apenas texto concatenado (várias frases em uma única string) ou uma lista de strings.
- Se for um modelo offline ou uma API que aceita listas de strings, defina como `False`.

3. **Defina os parâmetros (opcional):**

```python
params: Dict = {
    'api_key': '',  # Editor de texto para a chave da API
    'device': {    # Seletor para CPU ou CUDA
        'type': 'selector',
        'options': ['cpu', 'cuda'],
        'value': 'cpu'
    }
}
```

- Crie um dicionário `params` se o tradutor precisar de parâmetros configuráveis pelo usuário. Se não, deixe em branco ou atribua `None`.
- As chaves do dicionário são os nomes dos parâmetros exibidos na interface. Se o tipo de valor correspondente for str, será exibido no aplicativo como um editor de texto, no exemplo acima, o api_key será um editor de texto com um valor padrão vazio.
- Os valores podem ser strings (para editores de texto) ou dicionários (neste caso deve ser descrito por 'type', como exemplo acima. O parâmetro 'device' será exibido como um seletor no aplicativo, opções válidas são 'cpu' e 'cuda).

<p align="center">
<img src="./src/new_translator.png">
</p>
<p align="center">
params exibidos no painel de configuração do aplicativo.
</p>  

4. **Implemente o método `_setup_translator`:**

```python
def _setup_translator(self):
    '''
    faça a configuração aqui.
    as chaves de lang_map são aquelas opções de idiomas exibidas no aplicativo,
    atribua as chaves de idioma correspondentes aceitas pela API aos idiomas suportados.
    Apenas os idiomas suportados pelo tradutor são atribuídos aqui, este tradutor suporta apenas japonês e inglês.
    Para uma lista completa de idiomas, veja LANGMAP_GLOBAL em translator.__init__
    '''
    self.lang_map['日本語'] = 'ja'
    self.lang_map['English'] = 'en'
```

- Realize a configuração do tradutor (inicialização de modelos, autenticação na API, etc.).
- Mapeie os idiomas exibidos no app para os códigos de idioma aceitos pela API.
- Consulte `LANGMAP_GLOBAL` em `translator.__init__` para a lista completa de idiomas.

5. **Implemente o método `_translate`:**

```python
def _translate(self, src_list: List[str]) -> List[str]:
    '''
    faça a tradução aqui.
    Este tradutor não faz nada além de retornar o texto original.
    '''
    source = self.lang_map[self.lang_source]
    target = self.lang_map[self.lang_target]
    
    translation = text
    return translation
```

- Recebe uma lista de strings (`src_list`) a serem traduzidas.
- Se `concate_text` for `True`, as strings serão concatenadas antes de serem passadas para o tradutor.
- Realiza a tradução utilizando a API ou modelo.
- Retorna uma lista com as strings traduzidas.

### Métodos Opcionais

- **`updateParam(self, param_key: str, param_content)`:**
    - Implemente se precisar atualizar o estado do tradutor imediatamente após o usuário alterar os parâmetros.

- **`supported_tgt_list(self) -> List[str]`:**
    - Implemente se o suporte de idiomas do tradutor for assimétrico (por exemplo, só traduz de inglês para japonês).

- **`supported_src_list(self) -> List[str]`:**
    - Implemente se o suporte de idiomas do tradutor for assimétrico.

### Testes

Após implementar o tradutor, teste-o seguindo o exemplo em `tests/test_translators.py`.