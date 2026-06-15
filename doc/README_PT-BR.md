## BallonTranslator

[简体中文](/README.md) | [English](/README_EN.md) | [Русский](/doc/README_RU.md) | [日本語](/doc/README_JA.md) | [Español](/doc/README_ES.md) | [Français](/doc/README_FR.md) | [pt-BR](/doc/README_PT-BR.md) | [한국어](/doc/README_KO.md) | [Indonesia](/doc/README_ID.md) | [Tiếng Việt](/doc/README_VI.md)

BallonTranslator é mais uma ferramenta auxiliada por computador, alimentada por deep learning, para a tradução de quadrinhos/mangás.

<img src="https://github.com/user-attachments/assets/2140c402-dda2-47bc-9e7f-83ed41ce78af" div align=center>

<p align=center>
**Pré-Visualização**
</p>

## Recursos
* **Tradução totalmente automatizada:** 
  - Detecta, reconhece, remove e traduz textos automaticamente. O desempenho geral depende desses módulos.
  - A diagramação é baseada na estimativa de formatação do texto original.
  - Funciona bem com mangás e quadrinhos.
  - Diagramação aprimorada para mangás->inglês, inglês->chinês (baseado na extração de regiões de balões).
  
* **Edição de imagem:**
  - Permite editar máscaras e inpainting (similar à ferramenta Pincel de Recuperação para Manchas no Photoshop).
  - Adaptado para imagens com proporção de aspecto extrema, como webtoons.
  
* **Edição de texto:**
  - Suporta formatação de texto e [predefinições de estilo de texto](https://github.com/dmMaze/BallonsTranslator/pull/311). Textos traduzidos podem ser editados interativamente.
  - Permite localizar e substituir.
  - Permite exportar/importar para/de documentos do Word.

## Instalação

### No Windows

### No Windows

**Método A (Configuração automática do ambiente local em um clique, requer PowerShell)**:
O script instalará `BallonsTranslator` no diretório onde você o executar:
```powershell
irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex
```
Ou execute o seguinte comando no Prompt de Comando clássico (`cmd.exe`):
```cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex"
```

**Método B (Baixar pacote pré-configurado)**:
Baixe `Ballonstranslator_win_minium.zip` em [GitHub Releases](https://github.com/dmMaze/BallonsTranslator/releases), extraia-o e clique duas vezes em `launch_win.bat` para iniciar o aplicativo.

Esses métodos não oferecem suporte ao Windows 7; usuários do Windows 7 devem instalar o [Python 3.8](https://www.python.org/downloads/release/python-3810/) manualmente e executar a partir do código-fonte.

Se aparecerem erros envolvendo `msvcp140.dll`, `c10.dll` ou `[WinError 1114]`, instale ou atualize o [Microsoft Visual C++ Redistributable x64](https://aka.ms/vc14/vc_redist.x64.exe) (Visual Studio 2015-2022; [notas oficiais de download](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)).

## macOS / Linux

O script instalará `BallonsTranslator` no diretório onde você o executar:
```bash
curl -fLO https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.sh && chmod +x install.sh && ./install.sh
```

Se `curl` não estiver disponível, baixe o script com `wget -O ...`. O aplicativo inicia automaticamente após a instalação; depois, use `cd BallonsTranslator && ./launch.sh` para iniciá-lo novamente.

O aplicativo verifica as dependências principais na inicialização. Ao selecionar um módulo que precisa de bibliotecas extras, o aplicativo solicitará a instalação das dependências opcionais ausentes (você também pode ativar a instalação automática nas configurações). Se o download dos modelos falhar, verifique sua rede/proxy, ou baixe os modelos necessários pelo [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) ou [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) e coloque-os manualmente no diretório `data`.

O software possui verificação de atualização integrada; consulte Painel de configuração -> Startup & Update para detalhes.

# Utilização

**É recomendado executar o programa em um terminal caso ocorra alguma falha e não sejam fornecidas informações, como mostrado no gif a seguir.**
<img src="https://github.com/user-attachments/assets/ee92fbdc-718c-4e04-a876-0eff3ee2a989">  

- Na primeira execução, selecione o tradutor e defina os idiomas de origem e destino clicando no ícone de configurações.
- Abra uma pasta contendo as imagens do quadrinho (mangá/manhua/manhwa) que precisa de tradução clicando no ícone de pasta.
- Clique no botão `Run` e aguarde a conclusão do processo.

Os formatos de fonte, como tamanho e cor, são determinados automaticamente pelo programa neste processo. Você pode pré-determinar esses formatos alterando as opções correspondentes de "decidir pelo programa" para "usar configuração global" no painel de configurações->Diagramação. (As configurações globais são os formatos exibidos no painel de formatação de fonte à direita quando você não está editando nenhum bloco de texto na cena.)
<img src="https://github.com/user-attachments/assets/fb8a8b2c-54e4-4579-8319-42a172296c80">

## Edição de Imagem

### Ferramenta de Inpainting
<img src="https://github.com/user-attachments/assets/de0bc35d-6651-4f2f-985c-cfe9bfafb124">
<p align = "center">
**Modo de edição de imagem, ferramenta de Inpainting**
</p>

### Ferramenta Retângulo
<img src="https://github.com/user-attachments/assets/6c47f46f-ffd3-41fd-b667-5442be304c79">
<p align = "center">
**Ferramenta Retângulo**
</p>

Para 'apagar' resultados indesejados de inpainting, use a ferramenta de inpainting ou a ferramenta retângulo com o **botão direito do mouse** pressionado. O resultado depende da precisão com que o algoritmo ("método 1" e "método 2" no gif) extrai a máscara de texto. O desempenho pode ser pior em textos e fundos complexos.

## Edição de Texto
<img src="https://github.com/user-attachments/assets/0f688abe-41f7-416a-85c8-e0dd6968fd00">
<p align = "center">
**Modo de edição de texto**
</p>

<img src="https://github.com/user-attachments/assets/6d31c8a5-b909-4339-8036-7fc3ba2f014c" div align=center>
<p align=center>
**Formatação de texto em lote e layout automático**
</p>

<img src="https://github.com/user-attachments/assets/1b76c164-1454-4aa7-b60c-9fbdb0968350" div align=center>
<p align=center>
**OCR e tradução de área selecionada**
</p>

## Atalhos
* `A`/`D` ou `pageUp`/`Down` para virar a página
* `Ctrl+Z`, `Ctrl+Shift+Z` para desfazer/refazer a maioria das operações (a pilha de desfazer é limpa ao virar a página).
* `T` para o modo de edição de texto (ou o botão "T" na barra de ferramentas inferior).
* `W` para ativar o modo de criação de bloco de texto, arraste o mouse na tela com o botão direito pressionado para adicionar um novo bloco de texto (veja o gif de edição de texto).
* `P` para o modo de edição de imagem.
* No modo de edição de imagem, use o controle deslizante no canto inferior direito para controlar a transparência da imagem original.
* Desative ou ative qualquer módulo automático através da barra de título->executar. Executar com todos os módulos desativados irá refazer as letras e renderizar todo o texto de acordo com as configurações correspondentes.
* Defina os parâmetros dos módulos automáticos no painel de configuração.
* `Ctrl++`/`Ctrl+-` (Também `Ctrl+Shift+=`) para redimensionar a imagem.
* `Ctrl+G`/`Ctrl+F` para pesquisar globalmente/na página atual.
* `0-9` para ajustar a opacidade da camada de texto.
* Para edição de texto: negrito - `Ctrl+B`, sublinhado - `Ctrl+U`, itálico - `Ctrl+I`.
* Defina a sombra e a transparência do texto no painel de estilo de texto -> Efeito.

<img src="https://github.com/user-attachments/assets/084a250d-6a31-4344-94c0-2a5f4ba64b96">

## Modo Headless (Executar sem interface gráfica)

```python
python launch.py --headless --exec_dirs "[DIR_1],[DIR_2]..."
```

A configuração (idioma de origem, idioma de destino, modelo de inpainting, etc.) será carregada de config/config.json. Se o tamanho da fonte renderizada não estiver correto, especifique o DPI lógico manualmente através de `--ldpi`. Os valores típicos são 96 e 72.

## Módulos de Automação
Este projeto depende fortemente do [manga-image-translator](https://github.com/zyddnys/manga-image-translator). Serviços online e treinamento de modelos não são baratos, considere fazer uma doação ao projeto:
- Ko-fi: [https://ko-fi.com/voilelabs](https://ko-fi.com/voilelabs)
- Patreon: [https://www.patreon.com/voilelabs](https://www.patreon.com/voilelabs)
- 爱发电: [https://afdian.net/@voilelabs](https://afdian.net/@voilelabs)

O [Sugoi translator](https://sugoitranslator.com/) foi criado por [mingshiba](https://www.patreon.com/mingshiba).

## Detecção de Texto
* Suporta detecção de texto em inglês e japonês. O código de treinamento e mais detalhes podem ser encontrados em [comic-text-detector](https://github.com/dmMaze/comic-text-detector).
* Suporta o uso de detecção de texto do [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/). O nome de usuário e a senha precisam ser preenchidos, e o login automático será realizado a cada vez que o programa for iniciado.
  * Para instruções detalhadas, consulte [Manual do TuanziOCR](../doc/Manual_TuanziOCR_pt-BR.md).

## OCR
* Todos os modelos mit* são do manga-image-translator e suportam reconhecimento de inglês, japonês e coreano, além da extração da cor do texto.
* [manga_ocr](https://github.com/kha-white/manga-ocr) é de [kha-white](https://github.com/kha-white), reconhecimento de texto para japonês, com foco principal em mangás japoneses.
* Suporta o uso de OCR do [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/). O nome de usuário e a senha precisam ser preenchidos, e o login automático será realizado a cada vez que o programa for iniciado.
  * A implementação atual usa OCR em cada bloco de texto individualmente, resultando em velocidade mais lenta e sem melhoria significativa na precisão. Não é recomendado. Se necessário, use o Tuanzi Detector.
  * Ao usar o Tuanzi Detector para detecção de texto, recomenda-se definir o OCR como none_ocr para ler o texto diretamente, economizando tempo e reduzindo o número de solicitações.
  * Para instruções detalhadas, consulte [Manual do TuanziOCR](../doc/Manual_TuanziOCR_pt-BR.md).

## Inpainting
* O AOT é do [manga-image-translator](https://github.com/zyddnys/manga-image-translator).
* Todos os lama* são ajustados usando o [LaMa](https://github.com/advimman/lama).
* PatchMatch é um algoritmo do [PyPatchMatch](https://github.com/vacancy/PyPatchMatch). Este programa usa uma [versão modificada](https://github.com/dmMaze/PyPatchMatchInpaint) por mim.

## Tradutores
Tradutores disponíveis: Google, DeepL, ChatGPT, Sugoi, Caiyun, Baidu, Papago e Yandex.
* O Google desativou o serviço de tradução na China, defina a 'url' correspondente no painel de configuração para *.com.
* Os tradutores [Caiyun](https://dashboard.caiyunapp.com/), [ChatGPT](https://platform.openai.com/playground), [Yandex](https://yandex.com/dev/translate/), [Baidu](http://developers.baidu.com/) e [DeepL](https://www.deepl.com/docs-api/api-access) exigem um token ou chave de API.
* DeepL e Sugoi translator (e sua conversão CT2 Translation) graças a [Snowad14](https://github.com/Snowad14).
* Sugoi traduz do japonês para o inglês completamente offline.
* [Sakura-13B-Galgame](https://github.com/SakuraLLM/Sakura-13B-Galgame)

Para adicionar um novo tradutor, consulte [Como_add_um_novo_tradutor](../doc/Como_add_um_novo_tradutor.md). É simples como criar uma subclasse de uma classe base e implementar duas interfaces. Em seguida, você pode usá-lo no aplicativo. Contribuições para o projeto são bem-vindas.

## FAQ & Diversos
* Se o seu computador tiver uma GPU Nvidia ou Apple Silicon, o programa habilitará a aceleração de hardware.
* Adicione suporte para [saladict](https://saladict.crimx.com) (*Dicionário pop-up profissional e tradutor de páginas tudo-em-um*) no mini menu ao selecionar o texto. [Guia de instalação](../doc/saladict_pt-br.md).
* Acelere o desempenho se você tiver um dispositivo [NVIDIA CUDA](https://pytorch.org/docs/stable/notes/cuda.html) ou [AMD ROCm](https://pytorch.org/docs/stable/notes/hip.html), pois a maioria dos módulos usa o [PyTorch](https://pytorch.org/get-started/locally/).
* As fontes são do seu sistema.
* Agradecimentos a [bropines](https://github.com/bropines) pela localização para o russo.
* Adicionado script JSX de exportação para o Photoshop por [bropines](https://github.com/bropines). Para ler as instruções, melhorar o código e apenas explorar como funciona, vá para `scripts/export to photoshop` -> `install_manual.md`.
