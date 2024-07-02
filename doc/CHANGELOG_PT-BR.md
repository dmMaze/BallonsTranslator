# Changelogs

### 15/04/2023
Implementação de download de origem baseada em gallery-dl (#131) graças a [ROKOLYT](https://github.com/ROKOLYT)

### 27/02/2023
[v1.3.34](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.34) lançado
1. Corrige atribuição incorreta de orientação para CHT (#96)
2. Converte CHS para CHT se necessário para Caiyun e DeepL (#100)
3. Suporte para webp (#85)

### 23/02/2023
[v1.3.30](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.30) lançado
1. Migração para PyQt6 para melhor pré-visualização de renderização de texto e [compatibilidade](https://github.com/Nuitka/Nuitka/issues/251) com nuitka
2. Suporte para definir transparência da camada de texto (#88)
3. Exportação de logs para data/logs

### 27 de Janeiro de 2023
**[v1.3.26](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.26) lançado**
1. Adicionado suporte ao [saladict](https://saladict.crimx.com) (*Dicionário pop-up profissional e tradutor de páginas tudo-em-um*) no mini menu de seleção de texto. [Guia de Instalação](doc/saladict.md)
<img src = "./src/saladict_doc.jpg">

2. Adicionado substituição de palavras-chave para resultados de OCR e tradução automática [#78](https://github.com/dmMaze/BallonsTranslator/issues/78): Editar -> "Substituição de palavras-chave para tradução automática"
3. Adicionado importação de pastas por arrastar e soltar [#77](https://github.com/dmMaze/BallonsTranslator/issues/77)
4. Ocultar blocos de controle ao iniciar a edição de texto. [#81](https://github.com/dmMaze/BallonsTranslator/issues/81)
5. Correção de bugs

### 08 de Janeiro de 2023
**[v1.3.22](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.22) lançado**
1. Adicionado suporte para excluir e restaurar texto removido
2. Adicionado suporte para redefinir o ângulo
3. Correção de bugs

### 31 de Dezembro de 2022
**[v1.3.20](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.20) lançado**
1. Adaptado para imagens com proporção extrema, como webtoons
2. Adicionado suporte para colar texto em vários blocos de texto selecionados
3. Correção de bugs
4. OCR/Tradução/Inpainting de blocos de texto selecionados: O estilo da letra herdará do bloco selecionado correspondente. ctc_48px é mais recomendado para texto de linha única, mangocr para japonês de várias linhas; é necessário retreinar o modelo de detecção para que ctc48_px seja generalizado para várias linhas. Observe que, se você usar **ctc_48px**, certifique-se de que a caixa esteja no modo vertical e se ajuste o mais próximo possível da linha única de texto.
<img src="./src/ocrselected.gif" div align=center>

### 29 de Novembro de 2022
**[v1.3.15](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.15) lançado**
1. Correção de bugs
2. Otimização da lógica de salvamento
3. A forma da ferramenta Caneta/Inpaint pode ser definida como retângulo (experimental)

### 25 de Outubro de 2022
**[v1.3.14](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.14) lançado**
1. Correção de bugs

### 30 de Setembro de 2022
Suporte ao Modo Escuro desde a v1.3.13: Visualizar->Modo Escuro

### 24 de Setembro de 2022
**[v1.3.12](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.12) lançado**
1. Adicionado suporte para Pesquisa global (Ctrl+G) e pesquisa na página atual (Ctrl+F)
2. Pilhas de desfazer locais de cada editor de texto mescladas em uma pilha principal de edição de texto, agora separada da prancheta de desenho
3. Correção de bugs de importação/exportação de documentos do Word
4. Reformulação da janela sem moldura baseada em [https://github.com/zhiyiYo/PyQt-Frameless-Window](https://github.com/zhiyiYo/PyQt-Frameless-Window)

### 13 de Setembro de 2022
**[v1.3.8](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.8) lançado**

1. Correção de bugs e otimização da ferramenta Caneta
2. Correção de dimensionamento
3. Adicionado suporte para criação de predefinições de estilo de fonte e efeitos gráficos de texto (sombra e opacidade), veja [https://github.com/dmMaze/BallonsTranslator/pull/38](https://github.com/dmMaze/BallonsTranslator/pull/38)
4. Adicionado suporte para importação/exportação de documentos do Word (*.docx): [https://github.com/dmMaze/BallonsTranslator/pull/40](https://github.com/dmMaze/BallonsTranslator/pull/40)

### 31 de Agosto de 2022
**[v1.3.4](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.4) lançado**

1. Adicionado Sugoi Translator (apenas japonês-inglês, criado e autorizado por [mingshiba](https://www.patreon.com/mingshiba)): baixe o [modelo](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm) convertido por [@Snowad14](https://github.com/Snowad14) e coloque "sugoi_translator" na pasta "data".
2. Adicionado suporte para russo, graças a [bropines](https://github.com/bropines)
3. Adicionado ajuste de espaçamento entre letras
4. Reformulação do tipo vertical e correção de bugs de renderização de texto: [https://github.com/dmMaze/BallonsTranslator/pull/30](https://github.com/dmMaze/BallonsTranslator/pull/30)

### 17 de Agosto de 2022
**[v1.3.0](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.0) lançado**

1. Correção do tradutor DeepL, graças a [@Snowad14](https://github.com/Snowad14)
2. Correção de bug de tamanho e traçado de fonte que tornava o texto ilegível
3. Adicionado suporte para formato de fonte global (determina as configurações de formato de fonte usadas pelo modo de tradução automática): no painel de configuração->Diagramação, altere a opção correspondente de "decidir pelo programa" para "usar configuração global" para habilitar. Observe que as configurações globais são os formatos mostrados no painel de formato de fonte à direita quando você não está editando nenhum bloco de texto na cena.
4. Adicionado novo modelo de inpainting: lama-mpe e definido como padrão
5. Adicionado suporte para seleção e formatação de vários blocos de texto
6. Aprimorada a diagramação de mangá->inglês, inglês->chinês (**Layout automático** no painel de configuração->Diagramação, habilitado por padrão), também pode ser aplicado a blocos de texto selecionados usando a opção no menu do botão direito.

<img src="./src/multisel_autolayout.gif" div align=center>
<p align=center>
**formatação de texto em lote e auto layout**
</p>

### 19 de Maio de 2022
**[v1.2.0](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.2.0) lançado**
1. Adicionado suporte ao DeepL, graças a [@Snowad14](https://github.com/Snowad14)
2. Adicionado novo modelo OCR do manga-image-translator, com suporte a reconhecimento de coreano
3. Correção de bugs

### 17 de Abril de 2022
**[v1.1.0](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.1.0) lançado**
1. Utilização de qthread para gravar imagens editadas para evitar congelamento ao virar páginas
2. Otimização da política de inpainting
3. Adicionada ferramenta de retângulo
4. Mais atalhos
5. Correção de bugs

### 09 de Abril de 2022

1. v1.0.0 lançado