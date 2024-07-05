[简体中文](../doc/团子OCR说明.md) | pt-BR

## Referência de Parâmetros de Solicitação (Oficial)

<p align="center">
<img src="https://github.com/PiDanShouRouZhouXD/BallonsTranslator/assets/38401147/3c3985e9-f36e-41fb-af94-d6a8088e5ccd" width="85%" height="85%">
</p>

## Descrição do Tuanzi OCR

### Login
Ao fazer login pela primeira vez, você pode receber mensagens de erro de senha. Se tiver certeza de que a senha está correta, marque e desmarque a opção "force_refresh_token" para forçar um novo login. Salve as configurações e o problema deve ser resolvido.

### Detecção de Texto
A função de detecção de texto também extrai texto, mas de forma holística (identificação completa). Portanto, ao usar o TuanziOCR, recomendamos não usar a função OCR isoladamente, mas sim combinar a detecção de texto do TuanziOCR com a opção "none_ocr". O TuanziOCR possui filtros integrados para onomatopeias (Reprodução de sons por meio de fonemas/palavras. Alguns exemplos: Ruídos, gritos, sons de animais, etc.) e outros recursos. Para configurações detalhadas, consulte a "Referência de Parâmetros de Solicitação (Oficial)" acima.