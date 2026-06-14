> [!IMPORTANT]
> **Si planeas compartir públicamente los resultados de traducción automática generados con esta herramienta, y no han sido revisados o traducidos completamente por un traductor con experiencia, por favor indícalo claramente como traducción automática en un lugar visible.**

## BallonTranslator

[简体中文](/README.md) | [English](/README_EN.md) | [Русский](/doc/README_RU.md) | [日本語](/doc/README_JA.md) | [Español](/doc/README_ES.md) | [Français](/doc/README_FR.md) | [pt-BR](/doc/README_PT-BR.md) | [한국어](/doc/README_KO.md) | [Indonesia](/doc/README_ID.md) | [Tiếng Việt](/doc/README_VI.md) 

BallonTranslator es otra herramienta asistida por ordenador, basada en el aprendizaje profundo, para traducir cómics/manga.

<img src="https://github.com/user-attachments/assets/2140c402-dda2-47bc-9e7f-83ed41ce78af" div align=center>

<p align=center>
  <strong>Vista previa</strong>
</p>
  
## Recursos
* **Traducción totalmente automática:** 
  - Detecta, reconoce, elimina y traduce textos automáticamente. El rendimiento global depende de estos módulos.
  - La maquetación se basa en el formato estimado del texto original.
  - Funciona bien con manga y cómics.
  - Diseño mejorado para manga->inglés, inglés->chino (basado en la extracción de regiones de globos).
  
* **Edición de imágenes:**
  - Permite editar máscaras e inpainting (similar a la herramienta Pincel recuperador de imperfecciones de Photoshop).
  - Adaptado para imágenes con una relación de aspecto extrema, como los webtoons.
  
* **Edición de texto:**
  - Admite formato de texto y [preajustes de estilo de texto](https://github.com/dmMaze/BallonsTranslator/pull/311). Los textos traducidos pueden editarse interactivamente.
  - Buscar y reemplazar.
  - Exportación/importación a/desde documentos Word.

## Instalación

### En Windows
Si no desea instalar Python y configurar el entorno manualmente:

**Método A (Configuración automática del entorno local en un clic, requiere PowerShell)**:
El script creará automáticamente una carpeta `BallonsTranslator` en el directorio actual, descargará el código fuente más reciente, configurará un entorno aislado de Python 3.12, instalará todas las dependencias principales e iniciará la aplicación (requiere tener PowerShell habilitado en el sistema):
```powershell
irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex
```
O ejecute el siguiente comando en el símbolo del sistema clásico (`cmd.exe`):
```cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex"
```

**Método B (Descargar paquete preconfigurado)**:
Descargue el archivo `Ballonstranslator_win_minium.zip` más reciente desde [GitHub Releases](https://github.com/dmMaze/BallonsTranslator/releases), extráigalo en cualquier carpeta y haga doble clic en `launch_win.bat` para iniciar la aplicación.
Este paquete no es compatible con Windows 7; los usuarios de Windows 7 deben instalar [Python 3.8](https://www.python.org/downloads/release/python-3810/) manualmente y ejecutar desde el código fuente.

Los módulos PyTorch/deep learning pueden requerir [Microsoft Visual C++ Redistributable x64](https://aka.ms/vc14/vc_redist.x64.exe) (Visual Studio 2015-2022; [notas oficiales de descarga](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)). Instálelo o actualícelo si ve errores relacionados con `msvcp140.dll`, `c10.dll` o `[WinError 1114]`.

## macOS / Linux


```bash
curl -fLO https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.sh && chmod +x install.sh && ./install.sh
```

Si `curl` no está disponible, descargue el script con `wget -O ...` en su lugar.

Ejecute el instalador en el directorio donde desea crear la carpeta `BallonsTranslator`. La aplicación se inicia automáticamente después de la instalación; más tarde, use `cd BallonsTranslator && ./launch.sh` para iniciarla de nuevo.

La aplicación comprueba las dependencias principales al iniciar. Cuando seleccione un módulo que necesite bibliotecas adicionales, la aplicación le pedirá instalar las dependencias opcionales que falten (también puede activar la instalación automática en los ajustes). Si falla la descarga de modelos, revise la red/proxy, o descargue los modelos necesarios desde [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) o [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) y colóquelos manualmente en el directorio `data`.

El software incluye comprobación de actualizaciones; consulte Panel de configuración -> Inicio y actualización para más detalles.

# Utilización

**Se recomienda ejecutar el programa en un terminal en caso de que se produzca un fallo y no se proporcione información, como se muestra en el siguiente gif.**
<img src="https://github.com/user-attachments/assets/ee92fbdc-718c-4e04-a876-0eff3ee2a989">  

- En la primera ejecución, selecciona el traductor y establece los idiomas de origen y destino haciendo clic en el icono de configuración.
- Abre una carpeta que contenga las imágenes del cómic (manga/manhua/manhwa) que necesites traducir haciendo clic en el icono de la carpeta.
- Haz clic en el botón «Ejecutar» y espera a que se complete el proceso.

Los formatos de fuente, como el tamaño y el color, son determinados automáticamente por el programa en este proceso. Puede predeterminar estos formatos cambiando las opciones correspondientes de "decidir por el programa" a "utilizar configuración global" en el panel Configuración->Diagramación. (La configuración global son los formatos que se muestran en el panel de formato de fuente de la derecha cuando no está editando ningún bloque de texto en la escena).
<img src="https://github.com/user-attachments/assets/fb8a8b2c-54e4-4579-8319-42a172296c80">

## Edición de imágenes

### Herramienta para pintar
<img src="https://github.com/user-attachments/assets/de0bc35d-6651-4f2f-985c-cfe9bfafb124">
<p align = "center">
  <strong>Modo de edición de imágenes, herramienta Inpainting</strong>
</p>

### Herramienta rectángulo
<img src="https://github.com/user-attachments/assets/6c47f46f-ffd3-41fd-b667-5442be304c79">
<p align = "center">
  <strong>Herramienta rectángulo</strong>
</p>

Para 'borrar' los resultados de inpainting no deseados, utilice la herramienta inpainting o la herramienta rectángulo con el **botón derecho del ratón** pulsado. El resultado depende de la precisión con la que el algoritmo ("método 1" y "método 2" en el gif) extrae la máscara de texto. El rendimiento puede ser peor con texto y fondos complejos.

## Edición de texto
<img src="https://github.com/user-attachments/assets/0f688abe-41f7-416a-85c8-e0dd6968fd00">
<p align = "center">
  <strong>Modo de edición de texto</strong>
</p>

<img src="https://github.com/user-attachments/assets/6d31c8a5-b909-4339-8036-7fc3ba2f014c" div align=center>
<p align=center>
  <strong>Formato de texto por lotes y maquetación automática</strong>
</p>

<img src="https://github.com/user-attachments/assets/1b76c164-1454-4aa7-b60c-9fbdb0968350" div align=center>
<p align=center>
  <strong>OCR y traducción de áreas seleccionadas</strong>
</p>

## Atajos
* `A`/`D` o `pageUp`/`Down` para pasar de página
* `Ctrl+Z`, `Ctrl+Shift+Z` para deshacer/rehacer la mayoría de las operaciones (la pila de deshacer se borra al pasar página).
* `T` para el modo de edición de texto (o el botón "T" de la barra de herramientas inferior).
* `W` para activar el modo de creación de bloques de texto, arrastra el ratón por la pantalla con el botón derecho pulsado para añadir un nuevo bloque de texto (ver gif de edición de texto).
* `P` para el modo de edición de imágenes.
* En el modo de edición de imágenes, utiliza el control deslizante de la esquina inferior derecha para controlar la transparencia de la imagen original.
* Desactivar o activar cualquier módulo automático a través de la barra de título->ejecutar. Ejecutar con todos los módulos desactivados remapeará las letras y renderizará todo el texto según la configuración correspondiente.
* Establece los parámetros de los módulos automáticos en el panel de configuración.
* `Ctrl++`/`Ctrl+-` (También `Ctrl+Shift+=`) para redimensionar la imagen.
* `Ctrl+G`/`Ctrl+F` para buscar globalmente/en la página actual.
* `0-9` para ajustar la opacidad de la capa de texto.
* Para editar texto: negrita - `Ctrl+B`, subrayado - `Ctrl+U`, cursiva - `Ctrl+I`.
* Ajuste la sombra y la transparencia del texto en el panel de estilo de texto -> Efecto.
* ```Alt+Arrow Keys``` o ```Alt+WASD``` (```pageDown``` o ```pageUp``` mientras estás en el modo de edición de texto) para cambiar entre bloques de texto.

<img src="https://github.com/user-attachments/assets/084a250d-6a31-4344-94c0-2a5f4ba64b96">

## Modo Headless (ejecución sin interfaz gráfica)

```python
python launch.py --headless --exec_dirs "[DIR_1],[DIR_2]..."
```

La configuración (idioma de origen, idioma de destino, modelo de inpainting, etc.) se cargará desde config/config.json. Si el tamaño de la fuente renderizada no es correcto, especifique manualmente el DPI lógico mediante `--ldpi`. Los valores típicos son 96 y 72.

## Módulos de automatización
Este proyecto depende en gran medida de [manga-image-translator](https://github.com/zyddnys/manga-image-translator). Los servicios en línea y la formación de modelos no son baratos, así que por favor considere hacer una donación al proyecto:
- Ko-fi: [https://ko-fi.com/voilelabs](https://ko-fi.com/voilelabs)
- Patreon: [https://www.patreon.com/voilelabs](https://www.patreon.com/voilelabs)
- 爱发电: [https://afdian.net/@voilelabs](https://afdian.net/@voilelabs)

El [traductor de Sugoi](https://sugoitranslator.com/) fue creado por [mingshiba](https://www.patreon.com/mingshiba).

## Detección de texto
 * Permite detectar texto en inglés y japonés. El código de entrenamiento y más detalles en [comic-text-detector](https://github.com/dmMaze/comic-text-detector).
 * Admite el uso de la detección de texto de [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/). Es necesario rellenar el nombre de usuario y la contraseña, y el inicio de sesión automático se realizará cada vez que se inicie el programa.
   * Para obtener instrucciones detalladas, consulte el [Manual de TuanziOCR](../doc/Manual_TuanziOCR_ES.md).
 * Los modelos `YSGDetector` fueron entrenados por [lhj5426](https://github.com/lhj5426). Estos modelos filtran las onomatopeyas en CGs/Manga. Descarga los checkpoints desde [YSGYoloDetector](https://huggingface.co/YSGforMTL/YSGYoloDetector) y colócalos en la carpeta `data/models`.


## OCR
* Todos los modelos mit* proceden de manga-image-translator y admiten el reconocimiento en inglés, japonés y coreano, así como la extracción del color del texto.
* [manga_ocr](https://github.com/kha-white/manga-ocr) es de [kha-white](https://github.com/kha-white), reconocimiento de texto para japonés, centrado principalmente en el manga japonés.
* Admite el uso de OCR de [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/). Es necesario rellenar el nombre de usuario y la contraseña, y el inicio de sesión automático se realizará cada vez que se inicie el programa.
  * La implementación actual utiliza OCR en cada bloque de texto individualmente, lo que resulta en una velocidad más lenta y ninguna mejora significativa en la precisión. No se recomienda. Si es necesario, utilice el Detector Tuanzi.
  * Cuando se utiliza Tuanzi Detector para la detección de texto, se recomienda configurar el OCR a none_ocr para leer el texto directamente, ahorrando tiempo y reduciendo el número de peticiones.
  * Para obtener instrucciones detalladas, consulte el [Manual de TuanziOCR](doc/Manual_TuanziOCR_ES.md).
* Se añadió como un módulo opcional el soporte para PaddleOCR. En el modo Debug, verás un mensaje indicando que no está instalado. Puedes instalarlo fácilmente siguiendo las instrucciones que se muestran ahí. Si no quieres instalar el paquete manualmente, simplemente descomenta (elimina el `#`) las líneas correspondientes a paddlepaddle(gpu) y paddleocr. Hazlo bajo tu propia responsabilidad y riesgo. Si no se instaló correctamente, y genera errores; de ser así, repórtalo en Issues.
* Se añadió soporte para [OneOCR](https://github.com/b1tg/win11-oneocr). Es un modelo local de Windows, tomado de las aplicaciones Recortes (Snipping Tool) o Fotos `Win.PHOTOS`. Para usarlo, necesitas colocar el modelo y los archivos DLL en la carpeta 'data/models/one-ocr'. Es mejor colocar todos los archivos antes de ejecutar el programa. Puedes leer cómo encontrar y extraer los archivos DLL y del modelo aquí:
https://github.com/dmMaze/BallonsTranslator/discussions/859#discussioncomment-12876757. Agradecimientos a AuroraWright por el proyecto [OneOCR](https://github.com/AuroraWright/oneocr).

## Inpainting
* AOT es de [manga-image-translator](https://github.com/zyddnys/manga-image-translator).
* Todas las lama* se ajustan mediante [LaMa](https://github.com/advimman/lama).
* PatchMatch es un algoritmo de [PyPatchMatch](https://github.com/vacancy/PyPatchMatch). Este programa utiliza una [versión modificada](https://github.com/dmMaze/PyPatchMatchInpaint) por mí.

## Traductores disponibles
* **Google Translate**: El servicio de Google Translate ha sido desactivado en China. Para usarlo desde la China continental, debes configurar un proxy global y cambiar la URL en el panel de configuración a `*`.com
* **Caiyun**: Requiere que solicites un [token de acceso](https://dashboard.caiyunapp.com/).
* **Papago**: Compatible sin configuraciones adicionales.
* **DeepL y Sugoi (incluyendo su conversión con CT2 Translation)**: Agradecimientos a [Snowad14](https://github.com/Snowad14).
Si deseas usar el traductor Sugoi (solo soporta traducción del japonés al inglés), debes descargar el [modelo offline](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm) y mover la carpeta ```sugoi_translator``` dentro del directorio BallonsTranslator/ballontranslator/data/models.
* **Sugoi** traduce del japonés al inglés completamente sin conexión.
* Se admite [Sakura-13B-Galgame](https://github.com/SakuraLLM/Sakura-13B-Galgame). Si se ejecuta localmente en una sola tarjeta gráfica con poca memoria de video, puedes activar el ```low vram mode``` o Modo de bajo consumo de VRAM en el panel de configuración (activado por defecto).
* Para **DeepLX**, consulta [Vercel](https://github.com/bropines/Deeplx-vercel) o el [proyecto deeplx](https://github.com/OwO-Network/DeepLX).
* Se admiten dos versiones de traductores compatibles con **OpenAI**. Son compatibles tanto con el proveedor oficial como con proveedores de LLM de terceros que sigan la API de **OpenAI**. Es necesario configurarlo en el panel de ajustes.
   * La versión sin sufijo consume menos tokens, pero su estabilidad en la segmentación de oraciones es ligeramente peor, lo que puede causar problemas al traducir textos largos.
   * La versión con el sufijo **exp** consume más tokens, pero es más estable y usa técnicas tipo "jailbreak" en el prompt, adecuada para traducciones de textos largos.
* [m2m100](https://huggingface.co/facebook/m2m100_1.2B): Descarga y mueve la carpeta 'm2m100-1.2B-ctranslate2' al directorio 'data/models'.
* **Puedes encontrar información sobre los módulos de traductores [aquí](../doc/modules/translators.md)**.

Para otros modelos de traducción offline al inglés de buena calidad, consulta este [hilo de discusión](https://github.com/dmMaze/BallonsTranslator/discussions/515).
Para añadir un nuevo traductor, consulte [Cómo_añadir_un_nuevo_traductor](../doc/Como_añadir_un_nuevo_traductor.md). Es tan sencillo como crear una subclase de una clase base e implementar dos interfaces. Luego puedes usarla en la aplicación. Las contribuciones al proyecto son bienvenidas.

## FAQ & Varios
* Los ordenadores con tarjeta gráfica Nvidia o chip Apple Silicon activan por defecto la aceleración por GPU.
* Gracias a [bropines](https://github.com/bropines) por proporcionar la traducción al ruso.
* Los métodos de entrada de terceros pueden causar errores visuales en el cuadro de edición de la derecha. Véase el issue [#76](https://github.com/dmMaze/BallonsTranslator/issues/76); de momento no se planea solucionar esto.
* El menú flotante al seleccionar texto admite funciones como diccionarios agregados, traducción profesional palabra por palabra y [Saladict](https://saladict.crimx.com)(*Diccionario emergente profesional y traductor de páginas todo en uno*). Consulta las [instrucciones de instalación](../doc/saladict_es.md).
* Acelera el rendimiento si tienes un dispositivo [NVIDIA CUDA](https://pytorch.org/docs/stable/notes/cuda.html) o [AMD ROCm](https://pytorch.org/docs/stable/notes/hip.html), ya que la mayoría de los módulos utilizan [PyTorch](https://pytorch.org/get-started/locally/).
* Las fuentes son de tu sistema.
* Añadido script de exportación JSX para Photoshop por [bropines](https://github.com/bropines). Para leer las instrucciones, mejorar el código y simplemente explorar cómo funciona, vaya a `scripts/export to photoshop` -> `install_manual.md`.

<details>
  <summary><i>Pasos para habilitar la aceleración por GPU con tarjetas gráficas AMD (ROCm6)</i></summary>

1.  Actualiza el controlador de la tarjeta gráfica a la versión más reciente (se recomienda la versión 24.12.1 o superior). Descarga e instala [AMD HIP SDK 6.2](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html).
2.  Descarga [ZLUDA](https://github.com/lshqqytiger/ZLUDA/releases) (versión ROCm6) y descomprímelo dentro de una carpeta llamada 'zluda'.
Copia esta carpeta 'zluda' al disco del sistema, por ejemplo: 'C:\zluda'.
3.  Configura las variables de entorno del sistema (en **Windows 10**):
Ve a `Configuración → Propiedades del sistema → Configuración avanzada del sistema → Variables de entorno`.
En “Variables del sistema”, busca la variable **Path**, haz clic en editar y añade al final: `C:\zluda` y `%HIP_PATH_62%\bin`.
4.  Sustituye los archivos de enlace dinámico de la biblioteca CUDA: Copia los siguientes archivos desde 'C:\zluda' al escritorio: `cublas.dll`, `cusparse.dll` y `nvrtc.dll`. Luego, renómbralos de acuerdo con las siguientes reglas:

**Nota: Si usas el controlador AMD 25.5.1, asegúrate de actualizar ZLUDA a la versión 3.9.5 o superior.**

```
  Nombre original → Nuevo nombre

  cublas.dll → cublas64_11.dll

  cusparse.dll → cusparse64_11.dll

  nvrtc.dll → nvrtc64_112_0.dll
```
  Sustituye los archivos renombrados en el directorio: `BallonsTranslator\ballontrans_pylibs_win\Lib\site-packages\torch\lib\` reemplazando los archivos del mismo nombre.

5.  Inicia el programa y configura el OCR y la detección de texto para que usen CUDA **(la reparación de imágenes debe seguir usando la CPU)**.
6.  Al ejecutar OCR por primera vez, ZLUDA compilará los archivos PTX **(este proceso puede tardar entre 5 y 10 minutos dependiendo del rendimiento del CPU)**,**En las siguientes ejecuciones, no será necesario volver a compilar.**
</details>
