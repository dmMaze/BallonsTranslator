<p align="center">
  <img
    width="256"
    alt="Spinning fox animation"
    src="https://github.com/user-attachments/assets/fe44e9a6-c7da-4bc5-8421-87fd6c38a0ba"
  />
</p>

<h1 align="center">BallonsTranslator</h1>

<p align="center">BallonTranslator est un autre outil assisté par ordinateur, basé sur l'apprentissage profond (deep learning), permettant de traduire des comics/mangas.</p>

<p align="center">
  <a href="/README.md">简体中文</a> | <a href="/README_EN.md">English</a> | <a href="/doc/README_RU.md">Русский</a> | <a href="/doc/README_JA.md">日本語</a> | <a href="/doc/README_ES.md">Español</a> | Français | <a href="/doc/README_PT-BR.md">pt-BR</a> | <a href="/doc/README_KO.md">한국어</a> | <a href="/doc/README_ID.md">Indonesia</a> | <a href="/doc/README_VI.md">Tiếng Việt</a>
</p>

Prend en charge le formatage riche du texte et les préréglages de style. Les textes traduits peuvent être édités interactivement.

Prend en charge rechercher & remplacer

Prend en charge l’export/import vers/depuis des documents Word

# Fonctionnalités
> [!IMPORTANT]
> **Si vous partagez publiquement le résultat traduit et qu'aucun traducteur humain expérimenté n'a participé à la traduction ou à la relecture, veuillez indiquer clairement qu'il s'agit d'une traduction automatique.**

* Traduction entièrement automatisée
  - Prend en charge la détection, la reconnaissance, la suppression et la traduction automatiques du texte. Les performances globales dépendent de ces modules.
  - La composition typographique est basée sur l'estimation du formatage du texte original.
  - Fonctionne correctement avec les mangas et comics.
  - Amélioration du lettrage manga->Anglais, Anglais->Chinois (basé sur l'extraction des zones de bulles).
  
* Édition d’image  
  - Prise en charge de l'édition et de la retouche des masques (similaire à l'outil Pinceau correcteur dans Photoshop)
  - Adapté aux images à rapport hauteur/largeur extrême comme les webtoons
  
* Édition de texte
  - Prend en charge le formatage riche du texte et les [préréglages de style](https://github.com/dmMaze/BallonsTranslator/pull/311). Les textes traduits peuvent être édités interactivement.
  - [Transformations de texte](https://github.com/dmMaze/BallonsTranslator/pull/1238), recherche et remplacement
  - Prend en charge l’export/import vers/depuis des documents Word

* <details>
  <summary><i>Traduction LLM sensible au contexte et glossaires</i></summary>

  **Historique des traductions**

  - Réglez **LLM Context** sur **+history** pour montrer à `LLMTranslator` des exemples tirés des pages antérieures terminées. Cela peut améliorer la cohérence des noms, de la terminologie et du ton. Les reprises et plages sélectionnées peuvent aussi utiliser les pages antérieures admissibles.
  - **Token budget** contrôle la quantité de texte traduit antérieur incluse, en privilégiant les pages récentes. La page actuelle, les instructions, le glossaire et la réponse générée nécessitent de l’espace supplémentaire. La valeur par défaut est `4096`.
  - Un budget plus élevé fournit davantage de contexte narratif et supprime moins souvent les anciennes pages, mais envoie plus de texte et peut être plus lent. Les modèles locaux peuvent aussi nécessiter beaucoup plus de RAM/VRAM. La valeur par défaut `4096` est volontairement prudente ; les fournisseurs courants dotés d’une grande fenêtre de contexte, comme DeepSeek, acceptent souvent une limite supérieure. Environ 70 % de la limite de contexte du modèle constitue une limite supérieure raisonnable (`90000` pour 128K).
  - Le budget de l’historique influe aussi sur le cache de prompts. Tant que l’historique augmente sans dépasser ce budget, les requêtes consécutives gardent le même début, que des fournisseurs comme OpenAI et DeepSeek peuvent réutiliser à un tarif réduit par jeton d’entrée et parfois avec moins de latence. Lorsque le budget impose de supprimer d’anciennes pages, ce début change et la réutilisation du cache est réinitialisée. Un budget plus élevé réduit ces réinitialisations, mais envoie davantage d’historique et ne garantit donc pas un coût total inférieur.

  Le tableau ci-dessous donne une estimation approximative pour des pages de manga avec DeepSeek, où les jetons d’entrée mis en cache coûtent 10 % du prix des jetons d’entrée ordinaires. Les résultats réels varient selon le projet, le modèle et le fournisseur.

  | Token budget | Historique conservé estimé (pages) | Coût total estimé par rapport à l’absence d’historique |
  |---:|---:|---:|
  | `2048` | 3–4 | 1.65× |
  | `4096` | 6–9 | 1.79× |
  | `8192` | 12–19 | 2.10× |
  | `16384` | 23–38 | 2.66× |

  **Glossaires réutilisables**

  - Définissez **Glossary File** dans la boîte de dialogue d’exécution sur un fichier UTF-8 `.json`, `.txt` ou `.tsv`. Ce fichier est en lecture seule et peut être réutilisé dans plusieurs projets.
  - **Matching** envoie uniquement les entrées dont les termes sources figurent sur la page concernée. **All** envoie toutes les entrées et peut consommer beaucoup plus de jetons.
  - Les formats pris en charge comprennent :

    ```text
    # Texte au format Sakura
    source->traduction # note facultative

    # Texte séparé par des tabulations
    source<TAB>traduction<TAB>note facultative
    ```

    ```json
    [
      {"src": "source", "dst": "traduction", "info": "note facultative"}
    ]
    ```

  - La correspondance est littérale et insensible à la casse. Les entrées conflictuelles, les fichiers mal formés, les formats non pris en charge et les fichiers manquants interrompent la traduction avant l’envoi d’une requête au LLM.
  - Le contexte des pages antérieures et l’injection du glossaire ne concernent que `LLMTranslator` ; les autres traducteurs ignorent ces paramètres.

  </details>

# Installation

## Sous Windows

### Sur Windows

**Méthode A (Configuration automatique de l'environnement local en un clic, nécessite PowerShell)** :
Le script installe `BallonsTranslator` dans le répertoire où vous l'exécutez :
```powershell
irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex
```
Ou exécutez la commande suivante dans l'invite de commande classique (`cmd.exe`) :
```cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex"
```

**Méthode B (Télécharger le paquet préconfiguré)** :
Téléchargez `Ballonstranslator_win_minium.zip` depuis [GitHub Releases](https://github.com/dmMaze/BallonsTranslator/releases), extrayez-le et double-cliquez sur `launch_win.bat` pour démarrer l'application.

Ces méthodes ne prennent pas en charge Windows 7 ; les utilisateurs de Windows 7 doivent installer [Python 3.8](https://www.python.org/downloads/release/python-3810/) manuellement et exécuter depuis le code source.

Si vous voyez des erreurs liées à `msvcp140.dll`, `c10.dll` ou `[WinError 1114]`, installez ou mettez à jour [Microsoft Visual C++ Redistributable x64](https://aka.ms/vc14/vc_redist.x64.exe) (Visual Studio 2015-2022 ; [notes officielles de téléchargement](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)).

## macOS / Linux

Le script installe `BallonsTranslator` dans le répertoire où vous l'exécutez :
```bash
curl -fLO https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.sh && chmod +x install.sh && ./install.sh
```

Si `curl` n'est pas disponible, téléchargez plutôt le script avec `wget -O ...`. L'application démarre automatiquement après l'installation ; ensuite, utilisez `cd BallonsTranslator && ./launch.sh` pour la relancer.

L'application vérifie les dépendances principales au démarrage. Lorsque vous sélectionnez un module qui nécessite des bibliothèques supplémentaires, l'application vous proposera d'installer les dépendances optionnelles manquantes (vous pouvez aussi activer l'installation automatique dans les paramètres).

# Utilisation

**Il est conseillé de lancer le programme dans un terminal pour voir les messages en cas de plantage, voir le gif suivant.**
<img src="https://github.com/user-attachments/assets/ee92fbdc-718c-4e04-a876-0eff3ee2a989">  
- La première fois que vous lancez l'application, veuillez sélectionner le traducteur et définir les langues source et cible en cliquant sur l'icône des paramètres.
- Ouvrez un dossier contenant les images du manga/manhua/manhwa/comic à traduire en cliquant sur l’icône dossier.
- Cliquez sur le bouton `Run` et attendez la fin du processus.

Les formats de police, tels que la taille et la couleur, sont déterminés automatiquement par le programme au cours de ce processus. Vous pouvez prédéfinir ces formats en modifiant les options correspondantes de « Déterminer par programme » à « Utiliser les paramètres globaux » dans le panneau de configuration -> Composition typographique. (Les paramètres globaux sont les formats affichés dans le panneau de format de police de droite lorsque vous ne modifiez aucun bloc de texte dans la scène.)
<img src="https://github.com/user-attachments/assets/fb8a8b2c-54e4-4579-8319-42a172296c80">

## Édition d’image

### Outil de retouche
<img src="https://github.com/user-attachments/assets/de0bc35d-6651-4f2f-985c-cfe9bfafb124">
<p align = "center">
Mode d'édition d'image, outil de retouche
</p>

### Outil Rect
<img src="https://github.com/user-attachments/assets/6c47f46f-ffd3-41fd-b667-5442be304c79">
<p align = "center">
Outil Rect
</p>

Pour « effacer » les résultats indésirables de la retouche, utilisez l'outil de retouche ou l'outil de correction en maintenant le **clic droit** enfoncé.
Le résultat dépend de la précision avec laquelle l'algorithme (méthode 1 et méthode 2 dans le gif) extrait le masque de texte. Il peut être moins performant sur des textes et des arrière-plans complexes.  

## Édition de texte
<img src="https://github.com/user-attachments/assets/0f688abe-41f7-416a-85c8-e0dd6968fd00">
<p align = "center">
Mode édition de texte
</p>

<img src="https://github.com/user-attachments/assets/6d31c8a5-b909-4339-8036-7fc3ba2f014c" div align=center>
<p align=center>
Formatage de texte en lot & auto-mise en page
</p>

<img src="https://github.com/user-attachments/assets/1b76c164-1454-4aa7-b60c-9fbdb0968350" div align=center>
<p align=center>
OCR & traduction d’une zone sélectionnée
</p>

## Raccourcis
* ```A```/```D``` ou ```pageUp```/```Down``` pour changer de page.
* ```Ctrl+Z```, ```Ctrl+Shift+Z``` pour annuler/rétablir la plupart des opérations. (Remarque : la pile d'annulation sera effacée après avoir changé de page.)
* ```T``` pour le mode édition de texte (ou le bouton "T" en bas).
* ```W``` pour activer le mode de création de blocs de texte, cliquez avec le clic droit de la souris sur le canevas et faites glisser la souris pour ajouter un nouveau bloc de texte. (voir le gif sur l'édition de texte)
* ```P``` pour le mode édition d’image.
* En mode édition d'image, utilisez le curseur en bas à droite pour contrôler la transparence de l'image d'origine.
* Désactivez ou activez les modules automatiques via la barre de titre->Exécuter. L'exécution avec tous les modules désactivés réécrira et réaffichera tout le texte en fonction des paramètres correspondants.
* Définissez les paramètres des modules automatiques dans le panneau de configuration.
* ```Ctrl++```/```Ctrl+-``` (Aussi ```Ctrl+Shift+=```) pour redimensionner l’image.
* ```Ctrl+G```/```Ctrl+F``` pour faire une recherche globale/dans la page actuelle.
* ```0-9``` pour ajuster l'opacité du calque de texte.
* Pour l'édition de texte : gras - ```Ctrl+B```, souligné - ```Ctrl+U```, italique - ```Ctrl+I``` 
* Définissez l'ombre et la transparence du texte dans le panneau Style de texte -> Effet.
* ```Alt+Touches fléchées``` ou ```Alt+WASD``` (```pageDown``` ou ```pageUp``` en mode édition de texte) pour passer d'un bloc de texte à l'autre.
  
<img src="https://github.com/user-attachments/assets/084a250d-6a31-4344-94c0-2a5f4ba64b96">

## Mode sans interface (exécution sans interface graphique)
``` python
python launch.py --headless --exec_dirs "[DIR_1],[DIR_2]..."
```
Notez que la configuration (langue source, langue cible, modèle de retouche, etc.) sera chargée à partir du fichier config/config.json.  
Si la taille de la police rendue n'est pas correcte, spécifiez manuellement la résolution logique via ```--ldpi ```, les valeurs typiques sont 96 et 72.


# Modules d'automatisation
Ce projet dépend fortement de [manga-image-translator](https://github.com/zyddnys/manga-image-translator), un service en ligne et la formation des modèles n'est pas bon marché, veuillez envisager de faire un don au projet :
- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>  

[Sugoi translator](https://sugoitranslator.com/) est créé par [mingshiba](https://www.patreon.com/mingshiba).
  
## Détection de texte
 * Prise en charge de la détection de texte en anglais et en japonais. Le code source et plus de détails sont disponibles sur [comic-text-detector].
 * Prise en charge de la détection de texte à partir de [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/). Le nom d'utilisateur et le mot de passe doivent être renseignés, et la connexion automatique sera effectuée à chaque lancement du programme.

   * Pour obtenir des instructions détaillées, consultez le [Manuel TuanziOCR](../doc/Manual_TuanziOCR_FR.md)
 
 * Les Modèles`YSGDetector` sont entraînés par [lhj5426](https://github.com/lhj5426), filtrent les onomatopées dans CGs/mangas. Téléchargez depuis [YSGYoloDetector](https://huggingface.co/YSGforMTL/YSGYoloDetector) et placez dans `data/models`. 


## OCR
 * Les modèles mit* viennent de manga-image-translator, prennent en charge l’anglais, japonais, coréen et l’extraction de couleur du texte.
 * [manga_ocr](https://github.com/kha-white/manga-ocr) est un logiciel de reconnaissance de texte japonais développé par [kha-white](https://github.com/kha-white), principalement destiné aux mangas japonais.
 * Prise en charge de la reconnaissance optique de caractères (OCR) via [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/). Le nom d'utilisateur et le mot de passe doivent être renseignés, et la connexion automatique s'effectuera à chaque lancement du programme.
   * L’implémentation actuelle applique l’OCR sur chaque bloc, plus lente et pas plus précise, non recommandée. Préférez Tuanzi Detector.
   * Lorsque vous utilisez le Tuanzi Detector pour la détection de texte, il est recommandé de définir OCR sur none_ocr afin de lire directement le texte, ce qui permet de gagner du temps et de réduire le nombre de requêtes.
   * Pour obtenir des instructions détaillées, consultez le [Manuel TuanziOCR](../doc/Manual_TuanziOCR_FR.md)
* Ajouté en option sous forme de module PaddleOCR. En mode débogage, un message vous indiquera qu'il n'est pas présent. Vous pouvez simplement l'installer en suivant les instructions qui y sont décrites. Si vous ne souhaitez pas installer le paquet vous-même, il vous suffit de décommenter (supprimer le `#`) les lignes contenant paddlepaddle(gpu) et paddleocr. Tout cela se fait à vos propres risques et périls. Pour moi (bropines) et deux testeurs, tout s'est bien installé, mais vous pourriez rencontrer une erreur. Signalez-la dans le ticket et identifiez-moi.
* Ajouté [OneOCR](https://github.com/b1tg/win11-oneocr). Modèle WINDOWS local provenant des applications SnippingTOOL ou Win.PHOTOS. Pour l'utiliser, vous devez placer les fichiers du modèle et les fichiers DLL dans le dossier « data/models/one-ocr ». Avant de lancer le programme, il est préférable de copier tous les fichiers en une seule fois. Pour savoir comment trouver et obtenir les fichiers DLL et les fichiers de modèle, consultez : https://github.com/dmMaze/BallonsTranslator/discussions/859#discussioncomment-12876757 . Merci à AuroraWright pour le projet [OneOCR](https://github.com/AuroraWright/oneocr)

## Retouche
  * AOT provient de [manga-image-translator](https://github.com/zyddnys/manga-image-translator).
  * Tous les lama* sont affinés à l'aide de [LaMa](https://github.com/advimman/lama)
  * PatchMatch est un algorithme issu de [PyPatchMatch](https://github.com/vacancy/PyPatchMatch), ce programme utilise une [version modifiée](https://github.com/dmMaze/PyPatchMatchInpaint)
  
## Traducteurs

Traducteurs disponibles : Google, DeepL, ChatGPT, Sugoi, Caiyun, Baidu, Papago et Yandex.

* Vous trouverez des informations sur les modules Traducteurs [ici](../doc/modules/translators.md). *(Anglais)*

## FAQ & Divers
* Si vous avez une carte Nvidia ou une puce Apple, l’accélération matérielle sera activée.
* Ajout de la prise en charge de [saladict](https://saladict.crimx.com) (*Dictionnaire contextuel et traducteur de pages professionnel tout-en-un*) dans le mini-menu lors de la sélection de texte. [Guide d'installation](../doc/saladict_fr.md)
* Accélérez les performances si vous disposez d'un périphérique [NVIDIA's CUDA](https://pytorch.org/docs/stable/notes/cuda.html) ou [AMD's ROCm](https://pytorch.org/docs/stable/notes/hip.html), car la plupart des modules utilisent [PyTorch](https://pytorch.org/get-started/locally/).
* Les polices proviennent des polices de votre système.
* Merci à [bropines](https://github.com/bropines) pour l'adaptation en russe.
* Ajout du script JSX « Export vers Photoshop » par [bropines](https://github.com/bropines). </br> Pour lire les instructions, améliorer le code et simplement explorer son fonctionnement, rendez-vous dans `scripts/export vers Photoshop` -> `install_manual.md`.
