nuitka --standalone --mingw64 --show-memory --show-progress ^
    --enable-plugin=pyqt5 --include-qt-plugins=sensible,styles ^
        --follow-import-to=dl,utils,ui --include-plugin-directory=ballontranslator/dl,ballontranslator/ui,ballontranslator/utils ^
            --windows-product-version=1.3.9.0 --windows-company-name=DUMMY_WINDOWS_COMPANY_NAME --windows-product-name=BallonTranslator ^
                --output-dir=release  --nofollow-import-to=numpy,urllib3,jaconv,torch,torchvision,transformers,fugashi,unidic_lite,tqdm,shapely,pyclipper,einops,termcolor,bs4,deepl,qtpy,pkuseg,pandas,spacy_pkuseg,sentencepiece,ctranslate2,python-docx,docx2txt,piexif,docx,argparse,colorama,http,email,chardet,requests,pkg_resources,yaml,PIL,multiprocessing,dbm ^
                BallonTranslator