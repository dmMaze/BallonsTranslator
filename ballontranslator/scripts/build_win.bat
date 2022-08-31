nuitka --standalone --mingw64 --nofollow-imports --show-memory --show-progress ^
    --enable-plugin=pyqt5 --include-qt-plugins=sensible,styles ^
        --follow-import-to=dl,utils,ui --include-plugin-directory=ballontranslator/dl,ballontranslator/ui,ballontranslator/utils ^
            --windows-product-version=1.3.4.0 --windows-company-name=DUMMY_WINDOWS_COMPANY_NAME --windows-product-name=BallonTranslator ^
                --output-dir=release BallonTranslator 