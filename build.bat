cd ../
nuitka --standalone --mingw64 --nofollow-imports --show-memory --show-progress ^
    --enable-plugin=pyqt5 --include-qt-plugins=sensible,styles ^
        --follow-import-to=dl,utils,ui --include-plugin-directory=BallonTranslator/dl,BallonTranslator/ui,BallonTranslator/utils ^
            --windows-product-version=1.1.0.0 --windows-company-name=DUMMY_WINDOWS_COMPANY_NAME --windows-product-name=BallonTranslator ^
                --output-dir=BallonTranslatorRelease BallonTranslator 