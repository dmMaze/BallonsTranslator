set scriptpath=%~dp0
cd %scriptpath%..

set PATH=ballontrans_pylibs_win;ballontrans_pylibs_win\Scripts;PortableGit\cmd;%PATH%
git pull

pause