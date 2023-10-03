# Install Python 3.11.5 via pyenv
if command -v pyenv >/dev/null 2>&1; then
  pyenv install '3.11.5'
else
  echo "pyenv is not installed"
fi

# Activate Python 3.11.5 in global environment
pyenv global '3.11.5'

# Create and activate Python 3.11.5 virtual environment
python -m venv 'venv'
source 'venv/bin/activate'

# Install dependencies
pip install -r requirements.txt
pip install pyinstaller

# Copy required directories to data
ditto ../libs data/libs -V
ditto ../models data/models -V

# Comment lines 217-233 of launch.py
cp launch.py launch.py.bak # backup launch.py
sed -i '' '217,233s|^| \#|' launch.py # comment specfied lines
sed -n '217,233p' launch.py # check if comment is successful

# Build macOS app via pyinstaller
sudo pyinstaller launch.spec
