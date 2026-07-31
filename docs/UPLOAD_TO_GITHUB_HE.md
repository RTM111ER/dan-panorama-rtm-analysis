# העלאה ל־GitHub

לאחר חילוץ קובץ ה־ZIP:

```bash
cd dan-panorama-rtm-analysis
git init
git add .
git commit -m "Initial reproducible Dan Panorama RTM analysis"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

לפני ההעלאה מומלץ להריץ:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make test
make quick
```

אין צורך להעלות את `results/local/`; הוא מוחרג ב־`.gitignore`.
