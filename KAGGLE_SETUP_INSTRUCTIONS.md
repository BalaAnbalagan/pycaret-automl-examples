# Kaggle API Setup Instructions

## Step 1: Get Your Kaggle API Key

1. **Go to Kaggle**: https://www.kaggle.com/
2. **Login or Sign Up** with your email: `bala.anbalagan@sjsu.edu`
   - If new account, verify your email first
3. **Navigate to Account Settings**:
   - Click your profile picture (top right)
   - Click "Settings"
4. **Create API Token**:
   - Scroll down to "API" section
   - Click "Create New API Token"
   - A file named `kaggle.json` will download automatically

## Step 2: Install the API Key

The `kaggle.json` file contains your credentials. It looks like this:
```json
{
  "username": "your_kaggle_username",
  "key": "your_api_key_here"
}
```

### Option A: Automatic Installation (Run this command)

```bash
# Move the downloaded kaggle.json to the correct location
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Option B: Manual Installation

1. Open Finder
2. Press `Cmd + Shift + G` (Go to Folder)
3. Type: `~/.kaggle`
4. Copy your downloaded `kaggle.json` file here
5. Open Terminal and run:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Step 3: Verify Setup

Run this command to verify it works:
```bash
/Users/banbalagan/Library/Python/3.9/bin/kaggle datasets list
```

If you see a list of datasets, you're ready! ✅

## Step 4: Download All Datasets

Once verified, run this script to download all 6 datasets:

```bash
cd /Users/banbalagan/Projects/pycaret-automl-examples
./download_datasets.sh
```

---

## Quick Reference

| What | Command |
|------|---------|
| Test Kaggle API | `/Users/banbalagan/Library/Python/3.9/bin/kaggle datasets list` |
| Download dataset | `/Users/banbalagan/Library/Python/3.9/bin/kaggle datasets download -d DATASET_PATH` |
| List your datasets | `/Users/banbalagan/Library/Python/3.9/bin/kaggle datasets list -m` |

---

## Troubleshooting

### Error: "Could not find kaggle.json"
- Make sure `kaggle.json` is in `~/.kaggle/` directory
- Check permissions: `ls -la ~/.kaggle/kaggle.json` should show `-rw-------`

### Error: "403 Forbidden"
- Verify your Kaggle account email is confirmed
- Try regenerating API token on Kaggle website
- Make sure you're logged into Kaggle on the website

### Error: "kaggle: command not found"
- Use full path: `/Users/banbalagan/Library/Python/3.9/bin/kaggle`
- Or add to PATH: `export PATH=$PATH:/Users/banbalagan/Library/Python/3.9/bin`

---

**After completing these steps, let me know and I'll download all 6 datasets for you!**
