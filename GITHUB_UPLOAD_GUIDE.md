# GitHub Private Repository Setup Guide

## Quick Summary

This folder (`ProteusQUBO_GitHub`) contains **clean model framework code only** ready for GitHub upload.

- **Total files**: 16 files
- **Size**: ~236 KB
- **Sensitive info**: Removed ✓
- **Original files**: Not modified ✓

---

## Upload Steps

### Step 1: Navigate to the folder

```bash
cd ProteusQUBO_GitHub
```

### Step 2: Initialize Git repository

```bash
git init
git add .
git commit -m "Initial commit: ProteusQUBO model framework for peer review"
```

### Step 3: Create private repository on GitHub

1. Go to https://github.com/new
2. Set repository name: `ProteusQUBO`
3. **Important: Select "Private"** ✓
4. Do NOT initialize with README (we already have one)
5. Click "Create repository"

### Step 4: Push to GitHub

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/ProteusQUBO.git
git branch -M main
git push -u origin main
```

### Step 5: Add reviewers as collaborators

1. Go to repository settings: `https://github.com/YOUR_USERNAME/ProteusQUBO/settings/access`
2. Click "Add people"
3. Enter reviewer's GitHub username or email
4. Select permission level: **"Read"** (recommended for reviewers)
5. Send invitation

---

## What's Included

### ✅ Included Files (16 files)

```
ProteusQUBO_GitHub/
├── Block_bAE/                    (5 Python files)
│   ├── model_gru_transformer.py
│   ├── inference_latent.py
│   ├── latent_to_smiles.py
│   ├── train_gruencoder_transformerdecoder.py
│   └── datamodule.py
├── Qmol_FM/                      (6 Python files)
│   ├── opt.py
│   ├── convert_fm_to_qubo.py
│   ├── convert_ising_to_qubo.py
│   ├── sample.py               # ✓ Sensitive credentials removed
│   ├── utils_Qmol_FM.py
│   └── hubo.py
├── RBM/                          (2 Python files)
│   ├── train_rbm.py
│   └── sample_from_rbm.py
├── .gitignore                    # Prevents accidental data upload
├── README.md                     # English documentation
└── requirements.txt              # Python dependencies
```

### ❌ NOT Included (Intentionally excluded)

- Training data (`.h5`, `.csv` files)
- Model weights (`.ckpt`, `.pth` files)
- Experimental results (Results/, Val/ folders)
- Large images (`.png` files)
- Log files (`.log` files)
- IDE configurations (`.vscode`, `.claude`)

---

## Sharing with Reviewers

### Option 1: Direct Collaboration (Recommended)

Add reviewers as collaborators with **Read access**:
- They can view and clone the repository
- They cannot modify or push changes
- Access can be revoked after review

### Option 2: Generate Access Token

If reviewer doesn't have GitHub account:

1. Go to repository Settings → Security → Deploy keys
2. Create a read-only deploy key
3. Share the key with reviewer

### Option 3: Archive Download

As a backup option:
1. Click "Code" → "Download ZIP"
2. Send ZIP file via secure channel

---

## Security Checklist

- [x] Sensitive credentials removed from `sample.py`
- [x] Repository set to **Private**
- [x] `.gitignore` configured to block data files
- [x] No large files (all files < 1 MB)
- [x] No training data or model weights included
- [x] Original project files untouched

---

## After Review

### To update the repository:

```bash
cd ProteusQUBO_GitHub
# Make changes...
git add .
git commit -m "Update: description of changes"
git push
```

### To revoke reviewer access:

1. Go to Settings → Collaborators
2. Click "Remove" next to reviewer's name

### To delete repository:

1. Go to Settings → Danger Zone
2. Click "Delete this repository"
3. Type repository name to confirm

---

## Troubleshooting

### Error: "Repository already exists"

The repository name is taken. Choose a different name or delete the existing one.

### Error: "Permission denied"

Make sure you're authenticated:
```bash
# Use GitHub CLI (recommended)
gh auth login

# Or configure Git credentials
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Large file warning

If Git complains about large files:
```bash
# Check file sizes
find . -type f -size +10M -not -path "./.git/*"

# Remove large files from staging
git rm --cached path/to/large/file
```

---

## Contact Information

For questions about the code or repository setup, please contact:
- [Add your contact information]

---

**Last updated**: 2025-11-19
