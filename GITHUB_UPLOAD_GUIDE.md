# GitHub Upload Guide

## Step-by-Step Instructions

### Option 1: Using Git Command Line (Recommended)

#### 1. Initialize Git Repository
```bash
git init
```

#### 2. Add All Files
```bash
git add .
```

#### 3. Create Initial Commit
```bash
git commit -m "Initial commit: Complete surveillance system pipeline with YOLOv8"
```

#### 4. Create GitHub Repository
- Go to https://github.com/new
- Repository name: `surveillance-system-yolov8`
- Description: `Complete YOLOv8 pipeline for surveillance system with 5-class detection (Animal, Forest, Militant, UAV-Drone, Wildfire)`
- Choose Public or Private
- **DO NOT** initialize with README (we already have one)
- Click "Create repository"

#### 5. Link to GitHub Repository
```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/surveillance-system-yolov8.git
```

#### 6. Push to GitHub
```bash
git branch -M main
git push -u origin main
```

### Option 2: Using GitHub Desktop

1. Open GitHub Desktop
2. File → Add Local Repository
3. Choose this project folder
4. Click "Create Repository"
5. Click "Publish repository"
6. Choose repository name and visibility
7. Click "Publish Repository"

### Option 3: Using VS Code

1. Open this folder in VS Code
2. Click Source Control icon (left sidebar)
3. Click "Initialize Repository"
4. Stage all changes (+ icon)
5. Enter commit message: "Initial commit: Complete surveillance system pipeline"
6. Click ✓ to commit
7. Click "Publish Branch"
8. Choose repository name and visibility

## Verification

After uploading, verify on GitHub:
- ✅ All Python files visible
- ✅ README.md displays properly
- ✅ Documentation files present
- ✅ .gitignore working (no __pycache__, runs/, etc.)

## Repository Structure on GitHub

```
surveillance-system-yolov8/
├── .gitignore
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── USAGE_GUIDE.md
├── PROJECT_SUMMARY.md
├── example_metrics_output.md
├── requirements.txt
├── train_pipeline.py
├── evaluate_model.py
├── visualize_results.py
├── complete_pipeline.py
├── quick_start.py
├── data.yaml
├── README.dataset.txt
├── README.roboflow.txt
├── train/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Adding a Nice README Badge

Add these badges to the top of your README.md:

```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
```

## Recommended Repository Settings

### Topics (Add these tags)
- `yolov8`
- `object-detection`
- `surveillance`
- `computer-vision`
- `deep-learning`
- `pytorch`
- `machine-learning`
- `detection-pipeline`

### About Section
```
Complete YOLOv8 pipeline for surveillance system detecting Animals, Forest, Militants, UAV-Drones, and Wildfires. Includes training, evaluation, and visualization tools.
```

## Updating Repository Later

```bash
# After making changes
git add .
git commit -m "Description of changes"
git push
```

## Troubleshooting

### Large Files Error
If you get errors about large files:
```bash
# Remove large files from git
git rm --cached runs/ -r
git rm --cached *.pt

# Update .gitignore and commit
git add .gitignore
git commit -m "Update gitignore for large files"
git push
```

### Authentication Issues
If using HTTPS and getting authentication errors:
```bash
# Use personal access token instead of password
# Generate token at: https://github.com/settings/tokens
```

Or use SSH:
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/surveillance-system-yolov8.git
```

## Making Repository Stand Out

1. **Add Screenshots** - Create an `assets/` folder with visualization examples
2. **Add Demo GIF** - Show inference in action
3. **Star the repo** - Encourage others to star
4. **Add Wiki** - Detailed documentation
5. **Enable Issues** - For bug reports and feature requests
6. **Add Projects** - Track development progress

## Next Steps After Upload

1. Share repository link
2. Add to your GitHub profile README
3. Submit to awesome lists (awesome-yolo, awesome-computer-vision)
4. Write a blog post about it
5. Share on social media (LinkedIn, Twitter)

## Example Repository URL

After creation, your repository will be at:
```
https://github.com/YOUR_USERNAME/surveillance-system-yolov8
```

## Clone Command for Others

Others can clone your repository with:
```bash
git clone https://github.com/YOUR_USERNAME/surveillance-system-yolov8.git
cd surveillance-system-yolov8
pip install -r requirements.txt
```
