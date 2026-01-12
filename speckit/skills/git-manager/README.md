# Git Manager Skill - AI驱动的Git管理

## 快速开始

### 1. 设置环境变量

**Windows PowerShell:**
```powershell
$env:MINIMAX_API_KEY = "your-api-key"
$env:GIT_AI_MODEL = "MiniMax-M2.1"
```

**Windows CMD:**
```cmd
set MINIMAX_API_KEY=your-api-key
set GIT_AI_MODEL=MiniMax-M2.1
```

**Git Bash / WSL / Linux / macOS:**
```bash
export MINIMAX_API_KEY="your-api-key"
export GIT_AI_MODEL="MiniMax-M2.1"
```

### 2. 安装依赖

```powershell
cd speckit/skills/git-manager
npm install
```

### 3. 安装Git Hooks

```powershell
node lib/git-hooks.js install
```

### 4. 配置Git使用MiniMax API

```powershell
git config hooks.commitProvider minimax
git config hooks.apiKey $env:MINIMAX_API_KEY
git config hooks.apiBase "https://api.minimax.io/v1"
```

### 5. 测试提交

```powershell
git add .
git commit
```

AI会自动分析暂存的变更并生成提交信息。