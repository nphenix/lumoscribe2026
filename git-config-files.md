# Git配置文件集合

## .gitignore

```gitignore
# AI相关
*.model
*.checkpoint
*.log
.cache/
models/
datasets/

# 开发环境
node_modules/
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# 构建产物
dist/
build/
out/
coverage/

# IDE配置
.vscode/
.idea/
*.swp
*.swo

# 系统文件
.DS_Store
Thumbs.db
*.tmp

# 临时文件
*.tmp
*.temp
*.log

# 日志文件
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# 操作系统生成的文件
*.sublime-workspace
*.sublime-project
*.vscode
*.atom
```

## .gitattributes

```gitattributes
# 设置换行符处理
*.md text eol=lf
*.json text eol=lf
*.js text eol=lf
*.ts text eol=lf
*.py text eol=lf
*.yml text eol=lf
*.yaml text eol=lf

# 二进制文件
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.ico binary
*.pdf binary
*.zip binary
*.tar.gz binary

# 配置文件
*.gitignore text eol=lf
*.gitattributes text eol=lf
*.editorconfig text eol=lf

# 文档文件
*.md text eol=lf
*.txt text eol=lf
*.rst text eol=lf

# 代码文件
*.js text eol=lf
*.ts text eol=lf
*.tsx text eol=lf
*.jsx text eol=lf
*.py text eol=lf
*.java text eol=lf
*.c text eol=lf
*.cpp text eol=lf
*.h text eol=lf
*.hpp text eol=lf
*.cs text eol=lf
*.php text eol=lf
*.rb text eol=lf
*.go text eol=lf
*.rust text eol=lf
*.rs text eol=lf
```

## .editorconfig

```editorconfig
# EditorConfig is awesome: https://EditorConfig.org

# top-most EditorConfig file
root = true

# All files
[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 2

# Markdown files
[*.md]
trim_trailing_whitespace = false

# JSON files
[*.json]
indent_size = 2

# YAML files
[*.yml]
indent_size = 2

# Python files
[*.py]
indent_size = 4

# JavaScript/TypeScript files
[*.{js,ts,jsx,tsx}]
indent_size = 2

# Java files
[*.java]
indent_size = 4

# C/C++ files
[*.{c,cpp,h,hpp}]
indent_size = 4

# C# files
[*.cs]
indent_size = 4

# PHP files
[*.php]
indent_size = 4

# Ruby files
[*.rb]
indent_size = 2

# Go files
[*.go]
indent_size = 4

# Rust files
[*.{rs,rust}]
indent_size = 4
```

## .gitmessage.template

```text
# <type>(<scope>): <subject>
#
# <body>
#
# <footer>
#
# Type: feat, fix, docs, style, refactor, test, chore
# Scope: skill-generator, plan-creator, task-breakdown, change-manager, debug-diagnostic, templates
# Body: 72 characters max
# Footer: JIRA ticket, GitHub issue, or breaking changes
```

## husky配置（package.json）

```json
{
  "husky": {
    "hooks": {
      "pre-commit": "lint-staged",
      "commit-msg": "commitlint -E HUSKY_GIT_PARAMS"
    }
  },
  "lint-staged": {
    "*.{js,ts,jsx,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{json,md,yml,yaml}": [
      "prettier --write"
    ]
  }
}
```

## commitlint配置（commitlint.config.js）

```javascript
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat',     // 新功能
        'fix',      // 修复bug
        'docs',     // 文档更新
        'style',    // 代码格式
        'refactor', // 重构
        'test',     // 测试相关
        'chore'     // 构建或辅助工具更新
      ]
    ],
    'scope-enum': [
      2,
      'always',
      [
        'skill-generator',
        'plan-creator',
        'task-breakdown',
        'change-manager',
        'debug-diagnostic',
        'templates',
        'config',
        'docs'
      ]
    ],
    'subject-case': [2, 'never', ['upper-case']],
    'subject-max-length': [2, 'always', 72],
    'body-max-line-length': [2, 'always', 100]
  }
};
```

## ESLint配置（.eslintrc.js）

```javascript
module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true
  },
  extends: [
    'eslint:recommended',
    '@typescript-eslint/recommended'
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 12,
    sourceType: 'module'
  },
  plugins: [
    '@typescript-eslint'
  ],
  rules: {
    'indent': ['error', 2],
    'linebreak-style': ['error', 'unix'],
    'quotes': ['error', 'single'],
    'semi': ['error', 'always'],
    'no-unused-vars': ['error', { 'argsIgnorePattern': '^_' }],
    'no-console': 'warn'
  }
};
```

## Prettier配置（.prettierrc）

```json
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 80,
  "tabWidth": 2,
  "useTabs": false,
  "endOfLine": "lf"
}
```

## GitHub Actions工作流

### CI流程（.github/workflows/ci.yml）

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        node-version: [16.x, 18.x, 20.x]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Use Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
    
    - run: npm ci
    - run: npm run lint
    - run: npm run test
    - run: npm run build
```

### 发布流程（.github/workflows/release.yml）

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        registry-url: 'https://registry.npmjs.org'
    
    - run: npm ci
    - run: npm run build
    - run: npm test
    
    - name: Create Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false
```

## Git Hooks示例

### pre-commit钩子

```bash
#!/bin/sh
# 检查提交信息格式
commit_regex='^(feat|fix|docs|style|refactor|test|chore)\((skill-generator|plan-creator|task-breakdown|change-manager|debug-diagnostic|templates|config|docs)\): .{1,72}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "错误：提交信息格式不符合规范"
    echo "格式：type(scope): description"
    echo "示例：feat(skill-generator): add new template"
    exit 1
fi

# 检查是否有未跟踪的文件
if [ -n "$(git status --porcelain)" ]; then
    echo "警告：存在未跟踪的文件"
    git status --porcelain
fi
```

### commit-msg钩子

```bash
#!/bin/sh
# 验证提交信息
commit_regex='^(feat|fix|docs|style|refactor|test|chore)\((skill-generator|plan-creator|task-breakdown|change-manager|debug-diagnostic|templates|config|docs)\): .{1,72}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "错误：提交信息格式不符合规范"
    echo "格式：type(scope): description"
    echo "示例：feat(skill-generator): add new template"
    exit 1
fi
```

## 使用说明

1. **复制配置文件**：将上述配置文件复制到项目根目录
2. **安装依赖**：
   ```bash
   npm install --save-dev husky lint-staged @commitlint/cli @commitlint/config-conventional
   ```
3. **初始化Husky**：
   ```bash
   npx husky install
   ```
4. **添加钩子**：
   ```bash
   npx husky add .husky/pre-commit "lint-staged"
   npx husky add .husky/commit-msg 'npx --no -- commitlint --edit "$1"'
   ```
5. **设置提交模板**：
   ```bash
   git config commit.template .gitmessage.template
   ```

## 注意事项

- 根据项目实际技术栈调整配置
- CI/CD配置需要根据具体需求修改
- 团队成员需要安装相应的开发工具
- 定期更新依赖版本以保持安全性