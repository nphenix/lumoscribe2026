<#
.SYNOPSIS
    同步 speckit/ 目录到 .cursor/ 和 .roo/

.DESCRIPTION
    本脚本将 speckit/ 目录中的内容同步到：
    - .cursor/ (Cursor)
    - .roo/ (Roo Code)
    - .trae/ (Trae IDE)

    源目录：speckit/
    - speckit/skills/ - 技能定义库
    - speckit/templates/ - 项目模板库
    - speckit/specs/ - 技术规范和设计文档

    同步策略：
    - 使用 Robocopy 镜像模式（/MIR）确保目标目录与源完全一致
    - 自动创建目标目录结构
    - 删除目标中存在但源中不存在的文件

    PowerShell 兼容性：
    - 支持 Windows PowerShell 5.1 与 PowerShell 7+

.EXAMPLE
    .\sync-skills.ps1
    执行同步操作（同时同步到 Cursor 和 Roo Code）

.EXAMPLE
    .\sync-skills.ps1 -Target cursor
    只同步到 Cursor

.EXAMPLE
    .\sync-skills.ps1 -Target roo
    只同步到 Roo Code

.EXAMPLE
    .\sync-skills.ps1 -WhatIf
    预览同步操作，不实际执行

.EXAMPLE
    .\sync-skills.ps1 -Content skills
    只同步 skills 目录（可选值: skills, templates, specs, all）

.NOTES
    作者：AI Development Team
    版本：2.1.0
    日期：2026-01-13
#>

[CmdletBinding()]
param(
    # 目标工具：cursor, roo, 或 all（默认）
    [ValidateSet('cursor', 'roo', 'trae', 'all')]
    [string]$Target = 'all',

    # 同步内容：skills, templates, specs, 或 all（默认）
    [ValidateSet('skills', 'templates', 'specs', 'all')]
    [string]$Content = 'all',

    # 预览模式，不实际执行复制操作
    [switch]$WhatIf,

    # 详细输出
    [switch]$EnableVerbose
)

# 路径配置（相对于项目根目录）
$ProjectRoot = (Get-Item $PSScriptRoot).Parent.FullName
$SourceBase = Join-Path $ProjectRoot "speckit"

# 设置控制台编码为 UTF-8，确保中文正常显示
try {
    # 尝试设置控制台代码页为 65001 (UTF-8)
    if ([System.Console]::OutputEncoding.CodePage -ne 65001) {
        [System.Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    }
    # 设置 PowerShell 输出管道编码
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    Write-Warning "无法设置 UTF-8 编码，中文显示可能异常: $_"
}

# 定义要同步的源目录
$SourceDirs = @{
    skills = @{
        Source = Join-Path $SourceBase "skills"
        CursorPath = Join-Path $ProjectRoot ".cursor/skills"
        RooPath = Join-Path $ProjectRoot ".roo/skills"
        TraePath = Join-Path $ProjectRoot ".trae/skills"
    }
    templates = @{
        Source = Join-Path $SourceBase "templates"
        CursorPath = Join-Path $ProjectRoot ".cursor/templates"
        RooPath = Join-Path $ProjectRoot ".roo/templates"
        TraePath = Join-Path $ProjectRoot ".trae/templates"
    }
    specs = @{
        Source = Join-Path $SourceBase "specs"
        CursorPath = Join-Path $ProjectRoot ".cursor/specs"
        RooPath = Join-Path $ProjectRoot ".roo/specs"
        TraePath = Join-Path $ProjectRoot ".trae/specs"
    }
}

# 根据参数筛选要同步的内容
if ($Content -eq 'all') {
    $SelectedContent = @('skills', 'templates', 'specs')
}
else {
    $SelectedContent = @($Content)
}

# 目标目录配置
$Targets = @{
    cursor = @{
        Path = Join-Path $ProjectRoot ".cursor"
        Name = "Cursor"
    }
    roo = @{
        Path = Join-Path $ProjectRoot ".roo"
        Name = "Roo Code"
    }
    trae = @{
        Path = Join-Path $ProjectRoot ".trae"
        Name = "Trae IDE"
    }
}

# 根据参数筛选目标
if ($Target -eq 'cursor') {
    $SelectedTargets = @($Targets.cursor)
}
elseif ($Target -eq 'roo') {
    $SelectedTargets = @($Targets.roo)
}
elseif ($Target -eq 'trae') {
    $SelectedTargets = @($Targets.trae)
}
else {
    $SelectedTargets = @($Targets.cursor, $Targets.roo, $Targets.trae)
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Speckit 同步工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "同步内容: $($SelectedContent -join ', ')" -ForegroundColor White
Write-Host "目标: $(($SelectedTargets | ForEach-Object { $_.Name }) -join ', ')" -ForegroundColor White
Write-Host "模式: $(if ($WhatIf) { '预览 (WhatIf)' } else { '执行' })" -ForegroundColor White
Write-Host ""

$SyncResults = @()

foreach ($contentType in $SelectedContent) {
    $sourceInfo = $SourceDirs[$contentType]
    $SourcePath = $sourceInfo.Source
    
    # 验证源目录存在
    if (-not (Test-Path $SourcePath -PathType Container)) {
        Write-Warning "源目录不存在，跳过: $SourcePath"
        continue
    }
    
    foreach ($t in $SelectedTargets) {
        $TargetPath = $t.Path
        $TargetName = $t.Name

        # --- 特殊处理 Cursor/Trae 的 Skills 同步 (转换为 .mdc/.md) ---
        if ($contentType -eq "skills" -and ($TargetName -eq "Cursor" -or $TargetName -eq "Trae IDE")) {
            $RuleExt = if ($TargetName -eq "Cursor") { ".mdc" } else { ".md" }
            Write-Host "  [skills] 转换为 $TargetName Rules ($RuleExt)..." -ForegroundColor Cyan
            $RulesDir = Join-Path $TargetPath "rules"
            
            # 确保 rules 目录存在
            if (-not $WhatIf) {
                if (-not (Test-Path $RulesDir)) { New-Item -ItemType Directory -Path $RulesDir -Force | Out-Null }
            }

            # 遍历所有 skill
            $Skills = Get-ChildItem -Path $SourcePath -Directory
            foreach ($skill in $Skills) {
                $SkillName = $skill.Name
                $SkillFile = Join-Path $skill.FullName "SKILL.md"
                
                if (Test-Path $SkillFile) {
                    Write-Host "    处理技能: $SkillName" -ForegroundColor Gray
                    
                    # 读取内容
                    $SkillContent = Get-Content $SkillFile -Raw -Encoding UTF8
                    
                    # 尝试提取 description
                    $Description = "Agent Skill: $SkillName"
                    if ($SkillContent -match "description:\s*[`"']?([^`"'\r\n]+)[`"']?") {
                        $Description = $matches[1]
                    }

                    # 构建 Header
                    $Header = ""
                    if ($TargetName -eq "Cursor") {
                        $Header = @"
---
description: $Description
globs: 
alwaysApply: false
---

"@
                    } else {
                        # Trae 格式
                         $Header = @"
---
description: $Description
---

"@
                    }

                    $FinalContent = $Header + $SkillContent
                    $TargetRulePath = Join-Path $RulesDir "$SkillName$RuleExt"

                    if (-not $WhatIf) {
                        Set-Content -Path $TargetRulePath -Value $FinalContent -Encoding UTF8
                    }
                }
            }
            
            # 清理旧的 skills 目录（如果存在）
            $OldSkillsPath = Join-Path $TargetPath "skills"
            if (Test-Path $OldSkillsPath) {
                Write-Host "    清理旧的 $OldSkillsPath 目录..." -ForegroundColor Yellow
                if (-not $WhatIf) { Remove-Item -Path $OldSkillsPath -Recurse -Force }
            }

            continue # 跳过后续的标准 Robocopy 逻辑
        }
        # -----------------------------------------------------------

        $ContentTargetPath = if ($TargetPath -like "*.cursor*") {
            $sourceInfo.CursorPath
        }
        else {
            $sourceInfo.RooPath
        }
        
        Write-Host "[$contentType] 同步到 $TargetName..." -ForegroundColor Green
        
        # 确保目标目录存在
        if (-not (Test-Path $ContentTargetPath -PathType Container)) {
            Write-Host "  创建目录: $ContentTargetPath" -ForegroundColor Yellow
            if (-not $WhatIf) {
                New-Item -ItemType Directory -Path $ContentTargetPath -Force | Out-Null
            }
        }
        
        # 同步 Constitution
        $SourceConstitution = Join-Path $SourceBase "constitution.md"
        if (Test-Path $SourceConstitution) {
            Write-Host "  同步 Constitution..." -ForegroundColor Cyan
            $ConstitutionContent = Get-Content $SourceConstitution -Raw -Encoding UTF8
            
            if ($TargetName -eq "Cursor") {
                $TargetRuleFile = Join-Path $TargetPath "rules/constitution.mdc"
                $RuleDir = Join-Path $TargetPath "rules"
                
                # Cursor MDC Header
                $MDCHeader = @"
---
description: 项目宪章 - 核心规则（AI助手必须遵循）
globs: 
alwaysApply: true
---

"@
                $NewContent = $MDCHeader + $ConstitutionContent
                
                if (-not $WhatIf) {
                    if (-not (Test-Path $RuleDir)) { New-Item -ItemType Directory -Path $RuleDir -Force | Out-Null }
                    [System.IO.File]::WriteAllText($TargetRuleFile, $NewContent)
                }
            }
            elseif ($TargetName -eq "Trae IDE") {
                $TargetRuleFile = Join-Path $TargetPath "rules/constitution.md"
                $RuleDir = Join-Path $TargetPath "rules"
                
                # Trae Header
                $Header = @"
---
description: 项目宪章 - 核心规则（AI助手必须遵循）
---

"@
                $NewContent = $Header + $ConstitutionContent
                
                if (-not $WhatIf) {
                    if (-not (Test-Path $RuleDir)) { New-Item -ItemType Directory -Path $RuleDir -Force | Out-Null }
                    [System.IO.File]::WriteAllText($TargetRuleFile, $NewContent)
                }
            }
            elseif ($TargetName -eq "Roo Code") {
                $TargetRuleFile = Join-Path $TargetPath "rules/01-constitution.md"
                $RuleDir = Join-Path $TargetPath "rules"
                
                if (-not $WhatIf) {
                    if (-not (Test-Path $RuleDir)) { New-Item -ItemType Directory -Path $RuleDir -Force | Out-Null }
                    Copy-Item -Path $SourceConstitution -Destination $TargetRuleFile -Force
                }
            }
        }

        # 同步 Directory Structure
        $SourceDirStruct = Join-Path $SourceBase "directory-structure.md"
        if (Test-Path $SourceDirStruct) {
            Write-Host "  同步 Directory Structure..." -ForegroundColor Cyan
            $DirStructContent = Get-Content $SourceDirStruct -Raw -Encoding UTF8
            
            if ($TargetName -eq "Cursor") {
                $TargetRuleFile = Join-Path $TargetPath "rules/directory-structure.mdc"
                $RuleDir = Join-Path $TargetPath "rules"
                
                # Cursor MDC Header
                $MDCHeader = @"
---
description: 目录结构与文件命名规范
globs: 
alwaysApply: true
---

"@
                $NewContent = $MDCHeader + $DirStructContent
                
                if (-not $WhatIf) {
                    if (-not (Test-Path $RuleDir)) { New-Item -ItemType Directory -Path $RuleDir -Force | Out-Null }
                    [System.IO.File]::WriteAllText($TargetRuleFile, $NewContent)
                }
            }
            elseif ($TargetName -eq "Trae IDE") {
                $TargetRuleFile = Join-Path $TargetPath "rules/directory-structure.md"
                $RuleDir = Join-Path $TargetPath "rules"
                
                # Trae Header
                $Header = @"
---
description: 目录结构与文件命名规范
---

"@
                $NewContent = $Header + $DirStructContent
                
                if (-not $WhatIf) {
                    if (-not (Test-Path $RuleDir)) { New-Item -ItemType Directory -Path $RuleDir -Force | Out-Null }
                    Set-Content -Path $TargetRuleFile -Value $NewContent -Encoding UTF8
                }
            }
            elseif ($TargetName -eq "Roo Code") {
                $TargetRuleFile = Join-Path $TargetPath "rules/directory-structure.md"
                $RuleDir = Join-Path $TargetPath "rules"
                
                if (-not $WhatIf) {
                    if (-not (Test-Path $RuleDir)) { New-Item -ItemType Directory -Path $RuleDir -Force | Out-Null }
                    Copy-Item -Path $SourceDirStruct -Destination $TargetRuleFile -Force
                }
            }
        }

        # 同步 Prompt Standards
        $SourcePromptStandards = Join-Path $SourceBase "specs/prompt-management-standards.md"
        if (Test-Path $SourcePromptStandards) {
            Write-Host "  同步 Prompt Standards..." -ForegroundColor Cyan
            $PromptStandardsContent = Get-Content $SourcePromptStandards -Raw -Encoding UTF8
            
            if ($TargetName -eq "Cursor") {
                $TargetRuleFile = Join-Path $TargetPath "rules/prompt-management.mdc"
                $RuleDir = Join-Path $TargetPath "rules"
                
                # Cursor MDC Header
                $MDCHeader = @"
---
description: 提示词管理与开发规范（禁止硬编码，使用Registry）
globs: src/**/*.py
alwaysApply: true
---

"@
                $NewContent = $MDCHeader + $PromptStandardsContent
                
                if (-not $WhatIf) {
                    if (-not (Test-Path $RuleDir)) { New-Item -ItemType Directory -Path $RuleDir -Force | Out-Null }
                    [System.IO.File]::WriteAllText($TargetRuleFile, $NewContent)
                }
            }
            elseif ($TargetName -eq "Trae IDE") {
                $TargetRuleFile = Join-Path $TargetPath "rules/prompt-management.md"
                $RuleDir = Join-Path $TargetPath "rules"
                
                # Trae Header
                $Header = @"
---
description: 提示词管理与开发规范（禁止硬编码，使用Registry）
---

"@
                $NewContent = $Header + $PromptStandardsContent
                
                if (-not $WhatIf) {
                    if (-not (Test-Path $RuleDir)) { New-Item -ItemType Directory -Path $RuleDir -Force | Out-Null }
                    Set-Content -Path $TargetRuleFile -Value $NewContent -Encoding UTF8
                }
            }
            elseif ($TargetName -eq "Roo Code") {
                $TargetRuleFile = Join-Path $TargetPath "rules/02-prompt-management.md"
                $RuleDir = Join-Path $TargetPath "rules"
                
                if (-not $WhatIf) {
                    if (-not (Test-Path $RuleDir)) { New-Item -ItemType Directory -Path $RuleDir -Force | Out-Null }
                    Copy-Item -Path $SourcePromptStandards -Destination $TargetRuleFile -Force
                }
            }
        }

        # 清理 commands 目录（强制使用自然语言）
        $CommandsDir = Join-Path $TargetPath "commands"
        if (Test-Path $CommandsDir) {
            Write-Host "  清理旧的 commands 目录..." -ForegroundColor Yellow
            if (-not $WhatIf) {
                Remove-Item -Path $CommandsDir -Recurse -Force
            }
        }
        
        # 同步参数
        $RobocopyParams = @(
            $SourcePath,          # 源目录
            $ContentTargetPath,   # 目标目录
            "/MIR",               # 镜像模式
            "/MT:4",              # 多线程复制（4线程）
            "/R:3",               # 重试次数
            "/W:5",               # 重试等待时间（秒）
            "/LOG+:sync.log",     # 日志文件（追加）
            "/NFL",               # 不记录文件名
            "/NDL",               # 不记录目录名
            "/NJH",               # 不记录作业标头
            "/NJS"                # 不记录作业摘要
        )
        
        if ($WhatIf) {
            $RobocopyParams += "/L"  # 列表模式（仅预览）
        }
        
        if ($EnableVerbose) {
            $RobocopyParams += "/V"  # 详细输出
        }
        
        try {
            $result = Start-Process -FilePath "robocopy" -ArgumentList $RobocopyParams -NoNewWindow -Wait -PassThru
            $exitCode = $result.ExitCode
            
            $SyncResults += @{
                Content = $contentType
                Target = $TargetName
                ExitCode = $exitCode
            }
            
            # 显示结果
            switch ($exitCode) {
                0 { Write-Host "  [OK] No changes needed" -ForegroundColor Green }
                1 { Write-Host "  [OK] Copy completed" -ForegroundColor Green }
                2 { Write-Host "  [WARN] Some files were not copied (extra files exist in destination)" -ForegroundColor Yellow }
                3 { Write-Host "  [OK] Copy completed with some skipped files" -ForegroundColor Green }
                { $_ -ge 8 } { Write-Host "  [ERROR] Copy failed (Exit Code: $exitCode)" -ForegroundColor Red }
                default { Write-Host "  [INFO] Sync completed (Exit Code: $exitCode)" -ForegroundColor White }
            }
        }
        catch {
            Write-Error "  [$contentType] 同步到 $TargetName 失败: $_"
            $SyncResults += @{
                Content = $contentType
                Target = $TargetName
                ExitCode = -1
            }
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  同步完成" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "日志已保存: $(Join-Path $ProjectRoot 'sync.log')" -ForegroundColor Gray
Write-Host ""
Write-Host "建议下一步:" -ForegroundColor White
Write-Host "  1. 检查变更: git diff .cursor .roo .trae" -ForegroundColor Gray
Write-Host "  2. 提交变更: git add .cursor .roo .trae ; git commit -m 'chore: sync'" -ForegroundColor Gray

exit 0

