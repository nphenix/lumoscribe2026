#!/usr/bin/env node
/**
 * prepare-commit-msg hook script
 * 
 * This script generates commit messages using MiniMax API
 * and writes them to the commit message file.
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { MiniMaxProvider } from '../minimax-provider.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Get staged changes as formatted text
 * @returns {string} Formatted change description
 */
function getFormattedChanges() {
  try {
    const files = execSync('git diff --cached --name-only', { encoding: 'utf-8' })
      .trim().split('\n').filter(f => f);
    
    if (files.length === 0) {
      return 'No staged changes';
    }
    
    const diff = execSync('git diff --cached', { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
    
    return `Changed files:\n${files.join('\n')}\n\nGit diff:\n${diff}`;
  } catch (error) {
    return 'Error getting changes: ' + error.message;
  }
}

/**
 * Filter sensitive information from changes
 * @param {string} text - Text to filter
 * @returns {string} Filtered text
 */
function filterSensitiveData(text) {
  // Remove API keys
  let filtered = text.replace(/api[_-]?key["']?\s*[:=]\s*["']?([a-zA-Z0-9-_]+)["']?/gi, 'api_key="[REDACTED]"');
  
  // Remove tokens
  filtered = filtered.replace(/(token|bearer)\s+[a-zA-Z0-9-_]+/gi, '$1 [REDACTED]');
  
  // Remove passwords
  filtered = filtered.replace(/password["']?\s*[:=]\s*["']?[^"'\s]+["']?/gi, 'password="[REDACTED]"');
  
  return filtered;
}

/**
 * Generate commit message and write to file
 * @param {string} commitMsgFile - Path to commit message file
 */
async function generateCommitMessage(commitMsgFile) {
  console.log('🤖 AI is generating commit message...');
  
  try {
    // Get API configuration
    const apiKey = process.env.MINIMAX_API_KEY || 
                   execSync('git config hooks.apiKey', { encoding: 'utf-8' }).trim();
    
    const apiBase = process.env.MINIMAX_API_BASE || 
                    execSync('git config hooks.apiBase', { encoding: 'utf-8' }).trim() || 
                    'https://api.minimax.io/v1';
    
    const model = process.env.GIT_AI_MODEL || 
                  execSync('git config hooks.model', { encoding: 'utf-8' }).trim() || 
                  'MiniMax-M2.1';
    
    // Initialize provider
    const provider = new MiniMaxProvider({
      apiKey,
      apiBase,
      model
    });
    
    // Get and format changes
    const changes = getFormattedChanges();
    const filteredChanges = filterSensitiveData(changes);
    
    // System prompt for commit message generation
    const systemPrompt = `你是一个专业的Git助手，负责生成规范、清晰的提交信息。

提交信息规范：
1. 使用 Conventional Commits 格式：<type>(<scope>): <description>
2. 类型必须是：feat, fix, docs, style, refactor, test, chore
3. 作用域必须是以下之一：git-manager, skill-generator, plan-creator, task-breakdown, change-manager, debug-diagnostic, templates, config
4. 描述简洁明了，不超过72字符
5. 使用中文描述

只输出提交信息本身，不要其他内容。`;
    
    // Generate commit message
    const message = await provider.generateCommitMessage(filteredChanges, systemPrompt);
    
    // Write to commit message file
    fs.writeFileSync(commitMsgFile, message);
    
    console.log('✅ Commit message generated:');
    console.log(`   ${message}`);
    console.log('\n💡 You can edit the commit message in your editor');
    
  } catch (error) {
    console.error('❌ Error generating commit message:', error.message);
    console.log('📝 Please write commit message manually');
  }
}

// Main execution
const commitMsgFile = process.argv[2];

if (!commitMsgFile) {
  console.error('❌ Error: No commit message file specified');
  process.exit(1);
}

if (!fs.existsSync(commitMsgFile)) {
  console.error('❌ Error: Commit message file does not exist');
  process.exit(1);
}

// Check if already has content (amend or no-message commit)
if (fs.statSync(commitMsgFile).size > 0) {
  console.log('📝 Commit message already exists, skipping generation');
  process.exit(0);
}

// Generate commit message
generateCommitMessage(commitMsgFile);