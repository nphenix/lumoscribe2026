/**
 * Git Manager Skill - Main Entry Point
 * 
 * AI-powered Git management for lumoscribe2026
 * 
 * @module git-manager
 */

export { MiniMaxProvider } from './minimax-provider.js';
export { GitHooksManager } from './git-hooks.js';

/**
 * Create a new Git Manager instance
 * @param {Object} options - Configuration options
 * @returns {GitHooksManager} Configured GitHooksManager instance
 */
function createGitManager(options = {}) {
  return new GitHooksManager(options);
}

/**
 * Quick setup function for Git hooks installation
 * @param {Object} config - Configuration object
 * @param {string} config.apiKey - MiniMax API key
 * @param {string} [config.model] - Model name (default: MiniMax-M2.1)
 * @returns {Promise<boolean>} True if setup successful
 */
async function quickSetup(config) {
  const { apiKey, model = 'MiniMax-M2.1' } = config;
  
  if (!apiKey) {
    throw new Error('API key is required');
  }
  
  // Set environment variables
  process.env.MINIMAX_API_KEY = apiKey;
  process.env.GIT_AI_MODEL = model;
  
  // Install hooks
  const manager = new GitHooksManager();
  return manager.install();
}

/**
 * Generate commit message for current changes
 * @param {Object} options - Options for message generation
 * @param {string} [options.apiKey] - API key (optional, uses env var)
 * @returns {Promise<string>} Generated commit message
 */
async function generateCommitMessage(options = {}) {
  const provider = new MiniMaxProvider({
    apiKey: options.apiKey || process.env.MINIMAX_API_KEY
  });
  
  const { execSync } = await import('child_process');
  
  const diff = execSync('git diff --cached', { encoding: 'utf-8' });
  const files = execSync('git diff --cached --name-only', { encoding: 'utf-8' });
  
  const changes = `Changed files:\n${files}\n\nGit diff:\n${diff}`;
  
  const systemPrompt = `你是一个专业的Git助手，负责生成规范、清晰的提交信息。

提交信息规范：
1. 使用 Conventional Commits 格式：<type>(<scope>): <description>
2. 类型必须是：feat, fix, docs, style, refactor, test, chore
3. 作用域必须是以下之一：git-manager, skill-generator, plan-creator, task-breakdown, change-manager, debug-diagnostic, templates, config
4. 描述简洁明了，不超过72字符
5. 使用中文描述

只输出提交信息本身，不要其他内容。`;
  
  return provider.generateCommitMessage(changes, systemPrompt);
}

export default {
  createGitManager,
  quickSetup,
  generateCommitMessage,
  MiniMaxProvider,
  GitHooksManager
};