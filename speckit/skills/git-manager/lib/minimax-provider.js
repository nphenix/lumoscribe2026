/**
 * MiniMax AI Provider for Git Commit Message Generation
 * 
 * This module provides integration with MiniMax API (OpenAI compatible mode)
 * for generating intelligent commit messages.
 */

import OpenAI from 'openai';

/**
 * MiniMaxProvider class for generating commit messages using MiniMax API
 */
class MiniMaxProvider {
  /**
   * Create a new MiniMaxProvider instance
   * @param {Object} options - Configuration options
   * @param {string} options.apiKey - MiniMax API key
   * @param {string} [options.apiBase] - API base URL (default: https://api.minimax.io/v1)
   * @param {string} [options.model] - Model name (default: MiniMax-M2.1)
   */
  constructor(options = {}) {
    this.apiKey = options.apiKey || process.env.MINIMAX_API_KEY;
    this.apiBase = options.apiBase || 'https://api.minimax.io/v1';
    this.model = options.model || 'MiniMax-M2.1';
    
    if (!this.apiKey) {
      throw new Error('MiniMax API key is required. Set MINIMAX_API_KEY environment variable.');
    }
    
    this.client = new OpenAI({
      apiKey: this.apiKey,
      baseURL: this.apiBase
    });
  }
  
  /**
   * Get provider name
   * @returns {string} Provider name with model
   */
  getName() {
    return `MiniMax (${this.model})`;
  }
  
  /**
   * Generate a commit message based on code changes
   * @param {string} userPrompt - The prompt describing the changes
   * @param {string} [systemPrompt] - System-level prompt for AI behavior
   * @returns {Promise<string>} Generated commit message
   */
  async generateCommitMessage(userPrompt, systemPrompt = null) {
    const messages = [];
    
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    
    messages.push({
      role: 'user',
      content: this._buildCommitMessagePrompt(userPrompt)
    });
    
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages,
      temperature: 0.3,
      max_tokens: 100
    });
    
    return response.choices[0].message.content.trim();
  }
  
  /**
   * Build the commit message prompt
   * @param {string} changes - Description of code changes
   * @returns {string} Complete prompt for commit message generation
   */
  _buildCommitMessagePrompt(changes) {
    return `分析以下代码变更并生成规范的提交信息：

变更内容：
${changes}

要求：
1. 使用 Conventional Commits 格式：<type>(<scope>): <description>
2. 类型只能是以下之一：feat, fix, docs, style, refactor, test, chore
3. 作用域必须是以下之一：git-manager, skill-generator, plan-creator, task-breakdown, change-manager, debug-diagnostic, templates, config
4. 描述简洁明了，不超过72字符
5. 只输出提交信息，不要其他内容

示例输出：
feat(git-manager): add MiniMax API integration

请生成提交信息：`;}
}

export { MiniMaxProvider };
export default MiniMaxProvider;