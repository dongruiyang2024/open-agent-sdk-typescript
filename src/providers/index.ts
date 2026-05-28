/**
 * LLM Provider Factory
 *
 * Creates the appropriate provider based on API type configuration.
 */

export type { ApiType, LLMProvider, CreateMessageParams, CreateMessageResponse, NormalizedMessageParam, NormalizedContentBlock, NormalizedTool, NormalizedResponseBlock, ProviderStreamEvent, ReasoningEffort } from './types.js'

export { AnthropicProvider } from './anthropic.js'
export { OpenAIProvider } from './openai.js'
export { OpenAIResponsesProvider } from './openai-responses.js'

import type { ApiType, LLMProvider } from './types.js'
import { AnthropicProvider } from './anthropic.js'
import { OpenAIProvider } from './openai.js'
import { OpenAIResponsesProvider } from './openai-responses.js'

/**
 * Create an LLM provider based on the API type.
 *
 * @param apiType - 'anthropic-messages', 'openai-completions', or 'openai-responses'
 * @param opts - API credentials
 */
export function createProvider(
  apiType: ApiType,
  opts: { apiKey?: string; baseURL?: string },
): LLMProvider {
  switch (apiType) {
    case 'anthropic-messages':
      return new AnthropicProvider(opts)
    case 'openai-completions':
      return new OpenAIProvider(opts)
    case 'openai-responses':
      return new OpenAIResponsesProvider(opts)
    default:
      throw new Error(`Unsupported API type: ${apiType}. Use 'anthropic-messages', 'openai-completions', or 'openai-responses'.`)
  }
}
