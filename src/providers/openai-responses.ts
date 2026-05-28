/**
 * OpenAI Responses API Provider
 *
 * Converts between the SDK's internal Anthropic-like message format
 * and OpenAI's Responses API item format.
 */

import type {
  LLMProvider,
  CreateMessageParams,
  CreateMessageResponse,
  NormalizedMessageParam,
  NormalizedContentBlock,
  NormalizedTool,
  NormalizedResponseBlock,
  ProviderStreamEvent,
  ReasoningEffort,
} from './types.js'

type OpenAIResponsesInputItem =
  | { role: 'user' | 'assistant'; content: string }
  | { type: 'function_call'; call_id: string; name: string; arguments: string }
  | { type: 'function_call_output'; call_id: string; output: string }

interface OpenAIResponsesTool {
  type: 'function'
  name: string
  description: string
  parameters: Record<string, any>
  strict: boolean
}

interface OpenAIResponsesMessageItem {
  type: 'message'
  role?: 'assistant'
  content?: Array<{ type: string; text?: string }>
}

interface OpenAIResponsesFunctionCallItem {
  type: 'function_call'
  id?: string
  call_id?: string
  name: string
  arguments: string
}

interface OpenAIResponsesResponse {
  id?: string
  status?: string
  incomplete_details?: { reason?: string } | null
  output?: Array<OpenAIResponsesMessageItem | OpenAIResponsesFunctionCallItem | Record<string, any>>
  usage?: {
    input_tokens?: number
    output_tokens?: number
  }
}

export class OpenAIResponsesProvider implements LLMProvider {
  readonly apiType = 'openai-responses' as const
  private apiKey: string
  private baseURL: string

  constructor(opts: { apiKey?: string; baseURL?: string }) {
    this.apiKey = opts.apiKey || ''
    this.baseURL = (opts.baseURL || 'https://api.openai.com/v1').replace(/\/$/, '')
  }

  async createMessage(params: CreateMessageParams): Promise<CreateMessageResponse> {
    let finalResponse: CreateMessageResponse | undefined

    for await (const event of this.streamMessage(params)) {
      if (event.type === 'final_response') {
        finalResponse = event.response
      }
    }

    if (!finalResponse) {
      throw new Error('OpenAI Responses API stream ended without a final response')
    }

    return finalResponse
  }

  async *streamMessage(params: CreateMessageParams): AsyncGenerator<ProviderStreamEvent> {
    const response = await this.postResponse(params)
    const contentType = response.headers.get('content-type') || ''
    if (!contentType.includes('text/event-stream')) {
      const data = (await response.json()) as OpenAIResponsesResponse
      yield { type: 'final_response', response: this.convertResponse(data) }
      return
    }

    const textParts: string[] = []
    let completedResponse: OpenAIResponsesResponse | undefined

    for await (const event of this.streamSseEvents(response)) {
      if (event.type === 'response.output_text.delta' && typeof event.delta === 'string') {
        textParts.push(event.delta)
        yield { type: 'partial_text', text: textParts.join('') }
      } else if (event.type === 'response.completed' && event.response) {
        completedResponse = event.response as OpenAIResponsesResponse
      } else if (event.type === 'response.failed') {
        const message = event.response?.error?.message || event.error?.message || 'Responses stream failed'
        throw new Error(message)
      } else if (event.type === 'error') {
        throw new Error(event.error?.message || event.message || 'Responses stream error')
      }
    }

    yield {
      type: 'final_response',
      response: this.convertResponse(this.buildStreamedResponse(textParts, completedResponse)),
    }
  }

  private async postResponse(params: CreateMessageParams): Promise<Response> {
    const tools = params.tools ? this.convertTools(params.tools) : undefined
    const body: Record<string, any> = {
      model: params.model,
      max_output_tokens: params.maxTokens,
      input: this.convertMessages(params.messages),
      stream: true,
    }

    if (params.system) {
      body.instructions = params.system
    }

    if (tools && tools.length > 0) {
      body.tools = tools
    }

    const effort = this.normalizeReasoningEffort(params.reasoning?.effort)
    if (effort) {
      body.reasoning = { effort }
    }

    const response = await fetch(`${this.baseURL}/responses`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
      signal: params.abortSignal,
    })

    if (!response.ok) {
      const errBody = await response.text().catch(() => '')
      const err: any = new Error(
        `OpenAI Responses API error: ${response.status} ${response.statusText}: ${errBody}`,
      )
      err.status = response.status
      throw err
    }

    return response
  }

  private async readResponse(response: Response): Promise<OpenAIResponsesResponse> {
    const contentType = response.headers.get('content-type') || ''
    if (contentType.includes('text/event-stream')) {
      return this.readStreamedResponse(response)
    }

    return (await response.json()) as OpenAIResponsesResponse
  }

  private async readStreamedResponse(response: Response): Promise<OpenAIResponsesResponse> {
    const textParts: string[] = []
    let completedResponse: OpenAIResponsesResponse | undefined

    for await (const event of this.streamSseEvents(response)) {
      if (event.type === 'response.output_text.delta' && typeof event.delta === 'string') {
        textParts.push(event.delta)
      } else if (event.type === 'response.completed' && event.response) {
        completedResponse = event.response as OpenAIResponsesResponse
      } else if (event.type === 'response.failed') {
        const message = event.response?.error?.message || event.error?.message || 'Responses stream failed'
        throw new Error(message)
      } else if (event.type === 'error') {
        throw new Error(event.error?.message || event.message || 'Responses stream error')
      }
    }

    return this.buildStreamedResponse(textParts, completedResponse)
  }

  private buildStreamedResponse(
    textParts: string[],
    completedResponse: OpenAIResponsesResponse | undefined,
  ): OpenAIResponsesResponse {
    if (completedResponse?.output?.length) {
      return completedResponse
    }

    const text = textParts.join('')
    return {
      ...completedResponse,
      status: completedResponse?.status || 'completed',
      output: [
        {
          type: 'message',
          role: 'assistant',
          content: [{ type: 'output_text', text }],
        },
      ],
      usage: completedResponse?.usage,
    }
  }

  private async *streamSseEvents(response: Response): AsyncGenerator<Record<string, any>> {
    const reader = response.body?.getReader()
    if (!reader) {
      for (const event of this.parseSseEvents(await response.text())) {
        yield event
      }
      return
    }

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      const chunks = buffer.split(/\r?\n\r?\n/)
      buffer = chunks.pop() || ''

      for (const chunk of chunks) {
        const event = this.parseSseEventChunk(chunk)
        if (event) yield event
      }

      if (done) break
    }

    if (buffer.trim()) {
      const event = this.parseSseEventChunk(buffer)
      if (event) yield event
    }
  }

  private parseSseEvents(raw: string): Array<Record<string, any>> {
    const events: Array<Record<string, any>> = []

    for (const chunk of raw.split(/\r?\n\r?\n/)) {
      const event = this.parseSseEventChunk(chunk)
      if (event) events.push(event)
    }

    return events
  }

  private parseSseEventChunk(chunk: string): Record<string, any> | null {
    const dataLines: string[] = []

    for (const line of chunk.split(/\r?\n/)) {
      if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trimStart())
      }
    }

    if (dataLines.length === 0) {
      return null
    }

    const data = dataLines.join('\n')
    if (data === '[DONE]') {
      return null
    }

    try {
      return JSON.parse(data)
    } catch {
      // Ignore malformed SSE chunks from intermediate proxies.
      return null
    }
  }

  private convertMessages(messages: NormalizedMessageParam[]): OpenAIResponsesInputItem[] {
    const result: OpenAIResponsesInputItem[] = []

    for (const msg of messages) {
      if (typeof msg.content === 'string') {
        result.push({ role: msg.role, content: msg.content })
        continue
      }

      if (msg.role === 'user') {
        this.convertUserBlocks(msg.content, result)
      } else {
        this.convertAssistantBlocks(msg.content, result)
      }
    }

    return result
  }

  private convertUserBlocks(
    blocks: NormalizedContentBlock[],
    result: OpenAIResponsesInputItem[],
  ): void {
    const textParts: string[] = []

    for (const block of blocks) {
      if (block.type === 'text') {
        textParts.push(block.text)
      } else if (block.type === 'tool_result') {
        result.push({
          type: 'function_call_output',
          call_id: block.tool_use_id,
          output: typeof block.content === 'string' ? block.content : JSON.stringify(block.content),
        })
      }
    }

    if (textParts.length > 0) {
      result.push({ role: 'user', content: textParts.join('\n') })
    }
  }

  private convertAssistantBlocks(
    blocks: NormalizedContentBlock[],
    result: OpenAIResponsesInputItem[],
  ): void {
    const textParts: string[] = []

    for (const block of blocks) {
      if (block.type === 'text') {
        textParts.push(block.text)
      } else if (block.type === 'tool_use') {
        result.push({
          type: 'function_call',
          call_id: block.id,
          name: block.name,
          arguments: typeof block.input === 'string' ? block.input : JSON.stringify(block.input),
        })
      }
    }

    if (textParts.length > 0) {
      result.push({ role: 'assistant', content: textParts.join('\n') })
    }
  }

  private convertTools(tools: NormalizedTool[]): OpenAIResponsesTool[] {
    return tools.map((t) => ({
      type: 'function' as const,
      name: t.name,
      description: t.description,
      parameters: t.input_schema,
      strict: false,
    }))
  }

  private convertResponse(data: OpenAIResponsesResponse): CreateMessageResponse {
    const content: NormalizedResponseBlock[] = []

    for (const item of data.output || []) {
      if (item.type === 'message') {
        for (const part of (item as OpenAIResponsesMessageItem).content || []) {
          if (part.type === 'output_text' && part.text) {
            content.push({ type: 'text', text: part.text })
          }
        }
      } else if (item.type === 'function_call') {
        const call = item as OpenAIResponsesFunctionCallItem
        let input: any
        try {
          input = JSON.parse(call.arguments)
        } catch {
          input = call.arguments
        }
        content.push({
          type: 'tool_use',
          id: call.call_id || call.id || '',
          name: call.name,
          input,
        })
      }
    }

    if (content.length === 0) {
      content.push({ type: 'text', text: '' })
    }

    return {
      content,
      stopReason: this.mapStopReason(data, content),
      usage: {
        input_tokens: data.usage?.input_tokens || 0,
        output_tokens: data.usage?.output_tokens || 0,
      },
    }
  }

  private mapStopReason(
    data: OpenAIResponsesResponse,
    content: NormalizedResponseBlock[],
  ): 'end_turn' | 'max_tokens' | 'tool_use' | string {
    if (content.some((block) => block.type === 'tool_use')) {
      return 'tool_use'
    }
    if (data.incomplete_details?.reason === 'max_output_tokens') {
      return 'max_tokens'
    }
    return data.status && data.status !== 'completed' ? data.status : 'end_turn'
  }

  private normalizeReasoningEffort(
    effort: ReasoningEffort | undefined,
  ): Exclude<ReasoningEffort, 'max'> | undefined {
    if (!effort) return undefined
    return effort === 'max' ? 'xhigh' : effort
  }
}
