import assert from 'node:assert/strict'
import { afterEach, describe, it } from 'node:test'

import { createAgent } from '../src/agent.js'
import { createProvider } from '../src/providers/index.js'

const originalFetch = globalThis.fetch

function createSseResponse(events: Array<Record<string, any>>): Response {
  return new Response(
    events
      .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
      .join(''),
    {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    },
  )
}

function createDelayedSseResponse(events: Array<Record<string, any>>): Response {
  const encoder = new TextEncoder()
  return new Response(
    new ReadableStream({
      async start(controller) {
        for (const event of events) {
          controller.enqueue(
            encoder.encode(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`),
          )
          await new Promise((resolve) => setTimeout(resolve, 10))
        }
        controller.close()
      },
    }),
    {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    },
  )
}

afterEach(() => {
  globalThis.fetch = originalFetch
})

describe('OpenAI Responses provider', () => {
  it('posts normalized messages to /responses with reasoning effort', async () => {
    let capturedUrl = ''
    let capturedBody: any

    globalThis.fetch = async (url, init) => {
      capturedUrl = String(url)
      capturedBody = JSON.parse(String(init?.body))

      return new Response(
        JSON.stringify({
          output: [
            {
              type: 'message',
              role: 'assistant',
              content: [{ type: 'output_text', text: 'hello from responses' }],
            },
          ],
          usage: {
            input_tokens: 11,
            output_tokens: 7,
          },
        }),
        { status: 200 },
      )
    }

    const provider = createProvider('openai-responses', {
      apiKey: 'test-key',
      baseURL: 'https://gateway.example.test/v1/',
    })

    const response = await provider.createMessage({
      model: 'gpt-5.5-xhigh',
      maxTokens: 512,
      system: 'You are useful.',
      messages: [{ role: 'user', content: 'Say hello.' }],
      reasoning: { effort: 'xhigh' },
    })

    assert.equal(capturedUrl, 'https://gateway.example.test/v1/responses')
    assert.equal(capturedBody.model, 'gpt-5.5-xhigh')
    assert.equal(capturedBody.max_output_tokens, 512)
    assert.equal(capturedBody.stream, true)
    assert.equal(capturedBody.instructions, 'You are useful.')
    assert.deepEqual(capturedBody.reasoning, { effort: 'xhigh' })
    assert.deepEqual(capturedBody.input, [{ role: 'user', content: 'Say hello.' }])
    assert.deepEqual(response.content, [{ type: 'text', text: 'hello from responses' }])
    assert.equal(response.stopReason, 'end_turn')
    assert.deepEqual(response.usage, { input_tokens: 11, output_tokens: 7 })
  })

  it('parses streamed Responses text deltas', async () => {
    let capturedBody: any

    globalThis.fetch = async (_url, init) => {
      capturedBody = JSON.parse(String(init?.body))

      return createSseResponse([
        {
          type: 'response.output_text.delta',
          delta: 'hello ',
        },
        {
          type: 'response.output_text.delta',
          delta: 'from stream',
        },
        {
          type: 'response.completed',
          response: {
            status: 'completed',
            usage: {
              input_tokens: 13,
              output_tokens: 3,
            },
          },
        },
      ])
    }

    const provider = createProvider('openai-responses', {
      apiKey: 'test-key',
      baseURL: 'https://gateway.example.test/v1',
    })

    const response = await provider.createMessage({
      model: 'gpt-5.5-high',
      maxTokens: 256,
      system: '',
      messages: [{ role: 'user', content: 'Stream please.' }],
    })

    assert.equal(capturedBody.stream, true)
    assert.deepEqual(response.content, [{ type: 'text', text: 'hello from stream' }])
    assert.equal(response.stopReason, 'end_turn')
    assert.deepEqual(response.usage, { input_tokens: 13, output_tokens: 3 })
  })

  it('converts SDK tools and Responses function calls', async () => {
    let capturedBody: any

    globalThis.fetch = async (_url, init) => {
      capturedBody = JSON.parse(String(init?.body))

      return new Response(
        JSON.stringify({
          output: [
            {
              type: 'function_call',
              call_id: 'call_123',
              name: 'lookup_price',
              arguments: '{"sku":"ABC"}',
            },
          ],
          usage: {
            input_tokens: 20,
            output_tokens: 4,
          },
        }),
        { status: 200 },
      )
    }

    const provider = createProvider('openai-responses', { apiKey: 'test-key' })

    const response = await provider.createMessage({
      model: 'gpt-5.5-high',
      maxTokens: 256,
      system: '',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Check price.' },
            { type: 'tool_result', tool_use_id: 'call_prev', content: '{"ok":true}' },
          ],
        },
      ],
      tools: [
        {
          name: 'lookup_price',
          description: 'Look up price by SKU.',
          input_schema: {
            type: 'object',
            properties: { sku: { type: 'string' } },
            required: ['sku'],
          },
        },
      ],
    })

    assert.deepEqual(capturedBody.tools, [
      {
        type: 'function',
        name: 'lookup_price',
        description: 'Look up price by SKU.',
        parameters: {
          type: 'object',
          properties: { sku: { type: 'string' } },
          required: ['sku'],
        },
        strict: false,
      },
    ])
    assert.deepEqual(capturedBody.input, [
      {
        type: 'function_call_output',
        call_id: 'call_prev',
        output: '{"ok":true}',
      },
      { role: 'user', content: 'Check price.' },
    ])
    assert.deepEqual(response.content, [
      {
        type: 'tool_use',
        id: 'call_123',
        name: 'lookup_price',
        input: { sku: 'ABC' },
      },
    ])
    assert.equal(response.stopReason, 'tool_use')
  })

  it('lets createAgent call the Responses provider with effort', async () => {
    let capturedUrl = ''
    let capturedBody: any

    globalThis.fetch = async (url, init) => {
      capturedUrl = String(url)
      capturedBody = JSON.parse(String(init?.body))

      return new Response(
        JSON.stringify({
          output: [
            {
              type: 'message',
              role: 'assistant',
              content: [{ type: 'output_text', text: 'agent response' }],
            },
          ],
          usage: {
            input_tokens: 5,
            output_tokens: 2,
          },
        }),
        { status: 200 },
      )
    }

    const agent = createAgent({
      apiType: 'openai-responses',
      apiKey: 'test-key',
      baseURL: 'https://gateway.example.test/v1',
      cwd: '/tmp',
      effort: 'high',
      maxTurns: 1,
      model: 'gpt-5.5-high',
      systemPrompt: 'You are useful.',
      tools: [],
    })

    const events = []
    for await (const event of agent.query('Ping')) {
      events.push(event)
    }

    assert.equal(capturedUrl, 'https://gateway.example.test/v1/responses')
    assert.equal(capturedBody.reasoning.effort, 'high')
    assert.deepEqual(capturedBody.input, [{ role: 'user', content: 'Ping' }])
    assert.ok(events.some((event) => event.type === 'assistant'))
  })

  it('streams Responses text deltas through createAgent partial messages', async () => {
    globalThis.fetch = async () => {
      return createDelayedSseResponse([
        {
          type: 'response.output_text.delta',
          delta: 'hello ',
        },
        {
          type: 'response.output_text.delta',
          delta: 'stream',
        },
        {
          type: 'response.completed',
          response: {
            status: 'completed',
            usage: {
              input_tokens: 5,
              output_tokens: 2,
            },
          },
        },
      ])
    }

    const agent = createAgent({
      apiType: 'openai-responses',
      apiKey: 'test-key',
      baseURL: 'https://gateway.example.test/v1',
      cwd: '/tmp',
      includePartialMessages: true,
      maxTurns: 1,
      model: 'gpt-5.5-high',
      systemPrompt: 'You are useful.',
      tools: [],
    })

    const events = []
    const partialTexts: string[] = []
    for await (const event of agent.query('Ping')) {
      events.push(event.type)
      if (event.type === 'partial_message') {
        partialTexts.push(event.partial.text || '')
      }
    }

    const partialIndex = events.indexOf('partial_message')
    const assistantIndex = events.indexOf('assistant')
    assert.ok(partialIndex >= 0)
    assert.ok(assistantIndex > partialIndex)
    assert.deepEqual(partialTexts, ['hello ', 'hello stream'])
  })
})
