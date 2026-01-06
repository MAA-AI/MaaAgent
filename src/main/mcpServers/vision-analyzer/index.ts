/**
 * Vision Analyzer MCP Server
 *
 * 利用具有视觉能力的 LLM 来解析图像，支持基于会话 ID 的连续追问。
 * 直接使用 maa-ai 提供商的 API 和 Key 进行图像分析。
 */

import OpenAI from '@cherrystudio/openai'
import { loggerService } from '@logger'
import { reduxService } from '@main/services/ReduxService'
import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { CallToolRequestSchema, ErrorCode, ListToolsRequestSchema, McpError } from '@modelcontextprotocol/sdk/types.js'
import type { Provider } from '@types'

import { imageProcessor } from './ImageProcessor'
import { type SessionMessage, sessionStore } from './SessionStore'

const logger = loggerService.withContext('MCPServer:VisionAnalyzer')

/**
 * maa-ai 提供商 ID
 */
const MAA_AI_PROVIDER_ID = 'maa-ai'

/**
 * 默认配置
 */
const DEFAULT_CONFIG = {
  /** 默认模型名称 */
  defaultModel: 'gemini-3-flash-preview',
  /** 默认 API 地址 (当无法从 store 获取时使用) */
  fallbackApiHost: 'https://open.maa-ai.com/v1',
  /** 默认系统提示词 */
  defaultSystemPrompt:
    'You are a helpful AI assistant specialized in image analysis. ' +
    'Analyze the provided image carefully and provide detailed, accurate descriptions. ' +
    'If the user asks questions about the image, answer based on what you can see in the image.'
}

/**
 * VisionAnalyzerServer 配置选项
 */
interface VisionAnalyzerConfig {
  /** 模型名称，默认为 gemini-3-flash-preview */
  modelName?: string
}

/**
 * 从 Redux store 获取 maa-ai 提供商配置
 */
async function getMaaAiProvider(): Promise<Provider | null> {
  try {
    const providers = await reduxService.select<Provider[]>('state.llm.providers')
    if (!providers || !Array.isArray(providers)) {
      logger.warn('No providers found in Redux store')
      return null
    }

    const maaAiProvider = providers.find((p: Provider) => p.id === MAA_AI_PROVIDER_ID)
    if (!maaAiProvider) {
      logger.warn(`Provider '${MAA_AI_PROVIDER_ID}' not found`)
      return null
    }

    return maaAiProvider
  } catch (error) {
    logger.error('Failed to get maa-ai provider:', error as Error)
    return null
  }
}

/**
 * Vision Analyzer MCP Server 类
 */
class VisionAnalyzerServer {
  public server: Server
  private client: OpenAI | null = null
  private modelName: string
  private isInitialized = false
  private initializationPromise: Promise<void>

  constructor(config: VisionAnalyzerConfig = {}) {
    this.modelName = config.modelName || DEFAULT_CONFIG.defaultModel

    // 创建 MCP Server 实例
    this.server = new Server(
      {
        name: 'vision-analyzer',
        version: '1.0.0'
      },
      {
        capabilities: {
          tools: {}
        }
      }
    )

    // 异步初始化客户端
    this.initializationPromise = this.initializeClient()

    // 设置请求处理器
    this.setupRequestHandlers()

    logger.info('VisionAnalyzerServer created', { model: this.modelName })
  }

  /**
   * 异步初始化 OpenAI 客户端
   */
  private async initializeClient(): Promise<void> {
    try {
      logger.info('[Init] Starting client initialization...')

      const provider = await getMaaAiProvider()

      if (!provider) {
        logger.error('[Init] maa-ai provider not found, server will not be functional')
        return
      }

      logger.info('[Init] Provider found:', {
        id: provider.id,
        name: provider.name,
        apiHost: provider.apiHost,
        hasApiKey: !!provider.apiKey,
        apiKeyLength: provider.apiKey?.length || 0
      })

      if (!provider.apiKey) {
        logger.error('[Init] maa-ai provider API key not configured')
        return
      }

      const apiHost = provider.apiHost || DEFAULT_CONFIG.fallbackApiHost
      logger.info('[Init] Using apiHost:', { apiHost, fallback: !provider.apiHost })

      // 确保 baseURL 包含 /v1 后缀
      let baseURL = apiHost
      if (!baseURL.endsWith('/v1') && !baseURL.endsWith('/v1/')) {
        baseURL = baseURL.replace(/\/$/, '') + '/v1'
      }

      logger.info('[Init] Final baseURL:', { baseURL })

      this.client = new OpenAI({
        apiKey: provider.apiKey,
        baseURL: baseURL
      })

      this.isInitialized = true
      logger.info('[Init] VisionAnalyzerServer initialized successfully', {
        model: this.modelName,
        baseURL: baseURL
      })
    } catch (error) {
      logger.error('[Init] Failed to initialize VisionAnalyzerServer:', error as Error)
    }
  }

  /**
   * 确保客户端已初始化
   */
  private async ensureInitialized(): Promise<void> {
    await this.initializationPromise

    if (!this.isInitialized || !this.client) {
      // 尝试重新初始化
      await this.initializeClient()

      if (!this.isInitialized || !this.client) {
        throw new McpError(
          ErrorCode.InternalError,
          'Vision Analyzer is not properly configured. Please ensure the maa-ai provider is configured with a valid API key.'
        )
      }
    }
  }

  /**
   * 设置 MCP 请求处理器
   */
  private setupRequestHandlers(): void {
    // 注册工具列表处理器
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'analyze_image',
            description:
              'Analyze an image using a vision-capable LLM. ' +
              'Returns analysis results and a session_id for follow-up questions.',
            inputSchema: {
              type: 'object',
              properties: {
                image_path: {
                  type: 'string',
                  description:
                    'The absolute local path to the image file. ' +
                    'Supported formats: jpg, jpeg, png, gif, webp, bmp, svg.'
                },
                prompt: {
                  type: 'string',
                  description:
                    'The prompt or question to ask about the image. ' +
                    'Be specific about what aspects of the image you want analyzed.'
                }
              },
              required: ['image_path', 'prompt']
            }
          },
          {
            name: 'chat_image',
            description:
              'Continue a conversation about a previously analyzed image. ' +
              'Uses the session_id from a previous analyze_image call to maintain context.',
            inputSchema: {
              type: 'object',
              properties: {
                session_id: {
                  type: 'string',
                  description: 'The session ID returned from a previous analyze_image call.'
                },
                prompt: {
                  type: 'string',
                  description: 'The follow-up question or prompt about the image.'
                }
              },
              required: ['session_id', 'prompt']
            }
          }
        ]
      }
    })

    // 注册工具调用处理器
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params

      if (!args) {
        throw new McpError(ErrorCode.InvalidParams, `No arguments provided for tool: ${name}`)
      }

      try {
        switch (name) {
          case 'analyze_image':
            return await this.handleAnalyzeImage(args as { image_path: string; prompt: string })
          case 'chat_image':
            return await this.handleChatImage(args as { session_id: string; prompt: string })
          default:
            throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`)
        }
      } catch (error) {
        if (error instanceof McpError) {
          throw error
        }
        logger.error(`Error executing tool ${name}:`, error as Error)
        throw new McpError(
          ErrorCode.InternalError,
          `Error executing tool ${name}: ${error instanceof Error ? error.message : String(error)}`
        )
      }
    })
  }

  /**
   * 处理 analyze_image 工具调用
   */
  private async handleAnalyzeImage(args: { image_path: string; prompt: string }): Promise<{
    content: Array<{ type: 'text'; text: string }>
  }> {
    const { image_path, prompt } = args

    logger.info('[AnalyzeImage] Starting analyze_image...', {
      hasImagePath: !!image_path,
      hasPrompt: !!prompt,
      args: JSON.stringify(args)
    })

    // 参数验证
    if (!image_path) {
      throw new McpError(ErrorCode.InvalidParams, 'image_path is required')
    }
    if (!prompt) {
      throw new McpError(ErrorCode.InvalidParams, 'prompt is required')
    }

    logger.info('[AnalyzeImage] Parameters validated', {
      image_path,
      prompt: prompt.substring(0, 100)
    })

    // 确保已初始化
    logger.info('[AnalyzeImage] Ensuring client is initialized...')
    await this.ensureInitialized()
    logger.info('[AnalyzeImage] Client initialized, isInitialized:', { isInitialized: this.isInitialized })

    // 验证并处理图像
    logger.info('[AnalyzeImage] Validating image path...')
    const validation = await imageProcessor.validate(image_path)
    if (!validation.valid) {
      throw new McpError(ErrorCode.InvalidParams, `Invalid image path: ${validation.error}`)
    }

    logger.info('[AnalyzeImage] Image validated, processing...')
    const processedImage = await imageProcessor.process(image_path)
    logger.info('[AnalyzeImage] Image processed', {
      mimeType: processedImage.mimeType,
      urlPrefix: processedImage.url.substring(0, 50) + '...'
    })

    // 构建初始消息 (包含图像)
    logger.info('[AnalyzeImage] Building user message with image...')
    const userMessage: SessionMessage = {
      role: 'user',
      content: [
        {
          type: 'image_url',
          image_url: {
            url: processedImage.url
          }
        },
        {
          type: 'text',
          text: prompt
        }
      ]
    }

    // 调用 LLM
    logger.info('[AnalyzeImage] Calling LLM with system prompt and user message...')
    const response = await this.callLLM([{ role: 'system', content: DEFAULT_CONFIG.defaultSystemPrompt }, userMessage])
    logger.info('[AnalyzeImage] LLM call completed successfully')

    // 创建会话并保存
    const assistantMessage: SessionMessage = {
      role: 'assistant',
      content: response
    }

    const sessionId = await sessionStore.create(image_path, processedImage.url, DEFAULT_CONFIG.defaultSystemPrompt, [
      userMessage,
      assistantMessage
    ])

    logger.info('Image analyzed successfully', { sessionId, imagePath: image_path })

    // 返回结果
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(
            {
              session_id: sessionId,
              analysis: response,
              message: 'Image analyzed successfully. Use the session_id for follow-up questions.'
            },
            null,
            2
          )
        }
      ]
    }
  }

  /**
   * 处理 chat_image 工具调用
   */
  private async handleChatImage(args: { session_id: string; prompt: string }): Promise<{
    content: Array<{ type: 'text'; text: string }>
  }> {
    const { session_id, prompt } = args

    // 参数验证
    if (!session_id) {
      throw new McpError(ErrorCode.InvalidParams, 'session_id is required')
    }
    if (!prompt) {
      throw new McpError(ErrorCode.InvalidParams, 'prompt is required')
    }

    logger.debug('chat_image called', { session_id, prompt: prompt.substring(0, 100) })

    // 确保已初始化
    await this.ensureInitialized()

    // 检查会话是否存在
    const exists = await sessionStore.exists(session_id)
    if (!exists) {
      throw new McpError(
        ErrorCode.InvalidParams,
        `Session not found: ${session_id}. Please use analyze_image first to create a new session.`
      )
    }

    // 获取会话
    const session = await sessionStore.get(session_id)
    if (!session) {
      throw new McpError(ErrorCode.InternalError, `Failed to retrieve session: ${session_id}`)
    }

    // 构建新的用户消息
    const userMessage: SessionMessage = {
      role: 'user',
      content: prompt
    }

    // 获取完整的消息历史并添加新消息
    const messages = await sessionStore.getMessagesForLLM(session_id)
    messages.push(userMessage)

    // 调用 LLM
    const response = await this.callLLM(messages)

    // 保存新消息到会话
    const assistantMessage: SessionMessage = {
      role: 'assistant',
      content: response
    }

    await sessionStore.addMessage(session_id, userMessage)
    await sessionStore.addMessage(session_id, assistantMessage)

    logger.info('Chat message processed', { sessionId: session_id })

    // 返回结果
    return {
      content: [
        {
          type: 'text',
          text: response
        }
      ]
    }
  }

  /**
   * 调用 LLM 获取响应 (使用 OpenAI Responses API)
   */
  private async callLLM(messages: SessionMessage[]): Promise<string> {
    logger.info('[LLM] Starting LLM call...')

    if (!this.client) {
      logger.error('[LLM] OpenAI client not initialized')
      throw new McpError(ErrorCode.InternalError, 'OpenAI client not initialized')
    }

    // 记录客户端配置信息
    logger.info('[LLM] Client baseURL:', {
      baseURL: this.client.baseURL
    })

    try {
      // 转换消息格式为 Responses API 的 input 格式
      logger.info('[LLM] Converting messages to Responses API format...', {
        messageCount: messages.length
      })

      const input: OpenAI.Responses.ResponseInputItem[] = messages.map((msg, index) => {
        logger.debug(`[LLM] Processing message ${index}:`, {
          role: msg.role,
          contentType: typeof msg.content,
          isArray: Array.isArray(msg.content)
        })

        if (msg.role === 'system') {
          return {
            role: 'system' as const,
            content: typeof msg.content === 'string' ? msg.content : ''
          }
        } else if (msg.role === 'user') {
          // 处理用户消息（可能包含图像）
          if (Array.isArray(msg.content)) {
            const contentParts: OpenAI.Responses.ResponseInputContent[] = []
            for (const part of msg.content) {
              if (part.type === 'text') {
                contentParts.push({
                  type: 'input_text',
                  text: part.text || ''
                })
              } else if (part.type === 'image_url') {
                const imageUrl = part.image_url?.url || ''
                logger.debug('[LLM] Adding image to request:', {
                  hasUrl: !!imageUrl,
                  urlPrefix: imageUrl.substring(0, 50) + '...'
                })
                contentParts.push({
                  type: 'input_image',
                  image_url: imageUrl,
                  detail: 'auto'
                })
              }
            }
            return {
              role: 'user' as const,
              content: contentParts
            }
          } else {
            return {
              role: 'user' as const,
              content: typeof msg.content === 'string' ? msg.content : ''
            }
          }
        } else {
          // assistant 消息
          return {
            role: 'assistant' as const,
            content: typeof msg.content === 'string' ? msg.content : ''
          }
        }
      })

      logger.info('[LLM] Sending request to Responses API:', {
        model: this.modelName,
        inputItemCount: input.length,
        fullUrl: `${this.client.baseURL}/responses`
      })

      // 使用 Responses API
      const response = await this.client.responses.create({
        model: this.modelName,
        input: input
      })

      logger.info('[LLM] Response received:', {
        hasOutput: !!response.output,
        outputCount: response.output?.length || 0
      })

      // 从 response 中提取文本内容
      let content = ''
      if (response.output) {
        for (const item of response.output) {
          if (item.type === 'message' && item.content) {
            for (const contentItem of item.content) {
              if (contentItem.type === 'output_text') {
                content += contentItem.text || ''
              }
            }
          }
        }
      }

      if (!content) {
        logger.warn('[LLM] Empty content in response:', {
          responseOutput: JSON.stringify(response.output)
        })
        throw new Error('Empty response from LLM')
      }

      logger.info('[LLM] LLM response received successfully', {
        model: this.modelName,
        tokensUsed: response.usage?.total_tokens,
        contentLength: content.length
      })

      return content
    } catch (error) {
      const err = error as Error & { status?: number; code?: string }
      logger.error('[LLM] LLM call failed:', {
        message: err.message,
        status: err.status,
        code: err.code,
        stack: err.stack
      })
      throw new McpError(ErrorCode.InternalError, `LLM call failed: ${err.message || String(error)}`)
    }
  }
}

export default VisionAnalyzerServer
