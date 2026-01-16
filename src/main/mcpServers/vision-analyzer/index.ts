/**
 * Vision Analyzer MCP Server
 *
 * 利用具有视觉能力的 LLM 来解析图像。
 * 直接使用 maa-ai 提供商的 API 和 Key 进行图像分析。
 */

import OpenAI from '@cherrystudio/openai'
import { GoogleGenAI } from '@google/genai'
import { loggerService } from '@logger'
import { reduxService } from '@main/services/ReduxService'
import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { ErrorCode, McpError } from '@modelcontextprotocol/sdk/types.js'
import type { Provider } from '@types'
import * as z from 'zod'
import { zodToJsonSchema } from 'zod-to-json-schema'

import { imageProcessor } from './ImageProcessor'
import { VisualContentStructurerPrompt, VisualCoordinateExtractorPrompt } from './prompts'

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
  fallbackApiHost: 'https://open.maa-ai.com',
  /** 默认系统提示词 */
  defaultSystemPrompt: VisualContentStructurerPrompt
}

/**
 * 坐标提取结果的 Zod Schema
 */
const CoordinateExtractionSchema = z.object({
  results: z.array(
    z.object({
      target: z.string().describe('The name of the object to locate'),
      status: z.enum(['found', 'missing']).describe('Whether the target was found or missing'),
      count: z.number().int().nonnegative().describe('Number of instances found'),
      boxes: z
        .array(
          z
            .tuple([z.number(), z.number(), z.number(), z.number()])
            .describe('Bounding box in format [ymin, xmin, ymax, xmax]')
        )
        .describe('List of bounding boxes')
    })
  )
})

/**
 * VisionAnalyzerServer 配置选项
 */
interface VisionAnalyzerConfig {
  /** 模型名称，默认为 gemini-3-flash-preview */
  modelName?: string
}

/**
 * LLM 消息类型
 */
interface SessionMessage {
  role: 'system' | 'user' | 'assistant'
  content: string | Array<{ type: 'text' | 'image_url'; text?: string; image_url?: { url: string } }>
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
  private mcpServer: McpServer
  public get server(): Server {
    return this.mcpServer.server
  }

  private openaiClient: OpenAI | null = null
  private geminiClient: GoogleGenAI | null = null
  private modelName: string
  private isInitialized = false
  private initializationPromise: Promise<void>

  /**
   * 检测模型是否为 Gemini 模型
   */
  private isGeminiModel(modelName: string): boolean {
    return modelName.toLowerCase().startsWith('gemini')
  }

  private detectNormalizationScale(values: number[]): 1 | 1000 {
    const max = Math.max(...values.map((v) => Math.abs(v)))
    // 兼容浮点误差：1.0000001 仍视为 [0,1]
    if (max <= 1.000001) return 1
    if (max <= 1000) return 1000
    throw new McpError(
      ErrorCode.InvalidParams,
      `normalized values look out of range. Expected [0,1] or [0,1000], got max=${max}`
    )
  }

  private clamp(n: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, n))
  }

  constructor(config: VisionAnalyzerConfig = {}) {
    this.modelName = config.modelName || DEFAULT_CONFIG.defaultModel

    // 创建 MCP Server 实例
    this.mcpServer = new McpServer({
      name: 'vision-analyzer',
      version: '1.0.0'
    })

    // 异步初始化客户端
    this.initializationPromise = this.initializeClient()

    // 设置请求处理器
    this.setupRequestHandlers()

    logger.info('VisionAnalyzerServer created', { model: this.modelName })
  }

  /**
   * 异步初始化客户端（OpenAI 或 Gemini）
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

      // 根据模型类型初始化不同的客户端
      const isGemini = this.isGeminiModel(this.modelName)
      logger.info('[Init] Model type detection:', { modelName: this.modelName, isGemini })

      if (isGemini) {
        const baseURL = apiHost

        logger.info('[Init] Initializing Gemini client with baseURL:', { baseURL })

        this.geminiClient = new GoogleGenAI({
          vertexai: false,
          apiKey: provider.apiKey,
          httpOptions: {
            baseUrl: baseURL
          }
        })

        this.isInitialized = true
        logger.info('[Init] VisionAnalyzerServer initialized successfully with Gemini client', {
          model: this.modelName,
          baseURL: baseURL
        })
      } else {
        // 使用 OpenAI 客户端
        // 确保 baseURL 包含 /v1 后缀
        let baseURL = apiHost
        if (!baseURL.endsWith('/v1') && !baseURL.endsWith('/v1/')) {
          baseURL = baseURL.replace(/\/$/, '') + '/v1'
        }

        logger.info('[Init] Initializing OpenAI client with baseURL:', { baseURL })

        this.openaiClient = new OpenAI({
          apiKey: provider.apiKey,
          baseURL: baseURL
        })

        this.isInitialized = true
        logger.info('[Init] VisionAnalyzerServer initialized successfully with OpenAI client', {
          model: this.modelName,
          baseURL: baseURL
        })
      }
    } catch (error) {
      logger.error('[Init] Failed to initialize VisionAnalyzerServer:', error as Error)
    }
  }

  /**
   * 确保客户端已初始化
   */
  private async ensureInitialized(): Promise<void> {
    await this.initializationPromise

    const isGemini = this.isGeminiModel(this.modelName)
    const client = isGemini ? this.geminiClient : this.openaiClient

    if (!this.isInitialized || !client) {
      // 尝试重新初始化
      await this.initializeClient()

      const reinitializedClient = isGemini ? this.geminiClient : this.openaiClient
      if (!this.isInitialized || !reinitializedClient) {
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
    // 1. structure_visual_content
    this.mcpServer.registerTool(
      'structure_visual_content',
      {
        description:
          'Structure an image into a high-density, XML-tagged analysis for downstream AIs. ' +
          'Returns structured analysis. ' +
          'IMPORTANT: downstream LLMs must only rely on this tool output for image understanding; ' +
          'do NOT infer or fabricate spatial coordinates from it. If you need bounding boxes/coordinates, ' +
          'you MUST call `extract_visual_coordinates`. ' +
          'Required parameters: `image_path` (absolute local image path) and `prompt` (question/instructions).',
        inputSchema: {
          image_path: z
            .string()
            .describe(
              'The absolute local path to the image file. Supported formats: jpg, jpeg, png, gif, webp, bmp, svg.'
            ),
          prompt: z
            .string()
            .describe(
              'The prompt or question to ask about the image. Be specific about what aspects of the image you want analyzed.'
            )
        }
      },
      async (args) => {
        try {
          return await this.handleStructureVisualContent(args)
        } catch (error) {
          if (error instanceof McpError) throw error
          logger.error('Error executing structure_visual_content:', error as Error)
          throw new McpError(
            ErrorCode.InternalError,
            `Error executing structure_visual_content: ${error instanceof Error ? error.message : String(error)}`
          )
        }
      }
    )

    // 2. extract_visual_coordinates
    this.mcpServer.registerTool(
      'extract_visual_coordinates',
      {
        description:
          'Extract target bounding boxes from an image. ' +
          'Output is XML with `<image_width>`/`<image_height>` (pixels, -1 if unknown) and `<results>` containing `<result>` items. ' +
          'Each `<result>` includes `<target>`, `<status>` (found/missing), `<count>`, and `<boxes>` with `<box>` entries. ' +
          'Each `<box>` provides `<output_xywh>` (pixel coordinates [x, y, w, h]) ready for UI clicks/cropping. ' +
          'Required parameters: `image_path` (absolute local image path), `target_name` (what to locate). ' +
          'Optional: `hint` (extra constraints like color/shape/nearby text).',
        inputSchema: {
          image_path: z
            .string()
            .describe(
              'The absolute local path to the image file. Supported formats: jpg, jpeg, png, gif, webp, bmp, svg.'
            ),
          target_name: z
            .string()
            .describe('The target to locate in the image. Example: "Login button", "red apple", "search icon".'),
          hint: z
            .string()
            .optional()
            .describe(
              'Optional extra hints/constraints. Example: "top right", "next to the word Settings", "blue icon".'
            )
        }
      },
      async (args) => {
        try {
          return await this.handleExtractVisualCoordinates(args)
        } catch (error) {
          if (error instanceof McpError) throw error
          logger.error('Error executing extract_visual_coordinates:', error as Error)
          throw new McpError(
            ErrorCode.InternalError,
            `Error executing extract_visual_coordinates: ${error instanceof Error ? error.message : String(error)}`
          )
        }
      }
    )
  }

  /**
   * 处理 structure_visual_content 工具调用
   */
  private async handleStructureVisualContent(args: { image_path: string; prompt: string }): Promise<{
    content: Array<{ type: 'text'; text: string }>
  }> {
    const { image_path, prompt } = args

    logger.info('[StructureVisualContent] Starting structure_visual_content...', {
      hasImagePath: !!image_path,
      hasPrompt: !!prompt,
      args: JSON.stringify(args)
    })

    // Zod already validates params, but we can keep these checks or rely on Zod.
    // However, existing method expects params.

    // 确保已初始化
    logger.info('[StructureVisualContent] Ensuring client is initialized...')
    await this.ensureInitialized()
    logger.info('[StructureVisualContent] Client initialized, isInitialized:', { isInitialized: this.isInitialized })

    // 验证并处理图像
    logger.info('[StructureVisualContent] Validating image path...')
    const validation = await imageProcessor.validate(image_path)
    if (!validation.valid) {
      throw new McpError(ErrorCode.InvalidParams, `Invalid image path: ${validation.error}`)
    }

    logger.info('[StructureVisualContent] Image validated, processing...')
    const processedImage = await imageProcessor.process(image_path)
    logger.info('[StructureVisualContent] Image processed', {
      mimeType: processedImage.mimeType,
      urlPrefix: processedImage.url.substring(0, 50) + '...'
    })

    // 构建初始消息 (包含图像)
    logger.info('[StructureVisualContent] Building user message with image...')
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
    logger.info('[StructureVisualContent] Calling LLM with system prompt and user message...')
    const response = await this.callLLM(
      [{ role: 'system', content: DEFAULT_CONFIG.defaultSystemPrompt }, userMessage],
      0.3 // 描述图片内容使用较低但非零的温度，保持一定的创造性
    )
    logger.info('[StructureVisualContent] LLM call completed successfully')

    logger.info('Image structured successfully', { imagePath: image_path })

    // 返回结果
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(
            {
              analysis: response,
              message: 'Image structured successfully.'
            },
            null,
            2
          )
        }
      ]
    }
  }

  /**
   * 处理 extract_visual_coordinates 工具调用
   *
   * 使用 VisualCoordinateExtractorPrompt (1000x1000 虚拟网格) 输出 JSON：
   * { "results": [{ "target": "...", "status": "found|missing", "count": N, "boxes": [[ymin, xmin, ymax, xmax], ...] }] }
   *
   * 然后将归一化坐标转换为像素坐标 [x, y, w, h]，并输出 XML 格式
   */
  private async handleExtractVisualCoordinates(args: {
    image_path: string
    target_name: string
    hint?: string
  }): Promise<{ content: Array<{ type: 'text'; text: string }> }> {
    const { image_path, target_name, hint } = args

    // Zod handled required checks
    await this.ensureInitialized()

    const validation = await imageProcessor.validate(image_path)
    if (!validation.valid) {
      throw new McpError(ErrorCode.InvalidParams, `Invalid image path: ${validation.error}`)
    }

    const processedImage = await imageProcessor.process(image_path)
    const image_width = processedImage.width ?? -1
    const image_height = processedImage.height ?? -1

    const textPrompt =
      hint && hint.trim().length > 0
        ? `Find the following target in the image: ${target_name}\nAdditional hint: ${hint}`
        : `Find the following target in the image: ${target_name}`

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
          text: textPrompt
        }
      ]
    }

    const response = await this.callLLM(
      [{ role: 'system', content: VisualCoordinateExtractorPrompt }, userMessage],
      0, // 提取坐标需要精确的输出，使用温度 0
      true // 使用 JSON schema 限制输出格式
    )
    logger.info('[ExtractVisualCoordinates] LLM response:', { response })

    // 解析 JSON 响应
    let jsonResponse: { results: Array<{ target: string; status: string; count: number; boxes: number[][] }> }
    try {
      // 清理可能的 markdown 代码块标记
      const cleanedResponse = response
        .replace(/```json\s*/g, '')
        .replace(/```\s*/g, '')
        .trim()
      jsonResponse = JSON.parse(cleanedResponse)
    } catch (error) {
      logger.error('[ExtractVisualCoordinates] Failed to parse JSON response:', {
        response: response.substring(0, 200),
        error: error as Error
      })
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to parse LLM response as JSON: ${error instanceof Error ? error.message : String(error)}`
      )
    }

    // 输出完整的 JSON 响应用于调试
    logger.info('[ExtractVisualCoordinates] LLM returned JSON:', {
      jsonResponse: JSON.stringify(jsonResponse, null, 2),
      hasResults: !!jsonResponse.results,
      resultsType: typeof jsonResponse.results,
      resultsIsArray: Array.isArray(jsonResponse.results),
      resultsLength: Array.isArray(jsonResponse.results) ? jsonResponse.results.length : 'N/A'
    })

    // 验证 JSON 结构
    if (!jsonResponse.results || !Array.isArray(jsonResponse.results)) {
      logger.error('[ExtractVisualCoordinates] Invalid JSON structure:', {
        hasResults: !!jsonResponse.results,
        isResultsArray: Array.isArray(jsonResponse.results),
        jsonResponseKeys: Object.keys(jsonResponse),
        rawResponse: response.substring(0, 500)
      })
      throw new McpError(
        ErrorCode.InternalError,
        `Invalid JSON structure: 'results' field is missing or not an array. Response: ${response.substring(0, 200)}`
      )
    }

    // 转换坐标并构建 XML 输出
    const resultItems = jsonResponse.results.map((result) => {
      // 验证 boxes 字段
      if (!result.boxes || !Array.isArray(result.boxes)) {
        logger.warn('[ExtractVisualCoordinates] Invalid boxes in result:', {
          target: result.target,
          hasBoxes: !!result.boxes,
          isBoxesArray: Array.isArray(result.boxes)
        })
        // 返回一个空的结果项
        return (
          `  <result>\n` +
          `    <target>${result.target || 'unknown'}</target>\n` +
          `    <status>error</status>\n` +
          `    <count>0</count>\n` +
          `    <boxes>\n    </boxes>\n` +
          `  </result>`
        )
      }

      const convertedBoxes = result.boxes.map((box) => {
        const [ymin0, xmin0, ymax0, xmax0] = box
        const scale = this.detectNormalizationScale([ymin0, xmin0, ymax0, xmax0])

        const xmin = (xmin0 / scale) * image_width
        const ymin = (ymin0 / scale) * image_height
        const xmax = (xmax0 / scale) * image_width
        const ymax = (ymax0 / scale) * image_height

        let x = xmin
        let y = ymin
        let w = xmax - xmin
        let h = ymax - ymin

        // Clamp to image bounds
        const x2 = this.clamp(x + w, 0, image_width)
        const y2 = this.clamp(y + h, 0, image_height)
        x = this.clamp(x, 0, image_width)
        y = this.clamp(y, 0, image_height)
        w = Math.max(0, x2 - x)
        h = Math.max(0, y2 - y)

        // Round to integers
        const xywh: [number, number, number, number] = [Math.round(x), Math.round(y), Math.round(w), Math.round(h)]

        return {
          input: `${ymin0}, ${xmin0}, ${ymax0}, ${xmax0}`,
          detectedScale: scale,
          output: `${xywh[0]}, ${xywh[1]}, ${xywh[2]}, ${xywh[3]}`
        }
      })

      const boxesXml = convertedBoxes
        .map(
          (box, idx) =>
            `    <box index="${idx}">\n` +
            `      <input_ymin_xmin_ymax_xmax>${box.input}</input_ymin_xmin_ymax_xmax>\n` +
            `      <detected_scale>${box.detectedScale}</detected_scale>\n` +
            `      <output_xywh>${box.output}</output_xywh>\n` +
            `    </box>`
        )
        .join('\n')

      return (
        `  <result>\n` +
        `    <target>${result.target}</target>\n` +
        `    <status>${result.status}</status>\n` +
        `    <count>${result.count}</count>\n` +
        `    <boxes>\n${boxesXml}\n    </boxes>\n` +
        `  </result>`
      )
    })

    const xmlOutput =
      `<response>\n` +
      `  <image_width>${image_width}</image_width>\n` +
      `  <image_height>${image_height}</image_height>\n` +
      `  <results>\n${resultItems.join('\n')}\n  </results>\n` +
      `</response>`

    return {
      content: [
        {
          type: 'text',
          text: xmlOutput
        }
      ]
    }
  }

  /**
   * 调用 LLM 获取响应 (使用 OpenAI Responses API 或 Gemini API)
   * @param messages - 消息数组
   * @param temperature - 温度参数，控制输出的随机性 (0-1)，默认为 0
   * @param useJsonSchema - 是否使用 JSON schema 限制输出格式，默认为 false
   */
  private async callLLM(
    messages: SessionMessage[],
    temperature: number = 0,
    useJsonSchema: boolean = false
  ): Promise<string> {
    logger.info('[LLM] Starting LLM call...')

    const isGemini = this.isGeminiModel(this.modelName)

    if (isGemini) {
      return await this.callGeminiLLM(messages, temperature, useJsonSchema)
    } else {
      return await this.callOpenAILLM(messages, temperature, useJsonSchema)
    }
  }

  /**
   * 调用 OpenAI LLM 获取响应
   * @param messages - 消息数组
   * @param temperature - 温度参数，控制输出的随机性 (0-1)，默认为 0
   * @param useJsonSchema - 是否使用 JSON schema 限制输出格式，默认为 false
   */
  private async callOpenAILLM(
    messages: SessionMessage[],
    temperature: number = 0,
    useJsonSchema: boolean = false
  ): Promise<string> {
    logger.info('[LLM] Using OpenAI client...')

    if (!this.openaiClient) {
      logger.error('[LLM] OpenAI client not initialized')
      throw new McpError(ErrorCode.InternalError, 'OpenAI client not initialized')
    }

    // 记录客户端配置信息
    logger.info('[LLM] Client baseURL:', {
      baseURL: this.openaiClient.baseURL
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

      // 构建请求参数
      const requestParams: any = {
        model: this.modelName,
        input: input,
        temperature: temperature
      }

      // 如果需要使用 JSON schema 限制输出格式
      if (useJsonSchema) {
        logger.info('[LLM] Using JSON schema for structured output')
        requestParams.text = {
          format: {
            type: 'json_schema',
            name: 'coordinate_extraction',
            description: 'Extract visual coordinates from image',
            schema: zodToJsonSchema(CoordinateExtractionSchema as any),
            strict: true
          }
        }
      }

      logger.info('[LLM] Sending request to Responses API:', {
        model: this.modelName,
        inputItemCount: input.length,
        fullUrl: `${this.openaiClient.baseURL}/responses`,
        useJsonSchema
      })

      // 使用 Responses API
      const response = await this.openaiClient.responses.create(requestParams)

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

  /**
   * 调用 Gemini LLM 获取响应
   * @param messages - 消息数组
   * @param temperature - 温度参数，控制输出的随机性 (0-1)，默认为 0
   * @param useJsonSchema - 是否使用 JSON schema 限制输出格式，默认为 false
   */
  private async callGeminiLLM(
    messages: SessionMessage[],
    temperature: number = 0,
    useJsonSchema: boolean = false
  ): Promise<string> {
    logger.info('[LLM] Using Gemini client...')

    if (!this.geminiClient) {
      logger.error('[LLM] Gemini client not initialized')
      throw new McpError(ErrorCode.InternalError, 'Gemini client not initialized')
    }

    try {
      // 转换消息格式为 Gemini API 格式
      logger.info('[LLM] Converting messages to Gemini API format...', {
        messageCount: messages.length
      })

      const contents = messages
        .filter((msg) => msg.role !== 'system')
        .map((msg) => {
          if (msg.role === 'user' && Array.isArray(msg.content)) {
            const parts: Array<{ text?: string; inlineData?: { data: string; mimeType: string } }> = []
            for (const part of msg.content) {
              if (part.type === 'text') {
                parts.push({ text: part.text || '' })
              } else if (part.type === 'image_url') {
                const imageUrl = part.image_url?.url || ''
                // 解析 data URL
                const matches = imageUrl.match(/^data:(.+);base64,(.+)$/)
                if (matches && matches.length === 3) {
                  const mimeType = matches[1]
                  const base64Data = matches[2]
                  parts.push({
                    inlineData: {
                      data: base64Data,
                      mimeType: mimeType
                    }
                  })
                }
              }
            }
            return { role: 'user', parts }
          } else {
            return {
              role: msg.role === 'user' ? 'user' : 'model',
              parts: [{ text: typeof msg.content === 'string' ? msg.content : '' }]
            }
          }
        })

      // 提取系统提示词
      const systemInstruction = messages.find((msg) => msg.role === 'system')
      const systemPrompt =
        systemInstruction && typeof systemInstruction.content === 'string' ? systemInstruction.content : undefined

      // 对于 gemini-3 模型，将温度设置为 1
      const actualTemperature = this.modelName.toLowerCase().startsWith('gemini-3') ? 1 : temperature

      // 构建配置对象
      const config: any = {
        systemInstruction: systemPrompt,
        temperature: actualTemperature
      }

      // 如果需要使用 JSON schema 限制输出格式
      if (useJsonSchema) {
        logger.info('[LLM] Using JSON schema for structured output')
        config.responseMimeType = 'application/json'
        config.responseJsonSchema = zodToJsonSchema(CoordinateExtractionSchema as any)
      }

      logger.info('[LLM] Sending request to Gemini API:', {
        model: this.modelName,
        contentCount: contents.length,
        hasSystemPrompt: !!systemPrompt,
        temperature: actualTemperature,
        originalTemperature: temperature,
        useJsonSchema
      })

      // 使用 Gemini API
      const response = await this.geminiClient.models.generateContent({
        model: this.modelName,
        contents: contents,
        config: config
      })

      logger.info('[LLM] Response received:', {
        hasCandidates: !!response.candidates,
        candidateCount: response.candidates?.length || 0
      })

      // 从 response 中提取文本内容
      let content = ''
      if (response.candidates && response.candidates.length > 0) {
        for (const candidate of response.candidates) {
          if (candidate.content && candidate.content.parts) {
            for (const part of candidate.content.parts) {
              if (part.text) {
                content += part.text
              }
            }
          }
        }
      }

      if (!content) {
        logger.warn('[LLM] Empty content in response:', {
          response: JSON.stringify(response)
        })
        throw new Error('Empty response from LLM')
      }

      logger.info('[LLM] LLM response received successfully', {
        model: this.modelName,
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
