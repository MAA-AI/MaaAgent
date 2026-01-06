/**
 * SessionStore - 会话管理模块
 *
 * 负责管理图像分析的会话上下文，支持基于 session_id 的连续追问。
 * 会话数据持久化存储在 MCP 目录中，确保重启后可恢复。
 */

import type OpenAI from '@cherrystudio/openai'
import { loggerService } from '@logger'
import { getMcpDir } from '@main/utils/file'
import { promises as fs } from 'fs'
import path from 'path'
import { v4 as uuidv4 } from 'uuid'

const logger = loggerService.withContext('VisionAnalyzer:SessionStore')

/**
 * 消息类型定义 - 兼容 OpenAI 的消息格式
 */
export type SessionMessage = OpenAI.Chat.Completions.ChatCompletionMessageParam

/**
 * 会话数据结构
 */
export interface Session {
  /** 会话唯一标识 */
  id: string
  /** 创建时间 */
  createdAt: string
  /** 最后更新时间 */
  updatedAt: string
  /** 关联的图像路径 */
  imagePath: string
  /** 图像的 Data URI 或 URL (用于 LLM 调用) */
  imageDataUri: string
  /** 系统提示词 */
  systemPrompt: string
  /** 对话历史 */
  messages: SessionMessage[]
}

/**
 * 会话存储索引结构
 */
interface SessionIndex {
  sessions: Record<string, { filePath: string; createdAt: string; updatedAt: string }>
}

/**
 * SessionStore 类
 * 提供会话的 CRUD 操作和持久化存储
 */
export class SessionStore {
  private storageDir: string
  private indexPath: string
  private index: SessionIndex | null = null
  private isInitialized = false

  constructor() {
    this.storageDir = path.join(getMcpDir(), 'vision-analyzer-sessions')
    this.indexPath = path.join(this.storageDir, 'index.json')
  }

  /**
   * 初始化存储目录和索引
   */
  async init(): Promise<void> {
    if (this.isInitialized) return

    try {
      // 确保存储目录存在
      await fs.mkdir(this.storageDir, { recursive: true })

      // 加载或创建索引
      try {
        const indexData = await fs.readFile(this.indexPath, 'utf-8')
        this.index = JSON.parse(indexData)
      } catch {
        // 索引文件不存在，创建新索引
        this.index = { sessions: {} }
        await this.saveIndex()
      }

      this.isInitialized = true
      logger.debug(`SessionStore initialized at: ${this.storageDir}`)
    } catch (error) {
      logger.error('Failed to initialize SessionStore:', error as Error)
      throw error
    }
  }

  /**
   * 保存索引文件
   */
  private async saveIndex(): Promise<void> {
    if (!this.index) return
    await fs.writeFile(this.indexPath, JSON.stringify(this.index, null, 2))
  }

  /**
   * 生成会话文件路径
   */
  private getSessionFilePath(sessionId: string): string {
    return path.join(this.storageDir, `${sessionId}.json`)
  }

  /**
   * 创建新会话
   *
   * @param imagePath - 原始图像路径
   * @param imageDataUri - 图像的 Data URI
   * @param systemPrompt - 系统提示词
   * @param initialMessages - 初始消息（可选）
   * @returns 新会话的 ID
   */
  async create(
    imagePath: string,
    imageDataUri: string,
    systemPrompt: string,
    initialMessages: SessionMessage[] = []
  ): Promise<string> {
    await this.init()

    const sessionId = uuidv4()
    const now = new Date().toISOString()

    const session: Session = {
      id: sessionId,
      createdAt: now,
      updatedAt: now,
      imagePath,
      imageDataUri,
      systemPrompt,
      messages: initialMessages
    }

    // 保存会话文件
    const filePath = this.getSessionFilePath(sessionId)
    await fs.writeFile(filePath, JSON.stringify(session, null, 2))

    // 更新索引
    this.index!.sessions[sessionId] = {
      filePath,
      createdAt: now,
      updatedAt: now
    }
    await this.saveIndex()

    logger.debug(`Session created: ${sessionId}`)
    return sessionId
  }

  /**
   * 获取会话
   *
   * @param sessionId - 会话 ID
   * @returns 会话对象，如果不存在返回 null
   */
  async get(sessionId: string): Promise<Session | null> {
    await this.init()

    if (!this.index!.sessions[sessionId]) {
      return null
    }

    try {
      const filePath = this.getSessionFilePath(sessionId)
      const data = await fs.readFile(filePath, 'utf-8')
      return JSON.parse(data) as Session
    } catch (error) {
      logger.error(`Failed to read session ${sessionId}:`, error as Error)
      return null
    }
  }

  /**
   * 更新会话消息
   *
   * @param sessionId - 会话 ID
   * @param messages - 新的消息数组
   */
  async updateMessages(sessionId: string, messages: SessionMessage[]): Promise<void> {
    await this.init()

    const session = await this.get(sessionId)
    if (!session) {
      throw new Error(`Session not found: ${sessionId}`)
    }

    const now = new Date().toISOString()
    session.messages = messages
    session.updatedAt = now

    // 保存更新后的会话
    const filePath = this.getSessionFilePath(sessionId)
    await fs.writeFile(filePath, JSON.stringify(session, null, 2))

    // 更新索引中的时间戳
    this.index!.sessions[sessionId].updatedAt = now
    await this.saveIndex()

    logger.debug(`Session updated: ${sessionId}, messages: ${messages.length}`)
  }

  /**
   * 向会话添加消息
   *
   * @param sessionId - 会话 ID
   * @param message - 要添加的消息
   */
  async addMessage(sessionId: string, message: SessionMessage): Promise<void> {
    const session = await this.get(sessionId)
    if (!session) {
      throw new Error(`Session not found: ${sessionId}`)
    }

    session.messages.push(message)
    await this.updateMessages(sessionId, session.messages)
  }

  /**
   * 删除会话
   *
   * @param sessionId - 会话 ID
   */
  async delete(sessionId: string): Promise<void> {
    await this.init()

    if (!this.index!.sessions[sessionId]) {
      return
    }

    try {
      const filePath = this.getSessionFilePath(sessionId)
      await fs.unlink(filePath)
    } catch {
      // 文件可能已被删除，忽略错误
    }

    delete this.index!.sessions[sessionId]
    await this.saveIndex()

    logger.debug(`Session deleted: ${sessionId}`)
  }

  /**
   * 列出所有会话
   *
   * @returns 会话 ID 数组
   */
  async list(): Promise<string[]> {
    await this.init()
    return Object.keys(this.index!.sessions)
  }

  /**
   * 检查会话是否存在
   *
   * @param sessionId - 会话 ID
   * @returns 是否存在
   */
  async exists(sessionId: string): Promise<boolean> {
    await this.init()
    return !!this.index!.sessions[sessionId]
  }

  /**
   * 清理过期会话 (超过指定天数的会话)
   *
   * @param daysToKeep - 保留天数，默认 7 天
   */
  async cleanup(daysToKeep: number = 7): Promise<number> {
    await this.init()

    const cutoffDate = new Date()
    cutoffDate.setDate(cutoffDate.getDate() - daysToKeep)

    let deletedCount = 0

    for (const [sessionId, meta] of Object.entries(this.index!.sessions)) {
      const updatedAt = new Date(meta.updatedAt)
      if (updatedAt < cutoffDate) {
        await this.delete(sessionId)
        deletedCount++
      }
    }

    logger.info(`Cleaned up ${deletedCount} expired sessions`)
    return deletedCount
  }

  /**
   * 获取会话的消息历史（用于构建 LLM 请求）
   * 包含系统提示和用户/助手消息
   *
   * @param sessionId - 会话 ID
   * @returns 完整的消息数组
   */
  async getMessagesForLLM(sessionId: string): Promise<SessionMessage[]> {
    const session = await this.get(sessionId)
    if (!session) {
      throw new Error(`Session not found: ${sessionId}`)
    }

    // 构建消息数组：系统提示 + 历史消息
    const messages: SessionMessage[] = []

    // 添加系统提示
    if (session.systemPrompt) {
      messages.push({
        role: 'system',
        content: session.systemPrompt
      })
    }

    // 添加历史消息
    messages.push(...session.messages)

    return messages
  }
}

// 导出单例实例
export const sessionStore = new SessionStore()
