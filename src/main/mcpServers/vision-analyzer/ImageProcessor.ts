/**
 * ImageProcessor - 图像处理模块
 *
 * 负责处理不同来源的图像，将其转换为 LLM 可识别的格式。
 * 目前支持本地路径读取，预留 URL 处理接口以便后期扩展。
 */

import { loggerService } from '@logger'
import { promises as fs } from 'fs'
import path from 'path'

const logger = loggerService.withContext('ImageProcessor')

/**
 * 图像处理结果类型
 */
export interface ProcessedImage {
  /** 图像的 Data URI 或 URL */
  url: string
  /** MIME 类型 */
  mimeType: string
  /** 来源类型 */
  sourceType: 'local' | 'url'
}

/**
 * 支持的图像 MIME 类型映射
 */
const MIME_TYPE_MAP: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp',
  '.svg': 'image/svg+xml'
}

/**
 * ImageProcessor 类
 * 提供图像处理的抽象层，支持本地文件和 URL 两种来源
 */
export class ImageProcessor {
  /**
   * 处理图像路径，返回可用于 LLM 的图像表示
   *
   * @param imagePath - 本地文件路径或 URL
   * @returns ProcessedImage 对象
   */
  async process(imagePath: string): Promise<ProcessedImage> {
    // 判断是 URL 还是本地路径
    if (this.isUrl(imagePath)) {
      return await this.processUrl(imagePath)
    } else {
      return await this.processLocalFile(imagePath)
    }
  }

  /**
   * 判断给定路径是否为 URL
   */
  private isUrl(path: string): boolean {
    return path.startsWith('http://') || path.startsWith('https://')
  }

  /**
   * 处理本地文件，将其转换为 Base64 Data URI
   *
   * @param filePath - 本地文件的绝对路径
   * @returns ProcessedImage 对象
   */
  private async processLocalFile(filePath: string): Promise<ProcessedImage> {
    logger.debug(`Processing local file: ${filePath}`)

    // 验证文件路径是否为绝对路径
    if (!path.isAbsolute(filePath)) {
      throw new Error(`Image path must be absolute: ${filePath}`)
    }

    // 检查文件是否存在
    try {
      await fs.access(filePath)
    } catch {
      throw new Error(`Image file not found: ${filePath}`)
    }

    // 获取文件扩展名并确定 MIME 类型
    const ext = path.extname(filePath).toLowerCase()
    const mimeType = MIME_TYPE_MAP[ext]

    if (!mimeType) {
      throw new Error(`Unsupported image format: ${ext}. Supported formats: ${Object.keys(MIME_TYPE_MAP).join(', ')}`)
    }

    // 读取文件并转换为 Base64
    const fileBuffer = await fs.readFile(filePath)
    const base64Data = fileBuffer.toString('base64')

    // 构建 Data URI
    const dataUri = `data:${mimeType};base64,${base64Data}`

    logger.debug(`Image processed successfully: ${filePath}, size: ${fileBuffer.length} bytes`)

    return {
      url: dataUri,
      mimeType,
      sourceType: 'local'
    }
  }

  /**
   * 处理 URL 图像
   *
   * TODO: 当前直接返回 URL，后期可扩展为：
   * 1. 验证 URL 可访问性
   * 2. 下载图像并上传到 OSS
   * 3. 返回 OSS URL 供 LLM 使用
   *
   * @param imageUrl - 图像的 HTTP/HTTPS URL
   * @returns ProcessedImage 对象
   */
  private async processUrl(imageUrl: string): Promise<ProcessedImage> {
    logger.debug(`Processing URL: ${imageUrl}`)

    // TODO: 后期实现 URL 处理逻辑
    // 1. 验证 URL 格式
    // 2. 可选：下载图像并上传到 OSS
    // 3. 返回处理后的 URL

    // 当前实现：直接返回原始 URL
    // 注意：部分 LLM 可能不支持直接使用外部 URL
    return {
      url: imageUrl,
      mimeType: 'image/*', // URL 场景下 MIME 类型可能未知
      sourceType: 'url'
    }
  }

  /**
   * 批量处理多张图像
   *
   * @param imagePaths - 图像路径数组
   * @returns ProcessedImage 对象数组
   */
  async processMultiple(imagePaths: string[]): Promise<ProcessedImage[]> {
    const results: ProcessedImage[] = []

    for (const imagePath of imagePaths) {
      const processed = await this.process(imagePath)
      results.push(processed)
    }

    return results
  }

  /**
   * 验证图像路径是否有效
   *
   * @param imagePath - 待验证的图像路径
   * @returns 验证结果
   */
  async validate(imagePath: string): Promise<{ valid: boolean; error?: string }> {
    try {
      if (this.isUrl(imagePath)) {
        // URL 基本格式验证
        new URL(imagePath)
        return { valid: true }
      } else {
        // 本地文件验证
        if (!path.isAbsolute(imagePath)) {
          return { valid: false, error: 'Path must be absolute' }
        }

        const ext = path.extname(imagePath).toLowerCase()
        if (!MIME_TYPE_MAP[ext]) {
          return { valid: false, error: `Unsupported format: ${ext}` }
        }

        await fs.access(imagePath)
        return { valid: true }
      }
    } catch (error) {
      return {
        valid: false,
        error: error instanceof Error ? error.message : 'Unknown validation error'
      }
    }
  }
}

// 导出单例实例
export const imageProcessor = new ImageProcessor()
