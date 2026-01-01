import type { Model, SystemProviderId } from '@renderer/types'

export const glm45FlashModel: Model = {
  id: 'glm-4.5-flash',
  name: 'GLM-4.5-Flash',
  provider: 'maa-ai',
  group: 'GLM-4.5'
}

export const qwen38bModel: Model = {
  id: 'Qwen/Qwen3-8B',
  name: 'Qwen3-8B',
  provider: 'maa-ai',
  group: 'Qwen'
}

export const SYSTEM_MODELS: Record<SystemProviderId | 'defaultModel', Model[]> = {
  defaultModel: [
    // Default assistant model
    glm45FlashModel,
    // Default topic naming model
    qwen38bModel,
    // Default translation model
    glm45FlashModel,
    // Default quick assistant model
    glm45FlashModel
  ],
  'maa-ai': [glm45FlashModel, qwen38bModel]
}
