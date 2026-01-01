import CherryInProviderLogo from '@renderer/assets/images/providers/cherryin.png'
import type { AtLeast, Model, SystemProvider, SystemProviderId } from '@renderer/types'

import { glm45FlashModel, qwen38bModel } from './models/default'

export const SYSTEM_PROVIDERS_CONFIG: Record<SystemProviderId, SystemProvider> = {
  'maa-ai': {
    id: 'maa-ai',
    name: 'MaaAI',
    type: 'openai',
    apiKey: '',
    apiHost: 'https://open.maa-ai.com',
    anthropicApiHost: 'https://open.maa-ai.com',
    models: [glm45FlashModel, qwen38bModel] as Model[],
    isSystem: true,
    enabled: true
  }
} as const

export const SYSTEM_PROVIDERS: SystemProvider[] = Object.values(SYSTEM_PROVIDERS_CONFIG)

export const PROVIDER_LOGO_MAP: AtLeast<SystemProviderId, string> = {
  'maa-ai': CherryInProviderLogo
} as const

export function getProviderLogo(providerId: string) {
  return PROVIDER_LOGO_MAP[providerId as keyof typeof PROVIDER_LOGO_MAP]
}

// export const SUPPORTED_REANK_PROVIDERS = ['silicon', 'jina', 'voyageai', 'dashscope', 'aihubmix']
// export const NOT_SUPPORTED_RERANK_PROVIDERS = ['ollama', 'lmstudio'] as const satisfies SystemProviderId[]
export const NOT_SUPPORTED_RERANK_PROVIDERS = []
// export const ONLY_SUPPORTED_DIMENSION_PROVIDERS = ['ollama', 'infini'] as const satisfies SystemProviderId[]
export const ONLY_SUPPORTED_DIMENSION_PROVIDERS = []

type ProviderUrls = {
  api: {
    url: string
  }
  websites?: {
    official: string
    apiKey?: string
    docs: string
    models?: string
  }
}

export const PROVIDER_URLS: Record<SystemProviderId, ProviderUrls> = {
  'maa-ai': {
    api: {
      url: 'https://open.maa-ai.com'
    },
    websites: {
      official: 'https://open.maa-ai.com',
      apiKey: 'https://open.cherryin.ai/console/token',
      docs: 'https://open.cherryin.ai',
      models: 'https://open.cherryin.ai/pricing'
    }
  }
}
