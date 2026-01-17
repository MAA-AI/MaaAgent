// import { loggerService } from '@logger'
import { createMigrate } from 'redux-persist'

/**
 * Redux Persist 迁移配置
 *
 * 用于在应用更新时迁移旧版本的 persisted state 到新版本结构
 *
 * 使用说明：
 * 1. 每次修改 store 结构时，需要添加新的迁移版本号
 * 2. 迁移函数接收旧的 state，返回新的 state
 * 3. 版本号需要递增
 *
 * 示例配置：
 * const migrateConfig = {
 *   1: (state: any) => {
 *     // 从版本 0 迁移到版本 1 的逻辑
 *     return {
 *       ...state,
 *       newField: defaultValue
 *     }
 *   },
 *   2: (state: any) => {
 *     // 从版本 1 迁移到版本 2 的逻辑
 *     return {
 *       ...state,
 *       nested: {
 *         ...state.nested,
 *         updatedField: state.nested.oldField
 *       }
 *     }
 *   }
 * }
 *
 * 注意：添加新迁移时，记得同时更新 persistReducer 的 version 参数
 * file://./index.ts
 */

// const logger = loggerService.withContext('Migrate')

/**
 * 迁移配置对象
 * key 为版本号，value 为迁移函数
 */
const migrateConfig = {}

const migrate = createMigrate(migrateConfig as any)

export default migrate
