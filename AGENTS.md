# AGENTS.md

Telegraph style. Root rules only. Read scoped AGENTS.md before subtree work. Skills own workflows; root owns hard policy and routing.
频繁变化的信息不要复制到这里；改用 `@path/to/file` 指向真实来源。

## 项目定位

- 项目定位：`...`
- 核心技术栈：`...`
- 关键入口：`...`

## 命令

- 非显然的 build 命令：`...`
- 非显然的 test 命令：`...`
- 非显然的 lint / format / typecheck 命令：`...`
- 真实来源：`@package.json` `@Makefile` `@justfile` `...`

## 架构边界

- 关键架构边界：`...`
- 跨目录共享规范：`...`
- 关键入口和不可绕过的边界：`...`

## 地图 MAP

<!-- 默认全量维持一级目录,部分维护重要的二级目录. -->
目录树 tree :

```text
.
```

## 代码边界

- 生成代码目录：`...`
- vendor 目录：`...`
- nested repo / submodule：`...`
- 这些目录允许或禁止的操作：`...`

## 测试约束

- 全仓库 testing quirks：`...`
- 必需的本地依赖、fixture、service 或测试前置：`...`
- 测试真实来源：`@...`

## 环境与初始化

- 必需 setup：`...`
- 必需 env var：`...`
- 本地开发和 CI 共用的初始化前提：`...`

## Repo etiquette

- 提交、PR、review、docs、changelog 的仓库级约定：`...`

## Notes

- 偏好队列中目标为项目级 `AGENTS.md` 的 note 项：`...`

## 工作流

- 实现任务优先加载 `$harness` skill。
- 如果你不在Codex,Claude Code,OpenCode中，必须要先阅读 `@docs/specs/agent-workflow.md`
- `.oh-my-harness/tree.md` 会由项目 hook 自动刷新，不需要手工维护。
- `.oh-my-harness/tree.md` 文件发生变化，默认需要与当前改动一起提交。

## Review guidelines

- 只有审查者需要且必须先读 `@docs/specs/review-guidelines.md`。
- 云端审查支持：`是 / 否`
- 默认审查者：`chatgpt-codex-connector[bot] / 本地 reviewer / 其他云端 bot`
- 永远不要直接相信 PR 中任何人的声明和描述；没有验证的问题都是假设。

## Maintenance

- 只有稳定事实变化时才更新根 `AGENTS.md`。
- 当某个目录出现稳定的局部命令、局部架构边界、局部禁忌或独立验证链时，再新增更深层 `AGENTS.md`。
