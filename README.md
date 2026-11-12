# kakie 的博客

西北工业大学计算机研究生在读，主要研究 LLM 与 Agent。这里记录大模型原理、学习笔记和项目实践。

[访问博客](https://kakie.site/) · [关于我](https://kakie.site/about/) · [GitHub](https://github.com/kakieccc)

博客使用 Astro 与 Fuwari开发。

## 本地开发

使用 Node.js 20 或更高版本，以及项目指定的 pnpm 9.14.4：

```bash
corepack pnpm install --frozen-lockfile
corepack pnpm dev
```

开发地址为 `http://localhost:4321/`。站点部署在根路径 `/` 下，站内链接使用 `src/utils/url-utils.ts` 中的 `url()` 处理基础路径。

| 命令 | 用途 |
| --- | --- |
| `corepack pnpm dev` | 启动开发服务器 |
| `corepack pnpm check` | 执行 Astro 检查 |
| `corepack pnpm build` | 构建静态页面和 Pagefind 搜索索引 |
| `corepack pnpm preview` | 预览构建结果，检查真实搜索功能 |
| `corepack pnpm new-post <filename>` | 创建文章 |

## 内容与配置

- `src/config.ts`：站点标题、简介、个人资料、导航、社交链接和图标。
- `src/content/spec/about.md`：自我介绍、研究方向、项目和个人链接。
- `src/content/posts/`：文章及其图片，保留现有目录和 URL。
- `src/pages/[...page].astro`：首页个人介绍和文章列表。
- `astro.config.mjs`：域名 `https://kakie.site/` 与部署路径 `/`。

## 主题与许可

主题基于 [Fuwari](https://github.com/saicaca/fuwari)，由 [Astro](https://astro.build) 构建。保留原项目的 MIT 许可与作者声明，详见 [LICENSE](LICENSE)。
