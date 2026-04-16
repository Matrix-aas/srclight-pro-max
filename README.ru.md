# Srclight

**AI-first code intelligence для fullstack-репозиториев.**

`Srclight` в этом репозитории — это не просто зеркало оригинала, а сильно прокачанный fork, заточенный под реальные боли современных веб-проектов: Vue, Nuxt, Nitro, NestJS, Drizzle, MikroORM, Mongoose, GraphQL, BullMQ, RabbitMQ, Redis и большие TypeScript/JavaScript кодовые базы, где обычный grep быстро превращается в унылую археологию.

Главная цель форка простая: чтобы и нейросети, и люди быстрее понимали проект, тратили меньше времени на слепые поиски по файлам и чаще попадали сразу в нужную точку кода.

[English version](README.md)

## Зачем вообще нужен этот fork

У оригинального `srclight` была хорошая база.

Но для реальной fullstack-разработки нам было нужно больше:

- лучшее понимание Vue / Nuxt / Nitro
- лучшее понимание NestJS: роутов, модулей, сервисов, резолверов, очередей, cron jobs, транспортов
- более полезная индексация Drizzle, Mongoose и MikroORM
- лучшая топология проекта для AI-агентов
- stdio-first MCP workflow для локальной работы
- более внятный install / upgrade flow
- меньше тупых фейлов и меньше токенов, сожжённых на ориентацию

Короче: меньше боли, меньше хаоса, меньше “а где вообще живёт эта фича?”

## В чём плюсы нашего форка

| Область | Оригинальный srclight | Наш fork |
|---|---|---|
| TS / JS fullstack | Базовое извлечение символов | Глубокое framework-aware понимание современного веб-стека |
| Vue / Nuxt / Nitro | Ограниченно | SFC, server routes, middleware, plugins, app surfaces |
| NestJS | Частично | Controllers, modules, services, resolvers, microservices, schedulers, queues |
| Data layer | В основном generic | Лучше понимает Drizzle, Mongoose, MikroORM |
| AI workflow | Неплохой MCP сервер | Заточен под orientation, ownership, flow tracing и экономию токенов |
| MCP transport | Больше упор на SSE | `stdio-first` для локальной агентной работы, SSE оставлен как advanced path |
| Установка | Обычная | Нормальный upgrade path и защита от старого сломанного install |
| Онбординг | Generic docs | README и docs как у настоящего production-инструмента |

## Что эта штука реально делает

Srclight строит локальный intelligence layer поверх репозитория:

- парсит код через tree-sitter
- извлекает символы и framework-aware metadata
- строит локальные SQLite FTS5 индексы
- собирает callers / callees / dependents / impact
- добавляет semantic и hybrid search через embeddings
- отдаёт всё это через MCP, чтобы агент мог задавать более умные вопросы

То есть вместо двадцати бессмысленных `rg`-проходов и ручного чтения кучи файлов агент получает нормальные инструменты:

- `codebase_map()`
- `list_files()`
- `get_file_summary()`
- `api_surface()`
- `search_symbols()`
- `hybrid_search()`
- `get_symbol()`
- `get_signature()`
- `get_callers()`
- `get_callees()`
- `get_dependents()`
- `get_community()`
- `get_communities()`
- `get_execution_flows()`
- `detect_changes()`
- `recent_changes()`
- `git_hotspots()`

Для навигации по файлам обычно работает такой low-token путь: `list_files()` находит кандидатов, `get_file_summary()` быстро показывает суть файла, а `symbols_in_file()` даёт outline перед открытием секций. Когда нужно быстро понять backend surface, `api_surface()` отдаёт компактный список endpoints без ручного grep по контроллерам и роутерам. Графовые инструменты тоже сначала дают компактную сводку; `verbose=true` включай только когда реально нужен подробный состав сообщества или пошаговый trace. `get_community()` умеет file-aware fallback, так что даже miss может подсказать ближайший symbol или полезный file candidate.

## Почему этот fork крутой именно для AI

Без `srclight` агент обычно тратит время на:

- повторяющиеся поисковые проходы по репо
- случайные чтения файлов ради понимания структуры
- пропущенные async-цепочки и скрытые связи
- гадание, как связаны роуты, сервисы, очереди, стораджи и база

С `srclight` получается:

- быстрее понять топологию проекта
- быстрее находить входные точки фич
- точнее строить flow изменений
- реже жечь лимиты на бессмысленный поиск
- чаще попадать в правильный файл с первой попытки

Для AI-агентов это не косметика. Это буквально разница между “ориентируюсь” и “брожу по подвалу с фонариком”.

## Поддержка fullstack-стека

| Поверхность | Что умеет fork |
|---|---|
| Vue | SFC signals, template/script/style-aware indexing, component-level orientation |
| Nuxt / Nitro | route files, handlers, middleware, plugins |
| NestJS | controllers, routes, modules, services, resolvers, guards, pipes, filters, interceptors |
| Async systems | message patterns, event patterns, queues, workers, cron / interval / timeout jobs |
| ORM / DB | Drizzle tables and DB clients, Mongoose schemas/entities, MikroORM entities/repos/config |
| Search | keyword, semantic, hybrid, graph-based traversal |
| Orientation | topology, routes, ownership hints, execution flow context |

## Установка

### Самый простой путь

```bash
curl -fsSL https://raw.githubusercontent.com/Matrix-aas/srclight-pro-max/main/scripts/install.sh | bash
```

Installer:

- ставит через `pipx`
- умеет нормально обновляться
- ловит старый сломанный `0.15.x` install
- говорит, что именно удалить, если у тебя на машине ещё живёт тот древний вариант

### Ручная установка

Рекомендуемый вариант:

```bash
pipx install --force 'git+https://github.com/Matrix-aas/srclight-pro-max.git@main'
```

Из исходников:

```bash
git clone https://github.com/Matrix-aas/srclight-pro-max.git
cd srclight-pro-max
python3 -m pip install -e '.[dev]'
```

С доп. зависимостями:

```bash
python3 -m pip install -e '.[docs,pdf]'
python3 -m pip install -e '.[docs,pdf,ocr]'
python3 -m pip install -e '.[all]'
```

## Если у тебя стоит старый srclight

Если раньше ты делал:

```bash
pipx install srclight
```

и у тебя поселился старый `0.15.x`, сначала снести его к чёрту:

```bash
pipx uninstall srclight
pipx install --force 'git+https://github.com/Matrix-aas/srclight-pro-max.git@main'
```

Не надо пытаться смешивать старую ветку и этот fork в одной установке. Это плохая ветка реальности.

## Быстрый старт

```bash
# проиндексировать репо
cd /path/to/project
srclight index

# проиндексировать с embeddings
srclight index --embed

# поиск через CLI
srclight search "auth"
srclight symbols app/stores/auth.store.ts

# поднять MCP сервер для локальных агентов
srclight serve --transport stdio
```

Для workspace:

```bash
srclight workspace init fullstack
srclight workspace add /path/to/repo-a -w fullstack
srclight workspace add /path/to/repo-b -w fullstack
srclight workspace index -w fullstack --embed
srclight serve --workspace fullstack --transport stdio
```

## Embeddings

Рекомендуемая модель по умолчанию:

```bash
ollama pull qwen3-embedding:4b
srclight index --embed
```

Альтернатива:

```bash
ollama pull nomic-embed-text-v2-moe
srclight index --embed ollama:nomic-embed-text-v2-moe
```

В этом fork `--embed` по умолчанию использует `ollama:qwen3-embedding:4b`.

## MCP setup

### Claude Code

```bash
# single repo
claude mcp add srclight -- srclight serve --transport stdio

# workspace
claude mcp add srclight -- srclight serve --workspace fullstack --transport stdio
```

### Cursor

```json
{
  "mcpServers": {
    "srclight": {
      "command": "srclight",
      "args": ["serve", "--workspace", "fullstack", "--transport", "stdio"]
    }
  }
}
```

В этом fork основной локальный режим — `stdio`. `SSE` никуда не делся, но нужен только если ты осознанно хочешь long-lived shared server.

## Фокус развития

Этот fork намеренно оптимизирован под те репозитории, где AI-инструменты обычно сильнее всего тупят и теряют время:

- TypeScript / JavaScript
- Vue / Nuxt / Nitro
- NestJS
- GraphQL + REST
- Drizzle / MikroORM / Mongoose
- BullMQ / RabbitMQ / Redis

Более широкий language coverage из upstream остаётся, но основной упор здесь — на то, чтобы агенты хорошо понимали современные fullstack-системы end-to-end.

## Документация

- [Usage guide](docs/usage-guide.md)
- [English README](README.md)
- [Cursor MCP example](docs/cursor-mcp-example.json)
- [Releasing notes](docs/releasing.md)

## Лицензия

MIT.
