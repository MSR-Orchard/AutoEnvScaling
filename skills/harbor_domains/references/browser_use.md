# Browser workflows

Generate a complete Harbor task with a reproducible website or application state and a goal
that requires browser interaction: navigation, search, forms, filtering, editing records and
multi-page decisions. You may generate application code, pages, assets, data and configuration
within the permitted runtime; you do not need to reproduce a benchmark task.

Use only the browser actions and observations the supplied runtime documents, for example
screenshot and coordinate actions or DOM and accessibility-tree actions. Do not assume
Playwright, a particular MCP server or the Chrome DevTools protocol is installed. Do not invent a
host URL or start public services. A local application server needs a documented startup and
readiness check, even when it shares a container with the browser; check for your own page's
content, not just an open port, since the base image may already serve something. Declaring an
MCP server in `task.toml` does not restrict the solver's tools.

`solution/solve.sh` must reach the goal through the assigned browser interface, not by editing
the database, calling a private application API or evaluating JavaScript to set the answer.
Setup and verification may inspect state through their own separately authorized interfaces.
Keep reference answers and private checks out of the served assets.

`tests/test.sh` should check the application's final state or the required artifacts. A
recorded browser trajectory is additional evidence of interaction, not the success test. Plan a
wrong-output challenge, and check that the browser interaction really causes the verified state
change.

Reset both the browser session (cookies, local storage, downloads) and the server-side state; a
new browser context alone does not reset a database or a remote account. Prefer pinned local
services. Live websites need explicit network permission and a reproducibility plan. Do not add
an LLM judge or external API just because a reference environment uses one. Do not submit a
terminal task in place of a browser task.
