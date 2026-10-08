# Contributing

## Branches
There are exactly four branches: `main`, `prabodh`, `mirdul`, `anand`.

- Each person works only on their own branch.
- Only Prabodh merges into `main`.
- Before every push, run `git pull origin main --no-rebase`.

## Commits
Format: `type(scope): message`

- type: `feat`, `fix`, `docs`, `test`, `chore`
- scope: `ml`, `server`, `web`, `contract`, `demo`

Commit messages carry no tool or AI attribution lines.

## Never commit
`.db` files (except `contracts/fixtures/sample.db`), `.pt` files, `.env`, `node_modules`.

## Contracts
Contract changes are announced in the group chat first. Only Prabodh edits `contracts/`.
