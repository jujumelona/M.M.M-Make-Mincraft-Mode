# Repository Agent Policy

## Branch policy

- Work on `main` only.
- Never create a new branch for any reason, including temporary, feature, fix, review, automation, or recovery branches.
- Never switch to or write to any branch other than `main`.
- Do not open a pull request that requires creating a branch.
- Before any repository write, verify that the target ref is `main`.
- Before committing or pushing, refresh the latest `main` HEAD and preserve unrelated concurrent changes.
- Apply only the requested minimal patch on top of the latest `main`.
- If a non-`main` branch already exists, do not use it as a work target or source of new changes.

This policy is mandatory for every automated coding agent and repository maintenance action.
