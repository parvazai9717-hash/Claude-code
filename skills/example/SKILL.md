---
name: example
description: Summarise the text files in a workspace directory into a written report.
activation:
  - the user asks for an overview or summary of files in the workspace
  - the user asks what a directory contains
required_tools:
  - list_files
  - read_file
  - search_files
allowed_paths:
  - files
  - projects
  - outputs
inputs:
  - a workspace-relative directory to summarise
outputs:
  - a short written summary in the reply
  - optionally, a report written to outputs/ (requires approval)
requires_approval: false
limitations:
  - read-only; it never modifies the files it summarises
  - skips binary files, hidden files and anything above the configured size limit
  - does not follow symbolic links or read outside the workspace
---

# Example skill: summarise a directory

A read-only workflow. It grants no permissions of its own — every step below runs
through the same tool registry, path policy and approval checks as any other action.

## Steps

1. `list_files` on the target directory to see what is there. Start with `depth: 1`
   and increase only if the structure is unclear.
2. `read_file` on the files that look relevant. Read before summarising; never
   describe a file you have not opened.
3. `search_files` when looking for a specific topic across many files rather than
   reading each one.
4. Write a short summary: what the directory contains, what stands out, and
   anything that looks incomplete or inconsistent.
5. If the user asked for the summary as a file, propose a `write_file` call to
   `outputs/`. That step requires approval, and the write must then be checked
   with `verify_result` before you report it as done.

## Reporting

State what you actually read. If a file was skipped because it was binary, hidden
or too large, say so rather than guessing at its contents.
