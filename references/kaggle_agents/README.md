# Kaggle agent references

Store downloaded high-scoring agent files here for offline inspection. Put each
submission in its own subfolder because downloads often share a filename such
as `main.py`:

```text
references/kaggle_agents/
  rank_01_<submission-or-author>/
    main.py
    NOTES.md
  rank_02_<submission-or-author>/
    main.py
    NOTES.md
```

In `NOTES.md`, record the leaderboard score/rank, download date, and any
available submission or author identifier. Preserve the downloaded source
before making adaptations. Keep our agent in the project root as `submission.py`.

These files are local and ignored by Git. First inspect each source and its
imports/entry point. We can then summarize its strategy and decide whether its
ideas or its code can be compared offline; compatibility with the local
simulator is not assumed.
