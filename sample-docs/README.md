# Sample documents

Test fixtures for exercising ingestion, retrieval and the knowledge graph.

They describe one fictional company — **Northwind Labs** — deliberately spread
across formats, so entities and relationships (people, teams, projects,
locations) recur between documents and the graph has real edges to build rather
than isolated islands. Questions like *"who owns Project Atlas?"* or *"who
approves production access?"* are answerable only by retrieving across them.

| File | Format | Exercises |
| --- | --- | --- |
| `employee-handbook.md` | Markdown | Heading-based structural chunking |
| `engineering-onboarding.md` | Markdown | Ordered lists, team/reporting relationships |
| `security-policy.txt` | Plain text | Numbered sections, no markdown to lean on |
| `product-roadmap.html` | HTML | Script/style/nav stripping |
| `headcount.csv` | CSV | Row-to-`Column: value` flattening |
| `notion-export.zip` | Archive | Nested members, page-id suffix stripping, skipped binary |

`notion-export.zip` also contains an unsupported `.png` member, which should be
skipped rather than failing the whole archive.

## Ingesting them

Exclude this README — it is documentation, not test data:

```bash
for f in sample-docs/*; do
  [ "$(basename "$f")" = "README.md" ] && continue
  curl -s -X POST http://localhost:8001/api/v1/documents/upload-and-index -F "file=@$f"
done
```

Remove them again with `POST /api/v1/documents/delete` and the returned ids.
