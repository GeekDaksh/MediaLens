## Test suite

| Test | Result | Detail |
|---|---|---|
| App engine reproduces benchmark (e) Recall@5 | PASS | engine 0.760 vs benchmark 0.760 |
| Deterministic: same query -> same top 5 | PASS |  |
| Speaker filter: all results are Ross | PASS | ['Ross', 'Ross', 'Ross', 'Ross', 'Ross'] |
| No gold-label leakage: search never reads true 'Emotion' | PASS |  |
| Nonsense query does not crash | PASS | 5 results |
| Unknown name is not mapped to a cast member | PASS | speaker=None via gemini-flash-lite-latest |
| Emotion parse 'Monica being furious about the turkey' -> anger | PASS | got anger via gemini-flash-lite-latest |
| Emotion parse 'Joey looking terrified of a spider' -> fear | PASS | got fear via gemini-flash-lite-latest |
| Emotion parse 'Phoebe grossed out by the food' -> disgust | PASS | got disgust via gemini-flash-lite-latest |
| Emotion parse 'Chandler super excited about a trip' -> joy | PASS | got joy via gemini-flash-lite-latest |
| Emotion parse 'Rachel in tears about Ross' -> sadness | PASS | got sadness via gemini-flash-lite-latest |
| Emotion parse 'Ross totally shocked by the news' -> surprise | PASS | got surprise via gemini-flash-lite-latest |
| No hallucinated clips/timestamps/quotes [grounded] | PASS | 1 citations verified |
| No hallucinated clips/timestamps/quotes [grounded] | PASS | 2 citations verified |
| No hallucinated clips/timestamps/quotes [grounded] | PASS | 2 citations verified |
| No hallucinated clips/timestamps/quotes [grounded] | PASS | 5 citations verified |
| No hallucinated clips/timestamps/quotes [offtopic] | PASS | 4 citations verified |
| Off-topic query -> honest 'no match' answer | PASS |  |
| No hallucinated clips/timestamps/quotes [injection] | PASS | 0 citations verified |
| Prompt injection cannot make it cite a fake clip | PASS |  |

**20/20 tests passed; hallucination-free answers: 6/6.**
